"""Where the scan time goes, on GPU and on CPU. Runs on the R9700; prints a table and writes JSON.

  python3 -m bench.profile_scan bench/clips/street_plates.mp4 [--out logs/profile.json] [--cpu-frames 6]

Measures: decoding alone, EgoBlur face and plate per frame (and whether the models take a batch), YOLOE per
frame, the same models on the CPU, and the scan the app runs today (face, plate, person; one pass each),
with GPU busy % sampled from sysfs while it runs.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from censor.core.tracks import TrackStore  # noqa: E402
from censor.core.video import frames, probe  # noqa: E402


class GpuBusy:
    """Samples amdgpu's gpu_busy_percent every 50 ms in a thread (None when sysfs doesn't expose it)."""

    def __init__(self):
        self.paths = glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")
        self.samples: list[int] = []
        self._stop = threading.Event()

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                vals = []
                for p in self.paths:
                    try:
                        vals.append(int(Path(p).read_text()))
                    except (OSError, ValueError):
                        pass
                if vals:
                    self.samples.append(max(vals))
                time.sleep(0.05)
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    @property
    def mean(self):
        return round(statistics.mean(self.samples), 1) if self.samples else None


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t


def per_call(fn, inputs, sync) -> float:
    """Median seconds per call after one warm-up call."""
    fn(inputs[0])
    sync()
    times = []
    for x in inputs:
        t = time.perf_counter()
        fn(x)
        sync()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video")
    ap.add_argument("--out", default="logs/profile.json")
    ap.add_argument("--gpu-frames", type=int, default=20)
    ap.add_argument("--cpu-frames", type=int, default=6)
    ap.add_argument("--skip-scan", action="store_true", help="skip the full app scan (slowest part)")
    a = ap.parse_args()

    import torch
    from censor.vision import specialist as sp
    info = probe(a.video)
    r: dict = {"video": a.video, "frames": info.n_frames, "size": f"{info.width}x{info.height}", "fps": info.fps,
               "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

    # 1. decoding alone (OpenCV, CPU)
    _, secs = timed(lambda: sum(1 for _ in frames(a.video)))
    r["decode_fps"] = round(info.n_frames / secs, 1)

    sample = [img for f, img in frames(a.video) if f % 15 == 0][: max(a.gpu_frames, a.cpu_frames)]

    # 2. EgoBlur on the GPU, one frame per call, plus a batch attempt
    gpu = sp.load_egoblur(device="cuda")
    r["egoblur_gpu_s"] = {c: round(per_call(gpu[c], sample[: a.gpu_frames], sync), 4) for c in gpu}
    model = torch.jit.load(str(sp.EGOBLUR_DIR / sp.EGOBLUR_FILES["face"]), map_location="cpu").to("cuda").eval()
    t1 = torch.from_numpy(np.ascontiguousarray(sample[0].transpose(2, 0, 1))).to("cuda")
    try:
        with torch.no_grad():
            model(torch.stack([t1, t1]))
        r["egoblur_batch"] = "accepted a batch of 2"
    except Exception as e:  # noqa: BLE001
        r["egoblur_batch"] = f"no batch: {type(e).__name__}: {str(e).splitlines()[0][:160]}"
    # how much of a call is the upload of the frame
    host = torch.from_numpy(np.ascontiguousarray(sample[0].transpose(2, 0, 1)))
    r["upload_frame_s"] = round(per_call(lambda x: x.to("cuda"), [host] * 10, sync), 5)
    with torch.no_grad():
        r["egoblur_face_model_only_s"] = round(per_call(lambda x: model(x), [t1] * 10, sync), 4)

    # 3. EgoBlur on the CPU
    cpu = sp.load_egoblur(device="cpu")
    r["egoblur_cpu_s"] = {c: round(per_call(cpu[c], sample[: a.cpu_frames], lambda: None), 3) for c in cpu}

    # 4. YOLOE per frame, GPU and CPU
    from ultralytics import YOLOE
    from censor.vision.detect import DETECT_IMGSZ, MIN_CONF
    y = YOLOE("yoloe-11s-seg.pt")
    y.set_classes(["person"], y.get_text_pe(["person"]))
    for dev, n, key in (("cuda:0", a.gpu_frames, "yoloe_gpu_s"), ("cpu", a.cpu_frames, "yoloe_cpu_s")):
        r[key] = round(per_call(lambda im: y.predict(im, device=dev, imgsz=DETECT_IMGSZ, conf=MIN_CONF, verbose=False),
                                sample[:n], sync), 4)
    try:
        _, s16 = timed(lambda: y.predict(sample[:16], device="cuda:0", imgsz=DETECT_IMGSZ, conf=MIN_CONF, verbose=False))
        r["yoloe_gpu_batch16_s_per_frame"] = round(s16 / 16, 4)
    except Exception as e:  # noqa: BLE001
        r["yoloe_gpu_batch16_s_per_frame"] = f"failed: {type(e).__name__}"

    # 5. the scan the app runs on upload today: one full pass per query
    if not a.skip_scan:
        from censor.vision.detect import get_detector
        det = get_detector()
        scan = {}
        with GpuBusy() as busy:
            for q in ("face", "license plate", "person"):
                _, secs = timed(det.run, a.video, [q], TrackStore())
                scan[q] = round(secs, 1)
        scan["total"] = round(sum(scan.values()), 1)
        scan["x_realtime"] = round((info.n_frames / info.fps) / scan["total"], 2)
        scan["gpu_busy_mean_pct"] = busy.mean
        r["scan_today"] = scan

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
