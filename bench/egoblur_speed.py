"""Ways to make one EgoBlur call cheaper on the R9700, measured on real frames. EgoBlur is 87% of the upload
scan and the GPU is ~100% busy during it, so the per-call cost is what matters (bench/profile_scan.py).

  python3 -m bench.egoblur_speed bench/clips/street_plates.mp4

Tries: MIOpen autotuning (torch.backends.cudnn.benchmark), fp16 autocast, smaller input frames, and the
face and plate models at the same time on two streams. Prints seconds per frame and how far the boxes move
from the fp32 full-size result (mean IoU of matched boxes, and boxes lost), so a speedup that changes the
detections shows up here before it reaches the benchmark.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from censor.core.video import frames  # noqa: E402
from censor.vision import specialist as sp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="logs/egoblur_speed.json")
    a = ap.parse_args()

    import cv2
    import torch
    import torchvision
    imgs = [img for f, img in frames(a.video) if f % 25 == 0][: a.n]
    models = {c: torch.jit.load(str(sp.EGOBLUR_DIR / n), map_location="cpu").to("cuda").eval()
              for c, n in sp.EGOBLUR_FILES.items()}

    def call(cat, img, scale=1.0, amp=False, stream=None):
        x = img if scale == 1.0 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).to("cuda", non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            boxes, _, scores, _ = models[cat](t)
        boxes, scores = boxes.float(), scores.float()
        keep = torchvision.ops.nms(boxes, scores, 0.3)
        keep = keep[scores[keep] >= sp.EGOBLUR_CONF]
        return (boxes[keep] / scale).tolist(), scores[keep].tolist()

    def timed(fn, reps=None):
        fn(imgs[0])  # warm-up (and the autotuner's search, when on)
        torch.cuda.synchronize()
        ts = []
        for img in imgs:
            t = time.perf_counter()
            fn(img)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        return round(statistics.median(ts), 4)

    def agreement(ref, out):
        """Mean IoU of each reference box with its best match, and how many reference boxes have none >= 0.5."""
        ious, lost = [], 0
        for (rb, _), (ob, _) in zip(ref, out):
            for b in rb:
                best = max((sp._iou(tuple(b), tuple(o)) for o in ob), default=0.0)
                ious.append(best)
                lost += best < 0.5
        return {"mean_iou": round(statistics.mean(ious), 3) if ious else None, "lost": lost, "ref_boxes": len(ious)}

    r: dict = {"frames": len(imgs), "size": f"{imgs[0].shape[1]}x{imgs[0].shape[0]}"}
    ref = {c: [call(c, im) for im in imgs] for c in models}
    variants = {"fp32": {}, "fp16": {"amp": True}, "scale 0.75": {"scale": 0.75}, "scale 0.5": {"scale": 0.5},
                "fp16 + scale 0.75": {"amp": True, "scale": 0.75}}

    def run_all(label):
        out = {}
        for name, kw in variants.items():
            row = {}
            for c in models:
                try:
                    row[c] = {"s": timed(lambda im: call(c, im, **kw)),
                              **agreement(ref[c], [call(c, im, **kw) for im in imgs])}
                except Exception as e:  # noqa: BLE001
                    row[c] = f"failed: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
            out[name] = row
        r[label] = out

    torch.backends.cudnn.benchmark = False
    run_all("default")
    torch.backends.cudnn.benchmark = True  # MIOpen searches for the fastest kernels per input shape
    run_all("autotuned")

    # both models at once: one thread and one stream each, versus one after the other
    def both(img, parallel):
        if not parallel:
            for c in models:
                call(c, img)
            return
        streams = {c: torch.cuda.Stream() for c in models}

        def work(c):
            with torch.cuda.stream(streams[c]):
                call(c, img)
        ts = [threading.Thread(target=work, args=(c,)) for c in models]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    r["face+plate per frame"] = {"one after the other": timed(lambda im: both(im, False)),
                                 "two streams": timed(lambda im: both(im, True))}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
