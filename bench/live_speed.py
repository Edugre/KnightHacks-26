"""Could the real detector keep up with a livestream? Per-frame cost at stream sizes, tail included, and a paced run
where frames arrive on the clock whether or not the detector is done.

  python3 -m bench.live_speed bench/clips/street_plates.mp4 [--heights 720 1080] [--fps 30] [--seconds 20]

Per frame size (frames from the clip, resized, aspect kept):
1. One detection pass as a live loop would run it: EgoBlur face, EgoBlur plate and YOLOE person on the same frame,
   one after the other, each synced. p50 / p95 / p99 / max, since a stream misses its deadline on the slow frames,
   not the median ones.
2. A paced run: a feeder thread puts a frame into a one-frame slot every 1/fps s (the latest frame wins, older ones
   are dropped), and the detector loop always takes the newest. Counts the frames that got detections, the gap
   between them against the renderer's HOLD_FRAMES (a longer gap and a lost object's mask runs out before the next
   detection), and how old a frame is when its detections are ready.
From these, the plan: detect every N-th frame (N from the p95 pass) and hold the output back by a delay buffer long
enough that a frame's next detection and the renderer's lead-in exist before it goes out:
(N + LEAD_FRAMES) frames plus a p99 pass. An estimate to size a design with, not a guarantee.

Runs on the R9700 (EgoBlur models and Ultralytics). Frames are decoded before timing (decoding runs at ~1,450 fps,
IMPLEMENTATION.md section 9), so this is the GPU's side only. Writes logs/live_speed.json.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from censor.core.video import frames, probe  # noqa: E402
from censor.render.render import HOLD_FRAMES, LEAD_FRAMES  # noqa: E402

MODELS = ("face", "license plate", "person")


def stats_ms(secs: list[float]) -> dict:
    xs = sorted(secs)

    def pct(p: float) -> float:
        return xs[min(len(xs) - 1, math.ceil(p / 100 * len(xs)) - 1)]
    return {"p50": round(statistics.median(xs) * 1000, 1), "p95": round(pct(95) * 1000, 1),
            "p99": round(pct(99) * 1000, 1), "max": round(xs[-1] * 1000, 1)}


def plan(pass_secs: list[float], fps: float) -> dict:
    """Detection stride and delay buffer for a stream at `fps`, from the per-frame cost of one detection pass."""
    s = stats_ms(pass_secs)
    budget = 1000 / fps
    stride = max(1, math.ceil(s["p95"] / budget))
    return {"budget_ms": round(budget, 1), "every_n_frames": stride, "detections_per_s": round(1000 / s["p50"], 1),
            "keeps_up_every_frame": stride == 1, "delay_ms": round((stride + LEAD_FRAMES) * budget + s["p99"])}


def paced(detect: Callable[[np.ndarray], object], pool: list[np.ndarray], fps: float, seconds: float) -> dict:
    """Feed `pool` (cycled) at `fps` for `seconds` into a one-frame slot; `detect` always takes the newest frame."""
    n = int(fps * seconds)
    cv = threading.Condition()
    slot: list = [None]  # (index, published at) of the newest frame
    done = [False]

    def feed():
        t0 = time.perf_counter()
        for i in range(n):
            time.sleep(max(0.0, t0 + i / fps - time.perf_counter()))
            with cv:
                slot[0] = (i, time.perf_counter())
                cv.notify()
        with cv:
            done[0] = True
            cv.notify()

    feeder = threading.Thread(target=feed, daemon=True, name="feed")
    feeder.start()
    seen, stale, last = [], [], -1
    while True:
        with cv:
            cv.wait_for(lambda: done[0] or (slot[0] and slot[0][0] > last))
            if not (slot[0] and slot[0][0] > last):
                break
            i, at = slot[0]
        detect(pool[i % len(pool)])
        stale.append(time.perf_counter() - at)
        seen.append(i)
        last = i
    feeder.join()
    gaps = [b - a for a, b in zip(seen, seen[1:])] or [n]
    return {"frames": n, "detected": len(seen), "detected_share": round(len(seen) / n, 3),
            "gap_frames": {"mean": round(statistics.mean(gaps), 2), "max": max(gaps)},
            "max_gap_within_hold": max(gaps) <= HOLD_FRAMES, "staleness_ms": stats_ms(stale)}


def load_models(models: tuple[str, ...]) -> dict[str, Callable[[np.ndarray], object]]:
    """One call per model on a BGR frame, set up the way the app's scan runs them (EgoBlur fp16, YOLOE settings
    from censor/vision/detect.py)."""
    out: dict[str, Callable[[np.ndarray], object]] = {}
    if {"face", "license plate"} & set(models):
        from censor.vision import specialist as sp
        eg = sp.load_egoblur()
        out.update({c: eg[c] for c in ("face", "license plate") if c in models})
    if "person" in models:
        from ultralytics import YOLOE
        from censor.vision.detect import DETECT_IMGSZ, MIN_CONF
        y = YOLOE("yoloe-11s-seg.pt")
        y.set_classes(["person"], y.get_text_pe(["person"]))
        out["person"] = lambda img: y.predict(img, device=0, imgsz=DETECT_IMGSZ, conf=MIN_CONF, verbose=False)
    return out


def frame_pool(video: str, height: int, n: int) -> list[np.ndarray]:
    import cv2
    info = probe(video)
    step = max(1, info.n_frames // n)
    width = round(info.width * height / info.height / 2) * 2
    return [cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            for f, img in frames(video, every=step)][:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video")
    ap.add_argument("--heights", type=int, nargs="+", default=[720, 1080])
    ap.add_argument("--fps", type=float, default=30.0, help="stream frame rate to plan for")
    ap.add_argument("--frames", type=int, default=60, help="frames timed per size in part 1")
    ap.add_argument("--seconds", type=float, default=20.0, help="length of the paced run per size")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--out", default="logs/live_speed.json")
    a = ap.parse_args()

    import torch
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    models = load_models(tuple(a.models))

    def detect(img: np.ndarray) -> dict[str, float]:
        took = {}
        for c, fn in models.items():
            t = time.perf_counter()
            fn(img)
            sync()
            took[c] = time.perf_counter() - t
        return took

    r: dict = {"video": a.video, "fps": a.fps, "models": a.models, "lead_frames": LEAD_FRAMES,
               "hold_frames": HOLD_FRAMES, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "sizes": {}}
    for h in a.heights:
        pool = frame_pool(a.video, h, a.frames)
        for img in pool[:3]:  # a freshly loaded model is slow on its first calls at each size
            detect(img)
        runs = [detect(img) for img in pool]
        total = [sum(t.values()) for t in runs]
        size = f"{pool[0].shape[1]}x{h}"
        r["sizes"][size] = {"pass_ms": stats_ms(total), **{f"{c}_ms": stats_ms([t[c] for t in runs]) for c in models},
                            "plan": plan(total, a.fps), "paced": paced(detect, pool, a.fps, a.seconds)}
        print(f"[live] {size} done", file=sys.stderr)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(f"\n{'size':>10} {'pass p50/p95/p99 ms':>22} {'detect every':>13} {'delay ms':>9} "
          f"{'paced: detected':>16} {'max gap':>8} {'stale p95 ms':>13}")
    for size, s in r["sizes"].items():
        p, pl, pc = s["pass_ms"], s["plan"], s["paced"]
        gap = f"{pc['gap_frames']['max']}{'' if pc['max_gap_within_hold'] else '!'}"
        print(f"{size:>10} {p['p50']:>8} / {p['p95']} / {p['p99']:<6} {pl['every_n_frames']:>9} fr "
              f"{pl['delay_ms']:>9} {pc['detected_share']:>15.0%} {gap:>8} {pc['staleness_ms']['p95']:>13}")
    print(f"\nat {a.fps:g} fps; '!' = a gap longer than HOLD_FRAMES ({HOLD_FRAMES}). Full results: {a.out}")


if __name__ == "__main__":
    main()
