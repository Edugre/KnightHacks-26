"""Compare open-vocabulary detectors on people: how many person tracks each gives per clip, and how fragmented.

  python -m bench.compare_people bench/clips/street_walk.mp4 [more clips] [--models yoloe-11s-seg.pt yoloe-26s-seg.pt]

Runs each model the way the app does (Ultralytics track + BoT-SORT, text prompt "person", CENSOR_IMGSZ, conf
CENSOR_MIN_CONF), in fp32 and with half=True, and reports per clip: tracks, tracks shorter than 1 s (the scraps
that cost a VLM check each and get inconsistent answers), tracks covering half the clip or more, detections, and
seconds. Writes a crop sheet per model and clip to data/review/people/, one tile per track (its biggest box), so
merged or split people can be checked by eye. Runs on any CUDA GPU or the CPU (DETECTOR_OPTIONS.md, items 2 and 5).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from censor.core.video import probe, read_frame  # noqa: E402
from censor.vision.detect import DETECT_IMGSZ, MIN_CONF  # noqa: E402

OUT = ROOT / "data" / "review" / "people"


def run(model_name: str, clip: str, half: bool, device, conf: float = MIN_CONF, tracker: str = "botsort.yaml") -> dict:
    from ultralytics import YOLOE
    model = YOLOE(model_name)
    model.set_classes(["person"], model.get_text_pe(["person"]))
    tracks: dict[int, list[tuple[int, list[float], float]]] = {}
    untracked = 0
    t = time.perf_counter()
    for f, r in enumerate(model.track(source=clip, stream=True, persist=True, device=device, imgsz=DETECT_IMGSZ,
                                      conf=conf, half=half, verbose=False, tracker=tracker)):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        ids = r.boxes.id.tolist() if r.boxes.id is not None else [None] * len(r.boxes)
        for box, c, tid in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), ids):
            if tid is None:
                untracked += 1  # the app keeps these as one-frame tracks (fail closed)
            else:
                tracks.setdefault(int(tid), []).append((f, box, float(c)))
    secs = time.perf_counter() - t
    info = probe(clip)
    lengths = [(kf[-1][0] - kf[0][0] + 1) / info.fps for kf in tracks.values()]
    return {"model": model_name, "tracker": Path(tracker).stem, "half": half, "conf": conf, "seconds": round(secs, 1),
            "tracks": len(tracks) + untracked, "untracked_boxes": untracked,
            "under_1s": sum(1 for x in lengths if x < 1.0) + untracked,
            "half_clip_or_more": sum(1 for x in lengths if x >= 0.5 * info.n_frames / info.fps),
            "detections": sum(len(v) for v in tracks.values()) + untracked, "_tracks": tracks}


def sheet(clip: str, res: dict, out: Path) -> None:
    tiles = []
    for tid, kfs in sorted(res["_tracks"].items(), key=lambda kv: kv[1][0][0]):
        # start, middle and end of the track side by side: a track that changes person halfway (an id switch)
        # shows as different people in one tile
        tile = np.zeros((190, 3 * 74, 3), np.uint8)
        for n, (f, box, conf) in enumerate((kfs[0], kfs[len(kfs) // 2], kfs[-1])):
            x1, y1, x2, y2 = (int(max(0, v)) for v in box)
            crop = read_frame(clip, f)[y1:y2, x1:x2]
            if crop.size:
                s = min(72 / crop.shape[1], 160 / crop.shape[0])
                c = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
                tile[:c.shape[0], 74 * n:74 * n + c.shape[1]] = c
        cv2.putText(tile, f"#{tid} n{len(kfs)}  {kfs[0][0]}-{kfs[-1][0]}", (2, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(tile)
    if not tiles:
        return
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--models", nargs="+", default=["yoloe-11s-seg.pt", "yoloe-26s-seg.pt"])
    ap.add_argument("--no-half", action="store_true", help="fp32 only")
    ap.add_argument("--tracker", default="botsort.yaml", help="tracker config, e.g. bench/trackers/botsort_reid90.yaml")
    ap.add_argument("--save", action="store_true", help="also write each run's tracks to data/review/people/<clip>_<model>.json "
                                                        "(frame, box, conf per track id), for stitching experiments")
    a = ap.parse_args()
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip in a.clips:
        for m in a.models:
            for half in ((False,) if a.no_half or device == "cpu" else (False, True)):
                try:
                    res = run(m, clip, half, device, tracker=a.tracker)
                except Exception as e:  # noqa: BLE001  a model that won't load is a result too
                    rows.append({"clip": Path(clip).stem, "model": m, "half": half, "error": f"{type(e).__name__}: {e}"})
                    print(rows[-1], file=sys.stderr)
                    continue
                name = f"{Path(clip).stem}_{Path(m).stem}{'_half' if half else ''}" +                     (f"_{Path(a.tracker).stem}" if a.tracker != "botsort.yaml" else "")
                sheet(clip, res, OUT / f"{name}.jpg")
                if a.save:
                    (OUT / f"{name}.json").write_text(json.dumps({str(k): v for k, v in res["_tracks"].items()}))
                row = {"clip": Path(clip).stem, **{k: v for k, v in res.items() if not k.startswith("_")}}
                rows.append(row)
                print(json.dumps(row))
    (OUT / "compare.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {OUT / 'compare.json'} and crop sheets")


if __name__ == "__main__":
    main()
