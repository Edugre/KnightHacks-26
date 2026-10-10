"""Draw detector tracks on chosen frames, to see by eye what a detector change keeps or drops.

  python3 scripts/overlay_tracks.py clip.mp4 out.jpg --frames 0 60 140 --tracks before.json [after.json]

Each tracks file is a TrackStore JSON (python -m censor.vision.detect clip.mp4 "license plate" > tracks.json).
Boxes from the first file are red, from the second green; a track is drawn on a frame when it has a keyframe
within 2 frames of it. The frames are stacked into one image, two per row.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from censor.core.video import read_frame  # noqa: E402

COLORS = [(0, 0, 255), (0, 200, 0)]  # BGR: first file red, second green


def boxes_at(tracks: dict, f: int, near: int = 2) -> list[tuple]:
    out = []
    for t in tracks["tracks"].values():
        k = min(t["keyframes"], key=lambda k: abs(k["f"] - f))
        if abs(k["f"] - f) <= near:
            out.append((k["box"], k["conf"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--tracks", nargs="+", required=True, help="one or two TrackStore JSON files")
    a = ap.parse_args()
    stores = [json.loads(Path(p).read_text()) for p in a.tracks[:2]]
    tiles = []
    for f in a.frames:
        img = read_frame(a.video, f)
        for i, store in enumerate(stores):
            for (x1, y1, x2, y2), conf in boxes_at(store, f):
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[i], 4 - 2 * i)
                cv2.putText(img, f"{conf:.2f}", (int(x1), int(y1) - 6 - 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[i], 2)
        cv2.putText(img, f"frame {f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        tiles.append(cv2.resize(img, (960, int(960 * img.shape[0] / img.shape[1]))))
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
