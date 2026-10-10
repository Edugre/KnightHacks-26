"""Synthesize a test clip with known ground truth, so everything can be tested without real footage or a GPU.

Writes <out>.mp4 plus <out>.mp4.tracks.json (ground truth TrackStore, which the StubDetector replays):
- a "license plate" (white box with black digits) moving left to right,
- a "face" (skin-colored ellipse) bobbing in place,
- two "person" boxes: one red, one blue (for attribute requests),
- a "screen" that appears only in the second half.

Usage: python scripts/make_test_clip.py bench/clips/synthetic.mp4 [seconds]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from censor.core.tracks import Keyframe, Track, TrackStore  # noqa: E402

W, H, FPS = 1280, 720, 30


def main(out: str, seconds: float = 6.0):
    n = int(seconds * FPS)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(Path(out).with_suffix(".raw.mp4"))
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    gt = {k: [] for k in ("plate", "face", "red", "blue", "screen")}
    for f in range(n):
        img = np.full((H, W, 3), (60, 90, 60), np.uint8)
        cv2.rectangle(img, (0, 500), (W, H), (80, 80, 80), -1)  # road
        # plate
        x = int(50 + (W - 300) * f / n)
        plate = (x, 560, x + 160, 600)
        cv2.rectangle(img, plate[:2], plate[2:], (255, 255, 255), -1)
        cv2.putText(img, "ABC 1234", (x + 8, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        gt["plate"].append(plate)
        # face
        cy = int(150 + 20 * np.sin(f / 8))
        cv2.ellipse(img, (900, cy), (45, 60), 0, 0, 360, (140, 170, 220), -1)
        cv2.circle(img, (885, cy - 10), 6, (0, 0, 0), -1); cv2.circle(img, (915, cy - 10), 6, (0, 0, 0), -1)
        gt["face"].append((855, cy - 60, 945, cy + 60))
        # people
        rx = int(300 + 100 * np.sin(f / 20))
        cv2.rectangle(img, (rx, 250), (rx + 90, 480), (40, 40, 220), -1)  # red (BGR)
        gt["red"].append((rx, 250, rx + 90, 480))
        cv2.rectangle(img, (650, 260), (740, 480), (220, 80, 40), -1)  # blue
        gt["blue"].append((650, 260, 740, 480))
        # screen, second half only
        if f >= n // 2:
            cv2.rectangle(img, (1050, 300), (1230, 420), (30, 30, 30), -1)
            cv2.putText(img, "PASSWORD", (1060, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            gt["screen"].append((1050, 300, 1230, 420))
        vw.write(img)
    vw.release()

    # re-encode to H.264 so browsers can play it
    import subprocess, os
    ff = os.environ.get("CENSOR_FFMPEG", "ffmpeg")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", out], check=True)
    Path(tmp).unlink()

    store = TrackStore()
    def add(label, boxes, first=0):
        kfs = [Keyframe(f=first + i, box=b, conf=0.9, state="detected") for i, b in enumerate(boxes)]
        store.add(Track(track_id=store.next_id(), label=label, keyframes=kfs))
    add("license plate", gt["plate"]); add("face", gt["face"])
    add("person", gt["red"]); add("person", gt["blue"])
    add("screen", gt["screen"], first=n // 2)
    store.save(out + ".tracks.json")
    print(f"wrote {out} ({n} frames) and {out}.tracks.json ({len(store.tracks)} tracks)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 6.0)
