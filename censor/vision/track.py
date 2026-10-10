"""Draw-a-box tracking: a user-drawn box on one frame becomes a Track.

Uses OpenCV's MIL tracker forward from the drawn frame (CPU, no extra models). If the tracker
loses the target, the remaining frames fall back to the renderer's hold-after-loss rule.
Role A can swap this for a stronger tracker (e.g. SAM2-style propagation) behind the same function.
"""
from __future__ import annotations

import cv2

from censor.core.tracks import Keyframe, Track, TrackStore
from censor.core.video import frames

MAX_TRACK_FRAMES = 30 * 60  # cap at ~1 minute at 30 fps


def track_box(video: str, frame_index: int, box: tuple[float, float, float, float],
              store: TrackStore, label: str = "user box") -> Track:
    x1, y1, x2, y2 = box
    kfs = [Keyframe(f=frame_index, box=box, conf=1.0, state="detected")]
    tracker = None
    for f, frame in frames(video):
        if f < frame_index:
            continue
        if f == frame_index:
            tracker = cv2.TrackerMIL_create()
            tracker.init(frame, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            continue
        if f - frame_index > MAX_TRACK_FRAMES:
            break
        ok, (x, y, w, h) = tracker.update(frame)
        if not ok:
            break
        kfs.append(Keyframe(f=f, box=(x, y, x + w, y + h), conf=0.5, state="tracked"))
    t = Track(track_id=store.next_id(), label=label, source="user_box", keyframes=kfs)
    store.add(t)
    return t
