"""Attribute checks on track crops ("is this person wearing a red shirt?").

The real check runs on the shared Qwen3-VL model served by `censor.server.model_server`
(set CENSOR_MODEL_URL, e.g. http://127.0.0.1:8100). Without it, a stub answers "unknown",
and the agent treats "unknown" as a match: fail closed, more blur, never less.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

from censor.core.tracks import Track
from censor.core.video import read_frame

MODEL_URL = os.environ.get("CENSOR_MODEL_URL")
CROPS_PER_TRACK = 3
CROP_CONTEXT = 0.15   # widen each crop by this much per side, so clothes and hair at the edge of the box show
CROP_MIN_SIDE = 224   # upscale small crops: a 40 px tall figure is too few pixels for the VLM to judge


@dataclass
class AttributeAnswer:
    track_id: str
    answer: str  # "yes" | "no" | "unknown"
    conf: float
    p_yes: float = 0.5


def track_crops(video: str, track: Track, n: int = CROPS_PER_TRACK) -> list[np.ndarray]:
    """The clearest views of a track: the largest boxes, taken from different thirds of it so a turn of the head
    or a coat coming off is seen, each with some context and upscaled when small. Evenly spaced picks often
    landed on frames where the person was small, cut off or blurred."""
    kfs = track.keyframes
    thirds = [kfs[len(kfs) * i // n:len(kfs) * (i + 1) // n] or kfs for i in range(min(n, len(kfs)))]
    area = lambda k: (k.box[2] - k.box[0]) * (k.box[3] - k.box[1])  # noqa: E731
    crops = []
    for part in thirds:
        kf = max(part, key=area)
        frame = read_frame(video, kf.f)
        x1, y1, x2, y2 = kf.box
        dx, dy = (x2 - x1) * CROP_CONTEXT, (y2 - y1) * CROP_CONTEXT
        x1, y1 = int(max(0, x1 - dx)), int(max(0, y1 - dy))
        x2, y2 = int(min(frame.shape[1], x2 + dx)), int(min(frame.shape[0], y2 + dy))
        crop = frame[y1:y2, x1:x2]
        if not crop.size:
            continue
        short = min(crop.shape[:2])
        if short < CROP_MIN_SIDE:
            s = CROP_MIN_SIDE / short
            crop = cv2.resize(crop, (round(crop.shape[1] * s), round(crop.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
        crops.append(crop)
    return crops


def _ask_server(crops: list[np.ndarray], question: str) -> tuple[str, float, float]:
    imgs = [base64.b64encode(cv2.imencode(".jpg", c)[1].tobytes()).decode() for c in crops]
    body = json.dumps({"images": imgs, "question": question}).encode()
    req = urllib.request.Request(f"{MODEL_URL}/vlm/yesno", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.load(r)
    return out["answer"], float(out.get("conf", 0.5)), float(out.get("p_yes", 0.5))


def check_attribute(video: str, tracks: list[Track], question: str) -> list[AttributeAnswer]:
    answers = []
    for t in tracks:
        if not MODEL_URL:
            answers.append(AttributeAnswer(t.track_id, "unknown", 0.0))
            continue
        try:
            ans, conf, p_yes = _ask_server(track_crops(video, t), question)
        except Exception:
            ans, conf, p_yes = "unknown", 0.0, 0.5
        answers.append(AttributeAnswer(t.track_id, ans, conf, p_yes))
    return answers
