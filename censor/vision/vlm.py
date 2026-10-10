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


@dataclass
class AttributeAnswer:
    track_id: str
    answer: str  # "yes" | "no" | "unknown"
    conf: float


def track_crops(video: str, track: Track, n: int = CROPS_PER_TRACK) -> list[np.ndarray]:
    kfs = track.keyframes
    picks = [kfs[int(i)] for i in np.linspace(0, len(kfs) - 1, num=min(n, len(kfs)))]
    crops = []
    for kf in picks:
        frame = read_frame(video, kf.f)
        x1, y1, x2, y2 = (int(v) for v in kf.box)
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size:
            crops.append(crop)
    return crops


def _ask_server(crops: list[np.ndarray], question: str) -> tuple[str, float]:
    imgs = [base64.b64encode(cv2.imencode(".jpg", c)[1].tobytes()).decode() for c in crops]
    body = json.dumps({"images": imgs, "question": question}).encode()
    req = urllib.request.Request(f"{MODEL_URL}/vlm/yesno", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.load(r)
    return out["answer"], float(out.get("conf", 0.5))


def check_attribute(video: str, tracks: list[Track], question: str) -> list[AttributeAnswer]:
    answers = []
    for t in tracks:
        if not MODEL_URL:
            answers.append(AttributeAnswer(t.track_id, "unknown", 0.0))
            continue
        try:
            ans, conf = _ask_server(track_crops(video, t), question)
        except Exception:
            ans, conf = "unknown", 0.0
        answers.append(AttributeAnswer(t.track_id, ans, conf))
    return answers
