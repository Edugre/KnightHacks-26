"""Deterministic renderer: Timeline -> censored video. The only code that writes pixels.

Rules (IMPLEMENTATION.md section 3), all fail-closed:
- interpolate boxes between keyframes up to MAX_GAP frames apart;
- after a target's track is lost, hold the last box for HOLD_FRAMES, growing 5% per frame;
- 'uncertain' keyframes double the padding;
- an error while processing a frame blurs the whole frame and is counted in the stats.

CLI: python -m censor.render.render input.mp4 timeline.json out.mp4
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

from censor.core.timeline import Target, Timeline
from censor.core.video import frames

HOLD_FRAMES = 15
HOLD_GROWTH = 0.05
MAX_GAP = 15
FFMPEG = os.environ.get("CENSOR_FFMPEG", "ffmpeg")

Box = tuple[float, float, float, float]


@dataclass
class RenderStats:
    frames: int = 0
    masked_boxes: int = 0
    failed_frames: list[int] = field(default_factory=list)


def _lerp(a: Box, b: Box, t: float) -> Box:
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))  # type: ignore[return-value]


def expand_target(target: Target, n_frames: int) -> dict[int, tuple[Box, float]]:
    """Per-frame (box, padding) for one target, with interpolation and hold-after-loss applied."""
    out: dict[int, tuple[Box, float]] = {}
    kfs = sorted(target.keyframes, key=lambda k: k.f)
    if not kfs:
        return out

    def pad_for(state: str) -> float:
        return target.padding * (2.0 if state == "uncertain" else 1.0)

    def hold(after, start: int, limit: int):
        for k in range(1, HOLD_FRAMES + 1):
            f = start + k
            if f >= limit or f >= n_frames:
                break
            out[f] = (after.box, pad_for(after.state) + HOLD_GROWTH * k)

    for i, kf in enumerate(kfs):
        out[kf.f] = (kf.box, pad_for(kf.state))
        if i + 1 < len(kfs):
            nxt = kfs[i + 1]
            gap = nxt.f - kf.f
            if gap <= MAX_GAP:
                state = "uncertain" if "uncertain" in (kf.state, nxt.state) else kf.state
                for f in range(kf.f + 1, nxt.f):
                    out[f] = (_lerp(kf.box, nxt.box, (f - kf.f) / gap), pad_for(state))
            else:
                hold(kf, kf.f, nxt.f)
        else:
            hold(kf, kf.f, n_frames)
    return out


def _padded(box: Box, pad: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - bw * pad, x2 + bw * pad
    y1, y2 = y1 - bh * pad, y2 + bh * pad
    return (max(0, int(x1)), max(0, int(y1)), min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2))))


def apply_effect(frame: np.ndarray, rect: tuple[int, int, int, int], effect: str) -> None:
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    if effect == "box":
        roi[:] = 0
    elif effect == "pixelate":
        small = cv2.resize(roi, (max(1, w // 12), max(1, h // 12)), interpolation=cv2.INTER_LINEAR)
        roi[:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:  # blur: downscale hard then Gaussian, so nothing legible survives
        small = cv2.resize(roi, (max(1, w // 24), max(1, h // 24)), interpolation=cv2.INTER_AREA)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        k = max(3, (min(w, h) // 2) | 1)
        roi[:] = cv2.GaussianBlur(up, (k, k), 0)


def full_frame_blur(frame: np.ndarray) -> None:
    apply_effect(frame, (0, 0, frame.shape[1], frame.shape[0]), "blur")


class FrameMasker:
    """Precomputes per-frame boxes for every target; apply() masks one frame in place."""

    def __init__(self, timeline: Timeline):
        self.tl = timeline
        n = timeline.video.n_frames
        self.per_frame: dict[int, list[tuple[Box, float, str]]] = {}
        for target in timeline.targets.values():
            for f, (box, pad) in expand_target(target, n).items():
                self.per_frame.setdefault(f, []).append((box, pad, target.effect))

    def apply(self, frame: np.ndarray, f: int, stats: RenderStats | None = None) -> np.ndarray:
        h, w = frame.shape[:2]
        try:
            for box, pad, effect in self.per_frame.get(f, []):
                apply_effect(frame, _padded(box, pad, w, h), effect)
                if stats:
                    stats.masked_boxes += 1
        except Exception:  # fail closed
            full_frame_blur(frame)
            if stats:
                stats.failed_frames.append(f)
        return frame


def render_video(src: str, timeline: Timeline, dst: str, crf: int = 20) -> RenderStats:
    v = timeline.video
    masker = FrameMasker(timeline)
    stats = RenderStats()
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{v.width}x{v.height}", "-r", f"{v.fps}", "-i", "-",
           "-i", str(src), "-map", "0:v", "-map", "1:a?",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-shortest", str(dst)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f, frame in frames(src):
            masker.apply(frame, f, stats)
            proc.stdin.write(frame.tobytes())
            stats.frames += 1
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with {rc}")
    return stats


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    s = render_video(sys.argv[1], Timeline.load(sys.argv[2]), sys.argv[3])
    print(f"rendered {s.frames} frames, {s.masked_boxes} masked boxes, {len(s.failed_frames)} failed frames")
