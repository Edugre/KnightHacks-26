"""Deterministic renderer: Timeline -> censored video or photo. The only code that writes pixels.

Rules (IMPLEMENTATION.md section 3), all fail-closed:
- interpolate boxes between keyframes up to MAX_GAP frames apart;
- after a target's track is lost, keep masking for HOLD_FRAMES: the box keeps moving at the target's last
  speed and grows HOLD_GROWTH per frame;
- start masking LEAD_FRAMES before the first keyframe (detectors that check every few frames see an object
  up to that late), moving back at the first speed;
- 'uncertain' keyframes double the padding; every box gets at least MIN_PAD_PX on each side;
- an error while processing a frame blurs the whole frame and is counted in the stats.

A photo is a one-frame timeline (censor/core/video.py): render_image masks frame 0 with the same FrameMasker.

CLI: python -m censor.render.render input.mp4|input.jpg timeline.json out.mp4|out.jpg
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

from censor.core.timeline import Target, Timeline
from censor.core.video import frames, is_image, read_image

HOLD_FRAMES = 15
# Growth only covers what the speed estimate misses. 0.05 per frame, without following the motion, let a
# small plate slide out of its mask and still padded big plates by 75% a side (bench/review.py).
HOLD_GROWTH = 0.02
LEAD_FRAMES = 3
MAX_GAP = 15
MIN_PAD_PX = 3  # a 15x6 px plate at 720p moves a few pixels between checks; relative padding alone is 1 px
SPEED_KEYFRAMES = 4  # keyframes the speed for lead-in and hold is measured over
MAX_SPEED = 0.25     # box sizes per frame; a 13 px plate crossing the frame moved about 0.2
FFMPEG = os.environ.get("CENSOR_FFMPEG", "ffmpeg")

Box = tuple[float, float, float, float]


@dataclass
class RenderStats:
    frames: int = 0
    masked_boxes: int = 0
    failed_frames: list[int] = field(default_factory=list)


def _lerp(a: Box, b: Box, t: float) -> Box:
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))  # type: ignore[return-value]


def _speed(kfs: list, end: bool) -> Box:
    """Per-frame movement of the box at one end of a track (the last keyframes for end=True, the first ones
    otherwise): the center's speed over up to SPEED_KEYFRAMES keyframes, at most MAX_SPEED box sizes a frame,
    and no change in size. Two noisy boxes alone once sent a held face box flying across half the frame."""
    ks = kfs[-SPEED_KEYFRAMES:] if end else kfs[:SPEED_KEYFRAMES]
    a, b = ks[0], ks[-1]
    gap = b.f - a.f
    if gap <= 0 or gap > MAX_GAP * (len(ks) - 1):
        return (0.0, 0.0, 0.0, 0.0)
    vx = ((b.box[0] + b.box[2]) - (a.box[0] + a.box[2])) / 2 / gap
    vy = ((b.box[1] + b.box[3]) - (a.box[1] + a.box[3])) / 2 / gap
    ref = b.box if end else a.box
    cap = MAX_SPEED * max(ref[2] - ref[0], ref[3] - ref[1])
    vx, vy = max(-cap, min(cap, vx)), max(-cap, min(cap, vy))
    return (vx, vy, vx, vy)


def _moved(box: Box, v: Box, k: int) -> Box:
    return tuple(box[i] + v[i] * k for i in range(4))  # type: ignore[return-value]


def expand_target(target: Target, n_frames: int) -> dict[int, tuple[Box, float]]:
    """Per-frame (box, padding) for one target, with interpolation, lead-in and hold-after-loss applied."""
    out: dict[int, tuple[Box, float]] = {}
    kfs = sorted(target.keyframes, key=lambda k: k.f)
    if not kfs:
        return out

    def pad_for(state: str) -> float:
        return target.padding * (2.0 if state == "uncertain" else 1.0)

    def hold(i: int, limit: int):
        after = kfs[i]
        v = _speed(kfs[:i + 1], end=True)
        for k in range(1, HOLD_FRAMES + 1):
            f = after.f + k
            if f >= limit or f >= n_frames:
                break
            out[f] = (_moved(after.box, v, k), pad_for(after.state) + HOLD_GROWTH * k)

    first = kfs[0]
    v0 = _speed(kfs, end=False)
    for k in range(1, LEAD_FRAMES + 1):
        if first.f - k >= 0:
            out[first.f - k] = (_moved(first.box, v0, -k), pad_for(first.state) + HOLD_GROWTH * k)

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
                hold(i, nxt.f)
        else:
            hold(i, n_frames)
    return out


def _padded(box: Box, pad: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = max(bw * pad, MIN_PAD_PX), max(bh * pad, MIN_PAD_PX)
    x1, x2 = x1 - px, x2 + px
    y1, y2 = y1 - py, y2 + py
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
    else:  # blur: downscale hard so nothing legible survives, smooth at that size, scale back up
        # Smoothing the small image instead of the full-size one is ~80x cheaper per box (a 151 px kernel on a
        # padded face took 17 ms; a crowd's export took 193 s) and leaves no streaks of the original.
        sw, sh = max(1, w // 24), max(1, h // 24)
        small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
        k = max(3, (min(sw, sh) // 2) | 1)
        roi[:] = cv2.resize(cv2.GaussianBlur(small, (k, k), 0), (w, h), interpolation=cv2.INTER_LINEAR)


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


def render_image(src: str, timeline: Timeline, dst: str, jpeg_quality: int = 95) -> RenderStats:
    """Mask a photo and write it in dst's format. cv2.imwrite writes pixels only, so the export carries none of
    the source's metadata (EXIF GPS position, camera serial, capture time); the EXIF rotation is already in the
    pixels, since read_image applied it."""
    frame = read_image(src)
    stats = RenderStats(frames=1)
    FrameMasker(timeline).apply(frame, 0, stats)
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if str(dst).lower().endswith((".jpg", ".jpeg")) else []
    if not cv2.imwrite(str(dst), frame, params):
        raise RuntimeError(f"cannot write image {dst}")
    return stats


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    render = render_image if is_image(sys.argv[1]) else render_video
    s = render(sys.argv[1], Timeline.load(sys.argv[2]), sys.argv[3])
    print(f"rendered {s.frames} frames, {s.masked_boxes} masked boxes, {len(s.failed_frames)} failed frames")
