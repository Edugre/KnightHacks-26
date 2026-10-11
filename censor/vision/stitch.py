"""Join pieces of one person's track that the tracker split apart.

YOLOE + BoT-SORT loses a person for a few frames (blur, occlusion, a turn) and starts a new id when it finds
them again: on street_walk, 15 person tracks for about 3 people in focus. Each piece costs one VLM check per
request and gets its own answer, so one person could end up part blurred and part not.

A piece B continues piece A when B starts soon after A ends (or overlaps A's end by a few frames and sits on
top of it), where A was heading, at about A's size, and with similar colors. Joining two different people is
worse than leaving a piece: one VLM answer would then apply to both. So every test has to pass, and each piece
is joined at most once in each direction, best match first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from censor.core.tracks import Keyframe

MAX_GAP = 30          # frames between A's end and B's start (1.2 s at 25 fps)
MAX_OVERLAP = 5       # frames B may start before A ends, if it sits on top of A there
MAX_JUMP = 0.75      # B's first center vs where A was heading, in A's box heights (0.5 -> 0.75 added one correct join on street_walk; 1.0 added nothing more)
SIZE_RATIO = 1.4      # box heights within this factor of each other
MAX_COLOR_DIST = 0.35  # Bhattacharyya distance between color histograms (0 same, 1 nothing in common)
SPEED_FRAMES = 10     # frames at A's end used for its speed


def color_hist(img: np.ndarray, box) -> np.ndarray:
    """Hue-saturation histogram of the middle of a box: the torso and legs, less background at the edges."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    xa, xb = int(max(0, x1 + 0.2 * w)), int(min(img.shape[1], x2 - 0.2 * w))
    ya, yb = int(max(0, y1 + 0.1 * h)), int(min(img.shape[0], y2 - 0.1 * h))
    if xb <= xa or yb <= ya:
        return np.zeros((8 * 4,), np.float32)
    hsv = cv2.cvtColor(img[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).flatten()
    return (hist / max(hist.sum(), 1e-6)).astype(np.float32)


@dataclass
class Ends:
    """What the stitcher needs from each piece: mean color at its start and at its end (or None)."""
    start: Optional[np.ndarray] = None
    end: Optional[np.ndarray] = None


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def link_cost(a: list[Keyframe], b: list[Keyframe], ea: Ends, eb: Ends) -> Optional[float]:
    """How well B continues A (lower is better), or None if any test fails."""
    gap = b[0].f - a[-1].f
    if b[0].f <= a[0].f or gap > MAX_GAP or gap < -MAX_OVERLAP:
        return None
    last, first = a[-1], b[0]
    ha, hb = last.box[3] - last.box[1], first.box[3] - first.box[1]
    if ha <= 0 or hb <= 0 or max(ha, hb) / min(ha, hb) > SIZE_RATIO:
        return None
    if gap <= 0:  # B starts while A still runs: only a duplicate of the same person, sitting on top of it
        at = {k.f: k.box for k in a}
        shared = [_iou(at[k.f], k.box) for k in b if k.f in at]
        if not shared or min(shared) < 0.5:
            return None
        jump = 0.0
    else:
        tail = [k for k in a if k.f >= last.f - SPEED_FRAMES]
        span = max(1, tail[-1].f - tail[0].f)
        (x0, y0), (x1, y1) = _center(tail[0].box), _center(last.box)
        px, py = x1 + (x1 - x0) / span * gap, y1 + (y1 - y0) / span * gap
        bx, by = _center(first.box)
        jump = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5 / ha
        if jump > MAX_JUMP:
            return None
    color = 0.0
    if ea.end is not None and eb.start is not None:
        color = float(cv2.compareHist(ea.end, eb.start, cv2.HISTCMP_BHATTACHARYYA))
        if color > MAX_COLOR_DIST:
            return None
    return jump + color + max(gap, 0) / MAX_GAP


def stitch(tracks: Sequence[list[Keyframe]], ends: Optional[Sequence[Ends]] = None) -> list[list[Keyframe]]:
    """Join pieces that pass every test in link_cost, best match first, each piece joined at most once at each
    end. Returns the joined tracks, keyframes in frame order (where pieces overlap, A's boxes are kept)."""
    ends = list(ends) if ends is not None else [Ends() for _ in tracks]
    pairs = []
    for i, a in enumerate(tracks):
        for j, b in enumerate(tracks):
            if i != j and a and b:
                c = link_cost(a, b, ends[i], ends[j])
                if c is not None:
                    pairs.append((c, i, j))
    nxt: dict[int, int] = {}
    prv: dict[int, int] = {}
    for _, i, j in sorted(pairs):
        if i in nxt or j in prv or _same_chain(i, j, nxt, prv):
            continue
        nxt[i], prv[j] = j, i
    out = []
    for i in range(len(tracks)):
        if i in prv or not tracks[i]:
            continue
        chain, k = list(tracks[i]), i
        while k in nxt:
            k = nxt[k]
            seen = {kf.f for kf in chain}
            chain += [kf for kf in tracks[k] if kf.f not in seen]
        out.append(sorted(chain, key=lambda kf: kf.f))
    return out


def _same_chain(i: int, j: int, nxt: dict, prv: dict) -> bool:
    """Joining i -> j would close a loop (j already leads to i)."""
    k = j
    while k in nxt:
        k = nxt[k]
        if k == i:
            return True
    return False
