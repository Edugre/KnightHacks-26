"""Live replay engine: play a recorded clip as if it were a stream and censor it behind a short delay buffer.

One standing rule ("blur every face and license plate"), no request per object. Three threads:
- source: decodes the clip at its own fps (looping) into a ring of frames at the live size;
- detector: always runs on the newest ring frame (latest wins), so it runs back to back and skips frames;
- emitter: sends frame f out at t_arrive(f) + delay, masked from the detections around f.

Inside the delay window the detections *after* f are known, so f is masked the way the offline export would mask
it: link() the detections into tracks, then expand_target() (interpolation, lead-in, hold-after-loss) and
apply_effect() (censor/render/render.py). The emitter never waits for the detector. A frame is "vouched" only when
a detection exists at or before f and one at or after f, at most HOLD_FRAMES + 1 apart: with a longer gap a box
seen only at the earlier one runs out of hold before the later one (frames a+16..b-4 would be bare). An unvouched
frame is "blind": the last vouched frame goes out again with a DETECTOR LATE badge (nothing new is shown), and
before the first vouched frame the whole frame is blurred. A late detector therefore shows as a frozen picture,
never as a bare face. Any error while masking blurs the whole frame. A thread that dies stops the engine.

Camera / push mode (LiveReplay(..., push=True)): the source is the caller, not a clip. push(jpeg) decodes an upload and
puts it in the ring; the detector and emitter threads are unchanged and pace on each frame's arrival time. Frame
numbers follow elapsed time at a nominal 30 fps (f = max(last f + 1, round(elapsed * 30))), not the arrival count.
The vouch rule and HOLD_FRAMES are in frame numbers, so a Wi-Fi stall followed by a burst must not be numbered
consecutively: two detections 2 s apart in real time would sit 1 frame apart, and a box would be interpolated across
the stall, i.e. masked in the wrong place. Numbered by time, the stall is a ~60-frame gap that the vouch rule
refuses to bridge (frozen picture, DETECTOR LATE). The emitter takes the first slot at or after the next number;
numbers that never arrived count as gaps_in, frames lost from a full ring as dropped_out. Push mode also drops the
big-faint-face rule (face_rule=None): that rule is for car backs in street footage; a webcam scene has none and every
face there is "big", so it would turn a dim or turned face into "vouched, no box", i.e. bare and without a badge.

Only the shape half of the scan's confirmation rules runs here (faces: confirmed(); plates: plate_ok()), because the
history half needs the whole clip. The face rules are judged on the keyframes inside the window (WINDOW_BACK +
HOLD_FRAMES frames), so a big face whose only confident detections are older than the window could be dropped by the
big-faint-face rule; on the labeled clips every real big face scored 1.00 on every detection, so this has not been
seen, and it is listed as an open edge in docs/LIVE_REPLAY_PLAN.md. No FastAPI in this module;
censor/live_cli.py and censor/server/live_routes.py drive it.
"""
from __future__ import annotations

import inspect
import os
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from bench.live_speed import stats_ms
from censor.core.timeline import Target
from censor.core.tracks import Keyframe, TrackStore
from censor.core.video import probe
from censor.render.render import (HOLD_FRAMES, LEAD_FRAMES, Box, _padded, apply_effect, expand_target,
                                  full_frame_blur)
from censor.vision.specialist import (CONFIRM, EGOBLUR_CONF, FACE_EXPAND, LINK_GAP, Infer, _expand, _iou, _same,
                                      confirmed, link, plate_shaped)

Dets = dict[str, list[tuple[Box, float]]]        # category -> [(xyxy, score)]
Detect = Callable[[int, np.ndarray], Dets]       # (clip frame index, BGR frame at live size) -> Dets
Rect = tuple[int, int, int, int]
Embed = Callable[[np.ndarray, Box], "np.ndarray | None"]  # (BGR frame at live size, RAW face box) -> unit-norm vector | None
# on_emit(f, clip_f, blind, rects) or on_emit(f, clip_f, blind, rects, spared_rects): which one is called is decided once,
# from the callback's signature (see _takes_spared), so a 4-parameter callback such as the CLI's Recorder keeps working.
OnEmit = Callable[..., None]

LIVE_CATS = ("face", "license plate")
LIVE_DELAY_S = float(os.environ.get("CENSOR_LIVE_DELAY", "1.0"))
LIVE_HEIGHT = int(os.environ.get("CENSOR_LIVE_HEIGHT", "720"))
LIVE_OUT_W = int(os.environ.get("CENSOR_LIVE_OUT_W", "960"))
JPEG_QUALITY = 75
WINDOW_BACK = 2 * HOLD_FRAMES  # detections this far before f still matter to the tracks that reach f
UNCERTAIN_BELOW = 0.35
PADDING = 0.15
RING_SLACK_S = 0.5     # ring depth beyond the delay: room for a slow emit before frames start to drop
HISTORY = 4            # recent JPEGs kept for latest()
BASELINE_PASSES, RECENT_PASSES, SAMPLES = 50, 100, 300
BADGE_TEXT = "DETECTOR LATE"
PUSH_FPS = 30.0        # nominal rate of a pushed stream: frame numbers follow elapsed time at this rate; sizes the ring
INPUT_FPS_WINDOW_S = 2.0
CAMERA_CLIP = "camera"
YUNET_SCORE, YUNET_NMS, YUNET_TOP_K = 0.6, 0.3, 50
SPARE_COS = float(os.environ.get("CENSOR_SPARE_COS", "0.5"))  # stricter than OpenCV's documented 0.363 for SFace
SPARE_MIN_KF = 3        # a face track needs this many detections in the window before any bracket of it can be spared
SPARE_OVERLAP_IOU = 0.1 # an unspared face this close to a spared one at a bracket end means the two may have crossed
EMBED_MAX = 6           # per detector pass: only the largest faces are embedded; the rest are never spared
EMBED_MIN_H = 48        # raw face height (px at live size) below which a face is not embedded (never spared)
ENROLL_PASSES, ENROLL_PAIR_COS, ENROLL_MIN_H_FRAC = 3, 0.6, 0.08
SFACE_MARGIN = 0.30     # crop margin around the raw box handed to YuNet for the alignment landmarks
PLAIN_MARGIN = 0.10
SFACE_SIZE = 112
SFACE_URL = "https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")


@dataclass(frozen=True)
class Detection:
    f: int           # stream frame number (keeps counting across loops)
    dets: Dets       # may lack a category: use dets.get(cat, []); an empty Detection still vouches
    took_s: float
    # per category, same order as dets[cat]; only "face" ever holds True (that face matched the enrolled identity)
    spared: dict[str, tuple[bool, ...]] = field(default_factory=dict)
    gen: int = 0     # the engine's enrollment generation when this pass started; a stale one counts as unspared


@dataclass(frozen=True)
class _Slot:
    f: int
    clip_f: int
    frame: np.ndarray  # the detector's input: copy before drawing on it
    t_arrive: float


@dataclass(frozen=True)
class _Out:
    image: np.ndarray  # display size
    rects: list[Rect]
    blind: bool
    vouched: bool      # a good picture to freeze on later
    spared_rects: list[Rect] = field(default_factory=list)  # faces left unmasked because they are the enrolled person


# ---- masking rules (pure) -----------------------------------------------------------------------------------

def plate_ok(kfs: list[Keyframe], frame_w: int, frame_h: int) -> bool:
    """The shape kind of the scan's plate rules. confirmed() returns True before any shape check once min_conf is
    left out, so this repeats them: a plate cut off by the frame edge or tiny (a few pixels of noise) is kept,
    anything else must look like a plate. EgoBlur's false "plates" here are square stickers and signs."""
    rule = CONFIRM["license plate"]
    if any(k.box[0] < 3 or k.box[1] < 3 or k.box[2] > frame_w - 3 or k.box[3] > frame_h - 3 for k in kfs):
        return True
    widths = sorted(k.box[2] - k.box[0] for k in kfs)
    heights = sorted(k.box[3] - k.box[1] for k in kfs)
    w, h = widths[len(widths) // 2], heights[len(heights) // 2]
    return h < rule["shape_min_height"] or plate_shaped((0.0, 0.0, w, h), frame_h)


def _vouched(f: int, ready: list[Detection], hold: int) -> bool:
    before = [d.f for d in ready if d.f <= f]
    after = [d.f for d in ready if d.f >= f]
    return bool(before) and bool(after) and min(after) - max(before) <= hold + 1


def _unexpand(box: Box, by: float) -> Box:
    """Inverse of specialist._expand: the detectors return faces already widened by FACE_EXPAND, but the embedder
    wants the raw face. The centre is kept and each side shrinks by 1 / (1 + 2 * by)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (box[2] - box[0]) / (1 + 2 * by) / 2, (box[3] - box[1]) / (1 + 2 * by) / 2
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def _spared_flags(ready: list[Detection]) -> dict[tuple[int, Box], bool]:
    """(frame, box) -> spared, for every face. link() keeps a detection's box tuple unchanged in its Keyframe, so the
    key finds it again. Two identical boxes in one frame are ANDed: both must be spared for the key to be."""
    flags: dict[tuple[int, Box], bool] = {}
    for d in ready:
        marks = d.spared.get("face", ())
        for i, (box, _) in enumerate(d.dets.get("face", [])):
            flag = i < len(marks) and bool(marks[i])
            flags[(d.f, box)] = flags.get((d.f, box), True) and flag
    return flags


def _current(ready: list[Detection], gen: int) -> list[Detection]:
    """Detections from before the last enroll / forget carry stale identity flags: they count as all-unspared."""
    return [d if d.gen == gen or not d.spared else replace(d, spared={}) for d in ready]


def _bracket_spared(kfs: list[Keyframe], g: int, flags: dict[tuple[int, Box], bool],
                    faces_at: dict[int, list[Box]]) -> bool:
    """Is frame g of this face track spared? Only between two detections of the track (a = last at or before g,
    b = first at or after g; held and lead-in boxes are never spared), and only when all of: a and b are both flagged;
    the track has SPARE_MIN_KF detections in the window; a and b are the same face by link()'s own test and at most
    LINK_GAP apart; no unspared face box overlaps a or b at their frames (two people crossing). A track is never
    spared as a whole: a swap in the middle masks the frames around it."""
    if len(kfs) < SPARE_MIN_KF:
        return False
    before = [k for k in kfs if k.f <= g]
    after = [k for k in kfs if k.f >= g]
    if not before or not after:
        return False
    a, b = before[-1], after[0]
    if not (flags.get((a.f, a.box)) and flags.get((b.f, b.box))):
        return False
    if b.f - a.f > LINK_GAP or not _same(a.box, b.box):
        return False
    for k in (a, b):
        if any(not flags.get((k.f, o)) and _iou(k.box, o) >= SPARE_OVERLAP_IOU for o in faces_at.get(k.f, [])):
            return False
    return True


def _table(ready: list[Detection], frame_w: int, frame_h: int, n_frames: int,
           face_rule: dict | None = CONFIRM["face"], cats: Sequence[str] = LIVE_CATS
           ) -> tuple[dict[int, list], dict[int, list]]:
    """(boxes, spared), each frame -> [(box, padding, effect)], from the tracks the detections link into. A face box
    on a frame where _bracket_spared holds goes to `spared` (outlined, not masked); everything else to `boxes`.
    `face_rule` None keeps every face; only the categories in `cats` get tracks."""
    out: dict[int, list] = {}
    spared_out: dict[int, list] = {}
    flags = _spared_flags(ready) if any(d.spared for d in ready) else {}
    faces_at = {d.f: [b for b, _ in d.dets.get("face", [])] for d in ready} if flags else {}
    for cat in cats:
        seq = [(d.f, d.dets.get(cat, [])) for d in ready]
        for kfs in link(seq, uncertain_below=UNCERTAIN_BELOW):
            keep = ((face_rule is None or confirmed(kfs, face_rule, frame_h, frame_w)) if cat == "face"
                    else plate_ok(kfs, frame_w, frame_h))
            if not keep:
                continue
            # An object first seen at kfs[0] may have walked in right after the previous detection: lead back to it.
            prev = max((d.f for d in ready if d.f < kfs[0].f), default=None)
            lead = kfs[0].f - prev - 1 if prev is not None else 0
            target = Target(label=cat, source="rule", effect="blur", padding=PADDING, keyframes=kfs)
            check = cat == "face" and any(flags.get((k.f, k.box)) for k in kfs)
            for g, (box, pad) in expand_target(target, n_frames, max(LEAD_FRAMES, lead)).items():
                spare = check and _bracket_spared(kfs, g, flags, faces_at)
                (spared_out if spare else out).setdefault(g, []).append((box, pad, "blur"))
    return out, spared_out


def mask_and_spared(f: int, ready: list[Detection], frame_w: int, frame_h: int, hold: int = HOLD_FRAMES,
                    face_rule: dict | None = CONFIRM["face"], cats: Sequence[str] = LIVE_CATS
                    ) -> tuple[list[tuple[Box, float, str]] | None, list[tuple[Box, float, str]]]:
    """mask_boxes plus the boxes left unmasked on frame f because they are the enrolled person. (None, []) when f is
    not vouched."""
    if not _vouched(f, ready, hold):
        return None, []
    boxes, spared = _table(ready, frame_w, frame_h, f + hold + 1, face_rule, cats)
    return boxes.get(f, []), spared.get(f, [])


def mask_boxes(f: int, ready: list[Detection], frame_w: int, frame_h: int, hold: int = HOLD_FRAMES,
               face_rule: dict | None = CONFIRM["face"],
               cats: Sequence[str] = LIVE_CATS) -> list[tuple[Box, float, str]] | None:
    """Boxes to mask on frame f as (box, padding, effect), or None when f is not vouched. `ready` is windowed by
    the caller to [f - WINDOW_BACK, f + hold] and sorted by f. `hold` sets the vouch distance (b.f - a.f <= hold + 1)
    and the table horizon; how long a lost box is held is always render.py's HOLD_FRAMES. `face_rule` is the scan's
    face confirmation rule; None keeps every face track (camera mode). `cats` limits the tracks to those
    categories. Spared faces (see mask_and_spared) are not in the result."""
    return mask_and_spared(f, ready, frame_w, frame_h, hold, face_rule, cats)[0]


class BoxCache:
    """mask_boxes for a run of frames. link() + expand_target() over 25-40 detections is 10-20 ms of Python, which
    would fight the detector thread for the GIL at 30 fps, so the table is rebuilt only when the detections
    change (every 4-5 frames); the vouch check still runs per frame.

    `boxes(..., hold)`: as for mask_boxes, `hold` sets the vouch distance (b.f - a.f <= hold + 1) and the table
    horizon, while the hold-after-loss length itself is always render.py's HOLD_FRAMES."""

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._table: tuple[dict[int, list], dict[int, list]] = ({}, {})

    def boxes(self, f: int, ready: list[Detection], frame_w: int, frame_h: int, hold: int = HOLD_FRAMES,
              face_rule: dict | None = CONFIRM["face"], cats: Sequence[str] = LIVE_CATS,
              generation: int = 0) -> list[tuple[Box, float, str]] | None:
        return self.boxes_and_spared(f, ready, frame_w, frame_h, hold, face_rule, cats, generation)[0]

    def boxes_and_spared(self, f: int, ready: list[Detection], frame_w: int, frame_h: int, hold: int = HOLD_FRAMES,
                         face_rule: dict | None = CONFIRM["face"], cats: Sequence[str] = LIVE_CATS,
                         generation: int = 0
                         ) -> tuple[list[tuple[Box, float, str]] | None, list[tuple[Box, float, str]]]:
        """(masked boxes or None when f is not vouched, spared boxes). `generation` is the engine's enrollment
        counter: it is part of the key, so enrolling or forgetting rebuilds the table."""
        if not _vouched(f, ready, hold):
            return None, []
        key = (tuple(d.f for d in ready), frame_w, frame_h, hold, face_rule is None, tuple(cats), generation)
        if key != self._key:
            self._table = _table(ready, frame_w, frame_h, max(d.f for d in ready) + hold + 1, face_rule, cats)
            self._key = key
        return self._table[0].get(f, []), self._table[1].get(f, [])


# ---- detectors ----------------------------------------------------------------------------------------------

def egoblur_detect(infer: dict[str, Infer], cats: Sequence[str] = LIVE_CATS) -> Detect:
    """EgoBlur on a frame: scores at or above EGOBLUR_CONF, faces widened by FACE_EXPAND as the scan does. Only the
    models for `cats` run (faces only halves the GPU pass) and only those keys are returned."""
    def detect(clip_f: int, frame: np.ndarray) -> Dets:
        out: Dets = {}
        for cat in cats:
            kept = [(b, s) for b, s in infer[cat](frame) if s >= EGOBLUR_CONF]
            out[cat] = [(_expand(b, FACE_EXPAND), s) for b, s in kept] if cat == "face" else kept
        return out
    return detect


def sidecar_detect(tracks_json: Path, src_h: int, live_h: int, n_frames: int,
                   cats: Sequence[str] = LIVE_CATS) -> Detect:
    """Replay a TrackStore dump as if a detector had found it: no GPU. Boxes are scaled to the live size; the
    sidecar's faces are raw ground truth, so they get FACE_EXPAND like the real detector's. Only `cats` are returned."""
    scale = live_h / src_h
    table: dict[int, Dets] = {}
    for track in TrackStore.load(tracks_json).tracks.values():
        if track.label not in cats:
            continue
        for k in track.keyframes:
            box = tuple(c * scale for c in k.box)
            box = _expand(box, FACE_EXPAND) if track.label == "face" else box
            table.setdefault(k.f, {c: [] for c in cats})[track.label].append((box, k.conf))
    empty: Dets = {c: [] for c in cats}
    return lambda clip_f, frame: table.get(clip_f % n_frames, empty)


def yunet_detect(onnx_path: str | Path, score: float = YUNET_SCORE, cats: Sequence[str] = LIVE_CATS) -> Detect:
    """Faces only, on the CPU (OpenCV's YuNet, ~15-25 ms at 720p): the camera demo on a laptop with no GPU. Boxes are
    widened by FACE_EXPAND like EgoBlur's; there are no plates: if "license plate" is in `cats` its list is always
    empty, and without "face" in `cats` the net is skipped. The model file is fetched by hand, never here."""
    path = Path(onnx_path)
    if not path.is_file():
        raise FileNotFoundError(f"YuNet model not found: {path}. Download it once (about 230 KB) from {YUNET_URL}")
    net = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score_threshold=score, nms_threshold=YUNET_NMS,
                                    top_k=YUNET_TOP_K)
    lock, input_size = threading.Lock(), [(320, 320)]

    def detect(clip_f: int, frame: np.ndarray) -> Dets:
        h, w = frame.shape[:2]
        if "face" not in cats:
            return {c: [] for c in cats}
        with lock:  # the detector thread is the only caller; setInputSize + detect must not interleave anyway
            if input_size[0] != (w, h):
                net.setInputSize((w, h))
                input_size[0] = (w, h)
            _, faces = net.detect(frame)
        rows = [] if faces is None else faces  # N x 15: x, y, w, h, five landmarks (x, y), score
        out: Dets = {c: [] for c in cats}
        out["face"] = [(_expand((float(r[0]), float(r[1]), float(r[0] + r[2]), float(r[1] + r[3])), FACE_EXPAND),
                        float(r[-1])) for r in rows]
        return out
    return detect


def _crop(frame: np.ndarray, box: Box, margin: float) -> np.ndarray | None:
    """The box grown by `margin` of its size on each side, clamped to the frame; None when empty or not finite."""
    h, w = frame.shape[:2]
    if not all(np.isfinite(box)):
        return None
    bw, bh = box[2] - box[0], box[3] - box[1]
    x1, y1 = int(max(0, np.floor(box[0] - bw * margin))), int(max(0, np.floor(box[1] - bh * margin)))
    x2, y2 = int(min(w, np.ceil(box[2] + bw * margin))), int(min(h, np.ceil(box[3] + bh * margin)))
    return frame[y1:y2, x1:x2] if x2 - x1 >= 2 and y2 - y1 >= 2 else None


def sface_embed(onnx_path: str | Path, yunet_path: str | Path | None = None) -> Embed:
    """Face identity vectors on the CPU: OpenCV's SFace (128-d), L2-normalised (feature() itself is not unit-norm).
    Primary path (needs a YuNet model: `yunet_path`, else the CENSOR_YUNET environment variable): the RAW face box is
    cropped with a 30% margin, YuNet finds the face and its five landmarks inside the crop (no face found: None),
    alignCrop() aligns it to 112 x 112 and feature() embeds it. Fallback, only when CENSOR_SPARE_PLAIN=1 or no YuNet
    path is configured: the plain crop (10% margin) resized to 112 x 112 with no alignment. SFace is meant to see an
    aligned face, so that path is less accurate; SPARE_COS is strict to compensate (a wrong "no match" blurs, which is
    the safe direction). Models (SFace about 38 MB, YuNet about 230 KB) are fetched by hand, never here.
    FileNotFoundError when a configured model file is missing."""
    path = Path(onnx_path)
    if not path.is_file():
        raise FileNotFoundError(f"SFace model not found: {path}. Download face_recognition_sface_2021dec.onnx "
                                f"(about 38 MB) from {SFACE_URL}")
    if not hasattr(cv2, "FaceRecognizerSF"):
        raise RuntimeError("this OpenCV build has no cv2.FaceRecognizerSF")
    yunet = yunet_path if yunet_path is not None else os.environ.get("CENSOR_YUNET")
    plain = os.environ.get("CENSOR_SPARE_PLAIN") == "1" or not yunet
    if not plain and not Path(yunet).is_file():
        raise FileNotFoundError(f"YuNet model not found: {yunet}. Download it once (about 230 KB) from {YUNET_URL}")
    net = cv2.FaceRecognizerSF.create(str(path), "")
    finder: list = []   # created on first use, in the thread that embeds
    lock = threading.Lock()

    def aligned(frame: np.ndarray, box: Box) -> np.ndarray | None:
        crop = _crop(frame, box, SFACE_MARGIN)
        if crop is None:
            return None
        if not finder:
            finder.append(cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), score_threshold=YUNET_SCORE,
                                                    nms_threshold=YUNET_NMS, top_k=YUNET_TOP_K))
        ch, cw = crop.shape[:2]
        finder[0].setInputSize((cw, ch))
        _, rows = finder[0].detect(crop)
        if rows is None or len(rows) == 0:
            return None
        return net.alignCrop(crop, rows[int(np.argmax(rows[:, -1]))])

    def embed(frame: np.ndarray, box: Box) -> np.ndarray | None:
        with lock:
            if plain:
                crop = _crop(frame, box, PLAIN_MARGIN)
                face = None if crop is None else cv2.resize(crop, (SFACE_SIZE, SFACE_SIZE), interpolation=cv2.INTER_AREA)
            else:
                face = aligned(frame, box)
            if face is None:
                return None
            vec = np.asarray(net.feature(face), dtype=np.float32).ravel()
        norm = float(np.linalg.norm(vec))
        return vec / norm if vec.size and np.isfinite(norm) and norm > 0 else None
    embed.kind = "sface"  # type: ignore[attr-defined]
    return embed


def _takes_spared(cb: Callable | None) -> bool:
    """True when `cb` can be called with a 5th positional argument. Decided once from the signature: a callback with
    four parameters (the CLI's Recorder) keeps the old four-argument call; *args or five+ parameters get the spared
    rects. A callback whose signature cannot be read is treated as four-argument."""
    if cb is None:
        return False
    try:
        params = inspect.signature(cb).parameters.values()
    except (TypeError, ValueError):
        return False
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return True
    return sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params) >= 5


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b)) / (na * nb) if na > 0 and nb > 0 and a.shape == b.shape else -1.0


# ---- the engine ---------------------------------------------------------------------------------------------

def _pct(samples: Sequence[float], key: str) -> float | None:
    return stats_ms(list(samples))[key] if samples else None


def _badge(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    h = out.shape[0]
    scale = max(0.5, 0.8 * h / 720)
    (tw, th), base = cv2.getTextSize(BADGE_TEXT, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y, pad = 12, h - 12, 8
    cv2.rectangle(out, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), (110, 110, 110), -1)
    cv2.putText(out, BADGE_TEXT, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


MAX_JPEG_PIXELS = 4096 * 2304  # a bigger upload is refused before it is decoded (a tiny file can declare 20000 x 20000)


def _jpeg_pixels(b: bytes) -> int:
    """Pixels the first SOFn marker declares (0 if none is found or the bytes do not start like a JPEG), read without
    decoding anything. The magic check keeps a JPEG 2000 codestream (which has its own, unrelated markers) out."""
    if b[:3] != b"\xff\xd8\xff":
        return 0
    i = 2
    while i + 9 < len(b) and b[i] == 0xFF:
        if 0xC0 <= b[i + 1] <= 0xCF and b[i + 1] not in (0xC4, 0xC8, 0xCC):  # FF Cn len(2) prec(1) h(2) w(2)
            return int.from_bytes(b[i + 5:i + 7], "big") * int.from_bytes(b[i + 7:i + 9], "big")
        i += 2 + int.from_bytes(b[i + 2:i + 4], "big")
    return 0


def _live_size(src_w: int, src_h: int, height: int) -> tuple[int, int]:
    """(w, h) of a source scaled to `height` rows, aspect kept, width even."""
    return max(2, int(round(src_w * height / src_h / 2)) * 2), height


class Busy(RuntimeError):
    """push() found every decode slot taken (MAX_DECODES uploads are being decoded): the route answers 429."""


MAX_DECODES = 3   # concurrent JPEG decodes + resizes in push(); a fourth is refused at once, not queued


def _letterbox(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """`img` fitted inside a (w, h) canvas, aspect kept, centred, padded with black. An image already of that aspect is
    just resized, so a 16:9 camera keeps every pixel and any other shape (4:3, portrait) is never stretched."""
    w, h = size
    ih, iw = img.shape[:2]
    if (iw, ih) == (w, h):
        return img
    if abs(iw * h - ih * w) <= max(iw, w) * 2:   # same aspect to within a couple of pixels
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    scale = min(w / iw, h / ih)
    nw, nh = max(1, min(w, int(round(iw * scale)))), max(1, min(h, int(round(ih * scale))))
    canvas = np.zeros((h, w, 3), np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    canvas[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas


class LiveReplay:
    def __init__(self, clip: str | None, detect: Detect, height: int = LIVE_HEIGHT, delay_s: float = LIVE_DELAY_S,
                 out_w: int = LIVE_OUT_W, on_emit: OnEmit | None = None, push: bool = False,
                 cats: tuple[str, ...] = LIVE_CATS, embed: Embed | None = None) -> None:
        """Clip mode (default): `clip` is played back at its own fps, looping. Push mode (`push=True`, `clip` may be
        None): the caller feeds frames with push(); fps is a nominal 30 (it numbers the frames), n_frames is 0, and size
        is always the 16:9 canvas at `height` (every upload is letterboxed to it, so it matches the detector warm-up).
        The ring is sized by time at that nominal rate ((delay + RING_SLACK_S) * 30 frames); a camera that pushes
        slower simply fills it less. `cats` is what this run masks: a
        non-empty subset of LIVE_CATS (ValueError otherwise); build the detector with the same cats.
        `embed` (optional) turns on "blur everyone except me": see enroll(). Without it nothing is ever spared."""
        cats = tuple(cats)
        if not cats or len(set(cats)) != len(cats) or not set(cats) <= set(LIVE_CATS):
            raise ValueError(f"cats must be a non-empty subset of {LIVE_CATS}, got {cats!r}")
        self.cats = cats
        self.is_push = push
        self._face_rule = None if push else CONFIRM["face"]  # camera: every face is kept (see the module docstring)
        self._height, self._out_w = height, out_w
        if push:
            self.clip, self.fps, self.n_frames = str(clip) if clip else CAMERA_CLIP, PUSH_FPS, 0
            self.size = _live_size(16, 9, height)
        else:
            if clip is None:
                raise ValueError("a clip is required unless push=True")
            info = probe(str(clip))
            self.clip, self.fps, self.n_frames = str(clip), info.fps, max(1, info.n_frames)
            self.size = _live_size(info.width, info.height, height)  # (w, h)
        self.run, self.warming, self.delay_s = uuid.uuid4().hex[:8], False, delay_s
        self._detect, self._on_emit = detect, on_emit
        self._emit_spared = _takes_spared(on_emit)
        self._embed = embed
        self._decode_slots = threading.BoundedSemaphore(MAX_DECODES)
        self._embed_lock = threading.Lock()   # an embedder is not assumed thread-safe: detector thread and enroll()
        self._enrolled: np.ndarray | None = None   # memory only, never written anywhere; cleared by stop()
        self._gen = 0                              # bumped on enroll/forget: older Detections count as unspared
        self._spared_now = 0
        self._enroll_req: dict | None = None       # a pending enroll(), served by the detector thread
        self._embed_s: deque[float] = deque(maxlen=SAMPLES)
        self._skipped_embeds = 0
        self._display = self._display_for(self.size)
        self._stop = threading.Event()
        self._cv = threading.Condition()  # guards everything below
        self._ring: deque[_Slot] = deque(maxlen=int((delay_s + RING_SLACK_S) * self.fps) + 2)
        self._dets: list[Detection] = []
        self._history: deque[tuple[int, bytes]] = deque(maxlen=HISTORY)
        self._handed = -1  # highest frame number any poller has been given (latest() hands each out once)
        self._threads: list[threading.Thread] = []
        self._logged: set[tuple[str, str]] = set()
        self._n = dict(frames_in=0, frames_out=0, blind=0, dropped_out=0, emit_errors=0, det_errors=0, passes=0,
                       gaps=0, gap_sum=0, gap_max=0, last_det_f=-1, gaps_in=0)
        # push mode only (all under _cv)
        self._t0: float | None = None             # monotonic time of the first push: frame 0
        self._last_f = -1                         # number of the previous push
        self._arrivals: deque[float] = deque()    # push times of the last INPUT_FPS_WINDOW_S
        self._claimed_to = 0                      # the emitter has claimed every number below this
        self._evicted_unclaimed = 0               # slots evicted from the ring before the emitter claimed them
        self._first_passes: list[float] = []
        self._recent_passes: deque[float] = deque(maxlen=RECENT_PASSES)
        self._emit_s: deque[float] = deque(maxlen=SAMPLES)
        self._age_s: deque[float] = deque(maxlen=SAMPLES)

    def _display_for(self, size: tuple[int, int]) -> tuple[int, int]:
        """The size sent to viewers: `out_w` wide when the live size is wider, else the live size itself."""
        w, h = size
        return (self._out_w, int(round(h * self._out_w / w))) if self._out_w < w else size

    # -- control --
    @property
    def running(self) -> bool:
        return bool(self._threads) and not self._stop.is_set()

    def start(self) -> None:
        with self._cv:
            if self._threads:
                return
            steps = [("detector", self._detector), ("emitter", self._emitter)]
            if not self.is_push:
                steps.insert(0, ("source", self._pump))  # in push mode the caller is the source
            self._threads = [threading.Thread(target=self._guarded, args=(name, fn), daemon=True, name=f"live-{name}")
                             for name, fn in steps]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self.clear_enrolled()
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        for t in self._threads:
            t.join(timeout=2)  # never while holding the lock: the threads need it to leave their waits
        with self._cv:
            self._cv.notify_all()

    def latest(self, after: int, timeout: float) -> tuple[int, bytes] | None:
        """First retained frame with f > after that no poller has been given yet; a client ahead of the stream
        (after > newest) gets the newest now. None on timeout or once stopped. Each frame goes out once: the page
        keeps two polls in flight with the same `after`, and without this both woke on the same frame, so the
        second request carried a duplicate instead of the next frame (half the display rate through the proxy)."""
        with self._cv:
            newest = self._history[-1][0] if self._history else -1
            if after > newest >= 0:
                return self._history[-1]
            floor = max(after, self._handed)
            self._cv.wait_for(lambda: self._stop.is_set() or (bool(self._history) and self._history[-1][0] > floor),
                              timeout)
            if self._stop.is_set() or not self._history or self._history[-1][0] <= floor:
                return None
            frame = next(h for h in self._history if h[0] > floor)
            self._handed = frame[0]
            return frame

    def _publish(self, f: int, jpeg: bytes) -> None:
        with self._cv:
            self._history.append((f, jpeg))
            self._cv.notify_all()

    def _log_once(self, where: str, exc: Exception) -> None:
        key = (where, type(exc).__name__)
        if key not in self._logged:
            self._logged.add(key)
            print(f"[live] {where}: {type(exc).__name__}: {exc}", file=sys.stderr)

    def _guarded(self, name: str, fn: Callable[[], None]) -> None:
        """Run a thread body. An escaped exception ends the whole engine (running turns False, the routes notice)
        instead of leaving the other threads to look alive over a dead one."""
        try:
            fn()
        except Exception as e:  # noqa: BLE001  one dead thread is a dead engine: say so and stop
            self._log_once(name, e)
            self._stop.set()
            with self._cv:
                self._cv.notify_all()

    # -- source --
    def _pump(self) -> None:
        t0, f, clip_f, cap = time.monotonic(), 0, 0, None
        try:
            while not self._stop.is_set():
                if self._stop.wait(max(0.0, t0 + f / self.fps - time.monotonic())):  # late: no sleep
                    break
                if cap is None:
                    cap, clip_f = cv2.VideoCapture(self.clip), 0
                ok, img = cap.read()
                if not ok:
                    if clip_f == 0:
                        raise IOError(f"cannot read frames from {self.clip}")
                    cap.release()
                    cap = None  # end of the clip: loop
                    continue
                if img.shape[1::-1] != self.size:
                    img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
                with self._cv:
                    self._ring.append(_Slot(f, clip_f, img, time.monotonic()))
                    self._n["frames_in"] += 1
                    self._cv.notify_all()
                f, clip_f = f + 1, clip_f + 1
        finally:
            if cap is not None:
                cap.release()

    def push(self, jpeg: bytes) -> int:
        """Camera mode: decode one upload, scale it to the live height (aspect kept, even width; a smaller upload is
        upscaled, so a 640-wide frame at height 540 is stretched to 960) and put it in the ring. Returns its frame
        number. Any frame whose aspect is not 16:9 is letterboxed (black bars, centred) onto the 16:9 canvas, never
        stretched. Raises ValueError for bytes that are not a JPEG (nothing is counted), Busy (a RuntimeError) when
        MAX_DECODES uploads are already being decoded, and RuntimeError when the engine is not running.

        Frame numbers follow elapsed time, not the arrival count (see the module docstring): a stall leaves a gap."""
        if not self.is_push:
            raise RuntimeError("push() needs an engine built with push=True")
        if not self.running:
            raise RuntimeError("the engine is not running")
        if not 0 < _jpeg_pixels(jpeg) <= MAX_JPEG_PIXELS:
            raise ValueError("JPEG too large or malformed")  # before imdecode allocates what the header claims
        if not self._decode_slots.acquire(blocking=False):
            raise Busy("too many uploads being decoded")
        try:
            try:
                img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            except cv2.error:
                img = None
            if img is None:
                raise ValueError("not a decodable JPEG")
            img = _letterbox(img, self.size)
        finally:
            self._decode_slots.release()
        with self._cv:
            if self._stop.is_set():
                raise RuntimeError("the engine is not running")
            now = time.monotonic()
            self._t0 = now if self._t0 is None else self._t0
            f = max(self._last_f + 1, round((now - self._t0) * self.fps))
            self._last_f = f
            if len(self._ring) == self._ring.maxlen and self._ring[0].f >= self._claimed_to:
                self._n["dropped_out"] += 1  # a frame nobody emitted is about to leave the full ring
                self._evicted_unclaimed += 1
            self._ring.append(_Slot(f, f, img, now))
            self._n["frames_in"] += 1
            self._arrivals.append(now)
            while self._arrivals[0] < now - INPUT_FPS_WINDOW_S:
                self._arrivals.popleft()
            self._cv.notify_all()
            return f

    # -- detector --
    def _detector(self) -> None:
        last = -1
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._stop.is_set() or (self._ring and self._ring[-1].f > last))
                if self._stop.is_set():
                    return
                slot = self._ring[-1]
            last = slot.f
            with self._cv:
                enrolled, gen = self._enrolled, self._gen
            t = time.monotonic()
            try:
                dets = self._detect(slot.clip_f, slot.frame)
            except Exception as e:  # noqa: BLE001  the frames it missed go out blind
                self._log_once("detector", e)
                with self._cv:
                    self._n["det_errors"] += 1
                    self._fail_enroll("detector failed")
                continue
            spared = self._identify(slot.frame, dets, enrolled)
            self._record_pass(Detection(slot.f, dets, time.monotonic() - t, spared, gen))
            self._serve_enroll(slot.frame, dets)

    def _embed_one(self, frame: np.ndarray, box: Box) -> np.ndarray | None:
        """embed() of the RAW face (the detectors widen faces by FACE_EXPAND), as a flat float32 vector or None.
        An embedder that raises yields None: nobody is spared because of a failure."""
        try:
            with self._embed_lock:
                v = self._embed(frame, _unexpand(box, FACE_EXPAND))
        except Exception as e:  # noqa: BLE001
            self._log_once("embed", e)
            return None
        if v is None:
            return None
        v = np.asarray(v, dtype=np.float32).ravel()
        return v if v.size and np.all(np.isfinite(v)) and float(np.linalg.norm(v)) > 0 else None

    def _identify(self, frame: np.ndarray, dets: Dets, enrolled: np.ndarray | None) -> dict[str, tuple[bool, ...]]:
        """Which faces are the enrolled person. Runs only when someone is enrolled. At most EMBED_MAX faces per pass,
        largest first, and only those at least EMBED_MIN_H tall (raw): the rest are not spared and are counted in
        skipped_embeds. {} when nothing can be spared."""
        faces = dets.get("face") if self._embed is not None and enrolled is not None else None
        if not faces:
            return {}
        t = time.monotonic()
        area = lambda i: (faces[i][0][2] - faces[i][0][0]) * (faces[i][0][3] - faces[i][0][1])  # noqa: E731
        order = sorted(range(len(faces)), key=lambda i: -area(i))
        raw_h = lambda i: (faces[i][0][3] - faces[i][0][1]) / (1 + 2 * FACE_EXPAND)  # noqa: E731
        todo = [i for i in order if raw_h(i) >= EMBED_MIN_H][:EMBED_MAX]
        flags = [False] * len(faces)
        for i in todo:
            v = self._embed_one(frame, faces[i][0])
            flags[i] = v is not None and _cos(v, enrolled) >= SPARE_COS
        with self._cv:
            self._skipped_embeds += len(faces) - len(todo)
            self._embed_s.append(time.monotonic() - t)
        return {"face": tuple(flags)}

    # -- enrollment (served by the detector thread, so it sees the same frames and the same detector) --
    def _fail_enroll(self, reason: str, seen: int = 0, box: list | None = None) -> None:
        """Caller holds _cv."""
        req = self._enroll_req
        if req is not None and req["result"] is None:
            req["result"] = {"ok": False, "faces_seen": seen, "box": box, "reason": reason}
            self._cv.notify_all()

    def _serve_enroll(self, frame: np.ndarray, dets: Dets) -> None:
        """One step of a pending enroll() on this pass: exactly one face at least ENROLL_MIN_H_FRAC of the frame tall
        (raw), embedded on ENROLL_PASSES consecutive passes that agree pairwise (ENROLL_PAIR_COS); the mean becomes
        the enrolled vector. Any bad pass ends the request with its reason."""
        with self._cv:
            req = self._enroll_req
            if req is None or req["result"] is not None:
                return
        h = frame.shape[0]
        big = [b for b, _ in dets.get("face", []) if (b[3] - b[1]) / (1 + 2 * FACE_EXPAND) >= ENROLL_MIN_H_FRAC * h]
        seen = len(dets.get("face", []))
        with self._cv:
            if len(big) != 1:
                reason = "no face in view" if not big else f"{len(big)} faces in frame; only you may be visible"
                return self._fail_enroll(reason, seen)
        box = big[0]
        shown = [int(round(c)) for c in box]
        v = self._embed_one(frame, box)
        with self._cv:
            if req["result"] is not None:
                return
            if v is None:
                return self._fail_enroll("face too small or unusable", seen, shown)
            req["vecs"].append(v / np.linalg.norm(v))
            req["box"] = shown
            if len(req["vecs"]) < ENROLL_PASSES:
                return
            vecs = req["vecs"]
            if any(_cos(a, b) < ENROLL_PAIR_COS for i, a in enumerate(vecs) for b in vecs[i + 1:]):
                return self._fail_enroll("hold still and look at the camera", seen, shown)
            mean = np.mean(vecs, axis=0)
            self._enrolled, self._gen = (mean / np.linalg.norm(mean)).astype(np.float32), self._gen + 1
            req["result"] = {"ok": True, "faces_seen": 1, "box": shown, "reason": None}
            self._cv.notify_all()

    def _record_pass(self, d: Detection) -> None:
        with self._cv:
            n = self._n
            if n["last_det_f"] >= 0:
                n["gaps"] += 1
                n["gap_sum"] += d.f - n["last_det_f"]
                n["gap_max"] = max(n["gap_max"], d.f - n["last_det_f"])
            n["last_det_f"], n["passes"] = d.f, n["passes"] + 1
            if len(self._first_passes) < BASELINE_PASSES:
                self._first_passes.append(d.took_s)
            self._recent_passes.append(d.took_s)
            self._dets = [*self._dets, d]
            self._cv.notify_all()

    # -- emitter --
    def _emitter(self) -> None:
        next_f, cache, frozen = 0, BoxCache(), None
        while not self._stop.is_set():
            slot = self._next_slot(next_f)
            if slot is None or self._stop.wait(max(0.0, slot.t_arrive + self.delay_s - time.monotonic())):
                return
            t_start = time.monotonic()
            with self._cv:
                if not self.is_push:  # a clip's numbers are contiguous: a skipped one was evicted
                    self._n["dropped_out"] += slot.f - next_f
                self._dets = [d for d in self._dets if d.f >= slot.f - WINDOW_BACK]
                ready = [d for d in self._dets if d.f <= slot.f + HOLD_FRAMES]
            next_f = slot.f + 1
            try:
                out = self._render(slot, ready, cache, frozen)
                jpeg = self._encode(out.image)
            except Exception as e:  # noqa: BLE001  nothing safe to show: this frame is not sent
                self._log_once("emit", e)
                with self._cv:
                    self._n["emit_errors"] += 1
                continue
            frozen = out.image if out.vouched else frozen
            self._finish(slot, out, jpeg, t_start)

    def _next_slot(self, next_f: int) -> _Slot | None:
        with self._cv:
            self._cv.wait_for(lambda: self._stop.is_set() or (self._ring and self._ring[-1].f >= next_f))
            if self._stop.is_set():
                return None
            if self.is_push:
                # Numbers follow elapsed time, so the ring has holes: take the first slot at or after next_f. The
                # numbers before it never arrived (gaps_in), except the ones evicted unclaimed (counted in push()).
                slot = next(s for s in self._ring if s.f >= next_f)
                self._n["gaps_in"] += slot.f - next_f - self._evicted_unclaimed
                self._evicted_unclaimed, self._claimed_to = 0, slot.f + 1
                return slot
            first = self._ring[0].f  # evicted frames are skipped (and counted as dropped_out by the caller)
            return self._ring[max(next_f, first) - first]

    def _render(self, slot: _Slot, ready: list[Detection], cache: BoxCache, frozen: np.ndarray | None) -> _Out:
        w, h = self.size
        img = slot.frame.copy()  # the ring frame is the detector's input and apply_effect draws in place
        try:
            with self._cv:
                gen = self._gen
            boxes, spared = cache.boxes_and_spared(slot.f, _current(ready, gen), w, h, face_rule=self._face_rule,
                                                   cats=self.cats, generation=gen)
            if boxes is None:
                if frozen is not None:
                    return _Out(_badge(frozen), [], True, False)
                full_frame_blur(img)
                return _Out(self._to_display(img), [], True, False)
            rects = [_padded(b, p, w, h) for b, p, _ in boxes]
            for rect, (_, _, effect) in zip(rects, boxes):
                apply_effect(img, rect, effect)
            spared_rects = [_padded(b, p, w, h) for b, p, _ in spared]
            return _Out(self._to_display(img), rects, False, True, spared_rects)
        except Exception as e:  # noqa: BLE001  fail closed
            self._log_once("mask", e)
            with self._cv:
                self._n["emit_errors"] += 1
            img = slot.frame.copy()
            full_frame_blur(img)
            return _Out(self._to_display(img), [(0, 0, w, h)], False, False)

    def _to_display(self, img: np.ndarray) -> np.ndarray:
        return img if self._display == self.size else cv2.resize(img, self._display, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _encode(img: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise IOError("JPEG encode failed")
        return buf.tobytes()

    def _finish(self, slot: _Slot, out: _Out, jpeg: bytes, t_start: float) -> None:
        with self._cv:
            self._publish(slot.f, jpeg)
            now = time.monotonic()
            self._n["frames_out"] += 1
            self._n["blind"] += out.blind
            self._emit_s.append(now - t_start)
            self._age_s.append(now - slot.t_arrive)
            self._spared_now = len(out.spared_rects)
        if self._on_emit is not None:
            try:
                if self._emit_spared:
                    self._on_emit(slot.f, slot.clip_f, out.blind, out.rects, out.spared_rects)
                else:
                    self._on_emit(slot.f, slot.clip_f, out.blind, out.rects)
            except Exception as e:  # noqa: BLE001  a bad callback must not stop the stream
                self._log_once("on_emit", e)

    # -- identity ("blur everyone except me") --
    def enroll(self, timeout: float = 2.0) -> dict:
        """Remember the person in view as "me" (a vector in memory, nothing on disk, never updated afterwards). The
        detector thread serves it over its next ENROLL_PASSES passes: exactly one face of at least 8% of the frame's
        height must be in view on each, and the embeddings must agree. Returns {"ok", "faces_seen", "box", "reason"}
        ("box": the face as detected, widened by FACE_EXPAND like the masks, live-size pixels). Reasons: "no embedder",
        "no face in view", "N faces in frame; only you may be visible", "face too small or unusable", "hold still and
        look at the camera", "timed out". After a success a face is left unmasked only on frames between two
        detections of its track that both match this vector (SPARE_COS); see _bracket_spared."""
        if self._embed is None:
            return {"ok": False, "faces_seen": 0, "box": None, "reason": "no embedder"}
        req: dict = {"vecs": [], "box": None, "result": None}
        with self._cv:
            if self._enroll_req is not None and self._enroll_req["result"] is None:
                return {"ok": False, "faces_seen": 0, "box": None, "reason": "enrollment already in progress"}
            self._enroll_req = req
            self._cv.wait_for(lambda: req["result"] is not None or self._stop.is_set(), timeout)
            if req["result"] is None:
                req["result"] = {"ok": False, "faces_seen": 0, "box": None,
                                 "reason": "stopped" if self._stop.is_set() else "timed out"}
            if self._enroll_req is req:
                self._enroll_req = None
            return req["result"]

    def clear_enrolled(self) -> None:
        """Forget the enrolled face: every face is masked again from the next frame (older detections count as
        unspared through the generation counter)."""
        with self._cv:
            self._enrolled, self._gen, self._spared_now = None, self._gen + 1, 0

    # -- stats --
    def stats(self) -> dict:
        with self._cv:
            n = self._n
            out = n["frames_out"]
            return {
                "phase": "running" if self.running else "idle", "running": self.running, "warming": self.warming,
                "run": self.run, "clip": self.clip, "size": list(self.size), "fps": self.fps,
                "n_frames": self.n_frames, "delay_s": self.delay_s, "frames_out": out, "blind": n["blind"],
                "blind_share": n["blind"] / out if out else 0.0, "dropped_out": n["dropped_out"],
                "emit_errors": n["emit_errors"],
                "detector": {"passes": n["passes"], "p50_ms": _pct(self._recent_passes, "p50"),
                             "p95_ms": _pct(self._recent_passes, "p95"),
                             "baseline_p95_ms": _pct(self._first_passes, "p95"), "errors": n["det_errors"]},
                "emit": {"p50_ms": _pct(self._emit_s, "p50"), "p95_ms": _pct(self._emit_s, "p95")},
                # Over frames that have arrived, not frames sent: emits lag arrivals by the delay buffer
                "detected_share": n["passes"] / n["frames_in"] if n["frames_in"] else 0.0,
                "gap_frames": {"mean": n["gap_sum"] / n["gaps"] if n["gaps"] else None,
                               "max": n["gap_max"] if n["gaps"] else None},
                "hold_frames": HOLD_FRAMES, "frame_age_ms": {"p50": _pct(self._age_s, "p50")},
                "input_fps": self._input_fps() if self.is_push else None, "gaps_in": n["gaps_in"],
                "mode": "push" if self.is_push else "clip", "cats": list(self.cats),
                "identity": {"enabled": self._embed is not None, "enrolled": self._enrolled is not None,
                             "spared_now": self._spared_now, "embedder": "none" if self._embed is None else "sface",
                             "spare_cos": SPARE_COS, "skipped_embeds": self._skipped_embeds,
                             "embed_ms_p50": _pct(self._embed_s, "p50")},
            }

    def _input_fps(self) -> float:
        """Pushes in the last INPUT_FPS_WINDOW_S seconds, per second (under-reads for the first 2 s). Caller holds _cv."""
        since = time.monotonic() - INPUT_FPS_WINDOW_S
        return sum(1 for t in self._arrivals if t >= since) / INPUT_FPS_WINDOW_S
