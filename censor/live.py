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
never as a bare face. Any error while masking blurs the whole frame.

Only the shape half of the scan's confirmation rules runs here (faces: confirmed(); plates: plate_ok()), because the
history half needs the whole clip. No FastAPI in this module; censor/live_cli.py and censor/server/live_routes.py
drive it.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from bench.live_speed import stats_ms
from censor.core.timeline import Target
from censor.core.tracks import Keyframe, TrackStore
from censor.core.video import probe
from censor.render.render import (HOLD_FRAMES, LEAD_FRAMES, Box, _padded, apply_effect, expand_target,
                                  full_frame_blur)
from censor.vision.specialist import (CONFIRM, EGOBLUR_CONF, FACE_EXPAND, Infer, _expand, confirmed,
                                      link, plate_shaped)

Dets = dict[str, list[tuple[Box, float]]]        # category -> [(xyxy, score)]
Detect = Callable[[int, np.ndarray], Dets]       # (clip frame index, BGR frame at live size) -> Dets
Rect = tuple[int, int, int, int]
OnEmit = Callable[[int, int, bool, list[Rect]], None]

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


@dataclass(frozen=True)
class Detection:
    f: int           # stream frame number (keeps counting across loops)
    dets: Dets       # may lack a category: use dets.get(cat, []); an empty Detection still vouches
    took_s: float


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


def _table(ready: list[Detection], frame_w: int, frame_h: int, n_frames: int) -> dict[int, list]:
    """Frame -> [(box, padding, effect)] from the tracks the detections link into."""
    out: dict[int, list] = {}
    for cat in LIVE_CATS:
        seq = [(d.f, d.dets.get(cat, [])) for d in ready]
        for kfs in link(seq, uncertain_below=UNCERTAIN_BELOW):
            keep = (confirmed(kfs, CONFIRM["face"], frame_h, frame_w) if cat == "face"
                    else plate_ok(kfs, frame_w, frame_h))
            if not keep:
                continue
            # An object first seen at kfs[0] may have walked in right after the previous detection: lead back to it.
            prev = max((d.f for d in ready if d.f < kfs[0].f), default=None)
            lead = kfs[0].f - prev - 1 if prev is not None else 0
            target = Target(label=cat, source="rule", effect="blur", padding=PADDING, keyframes=kfs)
            for g, (box, pad) in expand_target(target, n_frames, max(LEAD_FRAMES, lead)).items():
                out.setdefault(g, []).append((box, pad, "blur"))
    return out


def mask_boxes(f: int, ready: list[Detection], frame_w: int, frame_h: int,
               hold: int = HOLD_FRAMES) -> list[tuple[Box, float, str]] | None:
    """Boxes to mask on frame f as (box, padding, effect), or None when f is not vouched. `ready` is windowed by
    the caller to [f - WINDOW_BACK, f + hold] and sorted by f."""
    if not _vouched(f, ready, hold):
        return None
    return _table(ready, frame_w, frame_h, f + hold + 1).get(f, [])


class BoxCache:
    """mask_boxes for a run of frames. link() + expand_target() over 25-40 detections is 10-20 ms of Python, which
    would fight the detector thread for the GIL at 30 fps, so the table is rebuilt only when the detections
    change (every 4-5 frames); the vouch check still runs per frame."""

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._table: dict[int, list] = {}

    def boxes(self, f: int, ready: list[Detection], frame_w: int, frame_h: int,
              hold: int = HOLD_FRAMES) -> list[tuple[Box, float, str]] | None:
        if not _vouched(f, ready, hold):
            return None
        key = (tuple(d.f for d in ready), frame_w, frame_h, hold)
        if key != self._key:
            self._table = _table(ready, frame_w, frame_h, max(d.f for d in ready) + hold + 1)
            self._key = key
        return self._table.get(f, [])


# ---- detectors ----------------------------------------------------------------------------------------------

def egoblur_detect(infer: dict[str, Infer]) -> Detect:
    """EgoBlur on a frame: scores at or above EGOBLUR_CONF, faces widened by FACE_EXPAND as the scan does."""
    def detect(clip_f: int, frame: np.ndarray) -> Dets:
        out: Dets = {}
        for cat in LIVE_CATS:
            kept = [(b, s) for b, s in infer[cat](frame) if s >= EGOBLUR_CONF]
            out[cat] = [(_expand(b, FACE_EXPAND), s) for b, s in kept] if cat == "face" else kept
        return out
    return detect


def sidecar_detect(tracks_json: Path, src_h: int, live_h: int, n_frames: int) -> Detect:
    """Replay a TrackStore dump as if a detector had found it: no GPU. Boxes are scaled to the live size; the
    sidecar's faces are raw ground truth, so they get FACE_EXPAND like the real detector's."""
    scale = live_h / src_h
    table: dict[int, Dets] = {}
    for track in TrackStore.load(tracks_json).tracks.values():
        if track.label not in LIVE_CATS:
            continue
        for k in track.keyframes:
            box = tuple(c * scale for c in k.box)
            box = _expand(box, FACE_EXPAND) if track.label == "face" else box
            table.setdefault(k.f, {c: [] for c in LIVE_CATS})[track.label].append((box, k.conf))
    empty: Dets = {c: [] for c in LIVE_CATS}
    return lambda clip_f, frame: table.get(clip_f % n_frames, empty)


# ---- the engine ---------------------------------------------------------------------------------------------

def _pct(samples, key: str) -> float | None:
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


class LiveReplay:
    def __init__(self, clip: str, detect: Detect, height: int = LIVE_HEIGHT, delay_s: float = LIVE_DELAY_S,
                 out_w: int = LIVE_OUT_W, on_emit: OnEmit | None = None):
        info = probe(str(clip))
        self.clip, self.fps, self.n_frames = str(clip), info.fps, max(1, info.n_frames)
        self.size = (int(round(info.width * height / info.height / 2)) * 2, height)  # even width (w, h)
        self.run, self.warming, self.delay_s = uuid.uuid4().hex[:8], False, delay_s
        self._detect, self._on_emit = detect, on_emit
        w, h = self.size
        self._display = (out_w, int(round(h * out_w / w))) if out_w < w else self.size
        self._stop = threading.Event()
        self._cv = threading.Condition()  # guards everything below
        self._ring: deque[_Slot] = deque(maxlen=int((delay_s + RING_SLACK_S) * self.fps) + 2)
        self._dets: list[Detection] = []
        self._history: deque[tuple[int, bytes]] = deque(maxlen=HISTORY)
        self._threads: list[threading.Thread] = []
        self._logged: set[tuple[str, str]] = set()
        self._n = dict(frames_out=0, blind=0, dropped_out=0, emit_errors=0, det_errors=0, passes=0,
                       gaps=0, gap_sum=0, gap_max=0, last_det_f=-1)
        self._first_passes: list[float] = []
        self._recent_passes: deque[float] = deque(maxlen=RECENT_PASSES)
        self._emit_s: deque[float] = deque(maxlen=SAMPLES)
        self._age_s: deque[float] = deque(maxlen=SAMPLES)

    # -- control --
    @property
    def running(self) -> bool:
        return bool(self._threads) and not self._stop.is_set()

    def start(self) -> None:
        with self._cv:
            if self._threads:
                return
            self._threads = [threading.Thread(target=fn, daemon=True, name=f"live-{fn.__name__}")
                             for fn in (self._source, self._detector, self._emitter)]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        for t in self._threads:
            t.join(timeout=2)  # never while holding the lock: the threads need it to leave their waits
        with self._cv:
            self._cv.notify_all()

    def latest(self, after: int, timeout: float) -> tuple[int, bytes] | None:
        """First retained frame with f > after; a client ahead of the stream (after > newest) gets the newest now.
        None on timeout or once stopped."""
        with self._cv:
            self._cv.wait_for(lambda: self._stop.is_set() or (bool(self._history) and self._history[-1][0] != after),
                              timeout)
            if self._stop.is_set() or not self._history or self._history[-1][0] == after:
                return None
            return next((h for h in self._history if h[0] > after), self._history[-1])

    def _publish(self, f: int, jpeg: bytes) -> None:
        with self._cv:
            self._history.append((f, jpeg))
            self._cv.notify_all()

    def _log_once(self, where: str, exc: Exception) -> None:
        key = (where, type(exc).__name__)
        if key not in self._logged:
            self._logged.add(key)
            print(f"[live] {where}: {type(exc).__name__}: {exc}", file=sys.stderr)

    # -- source --
    def _source(self) -> None:
        try:
            self._pump()
        except Exception as e:  # noqa: BLE001  nothing will arrive: say so and stop
            self._log_once("source", e)
            self._stop.set()
            with self._cv:
                self._cv.notify_all()

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
                    self._cv.notify_all()
                f, clip_f = f + 1, clip_f + 1
        finally:
            if cap is not None:
                cap.release()

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
            t = time.monotonic()
            try:
                dets = self._detect(slot.clip_f, slot.frame)
            except Exception as e:  # noqa: BLE001  the frames it missed go out blind
                self._log_once("detector", e)
                with self._cv:
                    self._n["det_errors"] += 1
                continue
            self._record_pass(Detection(slot.f, dets, time.monotonic() - t))

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
            first = self._ring[0].f  # evicted frames are skipped (and counted as dropped_out by the caller)
            return self._ring[max(next_f, first) - first]

    def _render(self, slot: _Slot, ready: list[Detection], cache: BoxCache, frozen: np.ndarray | None) -> _Out:
        w, h = self.size
        img = slot.frame.copy()  # the ring frame is the detector's input and apply_effect draws in place
        try:
            boxes = cache.boxes(slot.f, ready, w, h)
            if boxes is None:
                if frozen is not None:
                    return _Out(_badge(frozen), [], True, False)
                full_frame_blur(img)
                return _Out(self._to_display(img), [], True, False)
            rects = [_padded(b, p, w, h) for b, p, _ in boxes]
            for rect, (_, _, effect) in zip(rects, boxes):
                apply_effect(img, rect, effect)
            return _Out(self._to_display(img), rects, False, True)
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
        if self._on_emit is not None:
            try:
                self._on_emit(slot.f, slot.clip_f, out.blind, out.rects)
            except Exception as e:  # noqa: BLE001  a bad callback must not stop the stream
                self._log_once("on_emit", e)

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
                "detected_share": n["passes"] / out if out else 0.0,
                "gap_frames": {"mean": n["gap_sum"] / n["gaps"] if n["gaps"] else None,
                               "max": n["gap_max"] if n["gaps"] else None},
                "hold_frames": HOLD_FRAMES, "frame_age_ms": {"p50": _pct(self._age_s, "p50")},
            }
