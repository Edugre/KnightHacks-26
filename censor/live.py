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

import os
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
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
PUSH_FPS = 30.0        # nominal rate of a pushed stream: frame numbers follow elapsed time at this rate; sizes the ring
INPUT_FPS_WINDOW_S = 2.0
CAMERA_CLIP = "camera"
YUNET_SCORE, YUNET_NMS, YUNET_TOP_K = 0.6, 0.3, 50
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")


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


def _table(ready: list[Detection], frame_w: int, frame_h: int, n_frames: int,
           face_rule: dict | None = CONFIRM["face"]) -> dict[int, list]:
    """Frame -> [(box, padding, effect)] from the tracks the detections link into. `face_rule` None keeps every face."""
    out: dict[int, list] = {}
    for cat in LIVE_CATS:
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
            for g, (box, pad) in expand_target(target, n_frames, max(LEAD_FRAMES, lead)).items():
                out.setdefault(g, []).append((box, pad, "blur"))
    return out


def mask_boxes(f: int, ready: list[Detection], frame_w: int, frame_h: int, hold: int = HOLD_FRAMES,
               face_rule: dict | None = CONFIRM["face"]) -> list[tuple[Box, float, str]] | None:
    """Boxes to mask on frame f as (box, padding, effect), or None when f is not vouched. `ready` is windowed by
    the caller to [f - WINDOW_BACK, f + hold] and sorted by f. `hold` sets the vouch distance (b.f - a.f <= hold + 1)
    and the table horizon; how long a lost box is held is always render.py's HOLD_FRAMES. `face_rule` is the scan's
    face confirmation rule; None keeps every face track (camera mode)."""
    if not _vouched(f, ready, hold):
        return None
    return _table(ready, frame_w, frame_h, f + hold + 1, face_rule).get(f, [])


class BoxCache:
    """mask_boxes for a run of frames. link() + expand_target() over 25-40 detections is 10-20 ms of Python, which
    would fight the detector thread for the GIL at 30 fps, so the table is rebuilt only when the detections
    change (every 4-5 frames); the vouch check still runs per frame.

    `boxes(..., hold)`: as for mask_boxes, `hold` sets the vouch distance (b.f - a.f <= hold + 1) and the table
    horizon, while the hold-after-loss length itself is always render.py's HOLD_FRAMES."""

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._table: dict[int, list] = {}

    def boxes(self, f: int, ready: list[Detection], frame_w: int, frame_h: int, hold: int = HOLD_FRAMES,
              face_rule: dict | None = CONFIRM["face"]) -> list[tuple[Box, float, str]] | None:
        if not _vouched(f, ready, hold):
            return None
        key = (tuple(d.f for d in ready), frame_w, frame_h, hold, face_rule is None)
        if key != self._key:
            self._table = _table(ready, frame_w, frame_h, max(d.f for d in ready) + hold + 1, face_rule)
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


def yunet_detect(onnx_path: str | Path, score: float = YUNET_SCORE) -> Detect:
    """Faces only, on the CPU (OpenCV's YuNet, ~15-25 ms at 720p): the camera demo on a laptop with no GPU. Boxes are
    widened by FACE_EXPAND like EgoBlur's; there are no plates. The model file is fetched by hand, never here."""
    path = Path(onnx_path)
    if not path.is_file():
        raise FileNotFoundError(f"YuNet model not found: {path}. Download it once (about 230 KB) from {YUNET_URL}")
    net = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score_threshold=score, nms_threshold=YUNET_NMS,
                                    top_k=YUNET_TOP_K)
    lock, input_size = threading.Lock(), [(320, 320)]

    def detect(clip_f: int, frame: np.ndarray) -> Dets:
        h, w = frame.shape[:2]
        with lock:  # the detector thread is the only caller; setInputSize + detect must not interleave anyway
            if input_size[0] != (w, h):
                net.setInputSize((w, h))
                input_size[0] = (w, h)
            _, faces = net.detect(frame)
        rows = [] if faces is None else faces  # N x 15: x, y, w, h, five landmarks (x, y), score
        return {"face": [(_expand((float(r[0]), float(r[1]), float(r[0] + r[2]), float(r[1] + r[3])), FACE_EXPAND),
                          float(r[-1])) for r in rows],
                "license plate": []}
    return detect


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
    """Pixels the first SOFn marker declares (0 if none is found), read without decoding anything."""
    i = 2
    while i + 9 < len(b) and b[i] == 0xFF:
        if 0xC0 <= b[i + 1] <= 0xCF and b[i + 1] not in (0xC4, 0xC8, 0xCC):  # FF Cn len(2) prec(1) h(2) w(2)
            return int.from_bytes(b[i + 5:i + 7], "big") * int.from_bytes(b[i + 7:i + 9], "big")
        i += 2 + int.from_bytes(b[i + 2:i + 4], "big")
    return 0


def _live_size(src_w: int, src_h: int, height: int) -> tuple[int, int]:
    """(w, h) of a source scaled to `height` rows, aspect kept, width even."""
    return max(2, int(round(src_w * height / src_h / 2)) * 2), height


class LiveReplay:
    def __init__(self, clip: str | None, detect: Detect, height: int = LIVE_HEIGHT, delay_s: float = LIVE_DELAY_S,
                 out_w: int = LIVE_OUT_W, on_emit: OnEmit | None = None, push: bool = False) -> None:
        """Clip mode (default): `clip` is played back at its own fps, looping. Push mode (`push=True`, `clip` may be
        None): the caller feeds frames with push(); fps is a nominal 30 (it sizes the ring and numbers the frames),
        n_frames is 0 and size is (0, 0) until the first frame fixes it."""
        self.is_push = push
        self._face_rule = None if push else CONFIRM["face"]  # camera: every face is kept (see the module docstring)
        self._height, self._out_w = height, out_w
        if push:
            self.clip, self.fps, self.n_frames = str(clip) if clip else CAMERA_CLIP, PUSH_FPS, 0
            self.size = (0, 0)
        else:
            if clip is None:
                raise ValueError("a clip is required unless push=True")
            info = probe(str(clip))
            self.clip, self.fps, self.n_frames = str(clip), info.fps, max(1, info.n_frames)
            self.size = _live_size(info.width, info.height, height)  # (w, h)
        self.run, self.warming, self.delay_s = uuid.uuid4().hex[:8], False, delay_s
        self._detect, self._on_emit = detect, on_emit
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
        number. The first frame fixes `size`; a later frame of another aspect is stretched to it. Raises ValueError
        for bytes that are not a JPEG (nothing is counted) and RuntimeError when the engine is not running.

        Frame numbers follow elapsed time, not the arrival count (see the module docstring): a stall leaves a gap."""
        if not self.is_push:
            raise RuntimeError("push() needs an engine built with push=True")
        if not self.running:
            raise RuntimeError("the engine is not running")
        if not 0 < _jpeg_pixels(jpeg) <= MAX_JPEG_PIXELS:
            raise ValueError("JPEG too large or malformed")  # before imdecode allocates what the header claims
        try:
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            img = None
        if img is None:
            raise ValueError("not a decodable JPEG")
        size = self.size if self.size != (0, 0) else _live_size(img.shape[1], img.shape[0], self._height)
        if img.shape[1::-1] != size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        with self._cv:
            if self._stop.is_set():
                raise RuntimeError("the engine is not running")
            if self.size == (0, 0):
                self.size, self._display = size, self._display_for(size)
            elif img.shape[1::-1] != self.size:  # two first frames raced with different aspects: the loser conforms
                img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
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
            boxes = cache.boxes(slot.f, ready, w, h, face_rule=self._face_rule)
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
                # Over frames that have arrived, not frames sent: emits lag arrivals by the delay buffer
                "detected_share": n["passes"] / n["frames_in"] if n["frames_in"] else 0.0,
                "gap_frames": {"mean": n["gap_sum"] / n["gaps"] if n["gaps"] else None,
                               "max": n["gap_max"] if n["gaps"] else None},
                "hold_frames": HOLD_FRAMES, "frame_age_ms": {"p50": _pct(self._age_s, "p50")},
                "input_fps": self._input_fps() if self.is_push else None, "gaps_in": n["gaps_in"],
                "mode": "push" if self.is_push else "clip",
            }

    def _input_fps(self) -> float:
        """Pushes in the last INPUT_FPS_WINDOW_S seconds, per second (under-reads for the first 2 s). Caller holds _cv."""
        since = time.monotonic() - INPUT_FPS_WINDOW_S
        return sum(1 for t in self._arrivals if t >= since) / INPUT_FPS_WINDOW_S
