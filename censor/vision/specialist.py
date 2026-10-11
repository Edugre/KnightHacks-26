"""Specialist detectors for faces and license plates, plus the router that picks them over YOLOE.

YOLOE (open vocabulary) handles "censor anything", but on PP4AV it found 0% of faces and 7% of plates
(IMPLEMENTATION.md, M2). Meta's EgoBlur Gen1 models (Apache 2.0) reach 0.74 / 0.65 there, so the router
sends face and plate queries to EgoBlur and every other query to YOLOE.

EgoBlur is a still-image detector (~0.1 s per model per frame on the R9700), so it runs every
EGOBLUR_EVERY frames and a greedy IoU tracker links detections into tracks; the renderer interpolates
between keyframes and holds after loss. Face boxes are widened by FACE_EXPAND per side because EgoBlur
boxes the face tightly while a blur has to cover the whole head (0.32 -> 0.74 face recall on PP4AV).

Model files: ego_blur_face_gen1.jit and ego_blur_lp_gen1.jit in ~/datasets/egoblur (CENSOR_EGOBLUR_DIR).
Meta's CDN refuses the AUP server, so download them on a laptop and upload them.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from censor.core.tracks import Keyframe, Track, TrackStore
from censor.core.video import frames

EGOBLUR_DIR = Path(os.environ.get("CENSOR_EGOBLUR_DIR", Path.home() / "datasets" / "egoblur"))
# Which model file each category uses. Gen2 files (ego_blur_face_gen2.jit, ego_blur_lp_gen2.jit) load too; see
# DETECTOR_OPTIONS.md item 1 for how they compare before switching.
EGOBLUR_FILES = {"face": os.environ.get("CENSOR_EGOBLUR_FACE", "ego_blur_face_gen1.jit"),
                 "license plate": os.environ.get("CENSOR_EGOBLUR_LP", "ego_blur_lp_gen1.jit")}
# Gen2 models don't resize their input; 0 runs them at full resolution, N resizes the short side to N first.
EGOBLUR_GEN2_SHORT = int(os.environ.get("CENSOR_EGOBLUR_GEN2_SHORT", "0"))
EGOBLUR_CONF = float(os.environ.get("CENSOR_EGOBLUR_CONF", "0.1"))
EGOBLUR_EVERY = int(os.environ.get("CENSOR_EGOBLUR_EVERY", "3"))
# fp16 on the GPU: 0.147 -> 0.049 s per 1080p frame on the R9700 with the same boxes (mean IoU 0.999 faces,
# 0.98 plates against fp32, none lost; bench/egoblur_speed.py). EgoBlur is most of the upload scan.
EGOBLUR_FP16 = os.environ.get("CENSOR_EGOBLUR_FP16", "1") == "1"
DECODE_AHEAD = 8  # frames a CPU thread decodes ahead of the GPU
WARM_SIZES = ((1080, 1920), (720, 1280))  # frame sizes warm_up runs the models on (height, width)
FACE_EXPAND = 0.3     # per side; with the default 0.15 mask padding this matches the benchmark's face padding 0.5
LINK_IOU = 0.3
LINK_NEAR = 0.8  # or centers within this many box sizes (similar size): 0.8 kept every labeled plate, 1.0 added false links
# A track has to earn its blur: at least one confident detection, or seen in enough checked frames. Detection
# itself stays at EGOBLUR_CONF (0.1) so faint plates aren't missed; this drops isolated weak blips (signs, tail
# lights), which the renderer would otherwise pad twice and hold for HOLD_FRAMES. Weak frames inside a
# confirmed track are kept. Faces get their own rule below. Tune with bench/review.py and bench/pp4av.py --plate-conf.
# A weak-only plate track that persists must also look like a plate: wider than tall and not huge. Static
# things (signs, patterns on clothes) can be detected weakly in every frame; plates are about 2:1 to 5:1, two-line ones squarer.
# One confident detection isn't enough: on street_plates, 47 of the 52 false plate tracks the app kept were a
# single confident blip (tail light, sign, shoe) that the renderer then padded and held, while every real plate
# track had 6 or more (bench/review.py). Such a track is judged like a weak one instead.
CONFIRM = {"license plate": {
    "min_conf": float(os.environ.get("CENSOR_PLATE_CONFIRM_CONF", "0.5")),
    "min_confident": int(os.environ.get("CENSOR_PLATE_CONFIRM_HITS", "2")),  # detections >= min_conf
    "min_frames": int(os.environ.get("CENSOR_PLATE_CONFIRM_FRAMES", "3")),
    # A faint but persistent track (the "7 7" street sign: 31 frames, mean 0.26) blurred a sign the whole clip;
    # no real plate on the labeled clips relied on a track this faint.
    "weak_min_mean": float(os.environ.get("CENSOR_PLATE_WEAK_MEAN", "0.3")),
    "min_aspect": float(os.environ.get("CENSOR_PLATE_MIN_ASPECT", "1.2")),        # width / height; PP4AV: 96.7% of real plates pass
    "max_height_frac": float(os.environ.get("CENSOR_PLATE_MAX_HEIGHT", "0.25")),  # of the frame height
}, "face": {
    # A big "face" needs a confident detection. EgoBlur's false faces are car backs: 300-900 px boxes at
    # 0.1-0.68 that blurred half the frame on street_plates; real faces that big were all detected at 1.00
    # (bench/review.py on the three clips). Small faint faces stay: in a crowd many real faces never pass 0.3.
    "big_frac": float(os.environ.get("CENSOR_FACE_BIG_FRAC", "0.2")),        # of the frame height, median box
    "big_min_conf": float(os.environ.get("CENSOR_FACE_BIG_CONF", "0.8")),
}}
LINK_GAP = 3 * EGOBLUR_EVERY  # frames a track may go unseen and still be continued

# What people (or the LLM) type, mapped to the specialist category that handles it.
ALIASES = {"face": "face", "faces": "face", "human face": "face", "person's face": "face",
           "license plate": "license plate", "license plates": "license plate", "licence plate": "license plate",
           "plate": "license plate", "plates": "license plate", "number plate": "license plate",
           "number plates": "license plate", "car plate": "license plate"}

Box = tuple[float, float, float, float]
Infer = Callable[[np.ndarray], list[tuple[Box, float]]]  # BGR frame -> [(xyxy, score)]


def load_egoblur(model_dir: Optional[Path] = None, device: str = "cuda", nms_iou: float = 0.3,
                 fp16: Optional[bool] = None, files: Optional[dict[str, str]] = None) -> dict[str, Infer]:
    """Meta's TorchScript models. Gen1 is called the way its demo script does: BGR uint8 CHW in,
    (boxes, labels, scores, dims) out. Gen2 is a Detectron2 model: a list of {"image": BGR float CHW} in,
    instances out, no resizing inside. Both get NMS after. On the GPU they run under fp16 autocast unless
    CENSOR_EGOBLUR_FP16=0."""
    import cv2
    import torch
    import torchvision  # registers torchvision::nms, which both models call
    model_dir = model_dir or EGOBLUR_DIR
    amp = (EGOBLUR_FP16 if fp16 is None else fp16) and str(device).startswith("cuda")
    infer: dict[str, Infer] = {}
    for cat, name in (files or EGOBLUR_FILES).items():
        model = torch.jit.load(str(model_dir / name), map_location="cpu").to(device)
        try:
            model.eval()
        except RuntimeError:  # Gen2 is frozen, already in eval mode, and refuses the call
            pass
        gen2 = _gen2_boxes(model) if hasattr(model, "inference") else None

        def run(img: np.ndarray, model=model, gen2=gen2) -> list[tuple[Box, float]]:
            s = 1.0
            if gen2 is not None and EGOBLUR_GEN2_SHORT:
                s = EGOBLUR_GEN2_SHORT / min(img.shape[:2])
                img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                boxes, scores = gen2(t.float()) if gen2 is not None else model(t)[::2]
            boxes, scores = boxes.float() / s, scores.float()
            keep = torchvision.ops.nms(boxes, scores, nms_iou)
            return [(tuple(b), float(s)) for b, s in zip(boxes[keep].tolist(), scores[keep].tolist())]
        infer[cat] = run
    return infer


def _gen2_boxes(model):
    """Gen2 returns Detectron2 instances, a TorchScript class Python can't receive; this compiled wrapper hands
    back (boxes, scores) as plain tensors."""
    import torch
    from typing import Tuple

    class Gen2(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            inst = self.inner.inference([{"image": img}], None, False)[0]
            boxes, scores = inst._pred_boxes, inst._scores
            assert boxes is not None and scores is not None
            return boxes.tensor, scores

    return torch.jit.script(Gen2(model))


def egoblur_available(model_dir: Optional[Path] = None) -> bool:
    model_dir = model_dir or EGOBLUR_DIR
    return all((model_dir / name).exists() for name in EGOBLUR_FILES.values())


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _expand(box: Box, by: float) -> Box:
    w, h = box[2] - box[0], box[3] - box[1]
    return (box[0] - w * by, box[1] - h * by, box[2] + w * by, box[3] + h * by)


def _near(a: Box, b: Box) -> float:
    """How far apart two similar-sized boxes' centers are, in box sizes; inf when the sizes differ over 2x."""
    aa, ba = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    if aa <= 0 or ba <= 0 or not 0.5 <= ba / aa <= 2:
        return float("inf")
    size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    return max(abs(a[0] + a[2] - b[0] - b[2]), abs(a[1] + a[3] - b[1] - b[3])) / 2 / size


def _same(a: Box, b: Box) -> bool:
    """The test link() uses to continue a track: enough overlap, or a similar-sized box close by. (Any overlap at
    all is not enough: a huge false "face" overlaps every real face near it.)"""
    return _iou(a, b) >= LINK_IOU or _near(a, b) <= LINK_NEAR


def link(detections: list[tuple[int, list[tuple[Box, float]]]], uncertain_below: float) -> list[list[Keyframe]]:
    """Greedy tracker: each detection extends the best-overlapping recent track, or starts a new one. A small
    object moving fast barely overlaps itself between checked frames (a 13 px plate moving 3 px a frame), so a
    detection of about the same size within LINK_NEAR box sizes of the track's last box also continues it."""
    tracks: list[list[Keyframe]] = []
    for f, dets in detections:
        live = [t for t in tracks if f - t[-1].f <= LINK_GAP]
        free = list(range(len(dets)))
        for t in sorted(live, key=lambda t: -t[-1].conf):
            if not free:
                break
            j = max(free, key=lambda j: _iou(t[-1].box, dets[j][0]))
            if _iou(t[-1].box, dets[j][0]) < LINK_IOU:
                j = min(free, key=lambda j: _near(t[-1].box, dets[j][0]))
                if _near(t[-1].box, dets[j][0]) > LINK_NEAR:
                    continue
            box, conf = dets[j]
            t.append(Keyframe(f=f, box=box, conf=min(conf, 1.0), state="detected" if conf >= uncertain_below else "uncertain"))
            free.remove(j)
        for j in free:
            box, conf = dets[j]
            tracks.append([Keyframe(f=f, box=box, conf=min(conf, 1.0),
                                    state="detected" if conf >= uncertain_below else "uncertain")])
    return tracks


def plate_shaped(box: Box, frame_h: int, min_aspect: Optional[float] = None, max_height_frac: Optional[float] = None) -> bool:
    rule = CONFIRM["license plate"]  # defaults come from the configured rule, so the app and the benchmark agree
    min_aspect = rule["min_aspect"] if min_aspect is None else min_aspect
    max_height_frac = rule["max_height_frac"] if max_height_frac is None else max_height_frac
    w, h = box[2] - box[0], box[3] - box[1]
    return h > 0 and w / h >= min_aspect and h <= max_height_frac * frame_h


def confirmed(kfs: list[Keyframe], rule: Optional[dict], frame_h: int = 0) -> bool:
    """Tracks with enough confident detections always count. The rest need enough frames and, when the rule
    asks, the median box of the track has to be plate-shaped (a median, so one odd frame doesn't decide)."""
    if rule is None:
        return True
    if "big_frac" in rule and frame_h:
        heights = sorted(k.box[3] - k.box[1] for k in kfs)
        if heights[len(heights) // 2] > rule["big_frac"] * frame_h and max(k.conf for k in kfs) < rule["big_min_conf"]:
            return False
    if "min_conf" not in rule or sum(k.conf >= rule["min_conf"] for k in kfs) >= rule.get("min_confident", 1):
        return True
    if len(kfs) < rule["min_frames"] or sum(k.conf for k in kfs) / len(kfs) < rule.get("weak_min_mean", 0.0):
        return False
    if "min_aspect" not in rule or not frame_h:
        return True
    widths = sorted(k.box[2] - k.box[0] for k in kfs)
    heights = sorted(k.box[3] - k.box[1] for k in kfs)
    median = (0.0, 0.0, widths[len(widths) // 2], heights[len(heights) // 2])
    return plate_shaped(median, frame_h, rule["min_aspect"], rule["max_height_frac"])


def absorb(kept: list[list[Keyframe]], dropped: list[list[Keyframe]], gap: int = LINK_GAP) -> list[list[Keyframe]]:
    """Fold a dropped fragment into a kept track when it ends just before the track starts (or starts just after
    it ends) and the boxes match there as link() matches them: the tracker split one object, often where it sped up, and
    dropping the piece left the plate visible for those frames. Returns the fragments still dropped."""
    rest = sorted(dropped, key=lambda t: t[0].f)
    merged = True
    while merged:
        merged = False
        for d in list(rest):
            for t in kept:
                if 0 < t[0].f - d[-1].f <= gap and _same(d[-1].box, t[0].box):
                    t[:0] = d
                elif 0 < d[0].f - t[-1].f <= gap and _same(t[-1].box, d[0].box):
                    t.extend(d)
                else:
                    continue
                rest.remove(d)
                merged = True
                break
    return rest


def build_tracks(seq: list[tuple[int, list[tuple[Box, float]]]], rule: Optional[dict], frame_h: int,
                 uncertain_below: float = 0.35) -> tuple[list[list[Keyframe]], list[list[Keyframe]]]:
    """(kept, dropped) tracks from per-frame detections: link, apply the confirmation rule, fold fragments in.
    bench/review.py replays this on cached detections, so keep the app's logic here and only here."""
    kept, dropped = [], []
    for kfs in link(seq, uncertain_below):
        (kept if confirmed(kfs, rule, frame_h) else dropped).append(kfs)
    return kept, absorb(kept, dropped)


class EgoBlurDetector:
    """Faces and plates with EgoBlur on every EGOBLUR_EVERY-th frame, linked into tracks."""

    _cache: dict[str, dict[str, Infer]] = {}  # models stay loaded across runs (420 MB each)

    def __init__(self, infer: Optional[dict[str, Infer]] = None, every: int = EGOBLUR_EVERY,
                 conf: float = EGOBLUR_CONF, uncertain_below: float = 0.35,
                 confirm: Optional[dict[str, dict]] = None):
        self._infer, self.every, self.conf, self.uncertain_below = infer, every, conf, uncertain_below
        self.confirm = CONFIRM if confirm is None else confirm
        self.dropped: dict[str, int] = {}  # unconfirmed tracks per category in the last run, for logs and tests

    @property
    def infer(self) -> dict[str, Infer]:
        if self._infer is None:
            key = str(EGOBLUR_DIR)
            if key not in self._cache:
                self._cache[key] = load_egoblur()
            self._infer = self._cache[key]
        return self._infer

    _lock = threading.Lock()  # scans of different queries may run at once; the models are shared

    def run(self, video: str, queries: list[str], store: TrackStore) -> list[Track]:
        with self._lock:
            return self._run(video, queries, store)

    def warm_up(self, sizes=WARM_SIZES) -> None:
        """Load the models and run each twice per frame size on noise. A freshly loaded TorchScript model is
        slow on its first calls (with loading, about 6 s per model on the R9700), which the first upload
        would otherwise pay for. Scans wait for this to finish."""
        rng = np.random.default_rng(0)
        with self._lock:
            for h, w in sizes:
                img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
                for fn in self.infer.values():
                    fn(img)
                    fn(img)

    def _run(self, video: str, queries: list[str], store: TrackStore) -> list[Track]:
        wanted = {q: ALIASES[q.lower().strip()] for q in queries if q.lower().strip() in ALIASES}
        per_cat: dict[str, list[tuple[int, list[tuple[Box, float]]]]] = {c: [] for c in set(wanted.values())}
        frame_h = 0
        for f, img in frames(video, every=self.every, prefetch=DECODE_AHEAD):
            frame_h = img.shape[0]
            for cat in per_cat:
                dets = [(b, s) for b, s in self.infer[cat](img) if s >= self.conf]
                if cat == "face":
                    dets = [(_expand(b, FACE_EXPAND), s) for b, s in dets]
                per_cat[cat].append((f, dets))
        found = []
        self.dropped = {}
        for q, cat in wanted.items():
            kept, dropped = build_tracks(per_cat[cat], self.confirm.get(cat), frame_h, self.uncertain_below)
            if dropped:
                self.dropped[cat] = self.dropped.get(cat, 0) + len(dropped)
            for kfs in kept:
                t = Track(track_id=store.next_id(), label=q, source="detector", keyframes=kfs)
                store.add(t)
                found.append(t)
        if self.dropped:
            print(f"[egoblur] {video}: dropped unconfirmed tracks {self.dropped}", file=sys.stderr)
        return found


class RoutedDetector:
    """Faces and plates go to the specialist, every other query to the open-vocabulary detector."""

    def __init__(self, general, specialist):
        self.general, self.specialist = general, specialist

    def warm_up(self) -> None:
        for d in (self.specialist, self.general):
            if hasattr(d, "warm_up"):
                d.warm_up()

    def run(self, video: str, queries: list[str], store: TrackStore) -> list[Track]:
        special = [q for q in queries if q.lower().strip() in ALIASES]
        rest = [q for q in queries if q not in special]
        found = self.specialist.run(video, special, store) if special else []
        if rest:
            found += self.general.run(video, rest, store)
        return found


def routed_or(general) -> object:
    """Wrap `general` with EgoBlur when its model files are present; otherwise say so and use `general` alone."""
    if egoblur_available():
        return RoutedDetector(general, EgoBlurDetector())
    print(f"[detect] EgoBlur models not found in {EGOBLUR_DIR}; faces and plates use YOLOE "
          "(0% face recall on PP4AV)", file=sys.stderr)
    return general
