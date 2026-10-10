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
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from censor.core.tracks import Keyframe, Track, TrackStore
from censor.core.video import frames

EGOBLUR_DIR = Path(os.environ.get("CENSOR_EGOBLUR_DIR", Path.home() / "datasets" / "egoblur"))
EGOBLUR_FILES = {"face": "ego_blur_face_gen1.jit", "license plate": "ego_blur_lp_gen1.jit"}
EGOBLUR_CONF = float(os.environ.get("CENSOR_EGOBLUR_CONF", "0.1"))
EGOBLUR_EVERY = int(os.environ.get("CENSOR_EGOBLUR_EVERY", "3"))
FACE_EXPAND = 0.3     # per side; with the default 0.15 mask padding this matches the benchmark's face padding 0.5
LINK_IOU = 0.3
LINK_GAP = 3 * EGOBLUR_EVERY  # frames a track may go unseen and still be continued

# What people (or the LLM) type, mapped to the specialist category that handles it.
ALIASES = {"face": "face", "faces": "face", "human face": "face", "person's face": "face",
           "license plate": "license plate", "license plates": "license plate", "licence plate": "license plate",
           "plate": "license plate", "plates": "license plate", "number plate": "license plate",
           "number plates": "license plate", "car plate": "license plate"}

Box = tuple[float, float, float, float]
Infer = Callable[[np.ndarray], list[tuple[Box, float]]]  # BGR frame -> [(xyxy, score)]


def load_egoblur(model_dir: Optional[Path] = None, device: str = "cuda", nms_iou: float = 0.3) -> dict[str, Infer]:
    """Meta's Gen1 TorchScript models, called the way its demo script does: BGR uint8 CHW in,
    (boxes, labels, scores, dims) out, then NMS."""
    import torch
    import torchvision
    model_dir = model_dir or EGOBLUR_DIR
    infer: dict[str, Infer] = {}
    for cat, name in EGOBLUR_FILES.items():
        model = torch.jit.load(str(model_dir / name), map_location="cpu").to(device)
        model.eval()

        def run(img: np.ndarray, model=model) -> list[tuple[Box, float]]:
            t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).to(device)
            with torch.no_grad():
                boxes, _, scores, _ = model(t)
            keep = torchvision.ops.nms(boxes, scores, nms_iou)
            return [(tuple(b), float(s)) for b, s in zip(boxes[keep].tolist(), scores[keep].tolist())]
        infer[cat] = run
    return infer


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


def link(detections: list[tuple[int, list[tuple[Box, float]]]], uncertain_below: float) -> list[list[Keyframe]]:
    """Greedy IoU tracker: each detection extends the best-overlapping recent track, or starts a new one."""
    tracks: list[list[Keyframe]] = []
    for f, dets in detections:
        live = [t for t in tracks if f - t[-1].f <= LINK_GAP]
        free = list(range(len(dets)))
        for t in sorted(live, key=lambda t: -t[-1].conf):
            if not free:
                break
            j = max(free, key=lambda j: _iou(t[-1].box, dets[j][0]))
            if _iou(t[-1].box, dets[j][0]) >= LINK_IOU:
                box, conf = dets[j]
                t.append(Keyframe(f=f, box=box, conf=min(conf, 1.0), state="detected" if conf >= uncertain_below else "uncertain"))
                free.remove(j)
        for j in free:
            box, conf = dets[j]
            tracks.append([Keyframe(f=f, box=box, conf=min(conf, 1.0),
                                    state="detected" if conf >= uncertain_below else "uncertain")])
    return tracks


class EgoBlurDetector:
    """Faces and plates with EgoBlur on every EGOBLUR_EVERY-th frame, linked into tracks."""

    _cache: dict[str, dict[str, Infer]] = {}  # models stay loaded across runs (420 MB each)

    def __init__(self, infer: Optional[dict[str, Infer]] = None, every: int = EGOBLUR_EVERY,
                 conf: float = EGOBLUR_CONF, uncertain_below: float = 0.35):
        self._infer, self.every, self.conf, self.uncertain_below = infer, every, conf, uncertain_below

    @property
    def infer(self) -> dict[str, Infer]:
        if self._infer is None:
            key = str(EGOBLUR_DIR)
            if key not in self._cache:
                self._cache[key] = load_egoblur()
            self._infer = self._cache[key]
        return self._infer

    def run(self, video: str, queries: list[str], store: TrackStore) -> list[Track]:
        wanted = {q: ALIASES[q.lower().strip()] for q in queries if q.lower().strip() in ALIASES}
        per_cat: dict[str, list[tuple[int, list[tuple[Box, float]]]]] = {c: [] for c in set(wanted.values())}
        for f, img in frames(video):
            if f % self.every:
                continue
            for cat in per_cat:
                dets = [(b, s) for b, s in self.infer[cat](img) if s >= self.conf]
                if cat == "face":
                    dets = [(_expand(b, FACE_EXPAND), s) for b, s in dets]
                per_cat[cat].append((f, dets))
        found = []
        for q, cat in wanted.items():
            for kfs in link(per_cat[cat], self.uncertain_below):
                t = Track(track_id=store.next_id(), label=q, source="detector", keyframes=kfs)
                store.add(t)
                found.append(t)
        return found


class RoutedDetector:
    """Faces and plates go to the specialist, every other query to the open-vocabulary detector."""

    def __init__(self, general, specialist):
        self.general, self.specialist = general, specialist

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
