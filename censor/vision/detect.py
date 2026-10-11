"""Open-vocabulary detection + tracking -> Tracks.

Two implementations behind one interface:
- YoloeDetector: YOLOE-11s-seg with text prompts + Ultralytics tracker (runs on the R9700).
- StubDetector: replays a ground-truth file `<video>.tracks.json` (written by scripts/make_test_clip.py),
  so the UI, agent and renderer can be built on a laptop with no GPU.

Pick with CENSOR_STUB=1 (stub) or default (YOLOE, falling back to stub if Ultralytics is missing).
By default face and plate queries are routed to EgoBlur (censor/vision/specialist.py) when its models exist.
CLI: python -m censor.vision.detect clip.mp4 "license plate" ["face" ...]   # prints tracks JSON
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Protocol

from censor.core.tracks import Keyframe, Track, TrackStore
from censor.vision import stitch as stitch_mod

DETECT_IMGSZ = int(os.environ.get("CENSOR_IMGSZ", "1280"))
MIN_CONF = float(os.environ.get("CENSOR_MIN_CONF", "0.15"))
UNCERTAIN_BELOW = float(os.environ.get("CENSOR_UNCERTAIN_BELOW", "0.35"))
# Join pieces of one object's track that the tracker split (censor/vision/stitch.py). CENSOR_STITCH=0 turns it off.
STITCH = os.environ.get("CENSOR_STITCH", "1") == "1"


class Detector(Protocol):
    def run(self, video: str, queries: list[str], store: TrackStore) -> list[Track]: ...


class StubDetector:
    """Returns ground-truth tracks whose label matches a query. No GPU, no models."""

    def run(self, video: str, queries: list[str], store: TrackStore) -> list[Track]:
        gt_path = Path(str(video) + ".tracks.json")
        if not gt_path.exists():
            return []
        gt = TrackStore.load(gt_path)
        wanted = {q.lower().strip() for q in queries}
        found = []
        for t in gt.tracks.values():
            if t.label.lower() in wanted:
                t = t.model_copy(update={"track_id": store.next_id()})
                store.add(t)
                found.append(t)
        return found


class YoloeDetector:
    # Loading YOLOE and its text encoder took several seconds per search, so the model stays loaded.
    # One lock per process: runs share the model, and set_classes/track must not interleave.
    _models: dict = {}
    _lock = threading.Lock()

    def __init__(self, weights: str = "yoloe-11s-seg.pt", device: int | str = 0):
        from ultralytics import YOLOE  # imported lazily so laptops don't need it
        self._YOLOE = YOLOE
        self.weights, self.device = weights, device

    def run(self, video: str, queries: list[str], store: TrackStore) -> list[Track]:
        with self._lock:
            return self._run(video, queries, store)

    def _model(self):
        model = self._models.get(self.weights)
        if model is None:
            model = self._models[self.weights] = self._YOLOE(self.weights)
        return model

    def warm_up(self) -> None:
        """Load YOLOE and its text encoder and run one frame, so the first scan doesn't pay for it."""
        import numpy as np
        with self._lock:
            model = self._model()
            model.set_classes(["person"], model.get_text_pe(["person"]))
            model.predict(np.zeros((720, 1280, 3), np.uint8), device=self.device, imgsz=DETECT_IMGSZ, verbose=False)

    def _run(self, video: str, queries: list[str], store: TrackStore) -> list[Track]:
        model = self._model()
        model.predictor = None  # new predictor, so tracker state from the last video can't leak in
        model.set_classes(queries, model.get_text_pe(queries))
        by_key: dict[tuple[int, int], list[Keyframe]] = {}
        hists: dict[tuple[int, int], list] = {}  # colors at each track's first and last few boxes, for stitching
        untracked: list[tuple[int, Keyframe]] = []
        results = model.track(source=str(video), stream=True, persist=True, device=self.device,
                              imgsz=DETECT_IMGSZ, conf=MIN_CONF, verbose=False, tracker="botsort.yaml")
        for f, r in enumerate(results):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            ids = r.boxes.id.tolist() if r.boxes.id is not None else [None] * len(r.boxes)
            for box, conf, cls, tid in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist(), ids):
                state = "detected" if conf >= UNCERTAIN_BELOW else "uncertain"
                kf = Keyframe(f=f, box=tuple(box), conf=float(conf), state=state)
                if tid is None:
                    untracked.append((int(cls), kf))
                else:
                    key = (int(cls), int(tid))
                    by_key.setdefault(key, []).append(kf)
                    if STITCH:
                        h = hists.setdefault(key, [])
                        h.append(stitch_mod.color_hist(r.orig_img, box))
                        if len(h) > 6:  # keep the first 3 and the latest 3
                            del h[3]
        found = []
        for cls in sorted({c for c, _ in by_key}):
            keys = [k for k in by_key if k[0] == cls]
            pieces = [by_key[k] for k in keys]
            if STITCH:
                ends = [stitch_mod.Ends(start=_mean(hists[k][:3]), end=_mean(hists[k][-3:])) for k in keys]
                pieces = stitch_mod.stitch(pieces, ends)
            for kfs in sorted(pieces, key=lambda kfs: kfs[0].f):
                t = Track(track_id=store.next_id(), label=queries[cls], source="detector", keyframes=kfs)
                store.add(t)
                found.append(t)
        for cls, kf in untracked:  # fail closed: a detection without a track id still gets masked
            t = Track(track_id=store.next_id(), label=queries[cls], source="detector",
                      keyframes=[kf.model_copy(update={"state": "uncertain"})])
            store.add(t)
            found.append(t)
        return found


def _mean(hs: list):
    import numpy as np
    return np.mean(hs, axis=0).astype(np.float32) if hs else None


def get_detector() -> Detector:
    if os.environ.get("CENSOR_STUB") == "1":
        return StubDetector()
    try:
        from censor.vision.specialist import routed_or
        return routed_or(YoloeDetector())  # faces and plates -> EgoBlur when its models are present
    except ImportError:
        print("[detect] ultralytics not installed; using StubDetector", file=sys.stderr)
        return StubDetector()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    store = TrackStore()
    tracks = get_detector().run(sys.argv[1], sys.argv[2:], store)
    print(store.model_dump_json(indent=2))
    print(f"{len(tracks)} tracks", file=sys.stderr)
