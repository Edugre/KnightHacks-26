"""Open-vocabulary detection + tracking -> Tracks.

Two implementations behind one interface:
- YoloeDetector: YOLOE-11s-seg with text prompts + Ultralytics tracker (runs on the R9700).
- StubDetector: replays a ground-truth file `<video>.tracks.json` (written by scripts/make_test_clip.py),
  so the UI, agent and renderer can be built on a laptop with no GPU.

Pick with CENSOR_STUB=1 (stub) or default (YOLOE, falling back to stub if Ultralytics is missing).
CLI: python -m censor.vision.detect clip.mp4 "license plate" ["face" ...]   # prints tracks JSON
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Protocol

from censor.core.tracks import Keyframe, Track, TrackStore

DETECT_IMGSZ = int(os.environ.get("CENSOR_IMGSZ", "1280"))
MIN_CONF = float(os.environ.get("CENSOR_MIN_CONF", "0.15"))
UNCERTAIN_BELOW = float(os.environ.get("CENSOR_UNCERTAIN_BELOW", "0.35"))


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
    def __init__(self, weights: str = "yoloe-11s-seg.pt", device: int | str = 0):
        from ultralytics import YOLOE  # imported lazily so laptops don't need it
        self._YOLOE = YOLOE
        self.weights, self.device = weights, device

    def run(self, video: str, queries: list[str], store: TrackStore) -> list[Track]:
        model = self._YOLOE(self.weights)  # fresh model per run: tracker state must not leak
        model.set_classes(queries, model.get_text_pe(queries))
        by_key: dict[tuple[int, int], list[Keyframe]] = {}
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
                    by_key.setdefault((int(cls), int(tid)), []).append(kf)
        found = []
        for (cls, _), kfs in sorted(by_key.items(), key=lambda kv: kv[1][0].f):
            t = Track(track_id=store.next_id(), label=queries[cls], source="detector", keyframes=kfs)
            store.add(t)
            found.append(t)
        for cls, kf in untracked:  # fail closed: a detection without a track id still gets masked
            t = Track(track_id=store.next_id(), label=queries[cls], source="detector",
                      keyframes=[kf.model_copy(update={"state": "uncertain"})])
            store.add(t)
            found.append(t)
        return found


def get_detector() -> Detector:
    if os.environ.get("CENSOR_STUB") == "1":
        return StubDetector()
    try:
        return YoloeDetector()
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
