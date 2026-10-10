"""Ground-truth labels for the benchmark: bench/labels/<clip>.json.

Boxes are labeled on every `every`-th frame (default 10). Each object has a stable gt_id, so
bench/requests.yaml can name what a request must mask and what it must leave visible. A labeled frame
with no boxes is a claim that nothing is there, so it still counts toward over-blur.

Frames with reviewed=false are detector drafts nobody has checked yet. They are never scored: a draft
inherits the detector's misses, and scoring the detector against its own output would prove nothing.

CLI:
  python -m bench.gt from-sidecar bench/clips/synthetic.mp4 bench/labels/synthetic.json [--every 10] [--desc person_1=red]
  python -m bench.gt prelabel bench/clips/street_01.mp4 [--queries face "license plate"]   # GPU: detector drafts
Then check and fix the drafts with bench/label_server.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from censor.core.timeline import VideoInfo
from censor.core.tracks import TrackStore
from censor.core.video import probe

CATEGORIES = ("face", "license plate", "screen", "text", "person")
Category = Literal["face", "license plate", "screen", "text", "person"]
DRAFT_QUERIES = ("face", "license plate", "screen", "person")  # "text" is noisy as a draft prompt; add it with --queries
LABELS_DIR = Path(__file__).resolve().parent / "labels"


class GTObject(BaseModel):
    category: Category
    desc: str = ""


class GTBox(BaseModel):
    gt_id: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


class LabeledFrame(BaseModel):
    f: int = Field(ge=0)
    boxes: list[GTBox] = Field(default_factory=list)
    reviewed: bool = True  # False: detector draft, not scored until a person confirms it


class LabelSet(BaseModel):
    clip: str
    video: VideoInfo
    every: int = Field(10, ge=1)
    objects: dict[str, GTObject]
    frames: list[LabeledFrame]

    @model_validator(mode="after")
    def _consistent(self) -> "LabelSet":
        unknown = sorted({b.gt_id for fr in self.frames for b in fr.boxes} - set(self.objects))
        if unknown:
            raise ValueError(f"boxes reference gt_ids missing from objects: {unknown}")
        fs = [fr.f for fr in self.frames]
        if fs != sorted(set(fs)):
            raise ValueError("labeled frames must be unique and in increasing order")
        return self

    def frames_in(self, window: Optional[tuple[float, float]] = None) -> list[LabeledFrame]:
        """Reviewed frames only, optionally limited to a time window in seconds."""
        return [fr for fr in self.frames if fr.reviewed
                and (window is None or window[0] <= fr.f / self.video.fps <= window[1])]

    @property
    def n_drafts(self) -> int:
        return sum(not fr.reviewed for fr in self.frames)

    @classmethod
    def load(cls, path) -> "LabelSet":
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(json.load(f))

    def save(self, path) -> None:
        # One line per labeled frame keeps hand edits and diffs readable.
        d = self.model_dump()
        for fr in d["frames"]:
            if fr["reviewed"]:
                del fr["reviewed"]  # the default; only drafts carry the flag
        frames = ",\n".join("    " + json.dumps(fr) for fr in d.pop("frames"))
        head = json.dumps(d, indent=2)[:-2]  # drop the closing "\n}"
        Path(path).write_text(f'{head},\n  "frames": [\n{frames}\n  ]\n}}\n', encoding="utf-8")


def from_sidecar(video: str | Path, clip: Optional[str] = None, every: int = 10,
                 descs: Optional[dict[str, str]] = None) -> LabelSet:
    """Labels from the ground-truth sidecar make_test_clip.py writes. Exact, so frames are marked reviewed."""
    return from_store(video, TrackStore.load(f"{video}.tracks.json"), clip, every, descs, reviewed=True)


def prelabel(video: str | Path, queries: tuple[str, ...] = DRAFT_QUERIES, clip: Optional[str] = None,
             every: int = 10, detector=None) -> LabelSet:
    """Detector drafts for a real clip: every sampled frame is unreviewed until a person confirms it."""
    from censor.vision.detect import get_detector  # lazy: pulls in the detector stack
    store = TrackStore()
    (detector or get_detector()).run(str(video), list(queries), store)
    return from_store(video, store, clip, every, reviewed=False)


def from_store(video: str | Path, store: TrackStore, clip: Optional[str] = None, every: int = 10,
               descs: Optional[dict[str, str]] = None, reviewed: bool = True) -> LabelSet:
    """One object per track; gt_ids are <category>_<n> in track order."""
    info = probe(str(video))
    objects: dict[str, GTObject] = {}
    boxes: dict[str, dict[int, tuple]] = {}
    for t in store.tracks.values():
        if t.label not in CATEGORIES:
            raise ValueError(f"track {t.track_id} has label {t.label!r}, not one of {CATEGORIES}")
        gid = f"{t.label.replace(' ', '_')}_{sum(o.category == t.label for o in objects.values()) + 1}"
        objects[gid] = GTObject(category=t.label, desc=(descs or {}).get(gid, ""))
        boxes[gid] = {k.f: k.box for k in t.keyframes}
    frames = [LabeledFrame(f=f, boxes=[GTBox(gt_id=g, box=b[f]) for g, b in boxes.items() if f in b], reviewed=reviewed)
              for f in range(0, info.n_frames, every)]
    return LabelSet(clip=clip or Path(video).stem, video=info, every=every, objects=objects, frames=frames)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("from-sidecar", help="label a clip from its make_test_clip.py .tracks.json sidecar")
    p.add_argument("video")
    p.add_argument("out")
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--clip", help="clip id (default: video file stem)")
    p.add_argument("--desc", action="append", default=[], metavar="GT_ID=TEXT", help="describe an object (repeatable)")
    p = sub.add_parser("prelabel", help="detector drafts for a real clip (GPU server); confirm them in the label tool")
    p.add_argument("video")
    p.add_argument("--out", help="default: bench/labels/<clip>.json")
    p.add_argument("--queries", nargs="+", default=list(DRAFT_QUERIES), choices=CATEGORIES)
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--clip", help="clip id (default: video file stem)")
    p.add_argument("--force", action="store_true", help="overwrite existing labels")
    a = ap.parse_args()

    if a.cmd == "from-sidecar":
        labels, out = from_sidecar(a.video, a.clip, a.every, dict(d.split("=", 1) for d in a.desc)), Path(a.out)
    else:
        out = Path(a.out or LABELS_DIR / f"{a.clip or Path(a.video).stem}.json")
        if out.exists() and not a.force:
            raise SystemExit(f"{out} exists and may hold hand labels; pass --force to overwrite it")
        labels = prelabel(a.video, tuple(a.queries), a.clip, a.every)
    labels.save(out)
    print(f"wrote {out}: {len(labels.objects)} objects, {len(labels.frames)} frames ({labels.n_drafts} drafts)")
