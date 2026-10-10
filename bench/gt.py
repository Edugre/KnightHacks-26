"""Ground-truth labels for the benchmark: bench/labels/<clip>.json.

Boxes are labeled on every `every`-th frame (default 10). Each object has a stable gt_id, so
bench/requests.yaml can name what a request must mask and what it must leave visible. A labeled frame
with no boxes is a claim that nothing is there, so it still counts toward over-blur.

CLI (synthetic clip: labels straight from the make_test_clip.py sidecar):
  python -m bench.gt from-sidecar bench/clips/synthetic.mp4 bench/labels/synthetic.json [--every 10] [--desc person_1=red]
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


class GTObject(BaseModel):
    category: Category
    desc: str = ""


class GTBox(BaseModel):
    gt_id: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


class LabeledFrame(BaseModel):
    f: int = Field(ge=0)
    boxes: list[GTBox] = Field(default_factory=list)


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
        if window is None:
            return self.frames
        return [fr for fr in self.frames if window[0] <= fr.f / self.video.fps <= window[1]]

    @classmethod
    def load(cls, path) -> "LabelSet":
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(json.load(f))

    def save(self, path) -> None:
        # One line per labeled frame keeps hand edits and diffs readable.
        d = self.model_dump()
        frames = ",\n".join("    " + json.dumps(fr) for fr in d.pop("frames"))
        head = json.dumps(d, indent=2)[:-2]  # drop the closing "\n}"
        Path(path).write_text(f'{head},\n  "frames": [\n{frames}\n  ]\n}}\n', encoding="utf-8")


def from_sidecar(video: str | Path, clip: Optional[str] = None, every: int = 10,
                 descs: Optional[dict[str, str]] = None) -> LabelSet:
    """Labels from the ground-truth sidecar make_test_clip.py writes. gt_ids are <category>_<n> in track order."""
    store = TrackStore.load(f"{video}.tracks.json")
    info = probe(str(video))
    objects: dict[str, GTObject] = {}
    boxes: dict[str, dict[int, tuple]] = {}
    for t in store.tracks.values():
        if t.label not in CATEGORIES:
            raise ValueError(f"track {t.track_id} has label {t.label!r}, not one of {CATEGORIES}")
        gid = f"{t.label.replace(' ', '_')}_{sum(o.category == t.label for o in objects.values()) + 1}"
        objects[gid] = GTObject(category=t.label, desc=(descs or {}).get(gid, ""))
        boxes[gid] = {k.f: k.box for k in t.keyframes}
    frames = [LabeledFrame(f=f, boxes=[GTBox(gt_id=g, box=b[f]) for g, b in boxes.items() if f in b])
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
    a = ap.parse_args()
    labels = from_sidecar(a.video, a.clip, a.every, dict(d.split("=", 1) for d in a.desc))
    labels.save(a.out)
    print(f"wrote {a.out}: {len(labels.objects)} objects, {len(labels.frames)} labeled frames")
