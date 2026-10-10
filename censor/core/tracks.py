"""Track store: detector/tracker output cached per video (tracks.json).

The agent picks targets from here by id; the timeline copies keyframes from here.
"""
from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field

KeyframeState = Literal["detected", "tracked", "held", "uncertain"]


class Keyframe(BaseModel):
    f: int = Field(ge=0)
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    conf: float = Field(0.0, ge=0.0, le=1.0)
    state: KeyframeState = "detected"


class Track(BaseModel):
    track_id: str
    label: str  # the query that found it, e.g. "license plate"
    source: Literal["detector", "user_box", "rule"] = "detector"
    keyframes: list[Keyframe]
    thumbnail: Optional[str] = None

    @property
    def start(self) -> int:
        return self.keyframes[0].f

    @property
    def end(self) -> int:
        return self.keyframes[-1].f

    @property
    def max_conf(self) -> float:
        return max((k.conf for k in self.keyframes), default=0.0)


class TrackStore(BaseModel):
    queries: list[str] = Field(default_factory=list)  # prompts already run on this video
    tracks: dict[str, Track] = Field(default_factory=dict)

    def get(self, track_id: str) -> Optional[Track]:
        return self.tracks.get(track_id)

    def add(self, track: Track) -> None:
        self.tracks[track.track_id] = track

    def by_label(self, label: str) -> list[Track]:
        label = label.lower().strip()
        return [t for t in self.tracks.values() if t.label.lower() == label]

    def next_id(self) -> str:
        return f"trk_{len(self.tracks) + 1}"

    @classmethod
    def load(cls, path) -> "TrackStore":
        with open(path) as f:
            return cls.model_validate(json.load(f))

    def save(self, path) -> None:
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))
