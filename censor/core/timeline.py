"""Mask timeline: the shared contract (IMPLEMENTATION.md section 3).

Ops are an append-only log produced by the agent and the UI. The derived Timeline is
what the renderer reads. Undo pops the last op and re-folds. Nothing here touches pixels.
"""
from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from censor.core.tracks import Keyframe, TrackStore

SCHEMA_VERSION = 1

Effect = Literal["blur", "pixelate", "box"]
OpKind = Literal["add_targets", "remove_targets", "set_effect", "add_rule"]
Source = Literal["detector", "user_box", "rule"]


class VideoInfo(BaseModel):
    fps: float
    width: int
    height: int
    n_frames: int

    def frame_at(self, seconds: float) -> int:
        return max(0, min(self.n_frames - 1, int(round(seconds * self.fps))))


class Op(BaseModel):
    op_id: str
    request_id: str = "manual"
    kind: OpKind
    targets: list[str] = Field(default_factory=list)
    effect: Effect = "blur"
    padding: float = Field(0.15, ge=0.0, le=2.0)
    time_range: Optional[tuple[float, float]] = None  # seconds
    category: Optional[str] = None  # add_rule only (live mode, stretch)

    @field_validator("targets")
    @classmethod
    def _targets_for_kind(cls, v, info):
        kind = info.data.get("kind")
        if kind in ("add_targets", "remove_targets", "set_effect") and not v:
            raise ValueError(f"{kind} needs at least one target id")
        return v


class Target(BaseModel):
    label: str
    source: Source
    effect: Effect = "blur"
    padding: float = 0.15
    keyframes: list[Keyframe]


class Timeline(BaseModel):
    version: int = SCHEMA_VERSION
    video: VideoInfo
    targets: dict[str, Target] = Field(default_factory=dict)

    @classmethod
    def load(cls, path) -> "Timeline":
        with open(path) as f:
            return cls.model_validate(json.load(f))

    def save(self, path) -> None:
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))


class OpLog(BaseModel):
    ops: list[Op] = Field(default_factory=list)

    def next_id(self) -> str:
        return f"op_{len(self.ops) + 1}"

    def append(self, op: Op) -> Op:
        self.ops.append(op)
        return op

    def undo(self) -> Optional[Op]:
        return self.ops.pop() if self.ops else None


def fold(ops: list[Op], tracks: TrackStore, video: VideoInfo) -> Timeline:
    """Replay ops over the track store into the renderer's Timeline. Pure and deterministic."""
    tl = Timeline(video=video)
    for op in ops:
        if op.kind == "add_targets":
            for tid in op.targets:
                track = tracks.get(tid)
                if track is None:
                    raise KeyError(f"unknown track id {tid!r} in {op.op_id}")
                kfs = track.keyframes
                if op.time_range is not None:
                    f0, f1 = video.frame_at(op.time_range[0]), video.frame_at(op.time_range[1])
                    kfs = [k for k in kfs if f0 <= k.f <= f1]
                if tid in tl.targets:  # extend an existing target
                    merged = {k.f: k for k in tl.targets[tid].keyframes}
                    merged.update({k.f: k for k in kfs})
                    tl.targets[tid].keyframes = [merged[f] for f in sorted(merged)]
                    tl.targets[tid].effect, tl.targets[tid].padding = op.effect, op.padding
                else:
                    tl.targets[tid] = Target(label=track.label, source=track.source,
                                             effect=op.effect, padding=op.padding, keyframes=list(kfs))
        elif op.kind == "remove_targets":
            for tid in op.targets:
                tl.targets.pop(tid, None)
        elif op.kind == "set_effect":
            for tid in op.targets:
                if tid in tl.targets:
                    tl.targets[tid].effect = op.effect
        elif op.kind == "add_rule":
            pass  # standing rules are applied by the live pipeline (stretch), not here
    return tl


def export_json_schema() -> dict:
    return {
        "$comment": "Generated from censor/core/timeline.py. Do not edit by hand; run `python -m censor.core.timeline`.",
        "Timeline": Timeline.model_json_schema(),
        "Op": Op.model_json_schema(),
    }


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).resolve().parents[2] / "schema" / "timeline.schema.json"
    out.write_text(json.dumps(export_json_schema(), indent=2) + "\n")
    print(f"wrote {out}")
