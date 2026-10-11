"""Mask timeline: the shared contract (IMPLEMENTATION.md section 3).

Ops are an append-only log produced by the agent and the UI. The derived Timeline is
what the renderer reads. Undo pops the last op and re-folds. Nothing here touches pixels.
"""
from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from censor.core.tracks import Keyframe, TrackStore

SCHEMA_VERSION = 2  # 2: remove_targets honors time_range (unmask only that window); used to move a mask by hand

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
    time_range: Optional[tuple[float, float]] = Field(None, description=(
        "Seconds, inclusive. add_targets: mask only this window of the track. remove_targets: unmask only this "
        "window and keep the rest (moving a mask is remove_targets [f, end] on the old track plus add_targets "
        "[f, end] on a track re-tracked from the corrected box). None means the whole track."))
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
    undone: list[Op] = Field(default_factory=list)  # popped by undo, latest last; redo puts them back

    def next_id(self) -> str:
        return f"op_{len(self.ops) + 1}"

    def append(self, op: Op) -> Op:
        self.ops.append(op)
        self.undone.clear()  # a new edit starts a new history: what was undone can't come back on top of it
        return op

    def undo(self) -> Optional[Op]:
        if not self.ops:
            return None
        op = self.ops.pop()
        self.undone.append(op)
        return op

    def redo(self) -> Optional[Op]:
        """Put back the op the latest undo took, with its own id (ids follow the log's length, so it's free)."""
        if not self.undone:
            return None
        op = self.undone.pop()
        self.ops.append(op)
        return op


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
                if op.time_range is None or tid not in tl.targets:
                    tl.targets.pop(tid, None)
                    continue
                f0, f1 = video.frame_at(op.time_range[0]), video.frame_at(op.time_range[1])
                kept = [k for k in tl.targets[tid].keyframes if not f0 <= k.f <= f1]
                if kept:
                    tl.targets[tid].keyframes = kept
                else:
                    tl.targets.pop(tid)
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
