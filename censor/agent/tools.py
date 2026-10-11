"""The agent's toolbox (IMPLEMENTATION.md section 4). Every edit goes through the op log.

A Session holds one video's state: VideoInfo, TrackStore, OpLog. It persists to its work dir,
so a server restart doesn't lose edits. Tools return compact JSON-able dicts, never images.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from censor.core.timeline import Op, OpLog, Timeline, VideoInfo, fold
from censor.core.tracks import TrackStore
from censor.agent import intents
from censor.core.video import probe
from censor.vision import detect as detect_mod
from censor.vision import track as track_mod
from censor.vision import vlm as vlm_mod

EFFECTS = ("blur", "pixelate", "box")
RULE_CATEGORIES = ("license plate", "face", "person", "screen", "text")
COMPOUND_QUERY = re.compile(r"\b(wearing|with|in|holding|who|that|which|near|on|at|carrying|has|having)\b")


class ToolError(ValueError):
    pass


class Session:
    def __init__(self, video: str | Path, workdir: str | Path, detector=None):
        self.video = str(video)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()  # guards tracks, ops and the files; held briefly, never during a scan
        self._scans: dict[str, threading.Event] = {}  # query -> set when its scan is finished
        self._scan_lock = threading.Lock()
        self.info: VideoInfo = probe(self.video)
        self.detector = detector or detect_mod.get_detector()
        self.tracks = TrackStore.load(self._p("tracks.json")) if self._p("tracks.json").exists() else TrackStore()
        self.log = OpLog.model_validate_json(self._p("ops.json").read_text()) if self._p("ops.json").exists() else OpLog()

    def _p(self, name: str) -> Path:
        return self.workdir / name

    def save(self) -> None:
        with self.lock:
            self.tracks.save(self._p("tracks.json"))
            self._p("ops.json").write_text(self.log.model_dump_json(indent=2))

    @property
    def timeline(self) -> Timeline:
        with self.lock:
            return fold(self.log.ops, self.tracks, self.info)

    def _ensure_scanned(self, q: str) -> None:
        """Run the detector for q once. A scan of another query never blocks this one; a second caller for
        the same query (say the upload pre-scan and a request) waits for that scan instead of starting another."""
        with self._scan_lock:
            if q in self.tracks.queries:
                return
            done = self._scans.get(q)
            owner = done is None
            if owner:
                done = self._scans[q] = threading.Event()
        if not owner:
            t0 = time.perf_counter()
            done.wait()
            print(f"[scan] {q}: waited {time.perf_counter() - t0:.1f} s for the scan already running", file=sys.stderr)
            if q not in self.tracks.queries:
                raise ToolError(f"detector failed for {q!r} (see the server log)")
            return
        found = TrackStore()
        t0 = time.perf_counter()
        try:
            self.detector.run(self.video, [q], found)
        except Exception as e:  # surface to the agent/UI instead of a 500; nothing is marked as searched
            with self._scan_lock:
                self._scans.pop(q, None)
            done.set()
            raise ToolError(f"detector failed for {q!r}: {type(e).__name__}: {e}") from e
        with self.lock:  # new ids here: scans run in parallel, each into its own store
            for t in found.tracks.values():
                self.tracks.add(t.model_copy(update={"track_id": self.tracks.next_id()}))
            self.tracks.queries.append(q)
            self.save()
        print(f"[scan] {q}: {len(found.tracks)} tracks in {time.perf_counter() - t0:.1f} s", file=sys.stderr)
        done.set()

    def _append(self, request_id: str, **kw) -> str:
        with self.lock:
            op = self.log.append(Op(op_id=self.log.next_id(), request_id=request_id, **kw))
            self.save()
        return op.op_id

    def _check_ids(self, track_ids: list[str]) -> None:
        if not track_ids:
            raise ToolError("track_ids is empty: call find_objects or track_box first")
        missing = [t for t in track_ids if self.tracks.get(t) is None]
        if missing:
            raise ToolError(f"unknown track ids {missing}")

    # ---- tools -------------------------------------------------------------------------------

    def video_info(self) -> dict:
        v = self.info
        return {"duration": round(v.n_frames / v.fps, 2), "fps": v.fps, "n_frames": v.n_frames,
                "width": v.width, "height": v.height}

    def resolve_time(self, phrase: str) -> list[float]:
        dur = self.info.n_frames / self.info.fps
        p = phrase.lower().strip()
        if m := re.search(r"(\d+):(\d{2})", p):
            t = int(m[1]) * 60 + int(m[2])
            return [max(0.0, t - 2.0), min(dur, t + 2.0)]
        if "halfway" in p or "middle" in p:
            return [dur * 0.4, dur * 0.6]
        if m := re.search(r"first (\d+) ?(s|sec|seconds|min|minutes)", p):
            n = int(m[1]) * (60 if m[2].startswith("m") else 1)
            return [0.0, min(dur, float(n))]
        if m := re.search(r"last (\d+) ?(s|sec|seconds|min|minutes)", p):
            n = int(m[1]) * (60 if m[2].startswith("m") else 1)
            return [max(0.0, dur - n), dur]
        if re.search(r"\b(beginning|start)\b", p):
            return [0.0, dur * 0.2]
        if re.search(r"\bend\b", p):
            return [dur * 0.8, dur]
        raise ToolError(f"can't resolve time phrase {phrase!r}")

    def find_objects(self, query: str, time_range: Optional[list[float]] = None) -> list[dict]:
        q = query.lower().strip()
        if q in intents.SUBJECTS:  # the detector finds people; who they are is the VLM's question
            raise ToolError(f"query {query!r} is a kind of person. Call find_objects with 'person', then check_attribute "
                            f"with a question like 'Is this a {intents.SUBJECTS[q]}?' on the returned track ids.")
        if COMPOUND_QUERY.search(q) or len(q.split()) > 3:
            raise ToolError(f"query {query!r} mixes an object with attributes. Call find_objects with just the "
                            "object (e.g. 'person'), then check_attribute on the returned track ids.")
        self._ensure_scanned(q)
        with self.lock:
            tracks = list(self.tracks.by_label(q))
        out = []
        for t in tracks:
            start, end = t.start / self.info.fps, t.end / self.info.fps
            if time_range and (end < time_range[0] or start > time_range[1]):
                continue
            out.append({"track_id": t.track_id, "label": t.label, "start": round(start, 2),
                        "end": round(end, 2), "max_conf": round(t.max_conf, 2)})
        return out

    def check_attribute(self, track_ids: list[str], question: str) -> list[dict]:
        self._check_ids(track_ids)
        tracks = [self.tracks.get(t) for t in track_ids]
        return [{"track_id": a.track_id, "answer": a.answer, "conf": a.conf, "p_yes": round(a.p_yes, 3)}
                for a in vlm_mod.check_attribute(self.video, tracks, question)]

    def person_for(self, face_ids: list[str]) -> dict[str, Optional[str]]:
        """The person track each face belongs to (or None): the person whose box holds the face, near its top,
        over most of the frames they share. Lets "the face of the guy in red" be asked about the whole person."""
        self._ensure_scanned("person")
        with self.lock:
            people = [(p.track_id, {k.f: k.box for k in p.keyframes}) for p in self.tracks.by_label("person")]
            faces = {i: self.tracks.get(i) for i in face_ids}
        out: dict[str, Optional[str]] = {}
        for fid, face in faces.items():
            best, best_score = None, 0.0
            for pid, pboxes in people:
                scores = [_face_in_person(k.box, pboxes[k.f]) for k in face.keyframes if k.f in pboxes]
                score = sum(scores) / len(face.keyframes) if scores else 0.0
                if score > best_score:
                    best, best_score = pid, score
            out[fid] = best if best_score >= 0.5 else None
        return out

    def track_box(self, frame: int, box: list[float], label: str = "user box") -> dict:
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise ToolError("box must be [x1, y1, x2, y2] with x2 > x1 and y2 > y1")
        t = track_mod.track_box(self.video, int(frame), tuple(box), self.tracks, label)
        self.save()
        return {"track_id": t.track_id, "start": round(t.start / self.info.fps, 2), "end": round(t.end / self.info.fps, 2)}

    def add_masks(self, track_ids: Optional[list[str]] = None, effect: str = "blur", padding: float = 0.15,
                  time_range: Optional[list[float]] = None, labels: Optional[list[str]] = None,
                  request_id: str = "manual") -> str:
        # labels: mask every track found for these queries, so the LLM doesn't spell out dozens of ids
        track_ids = list(track_ids or []) + [t.track_id for q in labels or [] for t in self.tracks.by_label(q)
                                             if t.track_id not in (track_ids or [])]
        self._check_ids(track_ids)
        if effect not in EFFECTS:
            raise ToolError(f"effect must be one of {EFFECTS}")
        return self._append(request_id, kind="add_targets", targets=track_ids, effect=effect,
                            padding=padding, time_range=tuple(time_range) if time_range else None)

    def remove_masks(self, track_ids: list[str], request_id: str = "manual",
                     time_range: Optional[list[float]] = None) -> str:
        self._check_ids(track_ids)
        return self._append(request_id, kind="remove_targets", targets=track_ids,
                            time_range=tuple(time_range) if time_range else None)

    def move_box(self, track_id: str, frame: int, box: list[float], request_id: str = "manual") -> dict:
        """The user corrected a mask's box at `frame`: from that frame on, mask a track followed from the corrected
        box instead of the old one (same label, effect and padding), for as long as the new track lasts. Two ops
        under one request id, so the UI's Undo reverts both. Before `frame` nothing changes. Not offered to the LLM: it's a hand edit from the UI."""
        self._check_ids([track_id])
        old = self.timeline.targets.get(track_id)
        if old is None:
            raise ToolError(f"{track_id} isn't masked, so there is nothing to move")
        fps = self.info.fps
        if not 0 <= frame < self.info.n_frames:
            raise ToolError(f"frame {frame} is outside the video (0-{self.info.n_frames - 1})")
        new = self.track_box(frame, box, label=old.label)
        # Unmask the old track only while the new one follows the object: the box tracker stops when it loses it
        # (or after MAX_TRACK_FRAMES), and from there the old mask must come back, or the object shows (fail closed)
        window = [frame / fps, self.tracks.get(new["track_id"]).end / fps]
        removed = self.remove_masks([track_id], request_id=request_id, time_range=window)
        added = self.add_masks([new["track_id"]], effect=old.effect, padding=old.padding, time_range=window,
                               request_id=request_id)
        return {"track": new, "op_ids": [removed, added]}

    def set_effect(self, track_ids: list[str], effect: str, request_id: str = "manual") -> str:
        self._check_ids(track_ids)
        if effect not in EFFECTS:
            raise ToolError(f"effect must be one of {EFFECTS}")
        return self._append(request_id, kind="set_effect", targets=track_ids, effect=effect)

    def add_rule(self, category: str, effect: str = "blur", request_id: str = "manual") -> str:
        if category not in RULE_CATEGORIES:
            raise ToolError(f"category must be one of {RULE_CATEGORIES}")
        return self._append(request_id, kind="add_rule", category=category, effect=effect)

    def summarize(self) -> dict:
        tl = self.timeline
        by_label: dict[str, int] = {}
        uncertain = 0
        for t in tl.targets.values():
            by_label[t.label] = by_label.get(t.label, 0) + 1
            uncertain += sum(1 for k in t.keyframes if k.state == "uncertain")
        return {"n_targets": len(tl.targets), "by_label": by_label, "uncertain_keyframes": uncertain,
                "n_ops": len(self.log.ops)}

    def undo(self) -> Optional[str]:
        op = self.log.undo()
        self.save()
        return op.op_id if op else None


def _face_in_person(face, person) -> float:
    """How well a face box fits a person box: the share of the face inside it, if the face center is in the top
    40% of the person box (a head), else nothing."""
    ix = max(0.0, min(face[2], person[2]) - max(face[0], person[0]))
    iy = max(0.0, min(face[3], person[3]) - max(face[1], person[1]))
    area = (face[2] - face[0]) * (face[3] - face[1])
    cy = (face[1] + face[3]) / 2
    top = person[1] + 0.4 * (person[3] - person[1])
    return (ix * iy / area) if area > 0 and person[1] - 0.1 * (person[3] - person[1]) <= cy <= top else 0.0


# JSON schemas the LLM sees (role B: keep in sync with the methods above).
TOOL_SCHEMAS = [
    {"name": "video_info", "parameters": {"type": "object", "properties": {}}},
    {"name": "resolve_time", "parameters": {"type": "object", "properties": {"phrase": {"type": "string"}}, "required": ["phrase"]}},
    {"name": "find_objects", "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}, "time_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}},
        "required": ["query"]}},
    {"name": "check_attribute", "parameters": {"type": "object", "properties": {
        "track_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1}, "question": {"type": "string"}},
        "required": ["track_ids", "question"]}},
    {"name": "track_box", "parameters": {"type": "object", "properties": {
        "frame": {"type": "integer"}, "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}},
        "required": ["frame", "box"]}},
    {"name": "add_masks", "parameters": {"type": "object", "properties": {
        "track_ids": {"type": "array", "items": {"type": "string"}},
        "labels": {"type": "array", "items": {"type": "string"}, "description": "mask every track found for these queries"},
        "effect": {"enum": list(EFFECTS)}, "padding": {"type": "number"},
        "time_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}}}},
    {"name": "remove_masks", "parameters": {"type": "object", "properties": {
        "track_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1}}, "required": ["track_ids"]}},
    {"name": "set_effect", "parameters": {"type": "object", "properties": {
        "track_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1}, "effect": {"enum": list(EFFECTS)}},
        "required": ["track_ids", "effect"]}},
    {"name": "add_rule", "parameters": {"type": "object", "properties": {
        "category": {"enum": list(RULE_CATEGORIES)}, "effect": {"enum": list(EFFECTS)}}, "required": ["category"]}},
    {"name": "summarize", "parameters": {"type": "object", "properties": {}}},
    {"name": "undo", "parameters": {"type": "object", "properties": {}}},
]


def call_tool(session: Session, name: str, args: dict, request_id: str = "manual"):
    """Dispatch one tool call by name. Raises ToolError on bad calls (fed back to the LLM)."""
    if name not in {s["name"] for s in TOOL_SCHEMAS}:
        raise ToolError(f"unknown tool {name!r}")
    fn = getattr(session, name)
    if name in ("add_masks", "remove_masks", "set_effect", "add_rule"):
        args = {**args, "request_id": request_id}
    try:
        return fn(**args)
    except TypeError as e:
        raise ToolError(f"bad arguments for {name}: {e}") from e


def dumps(x) -> str:
    return json.dumps(x, separators=(",", ":"))
