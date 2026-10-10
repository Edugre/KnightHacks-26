"""Template fallback: turns common requests into tool-call plans without an LLM.

Handles: "blur all plates and faces", "pixelate the person in the red shirt around halfway",
"black box every screen in the first 10 seconds", "unblur the faces", "undo".
Anything else returns None and the agent asks the user to rephrase (it never guesses silently).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

OBJECTS = [  # (regex, canonical detector query)
    (r"\b(license plates?|number plates?|plates?)\b", "license plate"),
    (r"\bfaces?\b", "face"),
    (r"\b(screens?|monitors?|phones?|laptops?)\b", "screen"),
    (r"\b(text|signs?|captions?)\b", "text"),
    (r"\b(people|persons?|everyone|everybody|man|woman|guy|girl|someone|pedestrians?)\b", "person"),
]
ATTRIBUTE = re.compile(
    r"\b(?:person|man|woman|guy|girl|someone)\s+(?P<prep>in|wearing|with|holding)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?P<attr>.+?)(?=\s+(?:around|at|in the (?:first|last|beginning|middle|end)|during|from|and)\b|[,.]|$)")
TIME = re.compile(r"\b(halfway|half way|middle|\d+:\d{2}|first \d+ ?\w+|last \d+ ?\w+|beginning|start|end)\b")
REMOVE = re.compile(r"^\s*(unblur|un-blur|remove|stop blurring|don'?t blur|show|reveal)\b")
UNDO = re.compile(r"^\s*(undo|go back|revert)\b")
# A new noun phrase after "and" (or a comma) starts a new part of the request, and a time phrase applies to the
# objects in its own part: "blur all plates and the man in red around halfway" limits only the man. A time phrase
# before every object, or in a part with no object of its own (", around halfway"), applies to all of them.
CLAUSE_BREAK = re.compile(r",|\band\s+(?=(?:the|all|every|any|each|a|an|his|her|their|that|this|those|these)\b)")


@dataclass
class Plan:
    action: str  # "mask" | "remove" | "undo"
    queries: list[str] = field(default_factory=list)
    effect: str = "blur"
    attribute_question: Optional[str] = None
    time_phrase: Optional[str] = None  # the phrase found, if any
    times: dict[str, Optional[str]] = field(default_factory=dict)  # query -> the time phrase that applies to it


def parse(text: str) -> Optional[Plan]:
    t = text.lower().strip()
    if UNDO.match(t):
        return Plan(action="undo")
    action = "remove" if REMOVE.match(t) else "mask"
    effect = "pixelate" if "pixel" in t else "box" if re.search(r"black ?box|cover|black out", t) else "blur"

    queries: list[str] = []
    positions: dict[str, list[int]] = {}  # where each object is mentioned, to bind time phrases
    for rx, q in OBJECTS:
        hits = [m.start() for m in re.finditer(rx, t)]
        if hits:
            positions.setdefault(q, []).extend(hits)
            if q not in queries:
                queries.append(q)
    if not queries:
        return None

    question = None
    if m := ATTRIBUTE.search(t):
        prep = "wearing" if m["prep"] == "in" else m["prep"]
        question = f"Is this person {prep} {m['attr'].strip()}?"
        positions.setdefault("person", []).append(m.start())
        if "person" not in queries:
            queries.append("person")

    tm = TIME.search(t)
    return Plan(action=action, queries=queries, effect=effect, attribute_question=question,
                time_phrase=tm[0] if tm else None, times=_bind_time(t, tm, queries, positions))


def _bind_time(t: str, tm, queries: list[str], positions: dict[str, list[int]]) -> dict[str, Optional[str]]:
    if tm is None:
        return {q: None for q in queries}
    starts = [0] + [b.end() for b in CLAUSE_BREAK.finditer(t)]
    part = lambda pos: max(i for i, s in enumerate(starts) if s <= pos)  # noqa: E731
    own = [q for q in queries if any(part(p) == part(tm.start()) for p in positions[q])]
    if not own or tm.start() < min(p for ps in positions.values() for p in ps):
        return {q: tm[0] for q in queries}
    return {q: tm[0] if q in own else None for q in queries}
