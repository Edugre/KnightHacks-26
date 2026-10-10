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


@dataclass
class Plan:
    action: str  # "mask" | "remove" | "undo"
    queries: list[str] = field(default_factory=list)
    effect: str = "blur"
    attribute_question: Optional[str] = None
    time_phrase: Optional[str] = None


def parse(text: str) -> Optional[Plan]:
    t = text.lower().strip()
    if UNDO.match(t):
        return Plan(action="undo")
    action = "remove" if REMOVE.match(t) else "mask"
    effect = "pixelate" if "pixel" in t else "box" if re.search(r"black ?box|cover|black out", t) else "blur"

    queries: list[str] = []
    for rx, q in OBJECTS:
        if re.search(rx, t) and q not in queries:
            queries.append(q)
    if not queries:
        return None

    question = None
    if m := ATTRIBUTE.search(t):
        prep = "wearing" if m["prep"] == "in" else m["prep"]
        question = f"Is this person {prep} {m['attr'].strip()}?"
        if "person" not in queries:
            queries.append("person")

    tm = TIME.search(t)
    return Plan(action=action, queries=queries, effect=effect, attribute_question=question,
                time_phrase=tm[0] if tm else None)
