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
    (None, "person"),  # every word in SUBJECTS and NEUTRAL, plus "everyone"; filled in below
]
# A description of a person: who ("the woman", "the tall guy") and what they wear or hold ("in the red hoodie",
# "with a backpack"). All of it goes into the question the VLM is asked about each person track, so "blur the
# woman" blurs only women; before, the noun was dropped and every person was blurred.
SUBJECTS = {"man": "man", "men": "man", "guy": "man", "guys": "man", "gentleman": "man", "male": "man",
            "males": "man", "boy": "boy", "boys": "boy", "woman": "woman", "women": "woman", "lady": "woman",
            "ladies": "woman", "female": "woman", "females": "woman", "girl": "girl", "girls": "girl",
            "child": "child", "children": "child", "kid": "child", "kids": "child"}
NEUTRAL = ("person", "persons", "people", "someone", "pedestrian", "pedestrians")
NOT_ADJECTIVES = {"blur", "pixelate", "hide", "mask", "cover", "censor", "box", "black", "remove", "unblur", "show",
                  "only", "just", "and", "or", "of", "other", "every", "all", "any", "both", "each"}
OBJECTS[-1] = (r"\b(" + "|".join(sorted(list(SUBJECTS) + list(NEUTRAL) + ["everyone", "everybody"],
                                            key=len, reverse=True)) + r")\b", "person")
PERSON = re.compile(
    r"\b(?:(?P<det>the|a|an|that|this|those|these|any|every|all(?: the)?|both)\s+"
    r"(?P<adj>(?:(?!(?:the|a|an|and|or|all|every|any|of|with|in|on|at)\b)[a-z][a-z-]*\s+){0,3}?))?"
    r"(?P<noun>" + "|".join(sorted(list(SUBJECTS) + list(NEUTRAL), key=len, reverse=True)) + r")\b"
    # "the woman in the first 3 seconds": "in" followed by a time phrase is when, not what she wears
    r"(?:\s+(?:who is\s+|who's\s+|that is\s+)?(?P<prep>in(?!\s+(?:the\s+)?(?:first|last|beginning|middle|end|start)\b)"
    r"|wearing|with|holding|carrying)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?P<attr>.+?)(?=\s+(?:around|at|in the (?:first|last|beginning|middle|end)|during|from|and|or)\b|[,.]|$))?")
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
    attribute_question: Optional[str] = None  # the first of attribute_questions, if any
    attribute_questions: list[str] = field(default_factory=list)  # one per person described; a track matching any is kept
    time_phrase: Optional[str] = None  # the phrase found, if any
    times: dict[str, Optional[str]] = field(default_factory=dict)  # query -> the time phrase that applies to it


def parse(text: str) -> Optional[Plan]:
    t = text.lower().strip()
    if UNDO.match(t):
        return Plan(action="undo")
    action = "remove" if REMOVE.match(t) else "mask"
    effect = effect_of(t)

    people = list(PERSON.finditer(t))
    # Objects named inside a description are attributes, not targets: "the man holding a phone" blurs the man,
    # not every screen.
    inside = [m.span("attr") for m in people if m["attr"]]
    queries: list[str] = []
    positions: dict[str, list[int]] = {}  # where each object is mentioned, to bind time phrases
    for rx, q in OBJECTS:
        hits = [m.start() for m in re.finditer(rx, t) if not any(a <= m.start() < b for a, b in inside)]
        if hits:
            positions.setdefault(q, []).extend(hits)
            if q not in queries:
                queries.append(q)
    if not queries:
        return None

    questions = []
    for m in people:
        q = _person_question(m)
        if q and q not in questions:
            questions.append(q)
            positions.setdefault("person", []).append(m.start())
            if "person" not in queries:
                queries.append("person")

    tm = TIME.search(t)
    return Plan(action=action, queries=queries, effect=effect, attribute_question=questions[0] if questions else None,
                attribute_questions=questions, time_phrase=tm[0] if tm else None,
                times=_bind_time(t, tm, queries, positions))


def effect_of(text: str) -> str:
    t = text.lower()
    return "pixelate" if "pixel" in t else "box" if re.search(r"black ?box|cover|black out", t) else "blur"


PLURAL = {"men", "guys", "males", "boys", "women", "ladies", "females", "girls", "children", "kids", "persons", "people",
          "pedestrians"}


def one_person(text: str) -> Optional[bool]:
    """Whether the request means one particular person: "the lady", "the guy in red", "me" -> True; "the women",
    "people in red" -> False; no person described -> None. Read from the words, not left to the LLM, which marked
    "everybody but the lady" as a group."""
    t = text.lower()
    if re.search(r"\b(me|mine|myself|i'm|i am)\b", t):
        return True
    found = [m for m in PERSON.finditer(t) if _person_question(m)]
    if not found:
        return None
    if any(m["noun"] in PLURAL or (m["det"] or "") in ("all", "all the", "those", "these", "both", "every", "any")
           for m in found):
        return False
    return any((m["det"] or "") in ("the", "that", "this", "a", "an") for m in found)


def _person_question(m: re.Match) -> Optional[str]:
    """The yes/no question for one person description, or None when it describes nobody in particular."""
    adj = [w for w in (m["adj"] or "").split() if w not in NOT_ADJECTIVES]  # only after a determiner ("the tall guy")
    who = SUBJECTS.get(m["noun"])
    what = f"{'wearing' if m['prep'] == 'in' else m['prep']} {m['attr'].strip()}" if m["prep"] else ""
    if not (who or adj or what):
        return None
    if who or adj:
        noun = " ".join(adj + [who or "person"])
        return f"Is this {'an' if noun[0] in 'aeiou' else 'a'} {noun}{' ' + what if what else ''}?"
    return f"Is this person {what}?"


def _bind_time(t: str, tm, queries: list[str], positions: dict[str, list[int]]) -> dict[str, Optional[str]]:
    if tm is None:
        return {q: None for q in queries}
    starts = [0] + [b.end() for b in CLAUSE_BREAK.finditer(t)]
    part = lambda pos: max(i for i, s in enumerate(starts) if s <= pos)  # noqa: E731
    own = [q for q in queries if any(part(p) == part(tm.start()) for p in positions[q])]
    if not own or tm.start() < min(p for ps in positions.values() for p in ps):
        return {q: tm[0] for q in queries}
    return {q: tm[0] if q in own else None for q in queries}
