"""One request -> one plan -> deterministic execution with the toolbox.

The LLM writes the whole plan in a single call (a few seconds), instead of choosing one tool per step and
waiting for each result (over two minutes for "blur everyone except the guy in red"). The plan only says
what to find and which yes/no questions decide who is in or out; plain code below runs the tools, so the
LLM never invents track ids and every edit still goes through the op log. The template parser
(censor/agent/intents.py) produces the same plan for plain requests without any model.

Plan:
  {"action": "mask" | "remove" | "undo" | "clarify",
   "effect": "blur" | "pixelate" | "box",
   "targets": [{"object": "person", "only": ["Is this a woman?"], "except": [], "time": "around halfway"}],
   "message": "..."}   (the question back to the user, for "clarify")
"only": a track is kept if any question is answered yes or unknown. "except": a track is dropped if any is
answered yes; unknown keeps it masked. Either way an unsure answer means more blur, never less.
"one": the request means one particular person ("everyone but the lady"), so "except" leaves only the single
most likely match visible. Without it, every blurry passer-by the VLM thought might be a woman stayed unblurred.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from censor.agent import intents
from censor.agent.tools import EFFECTS, Session, ToolError

OBJECT_ALIASES = {"plate": "license plate", "plates": "license plate", "license plates": "license plate",
                  "licence plate": "license plate", "number plate": "license plate", "car plate": "license plate",
                  "faces": "face", "people": "person", "persons": "person", "everyone": "person",
                  "everybody": "person", "pedestrian": "person", "pedestrians": "person", "screens": "screen",
                  "phones": "screen", "phone": "screen", "monitor": "screen", "laptop": "screen", "signs": "text",
                  "sign": "text", "logos": "logo"}

SYSTEM = """You turn a request about censoring a video into a JSON plan. Reply with ONE JSON object and nothing else:
{"action": "mask"|"remove"|"undo"|"clarify", "effect": "blur"|"pixelate"|"box",
 "targets": [{"object": "<noun>", "only": ["<yes/no question>"], "except": ["<yes/no question>"], "one": true|false, "time": "<phrase or null>"}],
 "message": "<question for the user, only when action is clarify>"}
Rules:
- "object" is what a detector finds, as a plain noun: face, license plate, person, screen, text, car, logo, dog, ...
  Never put gender, age, clothing or colors in "object": a woman is object "person" with only ["Is this a woman?"].
- "only" limits the target to the tracks that match. "except" keeps matching tracks visible. Questions are yes/no
  about one track's crop: "Is this a woman?", "Is this a man wearing a red hoodie?", "Is this the face of a child?".
- A face belongs to a person, so questions about faces may mention clothing: "the face of the guy in red".
- "action": "mask" to hide (default), "remove" to unblur or stop hiding, "undo" to take back the last edit,
  "clarify" when you cannot tell what to hide (then ask in "message").
- "effect": "blur" unless the request says pixelate, black box, cover or black out ("box").
- "one": true when the request means one particular person or thing ("the guy in red", "the lady", "me"), false for
  groups ("the women", "people in red", "kids").
- "time": the time words from the request ("around halfway", "first 10 seconds", "last 5 seconds", "0:30") or null.
Examples:
"blur all plates" -> {"action":"mask","effect":"blur","targets":[{"object":"license plate","only":[],"except":[],"time":null}]}
"blur everyone except the guy in the red hoodie" -> {"action":"mask","effect":"blur","targets":[{"object":"person","only":[],"except":["Is this a man wearing a red hoodie?"],"one":true,"time":null}]}
"blur everyone but the kids" -> {"action":"mask","effect":"blur","targets":[{"object":"person","only":[],"except":["Is this a child?"],"one":false,"time":null}]}
"only blur the woman's face" -> {"action":"mask","effect":"blur","targets":[{"object":"face","only":["Is this the face of a woman?"],"except":[],"time":null}]}
"pixelate all faces but mine, I'm the one in the black t-shirt" -> {"action":"mask","effect":"pixelate","targets":[{"object":"face","only":[],"except":["Is this a person wearing a black t-shirt?"],"one":true,"time":null}]}
"hide the screens and the kid in the last 5 seconds" -> {"action":"mask","effect":"blur","targets":[{"object":"screen","only":[],"except":[],"time":"last 5 seconds"},{"object":"person","only":["Is this a child?"],"except":[],"time":"last 5 seconds"}]}
"unblur the plates" -> {"action":"remove","effect":"blur","targets":[{"object":"license plate","only":[],"except":[],"time":null}]}
"make it look cinematic" -> {"action":"clarify","effect":"blur","targets":[],"message":"What should I hide? For example faces, license plates or a person."}"""


@dataclass
class Target:
    object: str
    only: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    time: Optional[str | list[float]] = None
    one: bool = False  # one particular person or thing: "except" spares only the best match


@dataclass
class Plan:
    action: str = "mask"
    effect: str = "blur"
    targets: list[Target] = field(default_factory=list)
    message: str = ""


def from_template(p: intents.Plan) -> Plan:
    """The template parser's plan in planner form, so both run through the same executor."""
    if p.action == "undo":
        return Plan(action="undo")
    return Plan(action=p.action, effect=p.effect,
                targets=[Target(object=q, only=list(p.attribute_questions) if q == "person" else [],
                                time=p.times.get(q)) for q in p.queries])


def parse_plan(text: str) -> Plan:
    """Read and check the LLM's JSON. Raises ValueError with a reason when it isn't a usable plan."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    obj = json.loads(m.group(0))
    action = str(obj.get("action", "mask")).lower()
    if action not in ("mask", "remove", "undo", "clarify"):
        raise ValueError(f"unknown action {action!r}")
    effect = str(obj.get("effect") or "blur").lower()
    if effect not in EFFECTS:
        effect = "blur"
    targets = []
    for t in obj.get("targets") or []:
        noun = str(t.get("object") or "").lower().strip()
        if not noun:
            continue
        only = _questions(t.get("only"))
        if noun in intents.SUBJECTS:  # "woman" slipped into object: it's a person plus a question
            only = only or [f"Is this a {intents.SUBJECTS[noun]}?"]
            noun = "person"
        targets.append(Target(object=OBJECT_ALIASES.get(noun, noun), only=only, exclude=_questions(t.get("except")),
                              time=t.get("time") or None, one=bool(t.get("one"))))
    if action in ("mask", "remove") and not targets:
        raise ValueError("no targets in the plan")
    return Plan(action=action, effect=effect, targets=targets, message=str(obj.get("message") or ""))


def _questions(v) -> list[str]:
    if not v:
        return []
    return [str(q).strip() for q in (v if isinstance(v, list) else [v]) if str(q).strip()]


def execute(plan: Plan, s: Session, call: Callable[[str, dict], object]) -> str:
    """Run a plan with the toolbox. `call(tool, args)` runs one tool and records it (Agent._call)."""
    if plan.action == "clarify":
        return plan.message or "What should I hide? For example faces, license plates or a person."
    if plan.action == "undo":
        op = call("undo", {})
        return f"Undid {op}." if op else "Nothing to undo."
    done, notes = 0, []
    for t in plan.targets:
        tr = None
        if isinstance(t.time, list) and len(t.time) == 2:
            tr = [float(t.time[0]), float(t.time[1])]
        elif t.time:
            tr = call("resolve_time", {"phrase": str(t.time)})
        found = call("find_objects", {"query": t.object, **({"time_range": tr} if tr else {})})
        ids = [o["track_id"] for o in found]
        if ids and (t.only or t.exclude):
            ids, note = _filter(s, call, t, ids)
            notes.append(note)
        if not ids:
            continue
        if plan.action == "remove":
            live = [i for i in ids if i in s.timeline.targets]
            if live:
                call("remove_masks", {"track_ids": live})
                done += len(live)
        else:
            call("add_masks", {"track_ids": ids, "effect": plan.effect, **({"time_range": tr} if tr else {})})
            done += len(ids)
    what = ", ".join(t.object for t in plan.targets)
    if not done:
        return f"I looked for {what} but found nothing to change."
    verb = "Unmasked" if plan.action == "remove" else f"Applied {plan.effect} to"
    return f"{verb} {done} track(s) for: {what}." + (f" {' '.join(notes)}" if notes else "")


def _filter(s: Session, call, t: Target, ids: list[str]) -> tuple[list[str], str]:
    """Keep the tracks the questions select. Face tracks are asked about through the person they belong to, when
    one is found, so a question can mention clothing; otherwise the face crop itself is asked."""
    owner = s.person_for(ids) if t.object == "face" else {}
    ask = sorted({owner.get(i) or i for i in ids})

    def answers(questions: list[str]) -> dict[str, list[tuple[str, float]]]:
        out: dict[str, list[tuple[str, float]]] = {a: [] for a in ask}
        for q in questions:
            for a in call("check_attribute", {"track_ids": ask, "question": _for(q, owner)}):
                out[a["track_id"]].append((a["answer"], float(a.get("p_yes", 0.5))))
        return out

    keep = set(ids)
    if t.only:
        got = answers(t.only)
        keep = {i for i in keep if any(x in ("yes", "unknown") for x, _ in got[owner.get(i) or i])}
    if t.exclude:
        got = answers(t.exclude)
        spared = {a for a, res in got.items() if any(x == "yes" for x, _ in res)}  # unknown stays masked: fail closed
        if t.one and spared:  # "but the lady": only the most likely one, not every passer-by who might be
            # A tie (the VLM saturates at 1.0) goes to the person the video is about, the most screen time x size,
            # not a passer-by seen for a second: on street_talk the woman and a 1 s background track both got 1.0
            spared = {max(spared, key=lambda a: (round(max(p for _, p in got[a]), 2), _presence(s, a)))}
        keep = {i for i in keep if (owner.get(i) or i) not in spared}
    left = [i for i in ids if i in keep]
    return left, f"{len(left)} of {len(ids)} {t.object} track(s) matched."


def _presence(s: Session, track_id: str) -> float:
    """How much of the video a track takes up: its box area summed over its keyframes (0 if unknown)."""
    tracks = getattr(s, "tracks", None)
    t = tracks.get(track_id) if tracks is not None else None
    return sum((k.box[2] - k.box[0]) * (k.box[3] - k.box[1]) for k in t.keyframes) if t else 0.0


def _for(question: str, owner: dict) -> str:
    """A face question asked on a person crop reads better about the person: "the face of a woman" -> "a woman"."""
    return re.sub(r"\bthe face of\s+", "", question, flags=re.I) if any(owner.values()) else question
