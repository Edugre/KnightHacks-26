"""Agent: one user request -> one plan -> tool calls -> ops on the mask timeline.

Path 1 (template): plain requests ("blur all plates and the woman") are parsed by intents.parse() in
milliseconds, with no model.
Path 2 (planner): anything else, such as exceptions ("everyone except the guy in red") or wording the template
doesn't know, goes to Qwen3-VL once: it writes the whole plan as JSON (censor/agent/planner.py), and plain code
runs it with the toolbox. One call of a few seconds, where the old step-by-step loop took over two minutes.
Path 3 (clarify): neither understood the request, so ask the user. Never guess silently.
Path 4 (reference): the request comes with a target the user pointed at in the UI ("blur everyone except this
person"); the pointer replaces the description (Agent._run_ref).

CLI (stub detector, no GPU): CENSOR_STUB=1 python -m censor.agent.agent clip.mp4 "blur all plates"
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
import uuid
from dataclasses import asdict
from typing import Optional

from censor.agent import intents, planner
from censor.agent.tools import Session, ToolError, call_tool

# Wording the template parser would misread ("everyone except the guy in red"), so it goes to the planner.
NEEDS_LLM = re.compile(r"\b(except|but|not|unless|only|without|instead|other than|besides|apart from|other)\b", re.I)
# With a reference, these words mean "everything but it" ("blur everyone except this person"); without, "only it"
REF_EXCEPT = re.compile(r"\b(except|but|not|unless|without|other than|besides|apart from|aside from|else|others?)\b", re.I)
MODEL_URL = os.environ.get("CENSOR_MODEL_URL")
CLARIFY = "I couldn't tell what to hide. Try e.g. \"blur all license plates\" or draw a box."


def _chat(messages: list[dict]) -> str:
    body = json.dumps({"messages": messages, "max_new_tokens": 300}).encode()
    req = urllib.request.Request(f"{MODEL_URL}/llm/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["text"]


class Agent:
    def __init__(self, session: Session):
        self.s = session

    def run(self, text: str, ref: Optional[dict] = None) -> dict:
        rid = f"r_{uuid.uuid4().hex[:6]}"
        calls: list[dict] = []
        tp = intents.parse(text)
        if ref is not None and not (tp is not None and tp.action == "undo"):
            return self._run_ref(text, tp, ref, rid, calls)
        if tp is not None and not (MODEL_URL and NEEDS_LLM.search(text)):
            msg = self._execute(planner.from_template(tp), rid, calls)
            if msg is not None:
                return self._result(rid, "template", calls, msg)
        if MODEL_URL:
            try:
                plan = self._plan(text, calls)
                if plan.action == "clarify" and tp is not None:  # the planner gave up; the template didn't
                    plan = planner.from_template(tp)
                msg = self._execute(plan, rid, calls)
                if msg is not None:
                    return self._result(rid, "llm", calls, msg)
            except Exception as e:  # model server down, unusable plan twice...
                calls.append({"tool": "_plan", "error": str(e)})
            if tp is not None:
                msg = self._execute(planner.from_template(tp), rid, calls)
                if msg is not None:
                    return self._result(rid, "template", calls, msg)
        return self._result(rid, "clarify", calls, CLARIFY)

    def _result(self, rid, path, calls, message) -> dict:
        return {"request_id": rid, "path": path, "calls": calls, "message": message, "summary": self.s.summarize()}

    def _call(self, name: str, args: dict, rid: str, calls: list[dict]):
        t0 = time.perf_counter()
        ms = lambda: round(1000 * (time.perf_counter() - t0))  # noqa: E731
        try:
            res = call_tool(self.s, name, args, rid)
            calls.append({"tool": name, "args": args, "result": res, "ms": ms()})
            return res
        except ToolError as e:
            calls.append({"tool": name, "args": args, "error": str(e), "ms": ms()})
            raise

    def _plan(self, text: str, calls: list[dict]) -> planner.Plan:
        """One LLM call for the whole plan; if the reply isn't a usable plan, one retry with the reason."""
        messages = [{"role": "system", "content": planner.SYSTEM},
                    {"role": "user", "content": f"Video: {json.dumps(self.s.video_info())}\nRequest: {text}"}]
        for attempt in range(2):
            t0 = time.perf_counter()
            reply = _chat(messages)
            ms = round(1000 * (time.perf_counter() - t0))
            try:
                plan = planner.parse_plan(reply)
                one = intents.one_person(text)
                if one is not None:  # singular or plural is in the words; the LLM got "but the lady" wrong
                    for t in plan.targets:
                        t.one = one
                calls.append({"tool": "_plan", "ms": ms, "plan": asdict(plan)})
                return plan
            except (ValueError, json.JSONDecodeError) as e:
                calls.append({"tool": "_plan", "ms": ms, "error": str(e), "reply": reply[:300]})
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": f"That is not a usable plan: {e}. Reply with the JSON plan only."}]
        raise RuntimeError("no usable plan after a retry")

    def _run_ref(self, text: str, tp, ref: dict, rid: str, calls: list[dict]) -> dict:
        """Path 4 (reference): the user pointed at someone (a target they picked, or a box drawn on Source) and
        said what to do about them: "blur everyone except this person", "pixelate this face". The pointer stands
        in for the description, so no yes/no questions and no LLM: the words say what kind of thing, the effect,
        when, and whether it's "only this" or "all but this"."""
        objects = list(tp.queries) if tp is not None else []
        t0 = time.perf_counter()
        try:
            r = self.s.resolve_ref(prefer=objects or None, **ref)
        except ToolError as e:
            calls.append({"tool": "resolve_ref", "args": ref, "error": str(e)})
            return self._result(rid, "reference", calls, f"Couldn't use the reference: {e}")
        calls.append({"tool": "resolve_ref", "args": ref, "result": r, "ms": round(1000 * (time.perf_counter() - t0))})
        mode = "except" if REF_EXCEPT.search(text) else "only"
        if r["label"] == "user box" and mode == "only":
            objects = ["user box"]  # nothing detected there: hide the drawn box itself
        elif not objects:  # "blur everything but this": things of the reference's kind
            objects = [r["label"] if r["label"] != "user box" else "person"]
        action = "remove" if intents.REMOVE.match(text.lower()) else "mask"
        plan = planner.Plan(action=action, effect=intents.effect_of(text),
                            targets=[planner.Target(object=q, time=tp.times.get(q) if tp is not None else None)
                                     for q in objects])
        msg = self._execute(plan, rid, calls, ref={**r, "mode": mode})
        return self._result(rid, "reference", calls, msg if msg is not None else "That didn't work with the reference.")

    def _execute(self, plan: planner.Plan, rid: str, calls: list[dict], ref: Optional[dict] = None):
        try:
            return planner.execute(plan, self.s, lambda name, args: self._call(name, args, rid, calls), ref=ref)
        except ToolError as e:
            if "detector failed" in str(e):
                return f"Detection failed, so nothing was changed: {e}"
            return None


if __name__ == "__main__":
    import tempfile
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sess = Session(sys.argv[1], tempfile.mkdtemp(prefix="censor_"))
    print(json.dumps(Agent(sess).run(" ".join(sys.argv[2:])), indent=2))
