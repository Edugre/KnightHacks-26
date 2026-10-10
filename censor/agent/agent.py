"""Agent: one user request -> one plan -> tool calls -> ops on the mask timeline.

Path 1 (template): plain requests ("blur all plates and the woman") are parsed by intents.parse() in
milliseconds, with no model.
Path 2 (planner): anything else, such as exceptions ("everyone except the guy in red") or wording the template
doesn't know, goes to Qwen3-VL once: it writes the whole plan as JSON (censor/agent/planner.py), and plain code
runs it with the toolbox. One call of a few seconds, where the old step-by-step loop took over two minutes.
Path 3 (clarify): neither understood the request, so ask the user. Never guess silently.

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

from censor.agent import intents, planner
from censor.agent.tools import Session, ToolError, call_tool

# Wording the template parser would misread ("everyone except the guy in red"), so it goes to the planner.
NEEDS_LLM = re.compile(r"\b(except|but|not|unless|only|without|instead|other than|besides|apart from|other)\b", re.I)
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

    def run(self, text: str) -> dict:
        rid = f"r_{uuid.uuid4().hex[:6]}"
        calls: list[dict] = []
        tp = intents.parse(text)
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
                calls.append({"tool": "_plan", "ms": ms, "plan": asdict(plan)})
                return plan
            except (ValueError, json.JSONDecodeError) as e:
                calls.append({"tool": "_plan", "ms": ms, "error": str(e), "reply": reply[:300]})
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": f"That is not a usable plan: {e}. Reply with the JSON plan only."}]
        raise RuntimeError("no usable plan after a retry")

    def _execute(self, plan: planner.Plan, rid: str, calls: list[dict]):
        try:
            return planner.execute(plan, self.s, lambda name, args: self._call(name, args, rid, calls))
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
