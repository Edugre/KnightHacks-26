"""Agent: one user request -> tool calls -> ops on the mask timeline.

Path 1 (LLM): if CENSOR_MODEL_URL is set, Qwen3-VL plans step by step. Each turn it must reply with ONE
JSON object, either {"tool": name, "args": {...}} or {"done": true, "message": "..."}. Tool results
and errors are fed back. At most MAX_STEPS calls.
Path 2 (template): if the LLM is unavailable, fails, or makes no edits, intents.parse() handles it.
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

from censor.agent import intents
from censor.agent.tools import TOOL_SCHEMAS, Session, ToolError, call_tool

MAX_STEPS = 6
# Wording the template parser would misread ("everyone except the guy in red"), so it goes to the LLM.
NEEDS_LLM = re.compile(r"\b(except|but|not|unless|only|without|instead|other than|besides|apart from|other)\b", re.I)
MODEL_URL = os.environ.get("CENSOR_MODEL_URL")

SYSTEM = """You edit a video's censorship mask timeline by calling tools. You never edit pixels yourself.
Reply with exactly ONE JSON object per turn and nothing else:
  {"tool": "<name>", "args": {...}}   to call a tool, or
  {"done": true, "message": "<one sentence for the user>"}   when finished.
Rules:
- Always call find_objects (or track_box) first to get track ids. Never invent ids.
- For attributes (man or woman, child, age, hair, colors, clothing, what they hold, actions) call check_attribute on the candidate tracks; keep answers "yes" AND "unknown" (fail closed).
- Turn time phrases into seconds with resolve_time and pass them as time_range.
- find_objects takes a plain object noun only ("person", "face", "license plate", "screen", "text", "car").
  Never put gender, age, clothing, colors or actions in the query ("person", not "woman"); check those with check_attribute.
- Default effect is "blur".
- To mask everything a query found, call add_masks with "labels" (e.g. {"labels": ["person"]}) instead of listing ids.
  List track_ids only for a subset, such as the tracks check_attribute kept.
Example for "blur the man in the red shirt":
  {"tool": "find_objects", "args": {"query": "person"}}
  {"tool": "check_attribute", "args": {"track_ids": ["trk_1", "trk_2"], "question": "Is this person wearing a red shirt?"}}
  {"tool": "add_masks", "args": {"track_ids": ["trk_2"]}}      <- ids whose answer was yes or unknown
  {"done": true, "message": "Blurred 1 person in a red shirt."}
Tools (JSON schema):
""" + json.dumps(TOOL_SCHEMAS)


def _chat(messages: list[dict]) -> str:
    body = json.dumps({"messages": messages, "max_new_tokens": 200}).encode()
    req = urllib.request.Request(f"{MODEL_URL}/llm/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["text"]


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    return json.loads(m.group(0))


class Agent:
    def __init__(self, session: Session):
        self.s = session

    def run(self, text: str) -> dict:
        rid = f"r_{uuid.uuid4().hex[:6]}"
        calls: list[dict] = []
        n_ops_before = len(self.s.log.ops)
        if MODEL_URL and intents.parse(text) is not None and not NEEDS_LLM.search(text):
            # Cheapest path first: the template handles plain requests in milliseconds, while each LLM step
            # takes seconds. The LLM is for wording the template can't parse or exceptions it would get wrong.
            msg = self._run_template(text, rid, calls)
            if msg is not None:
                return self._result(rid, "template", calls, msg)
        if MODEL_URL:
            try:
                msg = self._run_llm(text, rid, calls)
                if len(self.s.log.ops) > n_ops_before or "undo" in [c["tool"] for c in calls]:
                    return self._result(rid, "llm", calls, msg)
            except Exception as e:  # model server down, bad output, too many steps...
                calls.append({"tool": "_llm", "error": str(e)})
        msg = self._run_template(text, rid, calls)
        path = "template" if msg is not None else "clarify"
        return self._result(rid, path, calls, msg or
                            "I couldn't tell what to hide. Try e.g. \"blur all license plates\" or draw a box.")

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

    def _run_llm(self, text: str, rid: str, calls: list[dict]) -> str:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Video: {json.dumps(self.s.video_info())}\nRequest: {text}"}]
        for _ in range(MAX_STEPS):
            t0 = time.perf_counter()
            reply = _chat(messages)
            calls.append({"tool": "_llm_step", "ms": round(1000 * (time.perf_counter() - t0))})
            messages.append({"role": "assistant", "content": reply})
            try:
                obj = _parse_json(reply)
            except ValueError as e:
                messages.append({"role": "user", "content": f"ERROR: {e}. Reply with one JSON object."})
                continue
            if obj.get("done"):
                return obj.get("message", "Done.")
            try:
                res = self._call(obj.get("tool", ""), obj.get("args") or {}, rid, calls)
                messages.append({"role": "user", "content": f"RESULT: {json.dumps(res)}"})
            except ToolError as e:
                messages.append({"role": "user", "content": f"ERROR: {e}"})
        raise RuntimeError(f"no 'done' after {MAX_STEPS} steps")

    def _run_template(self, text: str, rid: str, calls: list[dict]):
        plan = intents.parse(text)
        if plan is None:
            return None
        try:
            if plan.action == "undo":
                op = self._call("undo", {}, rid, calls)
                return f"Undid {op}." if op else "Nothing to undo."
            ranges = {p: self._call("resolve_time", {"phrase": p}, rid, calls) for p in set(plan.times.values()) if p}
            masked = 0
            for q in plan.queries:
                tr = ranges.get(plan.times.get(q))  # each object gets only the time phrase that applies to it
                found = self._call("find_objects", {"query": q, **({"time_range": tr} if tr else {})}, rid, calls)
                ids = [o["track_id"] for o in found]
                if not ids:
                    continue
                if plan.attribute_questions and q == "person":
                    keep: set[str] = set()  # "the woman and the kid": a person matching either description
                    for question in plan.attribute_questions:
                        answers = self._call("check_attribute", {"track_ids": ids, "question": question}, rid, calls)
                        keep |= {a["track_id"] for a in answers if a["answer"] in ("yes", "unknown")}
                    ids = [i for i in ids if i in keep]
                    if not ids:
                        continue
                if plan.action == "remove":
                    live = [i for i in ids if i in self.s.timeline.targets]
                    if live:
                        self._call("remove_masks", {"track_ids": live}, rid, calls)
                        masked += len(live)
                else:
                    self._call("add_masks", {"track_ids": ids, "effect": plan.effect,
                                             **({"time_range": tr} if tr else {})}, rid, calls)
                    masked += len(ids)
        except ToolError as e:
            if "detector failed" in str(e):
                return f"Detection failed, so nothing was changed: {e}"
            return None
        verb = "Unmasked" if plan.action == "remove" else f"Applied {plan.effect} to"
        return f"{verb} {masked} track(s) for: {', '.join(plan.queries)}." if masked else \
            f"I looked for {', '.join(plan.queries)} but found nothing to change."


if __name__ == "__main__":
    import tempfile
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sess = Session(sys.argv[1], tempfile.mkdtemp(prefix="censor_"))
    print(json.dumps(Agent(sess).run(" ".join(sys.argv[2:])), indent=2))
