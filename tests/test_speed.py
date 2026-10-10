"""Latency fixes: one detector pass per query even when the upload pre-scan and a request overlap; step timings."""
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from censor.agent.agent import Agent
from censor.agent.tools import Session
from censor.vision.detect import StubDetector

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which(os.environ.get("CENSOR_FFMPEG", "ffmpeg")) is None, reason="ffmpeg not on PATH")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("speed") / "c.mp4"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_clip.py"), str(out), "1"], check=True)
    return out


class SlowStub(StubDetector):
    def __init__(self):
        self.runs = []

    def run(self, video, queries, store):
        self.runs.append(tuple(queries))
        time.sleep(0.3)  # a full-video pass
        return super().run(video, queries, store)


def test_prescan_and_request_share_one_pass(clip, tmp_path, monkeypatch):
    import censor.server.app as app_mod
    det = SlowStub()
    s = Session(clip, tmp_path / "w", detector=det)
    scan = threading.Thread(target=app_mod.prescan, args=(s, ["license plate", "person"]))
    scan.start()
    time.sleep(0.05)  # the request arrives while the pre-scan is still on "license plate"
    r = Agent(s).run("blur all plates")
    scan.join()
    assert det.runs == [("license plate",), ("person",)]  # the request waited instead of scanning again
    assert r["summary"]["by_label"] == {"license plate": 1}


def test_tool_calls_are_timed(clip, tmp_path):
    r = Agent(Session(clip, tmp_path / "w", detector=StubDetector())).run("blur all plates")
    assert r["calls"] and all(isinstance(c["ms"], int) for c in r["calls"])


def test_plain_requests_skip_the_llm_and_exceptions_use_it(clip, tmp_path, monkeypatch):
    from censor.agent import agent as agent_mod
    steps = []

    def fake_chat(messages):
        steps.append(messages[-1]["content"])
        return ('{"action": "mask", "effect": "blur", "targets": [{"object": "person", "only": [], '
                '"except": ["Is this a man wearing red?"], "time": null}]}')

    monkeypatch.setattr(agent_mod, "MODEL_URL", "http://model")
    monkeypatch.setattr(agent_mod, "_chat", fake_chat)
    s = Session(clip, tmp_path / "w", detector=StubDetector())
    r = Agent(s).run("blur all plates")
    assert r["path"] == "template" and steps == []  # no LLM call for a plain request
    r = Agent(s).run("blur everyone except the guy in red")
    assert r["path"] == "llm" and len(steps) == 1  # one call for the whole plan, not one per tool
    assert r["summary"]["by_label"]["person"] == 2  # the stub VLM answers "unknown": both stay masked


def test_an_unusable_plan_is_retried_once_then_the_template_runs(clip, tmp_path, monkeypatch):
    from censor.agent import agent as agent_mod
    replies = iter(["I think you want plates blurred.", '{"action": "dance"}'])
    monkeypatch.setattr(agent_mod, "MODEL_URL", "http://model")
    monkeypatch.setattr(agent_mod, "_chat", lambda messages: next(replies))
    r = Agent(Session(clip, tmp_path / "w", detector=StubDetector())).run("only blur the plates")
    assert r["path"] == "template" and r["summary"]["by_label"] == {"license plate": 1}
    assert [c.get("error", "")[:20] for c in r["calls"] if c["tool"] == "_plan"][:2] ==         ["no JSON object in th", "unknown action 'danc"]


def test_nothing_is_not_an_exception_word():
    from censor.agent.agent import NEEDS_LLM
    assert not NEEDS_LLM.search("blur the button and nothing else on the other side".replace("other", "far"))
    assert NEEDS_LLM.search("blur everyone except me") and NEEDS_LLM.search("only the plates")


def test_a_finished_query_does_not_wait_for_another_scan(clip, tmp_path):
    import censor.server.app as app_mod
    det = SlowStub()
    s = Session(clip, tmp_path / "w", detector=det)
    s.find_objects("license plate")  # already scanned
    scan = threading.Thread(target=app_mod.prescan, args=(s, ["person"]))
    scan.start()
    time.sleep(0.05)  # "person" is mid-scan
    t0 = time.perf_counter()
    r = Agent(s).run("blur all car plates")
    assert time.perf_counter() - t0 < 0.2  # didn't sit behind the person scan
    scan.join()
    assert r["summary"]["by_label"] == {"license plate": 1}
