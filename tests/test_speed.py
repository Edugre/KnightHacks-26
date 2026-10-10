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
