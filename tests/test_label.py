"""Label format drafts, prelabel and the labeling server. No GPU: the stub detector stands in for YOLOE."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bench import run_bench as rb
from bench.gt import GTBox, GTObject, LabeledFrame, LabelSet, prelabel
from bench.label_server import create_app
from censor.core.timeline import VideoInfo
from censor.vision.detect import StubDetector

ROOT = Path(__file__).resolve().parents[1]
needs_ffmpeg = pytest.mark.skipif(shutil.which(os.environ.get("CENSOR_FFMPEG", "ffmpeg")) is None,
                                  reason="ffmpeg not on PATH")


def _labels(reviewed_second: bool) -> LabelSet:
    box = (10, 10, 50, 90)
    return LabelSet(clip="t", video=VideoInfo(fps=10, width=200, height=100, n_frames=30),
                    objects={"face_1": GTObject(category="face")},
                    frames=[LabeledFrame(f=0, boxes=[GTBox(gt_id="face_1", box=box)]),
                            LabeledFrame(f=10, boxes=[GTBox(gt_id="face_1", box=box)], reviewed=reviewed_second)])


def test_drafts_are_not_scored():
    hits, over = rb.score_frames(_labels(reviewed_second=False), {})
    assert hits == {"face_1": [False]} and len(over) == 1  # only the confirmed frame counts
    assert _labels(reviewed_second=False).n_drafts == 1


def test_save_round_trips_drafts(tmp_path):
    ls = _labels(reviewed_second=False)
    ls.save(tmp_path / "l.json")
    text = (tmp_path / "l.json").read_text()
    assert text.count('"reviewed"') == 1  # only the draft carries the flag
    assert LabelSet.load(tmp_path / "l.json") == ls


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("label") / "street.mp4"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_clip.py"), str(out), "1"], check=True)
    return out


@needs_ffmpeg
def test_prelabel_makes_unscored_drafts(clip):
    ls = prelabel(clip, detector=StubDetector())
    assert ls.clip == "street" and [fr.f for fr in ls.frames] == [0, 10, 20]
    assert ls.n_drafts == 3 and ls.frames_in() == []
    assert {o.category for o in ls.objects.values()} == {"face", "license plate", "person", "screen"}
    assert [fr.f for fr in ls.frames if any(b.gt_id == "screen_1" for b in fr.boxes)] == [20]  # second half only


@needs_ffmpeg
def test_label_server_round_trip(clip, tmp_path):
    out = tmp_path / "labels" / "street.json"
    c = TestClient(create_app(clip, out))
    assert "Bench Labeler" in c.get("/").text
    r = c.get("/api/labels").json()
    assert r["sampled"] == [0, 10, 20] and r["labels"]["objects"] == {} and "face" in r["categories"]
    assert c.get("/api/frame/10.jpg").headers["content-type"] == "image/jpeg"
    assert c.get("/api/frame/999.jpg").status_code == 404

    body = r["labels"] | {"objects": {"face_1": {"category": "face", "desc": "bobbing"}},
                          "frames": [{"f": 0, "boxes": [{"gt_id": "face_1", "box": [855, 90, 945, 210]}], "reviewed": True},
                                     {"f": 10, "boxes": [], "reviewed": True}]}
    assert c.put("/api/labels", json=body).json() == {"ok": True, "frames": 2, "drafts": 0}
    saved = LabelSet.load(out)
    assert saved.objects["face_1"].desc == "bobbing" and len(saved.frames_in()) == 2
    assert c.get("/api/labels").json()["labels"]["frames"][0]["boxes"][0]["gt_id"] == "face_1"

    bad = body | {"frames": [{"f": 0, "boxes": [{"gt_id": "nope", "box": [0, 0, 1, 1]}]}]}
    assert c.put("/api/labels", json=bad).status_code == 400
    assert LabelSet.load(out) == saved  # a rejected save leaves the file alone
