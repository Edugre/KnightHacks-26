"""End to end on a synthetic clip with the stub detector: request -> ops -> export. No GPU needed.

This is the M1 exit check (upload, request, export) minus the browser.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = os.environ.get("CENSOR_FFMPEG", "ffmpeg")
pytestmark = pytest.mark.skipif(shutil.which(FFMPEG) is None, reason="ffmpeg not on PATH")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("clips") / "synthetic.mp4"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_clip.py"), str(out), "3"], check=True)
    return out


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CENSOR_STUB", "1")
    monkeypatch.setenv("CENSOR_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("CENSOR_MODEL_URL", raising=False)
    import importlib
    import censor.server.app as app_mod
    importlib.reload(app_mod)
    from fastapi.testclient import TestClient
    return TestClient(app_mod.app), app_mod


def test_request_undo_export(clip, client):
    c, app_mod = client
    with open(clip, "rb") as f:
        vid = c.post("/api/videos", files={"file": ("synthetic.mp4", f, "video/mp4")}).json()["video_id"]
    with open(str(clip) + ".tracks.json", "rb") as f:
        assert c.post(f"/api/videos/{vid}/tracks-sidecar", files={"file": ("t.json", f)}).json()["ok"]

    r = c.post(f"/api/videos/{vid}/request", json={"text": "blur all plates and the person in the red shirt"}).json()
    assert r["path"] == "template"
    # stub VLM answers "unknown", so BOTH persons are kept: fail closed
    assert r["summary"]["by_label"] == {"license plate": 1, "person": 2}

    r = c.post(f"/api/videos/{vid}/request", json={"text": "pixelate the screen in the last 1 seconds"}).json()
    assert r["summary"]["by_label"].get("screen") == 1

    assert c.post(f"/api/videos/{vid}/undo").json()["summary"]["by_label"] == {"license plate": 1, "person": 2}

    r = c.post(f"/api/videos/{vid}/request", json={"text": "make it cinematic"}).json()
    assert r["path"] == "clarify"

    b = c.post(f"/api/videos/{vid}/box", json={"frame": 10, "box": [855, 90, 945, 210]}).json()
    assert b["summary"]["by_label"]["user box"] == 1

    assert c.get(f"/api/videos/{vid}/preview/30.jpg").headers["content-type"] == "image/jpeg"
    e = c.post(f"/api/videos/{vid}/export").json()
    assert e["stats"]["frames"] == 90 and e["stats"]["failed_frames"] == 0 and e["stats"]["masked_boxes"] > 0
    assert c.get("/" + e["url"]).status_code == 200


def test_plate_pixels_are_masked_in_export(clip, client):
    """The plate text region must change in the export; the sky far from targets must not change much."""
    import cv2
    import numpy as np
    from censor.agent.agent import Agent
    from censor.agent.tools import Session
    from censor.render.render import render_video
    from censor.vision.detect import StubDetector

    s = Session(clip, Path(os.environ["CENSOR_DATA"]) / "direct", detector=StubDetector())
    Agent(s).run("blur all plates")
    out = Path(s.workdir) / "out.mp4"
    render_video(str(clip), s.timeline, str(out))
    a, b = cv2.VideoCapture(str(clip)), cv2.VideoCapture(str(out))
    _, fa = a.read(); _, fb = b.read()
    plate = s.tracks.by_label("license plate")[0].keyframes[0].box
    x1, y1, x2, y2 = (int(v) for v in plate)
    diff_plate = np.abs(fa[y1:y2, x1:x2].astype(int) - fb[y1:y2, x1:x2].astype(int)).mean()
    diff_sky = np.abs(fa[0:80, 0:400].astype(int) - fb[0:80, 0:400].astype(int)).mean()
    assert diff_plate > 20 and diff_sky < 5


def test_sample_clips_load_without_a_file_picker(clip, client, tmp_path, monkeypatch):
    c, app_mod = client
    samples = tmp_path / "samples"
    samples.mkdir()
    shutil.copyfile(clip, samples / "street.mp4")
    shutil.copyfile(str(clip) + ".tracks.json", samples / "street.mp4.tracks.json")
    monkeypatch.setattr(app_mod, "SAMPLES_DIR", samples)
    assert [s["id"] for s in c.get("/api/samples").json()] == ["street"]
    vid = c.post("/api/samples/street").json()["video_id"]
    r = c.post(f"/api/videos/{vid}/request", json={"text": "blur all plates"}).json()
    assert r["summary"]["by_label"] == {"license plate": 1}  # the sidecar came along, so the stub finds the plate
    assert c.post("/api/samples/..%2Fsecret").status_code == 404
    assert c.post("/api/samples/missing").status_code == 404
