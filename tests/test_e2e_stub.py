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


def test_time_phrase_limits_only_its_own_object(clip, tmp_path):
    from censor.agent.agent import Agent
    from censor.agent.tools import Session
    from censor.vision.detect import StubDetector
    s = Session(clip, tmp_path / "w", detector=StubDetector())
    Agent(s).run("blur all plates and the person in the red shirt around halfway")
    by_label = {s.tracks.get(o.targets[0]).label: o.time_range for o in s.log.ops}
    assert by_label["license plate"] is None  # the whole clip, not just halfway
    assert by_label["person"] is not None


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


def test_reopening_a_video_skips_its_sidecar(clip, client, monkeypatch):
    """A video folder holds source.mp4 and, for samples, source.mp4.tracks.json; both match source.*. On the Strix
    Halo the glob listed the sidecar first and loading the synthetic sample failed (500, "cannot open video")."""
    _, app_mod = client
    d = app_mod.DATA_DIR / "abcd1234"
    d.mkdir(parents=True)
    shutil.copyfile(clip, d / "source.mp4")
    shutil.copyfile(str(clip) + ".tracks.json", d / "source.mp4.tracks.json")
    real = Path.glob
    monkeypatch.setattr(Path, "glob", lambda self, pat: reversed(sorted(real(self, pat))))  # sidecar first
    assert Path(app_mod.session("abcd1234").video).name == "source.mp4"


def _upload(c, clip):
    with open(clip, "rb") as f:
        vid = c.post("/api/videos", files={"file": ("synthetic.mp4", f, "video/mp4")}).json()["video_id"]
    with open(str(clip) + ".tracks.json", "rb") as f:
        c.post(f"/api/videos/{vid}/tracks-sidecar", files={"file": ("t.json", f)})
    return vid


def test_exports_are_never_cached(clip, client):
    """AUP's Cloudflare cached export.mp4 for 4 hours and served an old, unmasked render after a new one."""
    c, _ = client
    vid = _upload(c, clip)
    c.post(f"/api/videos/{vid}/request", json={"text": "blur all plates"})
    first = c.post(f"/api/videos/{vid}/export").json()["url"]
    r = c.get("/" + first)
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    assert c.get(f"/api/videos/{vid}/preview/0.jpg").headers["cache-control"] == "no-store"
    c.post(f"/api/videos/{vid}/request", json={"text": "blur the screen"})
    assert c.post(f"/api/videos/{vid}/export").json()["url"] != first  # a new render gets a new URL


def test_export_waits_for_a_running_request(clip, client):
    """An Export clicked mid-request (another tab, or after the proxy dropped the request) must include its masks."""
    import threading
    import time
    from censor.agent.agent import Agent
    c, app_mod = client
    vid = _upload(c, clip)
    started = threading.Event()

    def slow_request():
        with app_mod.turn(vid, "blur all plates"):
            started.set()
            time.sleep(0.5)
            Agent(app_mod.session(vid)).run("blur all plates")

    t = threading.Thread(target=slow_request)
    t.start()
    started.wait()
    assert c.get(f"/api/videos/{vid}").json()["busy"] == "blur all plates"
    e = c.post(f"/api/videos/{vid}/export").json()  # blocks until the request is done
    t.join()
    assert e["stats"]["masked_boxes"] > 0
    assert c.get(f"/api/videos/{vid}").json()["busy"] is None


def test_export_gives_up_with_a_message(clip, client, monkeypatch):
    import threading
    c, app_mod = client
    vid = _upload(c, clip)
    monkeypatch.setattr(app_mod, "EXPORT_WAIT", 0.1)
    held, release = threading.Event(), threading.Event()

    def hold():
        with app_mod.turn(vid, "blur everyone except me"):
            held.set()
            release.wait(5)

    t = threading.Thread(target=hold)
    t.start()
    held.wait()
    r = c.post(f"/api/videos/{vid}/export")
    release.set()
    t.join()
    assert r.status_code == 409 and "blur everyone except me" in r.json()["detail"]


def test_frames_skip_and_decode_ahead(clip):
    """Every n-th frame, and decoding on a thread, give exactly the frames a plain read gives."""
    from censor.core.video import frames
    plain = [(f, int(img.sum())) for f, img in frames(str(clip))]
    every3 = [x for x in plain if x[0] % 3 == 0]
    assert [(f, int(img.sum())) for f, img in frames(str(clip), every=3)] == every3
    assert [(f, int(img.sum())) for f, img in frames(str(clip), every=3, prefetch=4)] == every3
    it = frames(str(clip), prefetch=2)
    next(it)
    it.close()  # stopping early must not leave the decoder blocked


def test_vlm_crops_always_have_the_same_shape(clip):
    """A new input shape costs the VLM a kernel search on ROCm (~15 s), and a thin box scaled by its short side
    once became a giant image: every crop is fitted into one canvas, three per call."""
    from censor.core.tracks import Keyframe, Track
    from censor.vision import vlm
    thin = Track(track_id="t", label="person", keyframes=[Keyframe(f=0, box=(100, 0, 103, 700), conf=0.5)])
    small = Track(track_id="s", label="person", keyframes=[Keyframe(f=0, box=(100, 100, 120, 160), conf=0.5)])
    for t in (thin, small):
        crops = vlm.track_crops(str(clip), t)
        assert len(crops) == vlm.CROPS_PER_TRACK
        assert all(c.shape == (vlm.CROP_SIZE[1], vlm.CROP_SIZE[0], 3) for c in crops)
