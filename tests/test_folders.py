"""Folders on the home screen: create one, upload into it, list it."""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("clips") / "synthetic.mp4"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_clip.py"), str(out), "1"], check=True)
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


def upload(c, clip, folder=None, name="street.mp4"):
    with open(clip, "rb") as fh:
        return c.post("/api/videos", files={"file": (name, fh, "video/mp4")}, data={"folder": folder} if folder else {})


def test_no_folders_until_one_is_made(client):
    c, _ = client
    assert c.get("/api/folders").json() == []
    f = c.post("/api/folders", json={"name": "  Ride   along "}).json()
    assert f["name"] == "Ride along" and f["clipCount"] == 0  # whitespace tidied
    assert [x["id"] for x in c.get("/api/folders").json()] == [f["id"]]


def test_a_folder_needs_a_short_name(client):
    c, _ = client
    assert c.post("/api/folders", json={"name": "   "}).status_code == 400
    assert c.post("/api/folders", json={"name": "x" * 61}).status_code == 400


def test_upload_into_a_folder(client, clip):
    c, _ = client
    f = c.post("/api/folders", json={"name": "Dashcam"}).json()
    vid = upload(c, clip, f["id"], "monday.mp4").json()["video_id"]
    upload(c, clip)  # no folder: filed nowhere, as before
    assert c.get("/api/folders").json()[0]["clipCount"] == 1
    got = c.get(f"/api/folders/{f['id']}").json()
    assert [(x["video_id"], x["name"], x["kind"]) for x in got["clips"]] == [(vid, "monday.mp4", "video")]


def test_unknown_folder_saves_nothing(client, clip):
    c, app_mod = client
    assert upload(c, clip, "nope").status_code == 404
    assert c.get("/api/folders/nope").status_code == 404
    assert not any(app_mod.DATA_DIR.glob("*/source.*"))


def test_most_recently_used_first_and_gone_clips_dropped(client, clip):
    c, app_mod = client
    a = c.post("/api/folders", json={"name": "A"}).json()
    time.sleep(0.01)
    b = c.post("/api/folders", json={"name": "B"}).json()
    assert [x["name"] for x in c.get("/api/folders").json()] == ["B", "A"]
    vid = upload(c, clip, a["id"]).json()["video_id"]  # filing into A uses it
    assert [x["name"] for x in c.get("/api/folders").json()] == ["A", "B"]
    shutil.rmtree(app_mod.DATA_DIR / vid)  # its session was deleted
    assert c.get("/api/folders").json()[0]["clipCount"] == 0
    assert c.get(f"/api/folders/{a['id']}").json()["clips"] == []


def test_rename_and_delete(client, clip):
    c, app_mod = client
    f = c.post("/api/folders", json={"name": "Old"}).json()
    vid = upload(c, clip, f["id"]).json()["video_id"]
    assert c.patch(f"/api/folders/{f['id']}", json={"name": " New  name "}).json()["name"] == "New name"
    assert c.get(f"/api/folders/{f['id']}").json()["name"] == "New name"
    assert c.patch(f"/api/folders/{f['id']}", json={"name": ""}).status_code == 400
    assert c.patch("/api/folders/nope", json={"name": "x"}).status_code == 404
    assert c.delete(f"/api/folders/{f['id']}").json() == {"deleted": f["id"], "clips_kept": 1}
    assert c.get("/api/folders").json() == []
    assert c.get(f"/api/videos/{vid}").status_code == 200  # the clip itself is still there
    assert c.delete(f"/api/folders/{f['id']}").status_code == 404
