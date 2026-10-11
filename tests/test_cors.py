"""The page hosted apart from the server: CORS for its origin, the flags it can't get injected, and a Download
link that works cross-origin. No GPU or ffmpeg needed."""
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = "https://blurr.example.app"


def make_client(tmp_path, monkeypatch, origins=None):
    monkeypatch.setenv("CENSOR_STUB", "1")
    monkeypatch.setenv("CENSOR_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("CENSOR_MODEL_URL", raising=False)
    if origins is None:
        monkeypatch.delenv("CENSOR_CORS_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("CENSOR_CORS_ORIGINS", origins)
    import censor.server.app as app_mod
    importlib.reload(app_mod)
    from fastapi.testclient import TestClient
    return TestClient(app_mod.app)


def test_config_reports_the_flags_index_injects(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    assert c.get("/api/config").json() == {"offline": False, "stub": True}


def test_no_cors_headers_unless_an_origin_is_listed(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    r = c.get("/api/samples", headers={"Origin": PAGE})
    assert "access-control-allow-origin" not in r.headers


def test_a_listed_origin_may_call_the_api(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch, f" {PAGE}/ , http://localhost:5500")  # spaces and a trailing slash
    r = c.get("/api/samples", headers={"Origin": PAGE})
    assert r.headers["access-control-allow-origin"] == PAGE
    pre = c.options("/api/videos/x/request", headers={
        "Origin": PAGE, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
    assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] == PAGE
    other = c.get("/api/samples", headers={"Origin": "https://elsewhere.example"})
    assert "access-control-allow-origin" not in other.headers


def test_the_download_link_is_an_attachment(tmp_path, monkeypatch):
    photo = tmp_path / "synthetic.png"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_clip.py"), "--image", str(photo)], check=True)
    c = make_client(tmp_path, monkeypatch, PAGE)
    with open(photo, "rb") as f:
        vid = c.post("/api/videos", files={"file": ("synthetic.png", f, "image/png")}).json()["video_id"]
    c.post(f"/api/videos/{vid}/box", json={"frame": 0, "box": [10, 10, 60, 60]})
    url = c.post(f"/api/videos/{vid}/export").json()["url"]
    shown = c.get(f"/{url}", headers={"Origin": PAGE})  # what the Program monitor loads
    assert shown.status_code == 200 and "content-disposition" not in shown.headers
    assert shown.headers["access-control-allow-origin"] == PAGE
    saved = c.get(f"/{url}&download=1")
    assert saved.headers["content-disposition"].startswith("attachment") and "export.png" in saved.headers["content-disposition"]
