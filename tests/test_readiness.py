"""Detector status must describe real runtime state without invoking GPU work."""
import importlib
from importlib.machinery import ModuleSpec
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient
import numpy as np

from censor.core.tracks import TrackStore
from censor.vision import detect, specialist
from censor.vision import readiness


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.delenv("CENSOR_STUB", raising=False)
    monkeypatch.setenv("CENSOR_WARMUP", "0")
    monkeypatch.setenv("CENSOR_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("CENSOR_EGOBLUR_DIR", str(tmp_path))
    monkeypatch.setattr(specialist, "EGOBLUR_DIR", tmp_path)
    monkeypatch.setattr(specialist.EgoBlurDetector, "_cache", {})
    monkeypatch.setattr(detect.YoloeDetector, "_models", {})

    class Model:
        def __init__(self, weights):
            pass

        def get_text_pe(self, queries):
            return queries

        def set_classes(self, queries, embeddings):
            pass

        def predict(self, *args, **kwargs):
            pass

        def track(self, *args, **kwargs):
            return iter([SimpleNamespace(boxes=None)])

    module = ModuleType("ultralytics")
    module.__spec__ = ModuleSpec("ultralytics", loader=None)
    module.YOLOE = Model
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    for name in specialist.EGOBLUR_FILES.values():
        (tmp_path / name).write_bytes(b"fake model: never loaded by status")
    readiness.reset()
    yield tmp_path, Model
    readiness.reset()


def client():
    import censor.server.app as app_mod
    importlib.reload(app_mod)
    return TestClient(app_mod.app)


def rows(status):
    return {row["id"]: row for row in status["detectors"]}


def test_status_is_read_only_even_while_detector_locks_are_held(configured, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Reading status must not construct or load a detector")

    monkeypatch.setattr(detect, "get_detector", forbidden)
    monkeypatch.setattr(specialist, "load_egoblur", forbidden)
    c = client()
    with detect.YoloeDetector._lock, specialist.EgoBlurDetector._lock:
        response = c.get("/api/detectors")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["mode"] == "real"
    assert body["status"] == "not_loaded"
    assert {row["status"] for row in body["detectors"]} == {"not_loaded"}


def test_explicit_stub_never_reports_real_detection(configured, monkeypatch):
    monkeypatch.setenv("CENSOR_STUB", "1")
    readiness.reset()
    body = client().get("/api/detectors").json()
    assert body["mode"] == "stub"
    assert body["status"] != "ready"
    assert "stub" in body["message"].lower()
    assert all(row["status"] != "ready" for row in body["detectors"])


def test_missing_ultralytics_reports_automatic_stub(configured, monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    readiness.reset()
    assert isinstance(detect.get_detector(), detect.StubDetector)
    body = client().get("/api/detectors").json()
    assert body["mode"] == "stub"
    assert body["status"] != "ready"
    assert "ultralytics" in str(body).lower()


@pytest.mark.parametrize("missing", ["face", "license plate"])
def test_partial_egoblur_install_reports_both_fallbacks(configured, missing):
    directory, _ = configured
    (directory / specialist.EGOBLUR_FILES[missing]).unlink()
    readiness.reset()
    assert isinstance(detect.get_detector(), detect.YoloeDetector)
    body = readiness.snapshot()
    assert body["mode"] == "real" and body["status"] == "degraded"
    by_id = rows(body)
    for category in ("face", "plate"):
        assert by_id[category]["status"] == "fallback"
        assert "YOLOE" in by_id[category]["model"]


def test_loading_is_visible_without_waiting_for_warmup(configured, monkeypatch):
    _, model = configured
    entered, release = threading.Event(), threading.Event()

    def block(*args, **kwargs):
        entered.set()
        assert release.wait(5), "test did not release warmup"

    monkeypatch.setattr(model, "predict", block)
    detector = detect.get_detector()
    c = client()
    worker = threading.Thread(target=detector.general.warm_up)
    worker.start()
    try:
        assert entered.wait(2)
        body = c.get("/api/detectors").json()
        assert body["status"] == "loading"
        assert rows(body)["objects"]["status"] == "loading"
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert rows(readiness.snapshot())["objects"]["status"] == "ready"
    detect.get_detector()  # another Session wraps the same process-wide models
    assert rows(readiness.snapshot())["objects"]["status"] == "ready"


def test_general_failure_stays_visible_until_successful_retry(configured, monkeypatch):
    _, model = configured
    detector = detect.get_detector().general

    def broken(*args, **kwargs):
        raise RuntimeError("GPU warmup failed")

    monkeypatch.setattr(model, "predict", broken)
    with pytest.raises(RuntimeError, match="GPU warmup failed"):
        detector.warm_up()
    assert rows(readiness.snapshot())["objects"]["status"] == "unavailable"
    detect.get_detector()
    assert rows(readiness.snapshot())["objects"]["status"] == "unavailable"
    monkeypatch.setattr(model, "predict", lambda *a, **kw: None)
    detector.warm_up()
    assert rows(readiness.snapshot())["objects"]["status"] == "ready"


def test_egoblur_files_do_not_mean_ready_and_load_failure_recovers(configured, monkeypatch):
    detector = detect.get_detector().specialist
    assert rows(readiness.snapshot())["face"]["status"] == "not_loaded"

    def broken():
        raise RuntimeError("invalid TorchScript model")

    monkeypatch.setattr(specialist, "load_egoblur", broken)
    with pytest.raises(RuntimeError, match="invalid TorchScript"):
        detector.warm_up(sizes=((2, 2),))
    failed = rows(readiness.snapshot())
    assert failed["face"]["status"] == "unavailable"
    assert failed["plate"]["status"] == "unavailable"
    monkeypatch.setattr(specialist, "load_egoblur", lambda: {
        "face": lambda image: [], "license plate": lambda image: []})
    detector.warm_up(sizes=((2, 2),))
    recovered = rows(readiness.snapshot())
    assert recovered["face"]["status"] == "ready"
    assert recovered["plate"]["status"] == "ready"


def test_requests_update_readiness_when_startup_warmup_is_disabled(configured, monkeypatch):
    monkeypatch.setattr(specialist, "load_egoblur", lambda: {
        "face": lambda image: [], "license plate": lambda image: []})
    monkeypatch.setattr(specialist, "frames", lambda *a, **kw: iter([(0, np.zeros((2, 2, 3), np.uint8))]))
    detector = detect.get_detector()
    detector.run("clip.mp4", ["face", "license plate", "person"], TrackStore())
    assert readiness.snapshot()["status"] == "ready"
    assert {row["status"] for row in readiness.snapshot()["detectors"]} == {"ready"}
    detect.get_detector()
    assert readiness.snapshot()["status"] == "ready"


def test_empty_scans_do_not_claim_models_are_ready(configured, monkeypatch):
    _, model = configured
    monkeypatch.setattr(model, "track", lambda *a, **kw: iter(()))
    monkeypatch.setattr(specialist, "frames", lambda *a, **kw: iter(()))
    detector = detect.get_detector()
    detector.run("empty.mp4", ["face", "license plate", "person"], TrackStore())
    assert readiness.snapshot()["status"] == "not_loaded"
    assert {row["status"] for row in readiness.snapshot()["detectors"]} == {"not_loaded"}


def test_detector_initialization_error_is_reported(configured, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("detector initialization failed")

    monkeypatch.setattr(detect, "YoloeDetector", broken)
    with pytest.raises(RuntimeError, match="detector initialization failed"):
        detect.get_detector()
    body = readiness.snapshot()
    assert body["status"] in ("degraded", "unavailable")
    assert rows(body)["objects"]["status"] == "unavailable"
