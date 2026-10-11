"""The live replay routes, against a fake engine: no GPU, no clip, no ffmpeg."""
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from censor.server import live_routes

ROOT = Path(__file__).resolve().parents[1]
JPEG = b"\xff\xd8jpegbytes\xff\xd9"


class FakeReplay:
    """Same surface as censor.live.LiveReplay, scripted."""

    def __init__(self, frames=((5, JPEG),)):
        self.run = "abc12345"
        self.warming = False
        self.fps, self.size, self.n_frames = 30.0, (1280, 720), 90
        self.frames = list(frames)
        self.started = self.stopped = False
        self.alive = True  # a test flips this to False to play an engine whose thread died
        self.afters: list[int] = []

    @property
    def running(self):
        return self.alive and not self.stopped

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def latest(self, after, timeout):
        self.afters.append(after)
        if not self.frames or self.stopped:
            return None
        newer = [fr for fr in self.frames if fr[0] > after]
        return newer[0] if newer else self.frames[-1] if after > self.frames[-1][0] else None

    def stats(self):
        return {"run": self.run, "clip": "demo", "running": not self.stopped, "size": list(self.size),
                "fps": self.fps, "n_frames": self.n_frames, "delay_s": 1.0, "frames_out": 7, "blind": 0,
                "blind_share": 0.0, "dropped_out": 0, "emit_errors": 0,
                "detector": {"passes": 3, "p50_ms": 120.0, "p95_ms": 140.0, "baseline_p95_ms": 140.0, "errors": 0},
                "emit": {"p50_ms": 5.0, "p95_ms": 9.0}, "detected_share": 0.2,
                "gap_frames": {"mean": 5.0, "max": 6}, "hold_frames": 15, "frame_age_ms": {"p50": 1000.0}}


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch, tmp_path):
    """Every test starts idle, with one sample clip `demo` and EgoBlur 'present'."""
    monkeypatch.setitem(live_routes._state, "phase", "idle")
    monkeypatch.setitem(live_routes._state, "replay", None)
    monkeypatch.setitem(live_routes._state, "last_poll", 0.0)
    monkeypatch.setitem(live_routes._state, "clip", None)
    monkeypatch.setitem(live_routes._state, "error", None)
    monkeypatch.setattr(live_routes, "_detectors", {})
    samples = tmp_path / "clips"
    samples.mkdir()
    (samples / "demo.mp4").write_bytes(b"")  # the fake never opens it
    monkeypatch.setattr(live_routes, "SAMPLES_DIR", samples)
    monkeypatch.delenv("CENSOR_STUB", raising=False)
    monkeypatch.setattr(live_routes, "egoblur_available", lambda: True)
    yield
    replay = live_routes._state["replay"]
    if replay is not None:
        replay.stop()


@pytest.fixture()
def fake(monkeypatch):
    made = []

    def factory(clip, height, stub):
        r = FakeReplay()
        made.append((clip, height, stub, r))
        return r

    monkeypatch.setattr(live_routes, "make_replay", factory)
    return made


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(live_routes.router)
    return TestClient(app)


def _wait_phase(c, phase, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        s = c.get("/api/live/stats").json()
        if s["phase"] == phase:
            return s
        time.sleep(0.02)
    raise AssertionError(f"phase never became {phase}: {s}")


def _start(c, **body):
    return c.post("/api/live/start", json={"clip": "demo", **body})


def test_start_answers_202_then_runs(client, fake):
    r = _start(client)
    assert r.status_code == 202
    assert r.json()["phase"] in ("warming", "running")
    s = _wait_phase(client, "running")
    assert s["running"] is True and s["warming"] is False
    assert s["detector"]["passes"] == 3  # the engine's stats, merged
    clip, height, stub, replay = fake[0]
    assert (clip, height, stub) == (str(live_routes.SAMPLES_DIR / "demo.mp4"), 720, False)
    assert replay.started


def test_start_passes_the_requested_height(client, fake):
    _start(client, height=480)
    _wait_phase(client, "running")
    assert fake[0][1] == 480


def test_warming_reports_itself(client, monkeypatch):
    release = []

    def slow(clip, height, stub):
        while not release:
            time.sleep(0.01)
        return FakeReplay()

    monkeypatch.setattr(live_routes, "make_replay", slow)
    r = _start(client)
    assert r.status_code == 202
    assert r.json() == {"phase": "warming", "running": False, "warming": True, "clip": "demo"}
    assert client.get("/api/live/stats").json()["warming"] is True
    assert client.get("/api/live/frame").status_code == 204  # nothing to show yet
    release.append(1)
    _wait_phase(client, "running")


def test_second_start_is_409(client, fake):
    _start(client)
    r = _start(client)
    assert r.status_code == 409
    assert r.json() == {"detail": "a live replay is already running; stop it first"}
    assert len(fake) <= 1  # never a second engine


def test_unknown_clip_is_404(client, fake):
    assert client.post("/api/live/start", json={"clip": "nope"}).status_code == 404
    assert client.post("/api/live/start", json={"clip": "../etc/passwd"}).status_code == 404
    assert client.get("/api/live/stats").json()["phase"] == "idle"


@pytest.mark.parametrize("height", [2000, 100, 1081, 359])
def test_height_out_of_range_is_422(client, fake, height):
    assert _start(client, height=height).status_code == 422


def test_frame_when_idle_is_204(client):
    r = client.get("/api/live/frame")
    assert r.status_code == 204 and r.content == b""


def test_frame_after_start_is_a_jpeg_with_headers(client, fake):
    _start(client)
    _wait_phase(client, "running")
    r = client.get("/api/live/frame", params={"run": "abc12345", "after": -1})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.headers["X-Live-Frame"] == "5"
    assert r.headers["X-Live-Run"] == "abc12345"
    assert r.content == JPEG
    assert live_routes._state["last_poll"] > 0


def test_wrong_run_gets_the_newest_frame_now(client, fake):
    _start(client)
    _wait_phase(client, "running")
    replay = fake[0][3]
    r = client.get("/api/live/frame", params={"run": "stale000", "after": 3})
    assert r.status_code == 200 and r.headers["X-Live-Run"] == "abc12345"
    assert replay.afters[-1] >= 10**9  # "ahead of everything": the engine answers with the newest
    client.get("/api/live/frame")  # missing run counts as wrong too
    assert replay.afters[-1] >= 10**9


def test_right_run_passes_after_through(client, fake):
    _start(client)
    _wait_phase(client, "running")
    r = client.get("/api/live/frame", params={"run": "abc12345", "after": 5})
    assert r.status_code == 204  # nothing newer than 5
    assert fake[0][3].afters[-1] == 5


def test_stop_returns_final_stats_and_goes_idle(client, fake):
    _start(client)
    _wait_phase(client, "running")
    r = client.post("/api/live/stop")
    assert r.status_code == 200
    assert r.json()["frames_out"] == 7 and r.json()["phase"] == "idle" and r.json()["running"] is False
    assert fake[0][3].stopped
    assert client.get("/api/live/stats").json() == {"phase": "idle", "running": False}
    assert client.get("/api/live/frame").status_code == 204
    assert _start(client).status_code == 202  # can start again


def test_stop_when_idle_is_200(client):
    r = client.post("/api/live/stop")
    assert r.status_code == 200 and r.json() == {"phase": "idle", "running": False}


def test_stop_while_warming_discards_the_engine(client, monkeypatch):
    release, built = [], []

    def slow(clip, height, stub):
        while not release:
            time.sleep(0.01)
        r = FakeReplay()
        built.append(r)
        return r

    monkeypatch.setattr(live_routes, "make_replay", slow)
    _start(client)
    assert client.post("/api/live/stop").json()["phase"] == "idle"
    release.append(1)
    deadline = time.monotonic() + 2
    while not built and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)
    assert client.get("/api/live/stats").json()["phase"] == "idle"
    assert built[0].stopped and live_routes._state["replay"] is None


def _slow_factory(monkeypatch):
    """make_replay that blocks until `release` is set, then builds a FakeReplay. engines[i] belongs to call i."""
    release, engines, calls = threading.Event(), {}, []

    def slow(clip, height, stub):
        i = len(calls)
        calls.append(clip)
        assert release.wait(5)
        engines[i] = FakeReplay()
        return engines[i]

    monkeypatch.setattr(live_routes, "make_replay", slow)
    return release, engines, calls


def _wait_for(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        time.sleep(0.01)
    assert cond()


def test_engine_built_after_a_stop_is_never_started(client, monkeypatch):
    release, engines, calls = _slow_factory(monkeypatch)
    _start(client)
    _wait_for(lambda: len(calls) == 1)
    assert client.post("/api/live/stop").json()["phase"] == "idle"
    assert _start(client).status_code == 202  # a second warm-up begins while the first is still going
    _wait_for(lambda: len(calls) == 2)
    release.set()
    _wait_for(lambda: len(engines) == 2 and engines[0].stopped)
    assert _wait_phase(client, "running")["running"] is True
    assert not engines[0].started and engines[0].stopped  # built after its stop: never started
    assert engines[1].started and not engines[1].stopped
    assert live_routes._state["replay"] is engines[1]


def test_stop_during_engine_start_discards_it(client, monkeypatch):
    in_start, go = threading.Event(), threading.Event()
    made = []

    class SlowStart(FakeReplay):
        def start(self):
            in_start.set()
            assert go.wait(5)
            super().start()

    def factory(clip, height, stub):
        made.append(SlowStart())
        return made[0]

    monkeypatch.setattr(live_routes, "make_replay", factory)
    _start(client)
    assert in_start.wait(2)
    assert client.post("/api/live/stop").json()["phase"] == "idle"
    go.set()
    _wait_for(lambda: made[0].started)
    time.sleep(0.1)
    assert made[0].stopped
    assert client.get("/api/live/stats").json()["phase"] == "idle" and live_routes._state["replay"] is None


def test_stats_idle_shape(client):
    assert client.get("/api/live/stats").json() == {"phase": "idle", "running": False}


class FakeDetector:
    """Stands in for censor.vision.specialist.EgoBlurDetector: counts loads and warm-ups."""
    made: list = []
    warmed: list = []
    warm_gate: threading.Event | None = None

    def __init__(self):
        FakeDetector.made.append(self)
        self.infer = {"face": object()}

    def warm_up(self, sizes):
        if FakeDetector.warm_gate is not None:
            assert FakeDetector.warm_gate.wait(5)
        FakeDetector.warmed.append(tuple(sizes))


@pytest.fixture()
def real_mode(monkeypatch):
    """make_replay in real (non-stub) mode, with the detector, the probe and the engine faked."""
    FakeDetector.made, FakeDetector.warmed, FakeDetector.warm_gate = [], [], None
    monkeypatch.setattr("censor.vision.specialist.EgoBlurDetector", FakeDetector)
    monkeypatch.setattr("censor.core.video.probe", lambda path: types.SimpleNamespace(width=1920, height=1080, n_frames=90))
    engine_mod = types.ModuleType("censor.live")
    engine_mod.LiveReplay = lambda path, detect, height: types.SimpleNamespace(path=path, detect=detect, height=height)
    engine_mod.egoblur_detect = lambda infer: infer
    engine_mod.sidecar_detect = None
    monkeypatch.setitem(sys.modules, "censor.live", engine_mod)
    return FakeDetector


def test_make_replay_loads_and_warms_egoblur_once_per_size(real_mode):
    a = live_routes.make_replay("a.mp4", 720, False)
    b = live_routes.make_replay("b.mp4", 720, False)
    assert (a.path, b.path) == ("a.mp4", "b.mp4") and a.detect is b.detect  # same detector, same infer
    assert len(real_mode.made) == 1 and real_mode.warmed == [((720, 1280),)]
    live_routes.make_replay("a.mp4", 540, False)  # a new live size is a new warm-up
    assert len(real_mode.made) == 2 and real_mode.warmed[-1] == ((540, 960),)


def test_concurrent_warmups_share_one_detector(real_mode):
    real_mode.warm_gate = threading.Event()
    out = []
    threads = [threading.Thread(target=lambda: out.append(live_routes.make_replay("a.mp4", 720, False))) for _ in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.1)  # both are inside make_replay; one holds the detector lock in its warm-up
    real_mode.warm_gate.set()
    for t in threads:
        t.join(5)
    assert len(out) == 2 and len(real_mode.made) == 1 and len(real_mode.warmed) == 1


def test_a_failed_warmup_is_not_cached(real_mode, monkeypatch):
    def boom(self, sizes):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(real_mode, "warm_up", boom)
    with pytest.raises(RuntimeError):
        live_routes.make_replay("a.mp4", 720, False)
    assert live_routes._detectors == {}


def test_failed_start_goes_idle_and_reports_the_error(client, monkeypatch):
    def boom(clip, height, stub):
        raise RuntimeError("no GPU today")

    monkeypatch.setattr(live_routes, "make_replay", boom)
    assert _start(client).status_code == 202
    s = _wait_phase(client, "idle")
    assert s["running"] is False and "no GPU today" in s["error"]
    assert _start(client).status_code == 202  # not stuck; the old error is cleared on a new start


def test_without_egoblur_files_is_503(client, fake, monkeypatch):
    monkeypatch.setattr(live_routes, "egoblur_available", lambda: False)
    r = _start(client)
    assert r.status_code == 503
    assert r.json()["detail"] == f"EgoBlur models not found in {live_routes.EGOBLUR_DIR}; live replay needs them"
    assert live_routes._state["phase"] == "idle" and not fake


def test_stub_mode_skips_the_egoblur_check_but_needs_a_sidecar(client, fake, monkeypatch):
    monkeypatch.setenv("CENSOR_STUB", "1")
    monkeypatch.setattr(live_routes, "egoblur_available", lambda: False)
    r = _start(client)
    assert r.status_code == 400 and "tracks.json" in r.json()["detail"]
    (live_routes.SAMPLES_DIR / "demo.mp4.tracks.json").write_text("{}")
    assert _start(client).status_code == 202
    _wait_phase(client, "running")
    assert fake[0][2] is True  # stub=True reached the factory


def test_auto_stop_after_no_polls(client, fake, monkeypatch):
    _start(client)
    _wait_phase(client, "running")
    monkeypatch.setattr(live_routes, "AUTO_STOP_S", 0.05)
    time.sleep(0.1)
    live_routes._auto_stop_check()
    assert fake[0][3].stopped
    assert client.get("/api/live/stats").json()["phase"] == "idle"


def test_a_poll_keeps_it_alive(client, fake, monkeypatch):
    _start(client)
    _wait_phase(client, "running")
    client.get("/api/live/frame", params={"run": "abc12345", "after": -1})
    live_routes._auto_stop_check()
    assert not fake[0][3].stopped
    assert client.get("/api/live/stats").json()["phase"] == "running"


def test_a_frame_poll_refreshes_last_poll(client, fake, monkeypatch):
    _start(client)
    _wait_phase(client, "running")
    stale = time.monotonic() - live_routes.AUTO_STOP_S - 1
    monkeypatch.setitem(live_routes._state, "last_poll", stale)
    assert client.get("/api/live/frame", params={"run": "abc12345", "after": -1}).status_code == 200
    assert live_routes._state["last_poll"] > stale
    live_routes._auto_stop_check()
    assert not fake[0][3].stopped


def test_204_polls_do_not_keep_it_alive(client, fake, monkeypatch):
    _start(client)
    _wait_phase(client, "running")
    stale = time.monotonic() - live_routes.AUTO_STOP_S - 1
    monkeypatch.setitem(live_routes._state, "last_poll", stale)
    for _ in range(3):  # nothing newer than frame 5: the engine is not producing, the poll times out
        assert client.get("/api/live/frame", params={"run": "abc12345", "after": 5}).status_code == 204
    assert live_routes._state["last_poll"] == stale
    live_routes._auto_stop_check()
    assert fake[0][3].stopped
    assert client.get("/api/live/stats").json()["phase"] == "idle"


def test_idle_204_polls_leave_last_poll_alone(client):
    assert client.get("/api/live/frame").status_code == 204
    assert live_routes._state["last_poll"] == 0.0


def _start_and_kill(client, fake):
    _start(client)
    _wait_phase(client, "running")
    replay = fake[0][3]
    replay.alive = False  # a thread of the engine died
    return replay


def test_dead_engine_is_reaped_by_stats(client, fake):
    dead = _start_and_kill(client, fake)
    s = client.get("/api/live/stats").json()
    assert s == {"phase": "idle", "running": False, "error": live_routes.ENGINE_DIED}
    assert dead.stopped  # its leftover threads are stopped too
    assert client.get("/api/live/frame").status_code == 204
    assert _start(client).status_code == 202  # not 409
    assert _wait_phase(client, "running")["running"] is True
    assert fake[1][3] is not dead


def test_dead_engine_is_reaped_by_frame(client, fake):
    dead = _start_and_kill(client, fake)
    assert client.get("/api/live/frame", params={"run": "abc12345", "after": -1}).status_code == 204
    assert dead.stopped and live_routes._state["phase"] == "idle" and live_routes._state["replay"] is None
    assert client.get("/api/live/stats").json()["error"] == live_routes.ENGINE_DIED
    assert _start(client).status_code == 202


def test_dead_engine_is_reaped_by_the_watchdog_check(client, fake):
    dead = _start_and_kill(client, fake)
    live_routes._auto_stop_check()  # last_poll is fresh: only the death can stop it
    assert dead.stopped and client.get("/api/live/stats").json()["error"] == live_routes.ENGINE_DIED


def test_start_over_a_dead_engine_is_accepted_without_a_poll(client, fake):
    dead = _start_and_kill(client, fake)
    assert _start(client).status_code == 202  # nobody looked at /stats or /frame in between
    assert dead.stopped
    assert _wait_phase(client, "running")["running"] is True


def test_a_live_engine_is_not_reaped(client, fake):
    _start(client)
    _wait_phase(client, "running")
    assert client.get("/api/live/stats").json()["running"] is True
    live_routes._auto_stop_check()
    assert not fake[0][3].stopped and "error" not in client.get("/api/live/stats").json()


def test_live_page_is_served(client, monkeypatch, tmp_path):
    monkeypatch.setattr(live_routes, "WEB_DIR", tmp_path)
    r = client.get("/api/live/live")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]  # a short message, not a 404
    (tmp_path / "live.html").write_text("<html>LIVE PAGE</html>")
    assert "LIVE PAGE" in client.get("/api/live/live").text


def _live_routes_in_fresh_process(live_flag):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "CENSOR_STUB": "1", "CENSOR_WARMUP": "0"}
    env.pop("CENSOR_LIVE", None)
    if live_flag is not None:
        env["CENSOR_LIVE"] = live_flag
    code = "from censor.server.app import app; print('\\n'.join(app.openapi()['paths']))"
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-1500:]
    return [p for p in out.stdout.splitlines() if p.startswith("/api/live")]


def test_demo_app_has_no_live_routes_without_the_flag():
    assert _live_routes_in_fresh_process(None) == []
    assert _live_routes_in_fresh_process("0") == []


def test_demo_app_has_live_routes_with_the_flag():
    paths = _live_routes_in_fresh_process("1")
    assert {"/api/live/start", "/api/live/stop", "/api/live/stats", "/api/live/frame", "/api/live/live"} <= set(paths)
