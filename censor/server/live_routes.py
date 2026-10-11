"""Live replay routes: play a recorded clip as if it were a stream, masked behind a short delay buffer.

Included by censor.server.app only when CENSOR_LIVE=1 (the replay puts ~100% load on the GPU), so the demo app on
:8000 never exposes /api/live/start.

  POST /api/live/start   {"clip": str, "height": 360..1080 = 720}  -> 202 + stats; warms the detectors in the background
                         404 unknown clip, 409 already running, 503 no EgoBlur models (not in stub mode), 400 stub clip without sidecar
  POST /api/live/stop    -> final stats (or the idle dict)
  GET  /api/live/stats   engine stats + phase + warming, or {"phase": "idle", "running": false}
  GET  /api/live/frame?run=<id>&after=<n>   long-poll one JPEG (headers X-Live-Frame, X-Live-Run); 204 on timeout or idle;
                         a wrong or missing run id gets the newest frame now
  GET  /api/live/live    web/live.html (so the page's own URLs are relative to /api/live/)

A viewer that stops polling for AUTO_STOP_S seconds (a closed tab) gets the replay stopped, so it can't hold the GPU.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from censor.vision.specialist import EGOBLUR_DIR, egoblur_available

if TYPE_CHECKING:  # censor.live is imported lazily: this module must import even before the engine exists
    from censor.live import LiveReplay

router = APIRouter(prefix="/api/live")

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
# Same as censor/server/app.py:60-61 (not imported: app.py includes this router, so that would be circular)
SAMPLES_DIR = Path(__file__).resolve().parents[2] / "bench" / "clips"
SAMPLE_NAME = re.compile(r"[\w-]+")  # a bare file stem: no paths

AUTO_STOP_S = 30.0  # no /frame poll for this long -> stop
AUTO_STOP_CHECK_S = 5.0
FRAME_WAIT_S = 2.0  # how long one /frame poll waits for a newer frame
AHEAD = 10**12  # an `after` beyond any frame number: the engine answers with its newest frame

_lock = threading.Lock()
# phase: idle | warming | running. gen counts starts and stops, so a warm-up that was stopped meanwhile discards its
# engine. error: why the last start failed, shown by /stats once the phase is back to idle.
_state: dict = {"phase": "idle", "replay": None, "last_poll": 0.0, "clip": None, "error": None, "gen": 0}
_watchdog: threading.Thread | None = None


class StartIn(BaseModel):
    clip: str
    height: int = Field(720, ge=360, le=1080)


def make_replay(clip_path: str, height: int, stub: bool) -> "LiveReplay":
    """Build the engine for one clip (tests replace this with a fake). Real mode loads EgoBlur and warms it at the
    live size before the clock starts; stub mode replays the clip's tracks.json sidecar, no GPU."""
    from censor.core.video import probe
    from censor.live import LiveReplay, egoblur_detect, sidecar_detect

    if stub:
        v = probe(clip_path)
        detect = sidecar_detect(Path(f"{clip_path}.tracks.json"), src_h=v.height, live_h=height, n_frames=v.n_frames)
        return LiveReplay(clip_path, detect, height=height)
    from censor.vision.specialist import EgoBlurDetector

    v = probe(clip_path)
    width = max(2, int(round(v.width * height / v.height / 2)) * 2)
    det = EgoBlurDetector()
    det.warm_up(sizes=((height, width),))
    return LiveReplay(clip_path, egoblur_detect(det.infer), height=height)


def _stub_mode() -> bool:
    return os.environ.get("CENSOR_STUB") == "1"


def _idle_stats() -> dict:
    out: dict = {"phase": "idle", "running": False}
    if _state["error"]:
        out["error"] = _state["error"]
    return out


def _stats_locked() -> dict:
    """Current stats; the caller holds _lock."""
    phase, replay = _state["phase"], _state["replay"]
    if phase == "warming":
        return {"phase": "warming", "running": False, "warming": True, "clip": _state["clip"]}
    if replay is None:
        return _idle_stats()
    return {**replay.stats(), "phase": phase, "warming": False}


def _warm_and_run(gen: int, clip_path: str, height: int, stub: bool) -> None:
    replay = None
    try:
        replay = make_replay(clip_path, height, stub)
        replay.start()
    except Exception as e:  # noqa: BLE001  any failure (missing models, unreadable clip) ends in idle with a message
        print(f"[live] start failed: {type(e).__name__}: {e}", file=sys.stderr)
        with _lock:
            if _state["gen"] == gen:
                _state.update(phase="idle", replay=None, error=f"{type(e).__name__}: {e}")
        if replay is not None:
            replay.stop()
        return
    with _lock:
        if _state["gen"] == gen and _state["phase"] == "warming":
            _state.update(phase="running", replay=replay, last_poll=time.monotonic())
            return
    replay.stop()  # stopped while warming up: nobody wants this engine any more


def _auto_stop_check() -> None:
    with _lock:
        if _state["phase"] != "running" or time.monotonic() - _state["last_poll"] <= AUTO_STOP_S:
            return
        replay = _state["replay"]
        _state.update(phase="idle", replay=None, gen=_state["gen"] + 1)
    print(f"[live] no frame poll for {AUTO_STOP_S:.0f} s; stopping the replay", file=sys.stderr)
    if replay is not None:
        replay.stop()


def _watch() -> None:
    while True:
        time.sleep(AUTO_STOP_CHECK_S)
        try:
            _auto_stop_check()
        except Exception as e:  # noqa: BLE001  the watchdog must outlive one bad check
            print(f"[live] auto-stop check failed: {type(e).__name__}: {e}", file=sys.stderr)


def _ensure_watchdog() -> None:
    global _watchdog
    with _lock:
        if _watchdog is not None and _watchdog.is_alive():
            return
        _watchdog = threading.Thread(target=_watch, daemon=True, name="live-auto-stop")
        _watchdog.start()


@router.post("/start", status_code=202)
def start(body: StartIn):
    src = SAMPLES_DIR / f"{body.clip}.mp4"
    if not SAMPLE_NAME.fullmatch(body.clip) or not src.exists():
        raise HTTPException(404, f"no sample clip {body.clip!r}")
    stub = _stub_mode()
    with _lock:
        if _state["phase"] != "idle":
            raise HTTPException(409, "a live replay is already running; stop it first")
        if not stub and not egoblur_available():
            raise HTTPException(503, f"EgoBlur models not found in {EGOBLUR_DIR}; live replay needs them")
        if stub and not Path(f"{src}.tracks.json").exists():
            raise HTTPException(400, f"stub mode needs {src.name}.tracks.json next to the clip")
        gen = _state["gen"] + 1
        _state.update(phase="warming", replay=None, clip=body.clip, error=None, gen=gen)
        stats = _stats_locked()
    _ensure_watchdog()
    threading.Thread(target=_warm_and_run, args=(gen, str(src), body.height, stub), daemon=True,
                     name="live-start").start()
    return stats


@router.post("/stop")
def stop():
    with _lock:
        replay = _state["replay"]
        _state.update(phase="idle", replay=None, gen=_state["gen"] + 1)
    if replay is None:
        return _idle_stats()
    final = replay.stats()
    replay.stop()
    return {**final, "phase": "idle", "running": False, "warming": False}


@router.get("/stats")
def stats():
    with _lock:
        return _stats_locked()


@router.get("/frame")
def frame(run: str = "", after: int = -1):
    with _lock:
        replay = _state["replay"] if _state["phase"] == "running" else None
    if replay is None:
        return Response(status_code=204)
    if run != replay.run:  # a client from before a restart, or none yet: show the newest frame now
        after = AHEAD
    res = replay.latest(after, timeout=FRAME_WAIT_S)
    with _lock:
        _state["last_poll"] = time.monotonic()
    if res is None:
        return Response(status_code=204)
    f, jpeg = res
    return Response(content=jpeg, media_type="image/jpeg", headers={"X-Live-Frame": str(f), "X-Live-Run": replay.run})


@router.get("/live", response_class=HTMLResponse)
def live_page():
    page = WEB_DIR / "live.html"
    if not page.exists():
        return "<p>The live replay page (web/live.html) is not in this checkout yet. The API is at /api/live/.</p>"
    return page.read_text(encoding="utf-8")
