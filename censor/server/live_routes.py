"""Live replay routes: play a recorded clip as if it were a stream, masked behind a short delay buffer.

Included by censor.server.app only when CENSOR_LIVE=1 (the replay puts ~100% load on the GPU), so the demo app on
:8000 never exposes /api/live/start.

  POST /api/live/start   {"clip": str, "height": 540|720 = 720, "cats": ["face", "license plate"]}
                         -> 202 + stats; warms the detectors in the background
                         cats: what to mask this run; non-empty, each one of LIVE_CATS (duplicates dropped, order normalised
                         to LIVE_CATS order), else 422. The 202 / stats carry "cats": [..] while warming and running.
                         Stub mode with CENSOR_YUNET masking (clip or camera) and "license plate" requested still starts, but
                         YuNet only finds faces: the 202 / stats then also carry "detector_note":
                         "yunet: faces only, plates not detected" so the page can warn
                         404 unknown clip, 409 already running, 503 no EgoBlur models (not in stub mode), 400 stub clip with
                         neither a sidecar nor CENSOR_YUNET
                         stub mode (CENSOR_STUB=1, no GPU) masks a clip with its <clip>.mp4.tracks.json sidecar when there is one,
                         else with OpenCV's YuNet faces when CENSOR_YUNET names the ONNX file; the 202 / stats of a stub clip
                         run then carry "detector_kind": "sidecar" | "yunet" (real-mode clip runs carry none)
                         clip "camera" is the browser's own camera, not a sample file: the page pushes frames (no sidecar, no file
                         lookup) and the 202 / stats carry "detector_kind": "egoblur" | "yunet" | "none" ("none": nothing is masked)
  POST /api/live/push    one JPEG as the raw request body (camera runs only), header X-Live-Run must be the run id
                         -> {"f": n}; 409 not running / not a camera run / stale run id, 413 over 2 MB, 400 not a decodable JPEG
  POST /api/live/enroll  "That's me": enrolls the largest face in view as the identity to spare (blur everyone except me)
                         -> the engine's {"ok", "faces_seen", "box", "reason"} with 200 (ok false is a result, not an error);
                         409 not running, 503 {"detail": ...} when no face embedder is configured (CENSOR_SFACE unset)
  POST /api/live/enroll/clear   forget the enrolled face -> the engine's identity dict; 409 not running
  POST /api/live/stop    -> final stats (or the idle dict)
  GET  /api/live/stats   engine stats + phase + warming, or {"phase": "idle", "running": false}. Warming and running
                         stats carry "embedder": "sface" | "none" (CENSOR_SFACE set or not) and, running, the engine's
                         "identity": {"enrolled", "spared_now", "embedder"}
  GET  /api/live/frame?run=<id>&after=<n>   long-poll one JPEG (headers X-Live-Frame, X-Live-Run); 204 on timeout or idle;
                         a wrong or missing run id gets the newest frame now
  GET  /api/live/live    web/live.html (so the page's own URLs are relative to /api/live/)

The run id is a demo-grade guard against a stale page, not a secret: /stats shows it. The enroll routes have no auth
and rely on the server's 127.0.0.1 binding (plan section 6).

POST /api/live/start also takes "except_me": bool = false (needs CENSOR_SFACE, else 503 {"detail": "no embedder: set
CENSOR_SFACE"}); the 202 and stats carry "except_me". "height" is 540 or 720 (else 422). /push answers 429 when the
engine has too many decodes in flight (censor.live.Busy), 400 for a malformed Content-Length. /enroll/clear -> {"ok": true}.

A viewer that stops polling for AUTO_STOP_S seconds (a closed tab) gets the replay stopped, so it can't hold the GPU.
Only a /frame poll that returns a frame, or a /push the engine accepted, counts as a poll. An engine whose thread
died (LiveReplay.running is False) is noticed by /start, /stats, /frame, /push and the watchdog: the run ends in idle
with an error and a new start is accepted.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

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
CAMERA = "camera"  # the clip id for the browser's camera: frames arrive through /push instead of a sample file
PUSH_MAX_BYTES = 2_000_000  # one camera JPEG (960 wide at q0.6 is ~40 KB); more is not a frame from our page
# The categories a live run can mask; must equal censor.live.LIVE_CATS (duplicated: that module is imported lazily)
LIVE_CATS = ("face", "license plate")
YUNET_PLATES_NOTE = "yunet: faces only, plates not detected"
ENGINE_DIED = "engine stopped unexpectedly; see the server log"

_lock = threading.Lock()
# phase: idle | warming | running. gen counts starts and stops, so a warm-up that was stopped meanwhile discards its
# engine. error: why the last start failed, shown by /stats once the phase is back to idle. detector_kind: what masks a
# camera run ("egoblur" | "yunet" | "none") or a stub-mode clip ("sidecar" | "yunet"), None for real-mode clips. cats: what
# this run masks. detector_note: a warning for the page (YuNet asked for plates), else None. last_poll: the last accepted /frame poll or /push.
_state: dict = {"phase": "idle", "replay": None, "last_poll": 0.0, "clip": None, "error": None, "gen": 0,
                "detector_kind": None, "cats": LIVE_CATS, "detector_note": None, "embedder": "none", "except_me": False}
_watchdog: threading.Thread | None = None
# EgoBlur detectors by live (height, width): warmed once, then shared by every start. A warm-up that outlives its
# stop must not make the next start load a second 420 MB copy, so creation happens under _det_lock.
_detectors: dict[tuple[int, int], object] = {}
_det_lock = threading.Lock()
# SFace embedders by ONNX path: loaded once, shared by every start (same reasoning as _detectors)
_embedders: dict[str, object] = {}
_emb_lock = threading.Lock()
NO_EMBEDDER = "no embedder: set CENSOR_SFACE"  # <face_recognition_sface_2021dec.onnx> from opencv_zoo
BUSY = "too many frames in flight; slow down"


class StartIn(BaseModel):
    clip: str
    height: Literal[540, 720] = 720  # the only sizes the page offers; bounds the per-height detector cache
    cats: list[str] = list(LIVE_CATS)
    except_me: bool = False  # advisory for the page: needs an embedder (CENSOR_SFACE), else 503

    @field_validator("cats")
    @classmethod
    def _valid_cats(cls, cats: list[str]) -> list[str]:
        if not cats:
            raise ValueError(f"cats must not be empty; choose from {list(LIVE_CATS)}")
        bad = sorted(set(cats) - set(LIVE_CATS))
        if bad:
            raise ValueError(f"unknown cats {bad}; choose from {list(LIVE_CATS)}")
        return [c for c in LIVE_CATS if c in cats]  # deduplicated, in LIVE_CATS order


def _even(x: float) -> int:
    """The nearest even pixel count (at least 2): frame widths must be even for the JPEG and h264 paths."""
    return max(2, int(round(x / 2)) * 2)


def _detector_for(size: tuple[int, int]):
    """The EgoBlur detector warmed for one live (height, width). Loaded once per size and shared by every start (clips
    and camera alike); only a warmed detector is kept."""
    from censor.vision.specialist import EgoBlurDetector

    with _det_lock:
        det = _detectors.get(size)
        if det is None:
            det = EgoBlurDetector()
            det.warm_up(sizes=(size,))
            _detectors[size] = det
    return det


def _embedder_kind() -> str:
    """"sface" when CENSOR_SFACE names a model (the file itself is checked when the run is built), else "none"."""
    return "sface" if os.environ.get("CENSOR_SFACE") else "none"


def _embedder():
    """The shared SFace embed function for CENSOR_SFACE, or None when it is unset. A missing file raises
    FileNotFoundError (from sface_embed), which fails the start into idle with that message; nothing is cached then."""
    path = os.environ.get("CENSOR_SFACE")
    if not path:
        return None
    from censor.live import sface_embed

    with _emb_lock:
        emb = _embedders.get(path)
        if emb is None:
            emb = sface_embed(path)
            _embedders[path] = emb
    return emb


def _busy_type() -> type[BaseException]:
    """censor.live.Busy (imported lazily like the rest of the engine), or a type nothing raises."""
    try:
        from censor.live import Busy
    except ImportError:
        class Busy(Exception):  # noqa: N818  never raised: the engine has no Busy yet
            pass
    return Busy


def _no_detections(f: int, frame) -> dict:
    """The camera detector when nothing is loaded (stub mode without CENSOR_YUNET): the stream passes unmasked."""
    return {"face": [], "license plate": []}


def _camera_detector_kind(stub: bool) -> str:
    """What masks a camera run; /start reports it before make_replay runs, so both read the same two switches."""
    if not stub:
        return "egoblur"
    return "yunet" if os.environ.get("CENSOR_YUNET") else "none"


def _stub_clip_detector_kind(sidecar: Path) -> str | None:
    """What masks a stub-mode clip: its sidecar when it exists (that wins), else YuNet when CENSOR_YUNET is set, else
    None (nothing can mask it). /start and make_replay both ask, so they never disagree."""
    if sidecar.exists():
        return "sidecar"
    return "yunet" if os.environ.get("CENSOR_YUNET") else None


def _make_camera_replay(height: int, stub: bool, cats: tuple[str, ...], embed=None) -> LiveReplay:
    """The push-mode engine for the browser's camera. Real mode warms EgoBlur at the 16:9 size the page sends; stub mode
    uses OpenCV's YuNet face detector when CENSOR_YUNET names the ONNX file, else detects nothing."""
    from censor.live import LiveReplay, egoblur_detect

    if not stub:
        det = _detector_for((height, _even(height * 16 / 9)))
        return LiveReplay(None, egoblur_detect(det.infer, cats=cats), height=height, push=True,
                          cats=cats, embed=embed)
    onnx = os.environ.get("CENSOR_YUNET")
    if onnx:
        from censor.live import yunet_detect

        return LiveReplay(None, yunet_detect(onnx, cats=cats), height=height, push=True, cats=cats,
                          embed=embed)
    return LiveReplay(None, _no_detections, height=height, push=True, cats=cats, embed=embed)


def make_replay(clip_path: str, height: int, stub: bool, cats: tuple[str, ...] = LIVE_CATS) -> LiveReplay:
    """Build the engine for one clip (tests replace this with a fake). Real mode loads EgoBlur and warms it at the
    live size before the clock starts; stub mode (no GPU) replays the clip's tracks.json sidecar if it has one, else
    runs YuNet from CENSOR_YUNET on every frame. `cats` (a LIVE_CATS subset) goes to the detector and the engine. `clip_path == "camera"` builds a push-mode engine instead (see
    _make_camera_replay)."""
    embed = _embedder()  # None without CENSOR_SFACE; a missing model file raises before any detector is loaded
    if clip_path == CAMERA:
        return _make_camera_replay(height, stub, cats, embed)
    from censor.core.video import probe
    from censor.live import LiveReplay, egoblur_detect, sidecar_detect

    if stub:
        sidecar = Path(f"{clip_path}.tracks.json")
        kind = _stub_clip_detector_kind(sidecar)
        if kind == "sidecar":
            v = probe(clip_path)
            detect = sidecar_detect(sidecar, src_h=v.height, live_h=height, n_frames=v.n_frames, cats=cats)
        elif kind == "yunet":
            from censor.live import yunet_detect

            detect = yunet_detect(os.environ["CENSOR_YUNET"], cats=cats)
        else:
            raise FileNotFoundError(_stub_needs(Path(clip_path).name))
        return LiveReplay(clip_path, detect, height=height, cats=cats, embed=embed)
    v = probe(clip_path)
    det = _detector_for((height, _even(v.width * height / v.height)))
    return LiveReplay(clip_path, egoblur_detect(det.infer, cats=cats), height=height, cats=cats, embed=embed)


def _stub_needs(clip_name: str) -> str:
    return f"stub mode needs {clip_name}.tracks.json next to the clip, or CENSOR_YUNET=<onnx> for faces"


def _stub_mode() -> bool:
    return os.environ.get("CENSOR_STUB") == "1"


def _idle_stats() -> dict:
    # "embedder" is here too, so the page can enable "except me" before the first Start
    out: dict = {"phase": "idle", "running": False, "embedder": _embedder_kind()}
    if _state["error"]:
        out["error"] = _state["error"]
    return out


def _stats_locked() -> dict:
    """Current stats; the caller holds _lock."""
    phase, replay = _state["phase"], _state["replay"]
    # Not "detector": the engine's stats already use that key for its pass timings
    kind = {"detector_kind": _state["detector_kind"]} if _state["detector_kind"] else {}
    note = {"detector_note": _state["detector_note"]} if _state["detector_note"] else {}
    cats = {"cats": list(_state["cats"]), "embedder": _state["embedder"], "except_me": _state["except_me"]}
    if phase == "warming":
        return {"phase": "warming", "running": False, "warming": True, "clip": _state["clip"], **cats, **kind, **note}
    if replay is None:
        return _idle_stats()
    return {**replay.stats(), "phase": phase, "warming": False, **cats, **kind, **note}


def _reap_if_dead_locked() -> LiveReplay | None:
    """The caller holds _lock. If the running engine has died (LiveReplay.running turned False without a stop from
    here), end the run: back to idle with an error. Returns the dead engine so the caller can stop() it (which joins
    its leftover threads) after releasing the lock; None when there was nothing to reap."""
    replay = _state["replay"]
    if _state["phase"] != "running" or replay is None or replay.running:
        return None
    _state.update(phase="idle", replay=None, gen=_state["gen"] + 1, error=ENGINE_DIED)
    print(f"[live] {ENGINE_DIED}", file=sys.stderr)
    return replay


def _warm_and_run(gen: int, clip_path: str, height: int, stub: bool, cats: tuple[str, ...]) -> None:
    replay = None
    current = False
    try:
        replay = make_replay(clip_path, height, stub, cats)
        with _lock:
            current = _state["gen"] == gen and _state["phase"] == "warming"
            if current:
                _state["replay"] = replay  # /stop can reach it from here on
        if current:  # stopped while warming up: never start an engine nobody wants
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
        if current and _state["gen"] == gen and _state["phase"] == "warming":
            _state.update(phase="running", last_poll=time.monotonic())
            return
    replay.stop()  # stopped while warming up or starting: nobody wants this engine any more


def _auto_stop_check() -> None:
    with _lock:
        replay = _reap_if_dead_locked()
        idle_viewer = (replay is None and _state["phase"] == "running"
                       and time.monotonic() - _state["last_poll"] > AUTO_STOP_S)
        if idle_viewer:
            replay = _state["replay"]
            _state.update(phase="idle", replay=None, gen=_state["gen"] + 1)
    if idle_viewer:
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
    camera = body.clip == CAMERA  # checked before the sample lookup: the camera is not a file
    src = SAMPLES_DIR / f"{body.clip}.mp4"
    if not camera and (not SAMPLE_NAME.fullmatch(body.clip) or not src.exists()):
        raise HTTPException(404, f"no sample clip {body.clip!r}")
    stub = _stub_mode()
    with _lock:
        dead = _reap_if_dead_locked()  # a dead engine must not turn this start into a 409
    if dead is not None:
        dead.stop()
    with _lock:
        if _state["phase"] != "idle":
            raise HTTPException(409, "a live replay is already running; stop it first")
        if not stub and not egoblur_available():
            raise HTTPException(503, f"EgoBlur models not found in {EGOBLUR_DIR}; live replay needs them")
        if body.except_me and _embedder_kind() == "none":
            raise HTTPException(503, NO_EMBEDDER)
        kind = None  # real-mode clips carry no detector_kind
        if camera:
            kind = _camera_detector_kind(stub)
        elif stub:
            kind = _stub_clip_detector_kind(Path(f"{src}.tracks.json"))
            if kind is None:
                raise HTTPException(400, _stub_needs(src.name))
        cats = tuple(body.cats)  # normalised by StartIn
        note = YUNET_PLATES_NOTE if stub and kind == "yunet" and "license plate" in cats else None
        gen = _state["gen"] + 1
        _state.update(phase="warming", replay=None, clip=body.clip, error=None, gen=gen, detector_kind=kind, cats=cats,
                      detector_note=note, embedder=_embedder_kind(),
                      except_me=body.except_me)
        stats = _stats_locked()
    _ensure_watchdog()
    threading.Thread(target=_warm_and_run, args=(gen, CAMERA if camera else str(src), body.height, stub, cats), daemon=True,
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


def _running_replay() -> LiveReplay:
    """The running engine, or a 409 (a dead engine is reaped on the way)."""
    with _lock:
        dead = _reap_if_dead_locked()
        replay = _state["replay"] if _state["phase"] == "running" else None
    if dead is not None:
        dead.stop()
    if replay is None:
        raise HTTPException(409, "no live replay is running")
    return replay


@router.post("/enroll")
def enroll():
    """Enroll the largest face in view as "me". Blocks up to ~2 s (a threadpool worker, not the event loop)."""
    replay = _running_replay()
    if replay.stats().get("identity", {}).get("embedder") != "sface":
        raise HTTPException(503, NO_EMBEDDER)
    return replay.enroll()


@router.post("/enroll/clear")
def enroll_clear():
    replay = _running_replay()
    replay.clear_enrolled()
    return {"ok": True}


@router.get("/stats")
def stats():
    with _lock:
        dead = _reap_if_dead_locked()
        out = _stats_locked()
    if dead is not None:
        dead.stop()
    return out


@router.get("/frame")
def frame(run: str = "", after: int = -1):
    with _lock:
        dead = _reap_if_dead_locked()
        replay = _state["replay"] if _state["phase"] == "running" else None
    if dead is not None:
        dead.stop()
    if replay is None:
        return Response(status_code=204)
    if run != replay.run:  # a client from before a restart, or none yet: show the newest frame now
        after = AHEAD
    res = replay.latest(after, timeout=FRAME_WAIT_S)
    if res is None:  # a poll that got nothing must not keep a stalled or dead stream "alive"
        return Response(status_code=204)
    with _lock:
        _state["last_poll"] = time.monotonic()
    f, jpeg = res
    return Response(content=jpeg, media_type="image/jpeg", headers={"X-Live-Frame": str(f), "X-Live-Run": replay.run})


def _push_target(run: str | None) -> LiveReplay:
    """The engine a /push may feed, or an HTTPException saying why not (a dead engine is reaped on the way)."""
    with _lock:
        dead = _reap_if_dead_locked()
        replay = _state["replay"] if _state["phase"] == "running" else None
    if dead is not None:
        dead.stop()
    if replay is None:
        raise HTTPException(409, "no live replay is running")
    if not replay.is_push:
        raise HTTPException(409, 'this replay plays a clip; start it with clip "camera" to push frames')
    if run != replay.run:  # a page from before a restart: its frames belong to another run
        raise HTTPException(409, "stale run id; reload the page")
    return replay


async def _read_capped(request: Request, limit: int) -> bytes:
    """The raw request body, or a 413 as soon as it is known to be over `limit` bytes: the declared Content-Length
    first, then the running total while streaming (a chunked body declares nothing), so an oversized upload is never
    buffered whole."""
    too_big = HTTPException(413, f"frame too large (limit {limit // 1_000_000} MB)")
    declared = request.headers.get("content-length", "")
    if declared:
        if not (declared.isascii() and declared.isdigit()):
            raise HTTPException(400, "bad Content-Length")
        if int(declared) > limit:
            raise too_big
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise too_big
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/push")
async def push(request: Request):
    """One camera frame: the raw JPEG as the body (any content-type: toBlob sends image/jpeg, curl may send nothing).
    Answers {"f": n}, the stream frame number the engine gave it. An accepted frame also keeps the run alive."""
    replay = await run_in_threadpool(_push_target, request.headers.get("x-live-run"))
    data = await _read_capped(request, PUSH_MAX_BYTES)
    try:
        f = await run_in_threadpool(replay.push, data)  # decode + resize, a few ms: off the event loop
    except ValueError as e:
        raise HTTPException(400, "not a decodable JPEG") from e
    except _busy_type() as e:  # too many decodes in flight: the page should back off
        raise HTTPException(429, BUSY) from e
    except RuntimeError as e:  # the engine stopped between the check above and now
        raise HTTPException(409, "no live replay is running") from e
    with _lock:
        if _state["replay"] is replay:
            _state["last_poll"] = time.monotonic()
    return {"f": f}


@router.get("/live", response_class=HTMLResponse)
def live_page():
    page = WEB_DIR / "live.html"
    if not page.exists():
        return "<p>The live replay page (web/live.html) is not in this checkout yet. The API is at /api/live/.</p>"
    return page.read_text(encoding="utf-8")
