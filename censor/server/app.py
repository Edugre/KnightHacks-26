"""Main API (role D's UI talks to this). Sessions persist under DATA_DIR/<video_id>/.

  POST /api/videos                      multipart file  -> {video_id, info, prescan}  (starts a background scan)
  GET  /api/samples                     clips in bench/clips -> [{id, mb}]
  POST /api/samples/{id}                load one of them like an upload (no file picker needed)
  GET  /api/videos/{id}                 -> {info, summary, ops, busy}  (busy: the request still running, if any)
  GET  /api/videos/{id}/source          original video
  POST /api/videos/{id}/request         {"text": str}  -> agent result
  POST /api/videos/{id}/box             {"frame": int, "box": [x1,y1,x2,y2], "effect": "blur"} -> track + op
  POST /api/videos/{id}/targets         {"action": "add"|"remove"|"effect", "track_ids": [...], "effect": "blur"}
                                        mask, unmask or restyle chosen tracks by hand -> {op_id, summary}
  POST /api/videos/{id}/undo            -> {undone, summary}
  GET  /api/videos/{id}/timeline        derived Timeline JSON (what the renderer uses)
  GET  /api/videos/{id}/tracks          TrackStore JSON
  GET  /api/videos/{id}/masks           per-frame masked rectangles, exactly as the renderer applies them (UI preview)
  GET  /api/videos/{id}/preview/{f}.jpg masked frame f
  POST /api/videos/{id}/export          render -> {url, stats}  (waits for a running request; 409 if it takes too long)
  GET  /api/videos/{id}/export.mp4      rendered video (the url from export carries ?v=<render time>)

Requests, boxes, undo and exports on one video take turns, so an export always includes every finished edit.
Nothing is cached: AUP's Cloudflare keeps .mp4 and .jpg responses for 4 hours when the origin says nothing, which
served an old, unmasked export after a new one had been rendered.

Run: uvicorn censor.server.app:app --host 0.0.0.0 --port 8000   (see run.sh)
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from censor import netguard
from censor.agent.agent import Agent
from censor.agent.tools import Session, ToolError
from censor.core.timeline import Timeline
from censor.core.video import read_frame
from censor.render.render import FrameMasker, _padded, render_video

netguard.install_from_env()  # YOLOE runs in this process; refuse and log any outbound connection

DATA_DIR = Path(os.environ.get("CENSOR_DATA", "data")).resolve()
# Off in stub mode: the stub replays a sidecar that's uploaded after the video, so a scan now would cache nothing.
PRESCAN = [] if os.environ.get("CENSOR_STUB") == "1" else \
    [q.strip() for q in os.environ.get("CENSOR_PRESCAN", "face,license plate,person").split(",") if q.strip()]
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
SAMPLES_DIR = Path(__file__).resolve().parents[2] / "bench" / "clips"
SAMPLE_NAME = re.compile(r"[\w-]+")  # a bare file stem: no paths
# How long an export waits for a running request before giving up with a 409. Under the 100 s that AUP's
# Cloudflare waits for a response, so the user gets a message rather than a timeout page.
EXPORT_WAIT = float(os.environ.get("CENSOR_EXPORT_WAIT", "80"))
app = FastAPI(title="censor agent")


def _warm_up() -> None:
    """Load the detectors and run them once in the background at startup, so the first upload scans at the
    warm speed (17 s instead of 29 s for a 13 s 1080p clip on the R9700). Off with CENSOR_WARMUP=0."""
    try:
        from censor.vision.detect import get_detector
        det = get_detector()
        if hasattr(det, "warm_up"):
            t = time.perf_counter()
            det.warm_up()
            print(f"[warm-up] detectors ready in {time.perf_counter() - t:.1f} s", file=sys.stderr)
    except Exception as e:  # noqa: BLE001  a failed warm-up only means a slower first scan
        print(f"[warm-up] skipped: {type(e).__name__}: {e}", file=sys.stderr)


if os.environ.get("CENSOR_STUB") != "1" and os.environ.get("CENSOR_WARMUP", "1") == "1":
    threading.Thread(target=_warm_up, daemon=True, name="warm-up").start()
_sessions: dict[str, Session] = {}
_turns: dict[str, threading.Lock] = {}  # one per video: requests, boxes, undo and exports run one at a time
_working: dict[str, str] = {}  # video -> what it's busy with, for GET /api/videos/{id} and the 409 message


@app.middleware("http")
async def no_store(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@contextmanager
def turn(vid: str, what: str, timeout: float = -1):
    lock = _turns.setdefault(vid, threading.Lock())
    if not lock.acquire(timeout=timeout):
        raise HTTPException(409, f"Still working on \"{_working.get(vid, 'another edit')}\". "
                                 "Export again when it finishes.")
    _working[vid] = what
    try:
        yield
    finally:
        _working.pop(vid, None)
        lock.release()


def session(vid: str) -> Session:
    if vid not in _sessions:
        d = DATA_DIR / vid
        src = next(d.glob("source.*"), None) if d.exists() else None
        if src is None:
            raise HTTPException(404, f"no video {vid}")
        _sessions[vid] = Session(src, d)
    return _sessions[vid]


class RequestIn(BaseModel):
    text: str


class TargetsIn(BaseModel):
    action: Literal["add", "remove", "effect"]
    track_ids: list[str]
    effect: str = "blur"


class BoxIn(BaseModel):
    frame: int
    box: list[float]
    effect: str = "blur"


@app.get("/", response_class=HTMLResponse)
def index():
    page = WEB_DIR / "index.html"
    return page.read_text(encoding="utf-8") if page.exists() else "<p>censor agent API. See /docs</p>"


@app.post("/api/videos")
async def upload(file: UploadFile = File(...)):
    vid, d = _new_video_dir()
    src = d / f"source{Path(file.filename or 'x.mp4').suffix or '.mp4'}"
    with src.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return _start(vid)


@app.get("/api/samples")
def samples():
    """Clips already on the server (bench/clips), so a demo needs no file picker."""
    return [{"id": p.stem, "mb": round(p.stat().st_size / 1e6, 1)} for p in sorted(SAMPLES_DIR.glob("*.mp4"))]


@app.post("/api/samples/{name}")
def load_sample(name: str):
    src = SAMPLES_DIR / f"{name}.mp4"
    if not SAMPLE_NAME.fullmatch(name) or not src.exists():
        raise HTTPException(404, f"no sample clip {name!r}")
    vid, d = _new_video_dir()
    shutil.copyfile(src, d / "source.mp4")
    sidecar = Path(f"{src}.tracks.json")
    if sidecar.exists():  # stub mode replays it, so the sample works without a GPU too
        shutil.copyfile(sidecar, d / "source.mp4.tracks.json")
    return _start(vid)


def _new_video_dir() -> tuple[str, Path]:
    vid = uuid.uuid4().hex[:8]
    d = DATA_DIR / vid
    d.mkdir(parents=True)
    return vid, d


def _start(vid: str) -> dict:
    s = session(vid)
    if PRESCAN:
        threading.Thread(target=prescan, args=(s, PRESCAN), daemon=True).start()
    return {"video_id": vid, "info": s.video_info(), "prescan": PRESCAN}


def prescan(s: Session, queries: list[str]) -> None:
    """Scan for the usual targets right after upload, so the first request reads cached tracks instead of
    waiting for full-video passes. A request for one of these while it runs waits on the session lock."""
    for q in queries:
        try:
            s.find_objects(q)
        except ToolError as e:
            print(f"[prescan] {q}: {e}", file=sys.stderr)


@app.post("/api/videos/{vid}/tracks-sidecar")
async def upload_sidecar(vid: str, file: UploadFile = File(...)):
    """Stub mode only: attach ground-truth tracks (from make_test_clip.py) so StubDetector can 'find' them."""
    s = session(vid)
    Path(s.video + ".tracks.json").write_bytes(await file.read())
    return {"ok": True}


@app.get("/api/videos/{vid}")
def get_video(vid: str):
    s = session(vid)
    return {"info": s.video_info(), "summary": s.summarize(), "ops": [o.model_dump() for o in s.log.ops],
            "busy": _working.get(vid)}


@app.get("/api/videos/{vid}/source")
def source(vid: str):
    return FileResponse(session(vid).video)


@app.post("/api/videos/{vid}/request")
def request(vid: str, body: RequestIn):
    s = session(vid)
    with turn(vid, body.text):
        return Agent(s).run(body.text)


@app.post("/api/videos/{vid}/box")
def box(vid: str, body: BoxIn):
    s = session(vid)
    try:
        with turn(vid, "a drawn box"):
            t = s.track_box(body.frame, body.box)
            op = s.add_masks([t["track_id"]], effect=body.effect, request_id="ui_box")
    except ToolError as e:
        raise HTTPException(400, str(e))
    return {"track": t, "op_id": op, "summary": s.summarize()}


@app.post("/api/videos/{vid}/targets")
def targets(vid: str, body: TargetsIn):
    """Hand edits from the UI on tracks the user picked: mask a detected track that wasn't (add), unmask one
    (remove), or change its effect. Same toolbox calls as the agent, so each is one op that Undo reverts."""
    s = session(vid)
    try:
        with turn(vid, f"a hand edit ({body.action})"):
            if body.action == "add":
                op = s.add_masks(body.track_ids, effect=body.effect, request_id="ui_edit")
            elif body.action == "remove":
                op = s.remove_masks(body.track_ids, request_id="ui_edit")
            else:
                op = s.set_effect(body.track_ids, body.effect, request_id="ui_edit")
    except ToolError as e:
        raise HTTPException(400, str(e))
    return {"op_id": op, "summary": s.summarize()}


@app.post("/api/videos/{vid}/undo")
def undo(vid: str):
    s = session(vid)
    with turn(vid, "undo"):
        return {"undone": s.undo(), "summary": s.summarize()}


@app.get("/api/videos/{vid}/timeline")
def timeline(vid: str):
    return session(vid).timeline.model_dump()


@app.get("/api/videos/{vid}/masks")
def masks(vid: str):
    """Every masked rectangle per frame, from the renderer's own FrameMasker (interpolation, lead-in, hold after
    loss, padding), so the UI's live preview shows exactly where the export will blur. Integer pixels, grouped
    by target: {track_id: {effect, frames: [[f, x1, y1, x2, y2], ...]}}."""
    tl = session(vid).timeline
    v = tl.video
    out: dict[str, dict] = {}
    for tid, target in tl.targets.items():
        one = Timeline(video=v, targets={tid: target})
        frames = [[f, *_padded(box, pad, v.width, v.height)]
                  for f, items in sorted(FrameMasker(one).per_frame.items()) for box, pad, _ in items]
        out[tid] = {"effect": target.effect, "frames": frames}
    return {"fps": v.fps, "n_frames": v.n_frames, "width": v.width, "height": v.height, "targets": out}


@app.get("/api/videos/{vid}/tracks")
def tracks(vid: str):
    return session(vid).tracks.model_dump()


@app.get("/api/videos/{vid}/preview/{f}.jpg")
def preview(vid: str, f: int):
    s = session(vid)
    frame = read_frame(s.video, f)
    FrameMasker(s.timeline).apply(frame, f)
    return Response(cv2.imencode(".jpg", frame)[1].tobytes(), media_type="image/jpeg")


@app.post("/api/videos/{vid}/export")
def export(vid: str):
    s = session(vid)
    out = s.workdir / "export.mp4"
    with turn(vid, "an export", timeout=EXPORT_WAIT):
        stats = render_video(s.video, s.timeline, str(out))
    # A new URL per render, so no cache between here and the browser can hand back an older file
    return {"url": f"api/videos/{vid}/export.mp4?v={out.stat().st_mtime_ns}",
            "stats": {"frames": stats.frames, "masked_boxes": stats.masked_boxes,
                      "failed_frames": len(stats.failed_frames)}}


@app.get("/api/videos/{vid}/export.mp4")
def export_file(vid: str):
    out = session(vid).workdir / "export.mp4"
    if not out.exists():
        raise HTTPException(404, "not exported yet")
    return FileResponse(out, media_type="video/mp4")
