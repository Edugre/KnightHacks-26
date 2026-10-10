"""Main API (role D's UI talks to this). Sessions persist under DATA_DIR/<video_id>/.

  POST /api/videos                      multipart file  -> {video_id, info}
  GET  /api/videos/{id}                 -> {info, summary, ops}
  GET  /api/videos/{id}/source          original video
  POST /api/videos/{id}/request         {"text": str}  -> agent result
  POST /api/videos/{id}/box             {"frame": int, "box": [x1,y1,x2,y2], "effect": "blur"} -> track + op
  POST /api/videos/{id}/undo            -> {undone, summary}
  GET  /api/videos/{id}/timeline        derived Timeline JSON (what the renderer uses)
  GET  /api/videos/{id}/tracks          TrackStore JSON
  GET  /api/videos/{id}/preview/{f}.jpg masked frame f
  POST /api/videos/{id}/export          render -> {url, stats}
  GET  /api/videos/{id}/export.mp4      rendered video

Run: uvicorn censor.server.app:app --host 0.0.0.0 --port 8000   (see run.sh)
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from censor import netguard
from censor.agent.agent import Agent
from censor.agent.tools import Session, ToolError
from censor.core.video import read_frame
from censor.render.render import FrameMasker, render_video

netguard.install_from_env()  # YOLOE runs in this process; refuse and log any outbound connection

DATA_DIR = Path(os.environ.get("CENSOR_DATA", "data")).resolve()
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
app = FastAPI(title="censor agent")
_sessions: dict[str, Session] = {}


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


class BoxIn(BaseModel):
    frame: int
    box: list[float]
    effect: str = "blur"


@app.get("/", response_class=HTMLResponse)
def index():
    page = WEB_DIR / "index.html"
    return page.read_text() if page.exists() else "<p>censor agent API. See /docs</p>"


@app.post("/api/videos")
async def upload(file: UploadFile = File(...)):
    vid = uuid.uuid4().hex[:8]
    d = DATA_DIR / vid
    d.mkdir(parents=True)
    src = d / f"source{Path(file.filename or 'x.mp4').suffix or '.mp4'}"
    with src.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    s = session(vid)
    return {"video_id": vid, "info": s.video_info()}


@app.post("/api/videos/{vid}/tracks-sidecar")
async def upload_sidecar(vid: str, file: UploadFile = File(...)):
    """Stub mode only: attach ground-truth tracks (from make_test_clip.py) so StubDetector can 'find' them."""
    s = session(vid)
    Path(s.video + ".tracks.json").write_bytes(await file.read())
    return {"ok": True}


@app.get("/api/videos/{vid}")
def get_video(vid: str):
    s = session(vid)
    return {"info": s.video_info(), "summary": s.summarize(), "ops": [o.model_dump() for o in s.log.ops]}


@app.get("/api/videos/{vid}/source")
def source(vid: str):
    return FileResponse(session(vid).video)


@app.post("/api/videos/{vid}/request")
def request(vid: str, body: RequestIn):
    return Agent(session(vid)).run(body.text)


@app.post("/api/videos/{vid}/box")
def box(vid: str, body: BoxIn):
    s = session(vid)
    try:
        t = s.track_box(body.frame, body.box)
        op = s.add_masks([t["track_id"]], effect=body.effect, request_id="ui_box")
    except ToolError as e:
        raise HTTPException(400, str(e))
    return {"track": t, "op_id": op, "summary": s.summarize()}


@app.post("/api/videos/{vid}/undo")
def undo(vid: str):
    s = session(vid)
    return {"undone": s.undo(), "summary": s.summarize()}


@app.get("/api/videos/{vid}/timeline")
def timeline(vid: str):
    return session(vid).timeline.model_dump()


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
    stats = render_video(s.video, s.timeline, str(out))
    return {"url": f"api/videos/{vid}/export.mp4", "stats": {"frames": stats.frames, "masked_boxes": stats.masked_boxes,
                                                               "failed_frames": len(stats.failed_frames)}}


@app.get("/api/videos/{vid}/export.mp4")
def export_file(vid: str):
    out = session(vid).workdir / "export.mp4"
    if not out.exists():
        raise HTTPException(404, "not exported yet")
    return FileResponse(out, media_type="video/mp4")
