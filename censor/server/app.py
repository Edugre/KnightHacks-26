"""Main API (role D's UI talks to this). Sessions persist under DATA_DIR/<video_id>/.

  POST /api/videos                      multipart file (a video or a photo), optional form field folder=<id>
                                        -> {video_id, info, prescan}  (starts a background scan). A photo is a one-frame
                                        video: info.kind is "image". With folder, the clip is filed in that folder.
  GET  /api/folders                     the user's folders, most recently used first -> [{id, name, clipCount, lastUsed}]
  POST /api/folders                     {"name": str} -> a new, empty folder
  GET  /api/folders/{id}                one folder and its clips, newest first -> {id, name, clips: [{video_id, name,
                                        kind, added}]}  (counts as using it)
  PATCH  /api/folders/{id}              {"name": str} -> renamed
  DELETE /api/folders/{id}              removes the folder only; its clips stay on this machine, unfiled
  GET  /api/config                      -> {offline, stub}: what a separately hosted page can't get injected into it
  GET  /api/samples[?kind=image]        clips in bench/clips, or photos in bench/photos -> [{id, mb, ...}]
  GET  /api/samples/{id}/photo          a sample photo, for the home screen's thumbnails
  POST /api/samples/{id}[?kind=image]   load one of them like an upload (no file picker needed)
  GET  /api/videos/{id}                 -> {info, summary, ops, busy}  (busy: the request still running, if any)
  GET  /api/videos/{id}/source          original video
  POST /api/videos/{id}/request         {"text": str, "ref"?: {"track_id": str} | {"frame": int, "box": [x1,y1,x2,y2]}}
                                        -> agent result; ref is what "this person" means ("blur everyone except this person")
  POST /api/videos/{id}/box             {"frame": int, "box": [x1,y1,x2,y2], "effect": "blur"} -> track + op
  POST /api/videos/{id}/targets         {"action": "add"|"remove"|"effect", "track_ids": [...], "effect": "blur"}
                                        mask, unmask or restyle chosen tracks by hand -> {op_id, summary};
                                        {"action": "move", "track_ids": [one], "frame": f, "box": [x1,y1,x2,y2]}
                                        re-tracks from the corrected box from frame f on -> {op_ids, track, summary}
  POST /api/videos/{id}/undo            -> {undone, summary}
  POST /api/videos/{id}/redo            put back the op the latest undo took -> {redone, summary}  (summary.n_redo: how many can)
  GET  /api/videos/{id}/timeline        derived Timeline JSON (what the renderer uses)
  GET  /api/videos/{id}/tracks          TrackStore JSON
  GET  /api/videos/{id}/masks           per-frame masked rectangles, exactly as the renderer applies them (UI preview)
  GET  /api/videos/{id}/preview/{f}.jpg masked frame f
  POST /api/videos/{id}/export          render -> {url, kind, stats}  (waits for a running request; 409 if it takes too long)
  GET  /api/videos/{id}/export.<ext>    rendered video (.mp4) or photo (the source's format, no metadata); the url
                                        from export carries ?v=<render time>; &download=1 sends it as an attachment

Requests, boxes, undo and exports on one video take turns, so an export always includes every finished edit.
Nothing is cached: AUP's Cloudflare keeps .mp4 and .jpg responses for 4 hours when the origin says nothing, which
served an old, unmasked export after a new one had been rendered.

The page can also be hosted on its own (web/index.html, <meta name="censor-api">): set CENSOR_CORS_ORIGINS to its
origin(s), comma-separated, so the browser lets it call these routes and draw the videos on its canvases.

Run: uvicorn censor.server.app:app --host 0.0.0.0 --port 8000   (see run.sh)
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Optional

import cv2
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from censor import netguard
from censor.agent.agent import Agent
from censor.agent.tools import Session, ToolError
from censor.core.timeline import Timeline
from censor.core.video import IMAGE_EXT, is_image, probe, read_frame
from censor.render.render import FrameMasker, _padded, render_image, render_video

netguard.install_from_env()  # YOLOE runs in this process; refuse and log any outbound connection

DATA_DIR = Path(os.environ.get("CENSOR_DATA", "data")).resolve()
# Off in stub mode: the stub replays a sidecar that's uploaded after the video, so a scan now would cache nothing.
PRESCAN = [] if os.environ.get("CENSOR_STUB") == "1" else \
    [q.strip() for q in os.environ.get("CENSOR_PRESCAN", "face,license plate,person").split(",") if q.strip()]
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
SAMPLES_DIR = Path(__file__).resolve().parents[2] / "bench" / "clips"
PHOTOS_DIR = Path(__file__).resolve().parents[2] / "bench" / "photos"
# Photos cv2 can't read: say what to do instead of failing on a file the server can't open
UNSUPPORTED_PHOTO = {".heic": "HEIC", ".heif": "HEIF", ".avif": "AVIF", ".tif": "TIFF", ".tiff": "TIFF", ".gif": "GIF"}
SAMPLE_NAME = re.compile(r"[\w-]+")  # a bare file stem: no paths
# How long an export waits for a running request before giving up with a 409. Under the 100 s that AUP's
# Cloudflare waits for a response, so the user gets a message rather than a timeout page.
EXPORT_WAIT = float(os.environ.get("CENSOR_EXPORT_WAIT", "80"))
app = FastAPI(title="censor agent")
CORS_ORIGINS = [o.strip().rstrip("/") for o in os.environ.get("CENSOR_CORS_ORIGINS", "").split(",") if o.strip()]
if CORS_ORIGINS:  # no cookies or auth: the API has none, and "*" works as an origin for a quick test
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


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
        # The video, not its stub-mode sidecar (source.mp4.tracks.json also matches source.*, and on Linux
        # the glob can list it first, which made loading the synthetic sample fail with "cannot open video")
        src = next((p for p in sorted(d.glob("source.*")) if not p.name.endswith(".json")), None) if d.exists() else None
        if src is None:
            raise HTTPException(404, f"no video {vid}")
        _sessions[vid] = Session(src, d)
    return _sessions[vid]


class RefIn(BaseModel):
    track_id: Optional[str] = None  # a target the user picked
    frame: Optional[int] = None     # or a box they drew on this frame
    box: Optional[list[float]] = None


class RequestIn(BaseModel):
    text: str
    ref: Optional[RefIn] = None


class TargetsIn(BaseModel):
    action: Literal["add", "remove", "effect", "move"]
    track_ids: list[str]
    effect: str = "blur"
    frame: Optional[int] = None  # move only
    box: Optional[list[float]] = None  # move only: the corrected [x1, y1, x2, y2] at that frame


class BoxIn(BaseModel):
    frame: int
    box: list[float]
    effect: str = "blur"


@app.get("/", response_class=HTMLResponse)
def index():
    page = WEB_DIR / "index.html"
    if not page.exists():
        return "<p>censor agent API. See /docs</p>"
    html = page.read_text(encoding="utf-8")
    # The header says "network off" only when netguard is actually refusing outbound connections
    return html.replace("<body>", '<body data-offline="1">', 1) if netguard.active() else html


@app.get("/api/config")
def config():
    """The flags index() puts on <body>, for a copy of the page hosted elsewhere."""
    return {"offline": netguard.active(), "stub": os.environ.get("CENSOR_STUB") == "1"}


@app.post("/api/videos")
async def upload(file: UploadFile = File(...), folder: Optional[str] = Form(None)):
    suffix = Path(file.filename or "x.mp4").suffix.lower() or ".mp4"
    if suffix in UNSUPPORTED_PHOTO:
        raise HTTPException(415, f"{UNSUPPORTED_PHOTO[suffix]} photos aren't supported yet. Save it as JPEG or PNG and "
                                 "upload that.")
    if folder:
        _folder(_load_folders(), folder)  # 404 before saving anything
    vid, d = _new_video_dir()
    src = d / f"source{suffix}"
    with src.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        out = _start(vid)
    except IOError as e:  # not a video or photo OpenCV can read: don't keep it around as a broken session
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, f"Can't open {file.filename}: {e}")
    if folder:
        with _folders_lock:
            fs = _load_folders()
            f = _folder(fs, folder)
            f["clips"].append({"video_id": vid, "name": Path(file.filename or f"clip{suffix}").name,
                               "kind": out["info"].get("kind", "video"), "added": time.time()})
            f["last_used"] = time.time()
            _save_folders(fs)
    return out


# ---- folders: the user's clips, filed by name. One small JSON file next to the sessions (DATA_DIR/folders.json);
# a folder lists the video ids filed in it, and a clip whose session is gone is left out when listed.
_folders_lock = threading.Lock()
FOLDER_NAME_MAX = 60


class FolderIn(BaseModel):
    name: str


def _folders_path() -> Path:
    return DATA_DIR / "folders.json"


def _load_folders() -> list[dict]:
    try:
        return json.loads(_folders_path().read_text())
    except FileNotFoundError:
        return []


def _save_folders(fs: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _folders_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(fs, indent=1))
    tmp.replace(_folders_path())  # atomic: a crash mid-write never leaves half a file


def _folder(fs: list[dict], fid: str) -> dict:
    f = next((f for f in fs if f["id"] == fid), None)
    if f is None:
        raise HTTPException(404, f"no folder {fid!r}")
    return f


def _live_clips(f: dict) -> list[dict]:
    return [c for c in f["clips"] if (DATA_DIR / c["video_id"]).is_dir()]


@app.get("/api/folders")
def list_folders():
    fs = sorted(_load_folders(), key=lambda f: f["last_used"], reverse=True)
    return [{"id": f["id"], "name": f["name"], "clipCount": len(_live_clips(f)), "lastUsed": f["last_used"]} for f in fs]


def _folder_name(raw: str) -> str:
    name = " ".join(raw.split())
    if not name:
        raise HTTPException(400, "A folder needs a name.")
    if len(name) > FOLDER_NAME_MAX:
        raise HTTPException(400, f"Folder names are at most {FOLDER_NAME_MAX} characters.")
    return name


@app.post("/api/folders")
def new_folder(body: FolderIn):
    name = _folder_name(body.name)
    now = time.time()
    f = {"id": uuid.uuid4().hex[:8], "name": name, "created": now, "last_used": now, "clips": []}
    with _folders_lock:
        fs = _load_folders()
        fs.append(f)
        _save_folders(fs)
    return {"id": f["id"], "name": name, "clipCount": 0, "lastUsed": now}


@app.get("/api/folders/{fid}")
def get_folder(fid: str):
    with _folders_lock:
        fs = _load_folders()
        f = _folder(fs, fid)
        f["last_used"] = time.time()
        _save_folders(fs)
    clips = sorted(_live_clips(f), key=lambda c: c["added"], reverse=True)
    return {"id": f["id"], "name": f["name"], "clips": clips}


@app.patch("/api/folders/{fid}")
def rename_folder(fid: str, body: FolderIn):
    name = _folder_name(body.name)
    with _folders_lock:
        fs = _load_folders()
        f = _folder(fs, fid)
        f["name"] = name
        _save_folders(fs)
    return {"id": fid, "name": name, "clipCount": len(_live_clips(f)), "lastUsed": f["last_used"]}


@app.delete("/api/folders/{fid}")
def delete_folder(fid: str):
    """Only the folder goes: the clips filed in it are sessions of their own and stay (reopen them with ?video=)."""
    with _folders_lock:
        fs = _load_folders()
        f = _folder(fs, fid)
        fs.remove(f)
        _save_folders(fs)
    return {"deleted": fid, "clips_kept": len(_live_clips(f))}


@app.get("/api/samples")
def samples(kind: str = Query("video", pattern="^(video|image)$")):
    """Clips already on the server (bench/clips), so a demo needs no file picker. Length and size for the home
    screen's list, and whether a stub-mode sidecar (ground-truth tracks) comes along. kind=image: the photos in
    bench/photos, with their size and a thumbnail URL instead of a length."""
    hidden = _synthetic_clips()
    if kind == "image":
        out = []
        for p in _sample_photos():
            if p.stem in hidden:
                continue
            c = {"id": p.stem, "mb": round(p.stat().st_size / 1e6, 1), "sidecar": Path(f"{p}.tracks.json").exists(),
                 "url": f"api/samples/{p.stem}/photo"}
            try:
                v = probe(str(p))
                c.update(width=v.width, height=v.height)
            except IOError:
                pass
            out.append(c)
        return out
    out = []
    for p in sorted(SAMPLES_DIR.glob("*.mp4")):
        if p.stem in hidden:
            continue
        c = {"id": p.stem, "mb": round(p.stat().st_size / 1e6, 1), "sidecar": Path(f"{p}.tracks.json").exists()}
        try:
            v = probe(str(p))
            c.update(duration=round(v.n_frames / v.fps, 1), width=v.width, height=v.height)
        except IOError:  # unreadable: still listed, by size only
            pass
        out.append(c)
    return out


def _synthetic_clips() -> set[str]:
    """Test clips (synthetic: true in the bench manifest) stay off the home screen, and so does the photo
    make_test_clip.py writes under the same name (bench/photos/synthetic.png). The bench and stub mode still use
    them, and they still load by id."""
    try:
        clips = yaml.safe_load((SAMPLES_DIR / "manifest.yaml").read_text()).get("clips") or []
    except (OSError, AttributeError, yaml.YAMLError):
        return set()
    return {c["id"] for c in clips if isinstance(c, dict) and c.get("synthetic") and "id" in c}


def _sample_photos() -> list[Path]:
    return sorted(p for p in PHOTOS_DIR.glob("*") if p.suffix.lower() in IMAGE_EXT) if PHOTOS_DIR.is_dir() else []


def _sample_photo(name: str) -> Path:
    src = next((p for p in _sample_photos() if p.stem == name), None) if SAMPLE_NAME.fullmatch(name) else None
    if src is None:
        raise HTTPException(404, f"no sample photo {name!r}")
    return src


@app.get("/api/samples/{name}/photo")
def sample_photo(name: str):
    return FileResponse(_sample_photo(name))


@app.post("/api/samples/{name}")
def load_sample(name: str, kind: str = Query("video", pattern="^(video|image)$")):
    if kind == "image":
        src = _sample_photo(name)
    else:
        src = SAMPLES_DIR / f"{name}.mp4"
        if not SAMPLE_NAME.fullmatch(name) or not src.exists():
            raise HTTPException(404, f"no sample clip {name!r}")
    vid, d = _new_video_dir()
    dst = d / f"source{src.suffix.lower()}"
    shutil.copyfile(src, dst)
    sidecar = Path(f"{src}.tracks.json")
    if sidecar.exists():  # stub mode replays it, so the sample works without a GPU too
        shutil.copyfile(sidecar, f"{dst}.tracks.json")
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
    t0 = time.perf_counter()
    with turn(vid, body.text):
        waited = time.perf_counter() - t0
        out = Agent(s).run(body.text, ref=body.ref.model_dump(exclude_none=True) if body.ref else None)
    print(_request_log(vid, body.text, waited, time.perf_counter() - t0, out), file=sys.stderr)
    return out


def _request_log(vid: str, text: str, waited: float, total: float, out: dict) -> str:
    """One line per request with where its time went, so a slow one (a first request once took 59 s, and the
    AUP proxy gives up at ~100 s) shows in the log whether it waited for another edit, a scan or the models.
    A find_objects that waited for the upload scan shows as a slow find_objects, with a [scan] line above."""
    steps = ", ".join(f"{c['tool']} {c['ms'] / 1000:.1f} s" + (" (error)" if "error" in c else "")
                      for c in out.get("calls", []) if "ms" in c)
    return (f"[request] {vid} {text!r}: {total:.1f} s, {out.get('path')}; waited {waited:.1f} s for the "
            f"previous edit; {steps or 'no steps'}")


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
            if body.action == "move":
                if len(body.track_ids) != 1 or body.frame is None or body.box is None:
                    raise ToolError("move takes one track id, a frame and a box")
                moved = s.move_box(body.track_ids[0], body.frame, body.box, request_id=f"ui_move_{uuid.uuid4().hex[:6]}")
                return {**moved, "summary": s.summarize()}
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


@app.post("/api/videos/{vid}/redo")
def redo(vid: str):
    s = session(vid)
    with turn(vid, "redo"):
        return {"redone": s.redo(), "summary": s.summarize()}


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
    try:
        frame = read_frame(s.video, f)
    except IndexError as e:
        raise HTTPException(404, str(e))
    FrameMasker(s.timeline).apply(frame, f)
    return Response(cv2.imencode(".jpg", frame)[1].tobytes(), media_type="image/jpeg")


@app.post("/api/videos/{vid}/export")
def export(vid: str):
    s = session(vid)
    photo = is_image(s.video)
    out = s.workdir / f"export{Path(s.video).suffix.lower() if photo else '.mp4'}"
    with turn(vid, "an export", timeout=EXPORT_WAIT):
        stats = (render_image if photo else render_video)(s.video, s.timeline, str(out))
    # A new URL per render, so no cache between here and the browser can hand back an older file
    return {"url": f"api/videos/{vid}/{out.name}?v={out.stat().st_mtime_ns}", "kind": s.info.kind,
            "stats": {"frames": stats.frames, "masked_boxes": stats.masked_boxes,
                      "failed_frames": len(stats.failed_frames)}}


@app.get("/api/videos/{vid}/export.{ext}")
def export_file(vid: str, ext: str, download: bool = False):
    if f".{ext}" not in IMAGE_EXT | {".mp4"}:
        raise HTTPException(404, f"no export.{ext}")
    out = session(vid).workdir / f"export.{ext}"
    if not out.exists():
        raise HTTPException(404, "not exported yet")
    # download: the Download button's link. A cross-origin page's <a download> is ignored, so the header asks instead
    return FileResponse(out, media_type=mimetypes.guess_type(out.name)[0] or "application/octet-stream",
                        filename=out.name if download else None)
