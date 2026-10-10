"""Labeling tool for benchmark clips: draw boxes on every Nth frame and save bench/labels/<clip>.json.

  python -m bench.label_server bench/clips/street_01.mp4 [--labels PATH] [--every 10] [--port 8020]

Then open http://localhost:8020 (on AUP: https://dub.aupcloud.io/user/<you>/proxy/8020/). Runs on a laptop,
no GPU. Detector drafts from `python -m bench.gt prelabel` show dashed until you confirm the frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import ValidationError

from bench.gt import CATEGORIES, LABELS_DIR, LabelSet
from censor.core.video import probe, read_frame

PAGE = Path(__file__).resolve().parent / "label.html"


def create_app(video: Path, labels_path: Path, every: int = 10) -> FastAPI:
    info = probe(str(video))
    app = FastAPI(title="bench labeler")

    def current() -> LabelSet:
        if labels_path.exists():
            return LabelSet.load(labels_path)
        return LabelSet(clip=labels_path.stem, video=info, every=every, objects={}, frames=[])

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE.read_text(encoding="utf-8")

    @app.get("/api/labels")
    def get_labels():
        ls = current()
        return {"labels": ls.model_dump(), "categories": CATEGORIES, "file": str(labels_path),
                "sampled": list(range(0, ls.video.n_frames, ls.every))}

    @app.get("/api/frame/{f}.jpg")
    def frame(f: int):
        if not 0 <= f < info.n_frames:
            raise HTTPException(404, f"frame {f} is outside 0..{info.n_frames - 1}")
        return Response(cv2.imencode(".jpg", read_frame(str(video), f), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(),
                        media_type="image/jpeg")

    @app.put("/api/labels")
    def save(body: dict):
        try:
            ls = LabelSet.model_validate(body)
        except ValidationError as e:
            raise HTTPException(400, str(e))
        if (ls.video.width, ls.video.height) != (info.width, info.height):
            raise HTTPException(400, f"labels are {ls.video.width}x{ls.video.height} but the clip is {info.width}x{info.height}")
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        ls.save(labels_path)
        return {"ok": True, "frames": len(ls.frames), "drafts": ls.n_drafts}

    return app


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser(description="Label a benchmark clip in the browser.")
    ap.add_argument("video")
    ap.add_argument("--labels", help="default: bench/labels/<video stem>.json")
    ap.add_argument("--every", type=int, default=10, help="label every Nth frame (new label files only)")
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    video = Path(a.video)
    labels = Path(a.labels) if a.labels else LABELS_DIR / f"{video.stem}.json"
    print(f"labeling {video} -> {labels}: open http://localhost:{a.port}")
    uvicorn.run(create_app(video, labels, a.every), host=a.host, port=a.port)
