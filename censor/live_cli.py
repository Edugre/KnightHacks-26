"""Headless live replay: run the live engine in real time, record what a viewer would see, score it.

    python -m censor.live_cli run <clip> [--seconds 30] [--height 720] [--delay 1.0] --out X.mp4 --report X.json [--stub]
    python -m censor.live_cli score X.json bench/labels/<clip>.json [--min-face 0.92] [--min-plate 0.95]

`run` plays the clip as if it were a live feed (censor.live.LiveReplay), writes the emitted frames to an MP4 with a
REPLAY / age / blind overlay, and dumps a JSON report of the rects masked on every emitted frame. `--stub`
replays <clip>.tracks.json instead of running EgoBlur (no GPU).

`score` is the go/no-go gate. A labeled face or plate box is covered when at least COVER_MIN of its area lies
inside the rects masked on that frame; frames the engine declared blind (it froze the last vouched frame) are
exempt, and labeled frames that were never emitted are skipped. The gate is recall per category, not "zero bare
boxes" (plan D6): live has no recheck_tiny / absorb passes, so zero bare boxes is out of reach by construction.

Exit codes: 0 ok, 1 score below a threshold, 2 usage or environment problem (missing file, models, ffmpeg).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from censor.core.timeline import VideoInfo
from censor.core.video import probe
from censor.render.render import FFMPEG
from censor.vision.specialist import egoblur_available

CATS = ("face", "license plate")
MIN_RECALL = {"face": 0.92, "license plate": 0.95}  # offline recall on the bench clips minus 0.03
POLL_S = 2.0          # latest() timeout
MAX_IDLE_POLLS = 5    # consecutive empty polls before the run is declared stalled
RECORD_WAIT_S = 2.0   # how long the video loop waits for on_emit to have recorded a frame it already holds
STATS_EVERY = 10      # refresh the age shown in the overlay every n frames (stats() sorts samples)
Rect = tuple[int, int, int, int]


def die(msg: str, code: int = 2) -> "SystemExit":
    print(msg, file=sys.stderr)
    return SystemExit(code)


# ---- score ----------------------------------------------------------------------------------------

def _label_index(labels, live_h: int) -> tuple[dict[int, list[tuple[str, str, Rect]]], dict[str, str]]:
    """clip_f -> [(gt_id, category, label box scaled to the live size and clipped)] for reviewed face/plate boxes."""
    from bench.run_bench import _clip_box
    scale = live_h / labels.video.height
    live_w = int(round(labels.video.width * scale))
    cats = {gid: o.category for gid, o in labels.objects.items() if o.category in CATS}
    index: dict[int, list[tuple[str, str, Rect]]] = {}
    for fr in labels.frames_in():
        for b in fr.boxes:
            rect = _clip_box([v * scale for v in b.box], live_w, live_h) if b.gt_id in cats else None
            if rect is not None:
                index.setdefault(fr.f, []).append((b.gt_id, cats[b.gt_id], rect))
    return index, cats


def score_report(report: dict, labels, live_h: int) -> dict:
    """Per category {boxes, covered, recall (None without boxes), failing: {gt_id: first bare frame},
    longest_run_s: {gt_id: seconds}} plus blind_share and frames. See the module docstring for the rules."""
    from bench.run_bench import COVER_MIN, coverage
    index, cats = _label_index(labels, live_h)
    fps, step = float(report["fps"]), labels.every
    out: dict = {c: {"boxes": 0, "covered": 0, "recall": None, "failing": {}, "longest_run_s": {}} for c in CATS}
    run_start: dict[str, Optional[int]] = {}  # gt_id -> stream f where the current bare run began
    for fr in report["frames"]:
        for gid, cat, rect in index.get(fr["clip_f"], []):
            res = out[cat]
            res["boxes"] += 1
            hit = fr["blind"] or coverage(rect, [tuple(r) for r in fr["rects"]]) >= COVER_MIN
            res["covered"] += hit
            if hit:
                run_start[gid] = None
                continue
            res["failing"].setdefault(gid, {"f": fr["f"], "clip_f": fr["clip_f"]})
            run_start[gid] = fr["f"] if run_start.get(gid) is None else run_start[gid]
            secs = (fr["f"] - run_start[gid] + step) / fps
            res["longest_run_s"][gid] = max(res["longest_run_s"].get(gid, 0.0), secs)
    for res in out.values():
        res["recall"] = res["covered"] / res["boxes"] if res["boxes"] else None
    n = len(report["frames"])
    out["blind_share"] = sum(fr["blind"] for fr in report["frames"]) / n if n else 0.0
    out["frames"] = n
    return out


def _print_score(res: dict) -> None:
    print(f"{res['frames']} frames emitted, blind share {res['blind_share']:.3f}")
    for cat in CATS:
        r = res[cat]
        recall = "n/a" if r["recall"] is None else f"{r['recall']:.3f}"
        print(f"{cat}: boxes {r['boxes']}, covered {r['covered']}, recall {recall}")
        for gid, first in r["failing"].items():
            print(f"  {gid}: first bare at clip frame {first['clip_f']} (stream frame {first['f']}), "
                  f"longest bare run {r['longest_run_s'][gid]:.2f} s")


def cmd_score(a: argparse.Namespace) -> int:
    from bench.gt import LabelSet
    try:
        report = json.loads(Path(a.report).read_text(encoding="utf-8"))
        labels = LabelSet.load(a.labels)
    except (OSError, ValueError) as e:  # ValueError covers bad JSON and pydantic validation
        raise die(f"cannot read inputs: {e}")
    res = score_report(report, labels, report["size"][1])
    _print_score(res)
    floors = {"face": a.min_face, "license plate": a.min_plate}
    low = [c for c in CATS if res[c]["recall"] is not None and res[c]["recall"] < floors[c]]
    for c in low:
        print(f"FAIL: {c} recall {res[c]['recall']:.3f} < {floors[c]:.2f}")
    return 1 if low else 0


# ---- run ------------------------------------------------------------------------------------------

class Recorder:
    """Collects the engine's on_emit calls; thread-safe, because the emitter thread calls it while the video
    loop reads it."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._by_f: dict[int, dict] = {}
        self.frames: list[dict] = []

    def __call__(self, f: int, clip_f: int, blind: bool, rects: list) -> None:
        rec = {"f": f, "clip_f": clip_f, "blind": bool(blind), "rects": [list(map(int, r)) for r in rects]}
        with self._cv:
            self._by_f[f] = rec
            self.frames.append(rec)
            self._cv.notify_all()

    def get(self, f: int, timeout: float) -> Optional[dict]:
        with self._cv:
            self._cv.wait_for(lambda: f in self._by_f, timeout)
            return self._by_f.get(f)

    def snapshot(self) -> list[dict]:
        with self._cv:
            return list(self.frames)


def _live_size(clip: str, height: int) -> tuple[int, int]:
    info = probe(clip)
    return int(round(info.width * height / info.height / 2)) * 2, height


def _build_detect(a: argparse.Namespace, src: VideoInfo, size: tuple[int, int]):
    from censor.live import egoblur_detect, sidecar_detect
    w, h = size
    if a.stub:
        return sidecar_detect(Path(f"{a.clip}.tracks.json"), src_h=src.height, live_h=h, n_frames=src.n_frames)
    from censor.vision.specialist import EgoBlurDetector
    det = EgoBlurDetector()
    det.warm_up(sizes=((h, w),))  # before the clock starts, or the first seconds are all blind
    return egoblur_detect(det.infer)


def _overlay(img: np.ndarray, age_ms, blind_n: int, frames_out: int, blind: bool) -> None:
    txt = f"REPLAY · age {age_ms} ms · blind {blind_n}/{frames_out}" + (" · DETECTOR LATE" if blind else "")
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (0, 0), (tw + 24, 40), (20, 20, 20), -1)
    cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def _decode(jpeg: bytes, size: tuple[int, int]) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("the engine returned an undecodable frame")
    # latest() serves the display size; this is a recording of what the viewer sees, scaled back up to the live size.
    return img if (img.shape[1], img.shape[0]) == size else cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def _ffmpeg_cmd(size: tuple[int, int], fps: float, out: str) -> list[str]:
    return [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
            "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", out]


def _record(replay, rec: Recorder, proc: subprocess.Popen, size: tuple[int, int], seconds: float) -> None:
    """Pull emitted frames until `seconds` of stream time have passed; one video frame per stream frame."""
    after, last, idle, blind_n, out_n, age = -1, None, 0, 0, 0, "?"
    while True:
        got = replay.latest(after, POLL_S)
        if got is None:
            idle += 1
            if idle >= MAX_IDLE_POLLS:
                raise RuntimeError(f"no frame for {idle * POLL_S:.0f} s; the engine stalled")
            continue
        idle, (f, jpeg) = 0, got
        if f / replay.fps >= seconds:
            return
        meta = rec.get(f, RECORD_WAIT_S)
        if meta is None:
            raise RuntimeError(f"frame {f} was served but never reported through on_emit")
        for _ in range(f - after - 1 if last is not None else 0):  # slots the engine dropped: repeat the last frame
            proc.stdin.write(last.tobytes())
            out_n += 1
        blind_n += meta["blind"]
        out_n += 1
        if out_n % STATS_EVERY == 1:
            p50 = replay.stats()["frame_age_ms"]["p50"]
            age = "?" if p50 is None else int(round(p50))
        last = _decode(jpeg, size)
        _overlay(last, age, blind_n, out_n, meta["blind"])
        proc.stdin.write(last.tobytes())
        after = f


def cmd_run(a: argparse.Namespace) -> int:
    if not Path(a.clip).is_file():
        raise die(f"clip not found: {a.clip}")
    if not a.stub and not egoblur_available():
        raise die("EgoBlur models not found; live replay needs them (run with --stub for a GPU-free replay)")
    if a.stub and not Path(f"{a.clip}.tracks.json").is_file():
        raise die(f"--stub needs the sidecar {a.clip}.tracks.json")
    if shutil.which(FFMPEG) is None:
        raise die(f"ffmpeg not found ({FFMPEG}); set CENSOR_FFMPEG or put it on PATH")
    size = _live_size(a.clip, a.height)
    detect = _build_detect(a, probe(a.clip), size)
    from censor.live import LiveReplay
    rec = Recorder()
    replay = LiveReplay(a.clip, detect, height=a.height, delay_s=a.delay, on_emit=rec)
    for p in (a.out, a.report):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(_ffmpeg_cmd(replay.size, replay.fps, a.out), stdin=subprocess.PIPE)
    replay.start()
    try:
        _record(replay, rec, proc, replay.size, a.seconds)
    finally:
        replay.stop()
        proc.stdin.close()
        rc = proc.wait()
    stats = replay.stats()
    doc = {"clip": a.clip, "size": list(replay.size), "fps": replay.fps, "n_frames": replay.n_frames,
           "delay_s": a.delay, "stats": stats, "frames": rec.snapshot()}
    Path(a.report).write_text(json.dumps(doc), encoding="utf-8")
    d = stats["detector"]
    print(f"{len(doc['frames'])} frames emitted in {a.seconds:g} s of stream, ffmpeg exit {rc}\n"
          f"blind share {stats['blind_share']:.3f} ({stats['blind']} frames), dropped {stats['dropped_out']}\n"
          f"detector p50 {d['p50_ms']} ms, p95 {d['p95_ms']} ms over {d['passes']} passes")
    return 0 if rc == 0 else 2


# ---- cli ------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m censor.live_cli", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="replay a clip as a live feed and record the masked output")
    r.add_argument("clip")
    r.add_argument("--seconds", type=float, default=30.0)
    r.add_argument("--height", type=int, default=720)
    r.add_argument("--delay", type=float, default=1.0)
    r.add_argument("--out", required=True)
    r.add_argument("--report", required=True)
    r.add_argument("--stub", action="store_true", help="replay <clip>.tracks.json instead of running EgoBlur")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("score", help="check a run report against ground-truth labels (exit 1 below the recall floors)")
    s.add_argument("report")
    s.add_argument("labels")
    s.add_argument("--min-face", type=float, default=MIN_RECALL["face"])
    s.add_argument("--min-plate", type=float, default=MIN_RECALL["license plate"])
    s.set_defaults(fn=cmd_score)
    return ap


def main(argv: Optional[list[str]] = None) -> None:
    a = build_parser().parse_args(argv)
    raise SystemExit(a.fn(a))


if __name__ == "__main__":
    main()
