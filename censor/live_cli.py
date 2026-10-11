"""Headless live replay: run the live engine in real time, record what a viewer would see, score it.

    python -m censor.live_cli run <clip> [--seconds 30] [--height 720] [--delay 1.0] --out X.mp4 --report X.json
                                 [--cats "face,license plate"] [--stub [--yunet MODEL.onnx]]
    python -m censor.live_cli score X.json bench/labels/<clip>.json [--min-face 0.92] [--min-plate 0.95]
                                  [--max-blind 0.01]

`run` plays the clip as if it were a live feed (censor.live.LiveReplay), writes the emitted frames to an MP4 with a
REPLAY / age / blind overlay, and dumps a JSON report of the rects masked on every emitted frame. `--stub`
needs no GPU: it replays <clip>.tracks.json when that sidecar exists, else runs OpenCV's YuNet face detector
(faces only, CPU) with the ONNX file named by --yunet or, failing that, the CENSOR_YUNET environment variable. A
sidecar wins over YuNet; --yunet wins over CENSOR_YUNET. With neither, or a model file that is missing, `run` exits 2.

`score` is the go/no-go gate. A labeled face or plate box is covered when at least COVER_MIN of its area lies
inside the rects masked on that frame; frames the engine declared blind (it froze the last vouched frame) count as
covered, and labeled frames that were never emitted are skipped. Because blindness is free for recall, the gate
has three parts, and any one failing exits 1: recall per category at or above its floor; the share of emitted
frames that were blind at or below --max-blind (a dead detector is all blind frames and would otherwise score
1.0); and at least one scored box in every category the labels contain (a run that emitted none of the labeled
frames did not cover the clip). It is recall, not "zero bare boxes" (plan D6): live has no recheck_tiny / absorb
passes, so zero bare boxes is out of reach by construction.

`run --cats` picks what is masked: a comma-separated subset of "face" and "license plate" (default both; anything
else, or an empty list, exits 2). The report's stats.cats records it, and `score` then scores only the categories
listed there (a faces-only run is not failed on plates, and says so); a report without stats.cats scores both.

A failed `run` (ffmpeg died, the engine stalled) still writes the report with the frames collected so far.

Exit codes: 0 ok, 1 score fails a gate, 2 usage, environment or run problem (missing file, models, ffmpeg,
a malformed report, a run that failed part-way).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from censor.core.timeline import VideoInfo
from censor.core.video import probe
from censor.render.render import FFMPEG
from censor.vision.specialist import egoblur_available

if TYPE_CHECKING:  # annotations only: bench and the engine are imported lazily where used
    from bench.gt import LabelSet
    from censor.live import Detect, LiveReplay

CATS = ("face", "license plate")
MIN_RECALL = {"face": 0.92, "license plate": 0.95}  # offline recall on the bench clips minus 0.03
POLL_S = 2.0          # latest() timeout
MAX_IDLE_POLLS = 5    # consecutive empty polls before the run is declared stalled
RECORD_WAIT_S = 2.0   # how long the video loop waits for on_emit to have recorded a frame it already holds
STATS_EVERY = 10      # refresh the age shown in the overlay every n frames (stats() sorts samples)
FFMPEG_WAIT_S = 30.0  # how long to wait for ffmpeg to finish the file once its stdin is closed
MAX_BLIND = 0.01      # at most this share of emitted frames may be blind
Rect = tuple[int, int, int, int]


def parse_cats(text: str) -> tuple[str, ...]:
    """"face,license plate" -> ("face", "license plate"); ValueError on an unknown, repeated or empty category."""
    cats = tuple(c.strip() for c in text.split(","))
    if not text.strip() or any(c not in CATS for c in cats) or len(set(cats)) != len(cats):
        raise ValueError(f"--cats must be a comma-separated subset of {', '.join(CATS)}, got {text!r}")
    return cats


def die(msg: str, code: int = 2) -> "SystemExit":
    print(msg, file=sys.stderr)
    return SystemExit(code)


# ---- score ----------------------------------------------------------------------------------------

def _label_index(labels: "LabelSet", live_h: int) -> tuple[dict[int, list[tuple[str, str, Rect]]], dict[str, str]]:
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


def score_report(report: dict, labels: "LabelSet", live_h: int) -> dict:
    """Per category {labeled (label boxes in the file), boxes (scored on emitted frames), covered, recall (None
    without boxes), failing: {gt_id: first bare frame}, longest_run_s: {gt_id: seconds}} plus blind_share (over the
    report's frames), engine_blind_share (None if the report has none) and frames. See the module docstring."""
    from bench.run_bench import COVER_MIN, coverage
    index, cats = _label_index(labels, live_h)
    fps, step = float(report["fps"]), labels.every
    out: dict = {c: {"labeled": 0, "boxes": 0, "covered": 0, "recall": None, "failing": {}, "longest_run_s": {}}
                 for c in CATS}
    for entries in index.values():
        for _, cat, _ in entries:
            out[cat]["labeled"] += 1
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
    out["engine_blind_share"] = (report.get("stats") or {}).get("blind_share")  # the engine's own count, if recorded
    out["frames"] = n
    return out


def _run_cats(report: dict) -> tuple[str, ...]:
    """The categories the run masked (stats.cats), both when the report does not say."""
    cats = (report.get("stats") or {}).get("cats")
    return tuple(c for c in CATS if c in cats) if cats else CATS


def _print_score(res: dict, cats: tuple[str, ...] = CATS) -> None:
    engine = "" if res["engine_blind_share"] is None else f" (engine blind share {res['engine_blind_share']:.3f})"
    print(f"{res['frames']} frames emitted, blind share {res['blind_share']:.3f}{engine}")
    for cat in CATS:
        if cat not in cats:
            print(f"{cat}: not run, {'plates' if cat == 'license plate' else 'faces'} were not run (not scored)")
            continue
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
    try:
        res = score_report(report, labels, report["size"][1])
    except (KeyError, IndexError, TypeError, ValueError) as e:  # a report missing keys, or the wrong shape
        raise die(f"malformed report {a.report}: {type(e).__name__} {e}")
    cats = _run_cats(report)
    _print_score(res, cats)
    floors = {"face": a.min_face, "license plate": a.min_plate}
    fails = []
    if res["blind_share"] > a.max_blind:  # blind frames count as covered, so this is what catches a dead detector
        fails.append(f"blind share {res['blind_share']:.3f} > {a.max_blind:.3f}")
    for c in cats:
        r = res[c]
        if r["labeled"] and not r["boxes"]:  # no labeled frame was emitted: the run did not cover the clip
            fails.append(f"{c}: labels contain {r['labeled']} boxes but none were scored")
        elif r["recall"] is not None and r["recall"] < floors[c]:
            fails.append(f"{c} recall {r['recall']:.3f} < {floors[c]:.2f}")
    for msg in fails:
        print(f"FAIL: {msg}")
    return 1 if fails else 0


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


def _stub_source(a: argparse.Namespace) -> tuple[str, Optional[str]]:
    """What masks a --stub run: ("sidecar", None) when <clip>.tracks.json exists, else ("yunet", onnx path) from
    --yunet or CENSOR_YUNET, else ("none", None)."""
    if Path(f"{a.clip}.tracks.json").is_file():
        return "sidecar", None
    onnx = a.yunet or os.environ.get("CENSOR_YUNET")
    return ("yunet", onnx) if onnx else ("none", None)


def _build_detect(a: argparse.Namespace, src: VideoInfo, size: tuple[int, int],
                  cats: tuple[str, ...] = CATS) -> "Detect":
    from censor.live import egoblur_detect, sidecar_detect
    w, h = size
    if a.stub:
        kind, onnx = _stub_source(a)
        if kind == "yunet":
            from censor.live import yunet_detect
            try:
                return yunet_detect(onnx, cats=cats)
            except FileNotFoundError as e:  # the message names the path and where to get opencv_zoo's model
                raise die(str(e))
        return sidecar_detect(Path(f"{a.clip}.tracks.json"), src_h=src.height, live_h=h, n_frames=src.n_frames,
                              cats=cats)
    from censor.vision.specialist import EgoBlurDetector
    det = EgoBlurDetector()
    det.warm_up(sizes=((h, w),))  # before the clock starts, or the first seconds are all blind
    return egoblur_detect(det.infer, cats=cats)


def _overlay(img: np.ndarray, age_ms: int | str, blind_n: int, frames_out: int, blind: bool) -> None:
    txt = f"REPLAY · age {age_ms} ms · blind {blind_n}/{frames_out}" + (" · DETECTOR LATE" if blind else "")
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (0, 0), (tw + 24, 40), (20, 20, 20), -1)
    cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def _decode(jpeg: bytes, size: tuple[int, int]) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("the engine returned an undecodable frame")
    # latest() serves the display size; this is a recording of what the viewer sees, scaled back up to the live size.
    img = img if (img.shape[1], img.shape[0]) == size else cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(img)  # the pipe takes the array's buffer as is, which must be C-contiguous


def _ffmpeg_cmd(size: tuple[int, int], fps: float, out: str) -> list[str]:
    return [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
            "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", out]


def _record(replay: "LiveReplay", rec: Recorder, proc: subprocess.Popen, size: tuple[int, int], seconds: float) -> None:
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
            proc.stdin.write(last)
            out_n += 1
        blind_n += meta["blind"]
        out_n += 1
        if out_n % STATS_EVERY == 1:
            p50 = replay.stats()["frame_age_ms"]["p50"]
            age = "?" if p50 is None else int(round(p50))
        last = _decode(jpeg, size)
        _overlay(last, age, blind_n, out_n, meta["blind"])
        proc.stdin.write(last)
        after = f


def _close_ffmpeg(proc: subprocess.Popen) -> int:
    """Close ffmpeg's stdin and wait for it to finish the file. Neither a dead pipe nor an ffmpeg that hangs may
    mask the error that got us here: the exit status is what the caller reports."""
    try:
        proc.stdin.close()
    except OSError:  # BrokenPipeError flushing into an ffmpeg that already died; its exit status says why
        pass
    try:
        return proc.wait(timeout=FFMPEG_WAIT_S)
    except subprocess.TimeoutExpired:
        print(f"ffmpeg did not finish within {FFMPEG_WAIT_S:.0f} s; killing it", file=sys.stderr)
        proc.kill()
        return proc.wait()


def _write_report(a: argparse.Namespace, replay: "LiveReplay", rec: Recorder) -> dict:
    """Dump the run report (the rects masked on every emitted frame, plus the engine's stats); returns the stats."""
    stats = replay.stats()
    doc = {"clip": a.clip, "size": list(replay.size), "fps": replay.fps, "n_frames": replay.n_frames,
           "delay_s": a.delay, "stats": stats, "frames": rec.snapshot()}
    Path(a.report).write_text(json.dumps(doc), encoding="utf-8")
    return stats


def cmd_run(a: argparse.Namespace) -> int:
    try:
        cats = parse_cats(a.cats)
    except ValueError as e:
        raise die(str(e))
    if not Path(a.clip).is_file():
        raise die(f"clip not found: {a.clip}")
    if not a.stub and not egoblur_available():
        raise die("EgoBlur models not found; live replay needs them (run with --stub for a GPU-free replay)")
    if a.yunet and not a.stub:
        raise die("--yunet only applies with --stub (a real run uses EgoBlur)")
    if a.stub and _stub_source(a)[0] == "none":
        raise die(f"--stub needs the sidecar {a.clip}.tracks.json next to the clip, or CENSOR_YUNET=<onnx> / "
                  f"--yunet <onnx> for faces")
    if shutil.which(FFMPEG) is None:
        raise die(f"ffmpeg not found ({FFMPEG}); set CENSOR_FFMPEG or put it on PATH")
    size = _live_size(a.clip, a.height)
    detect = _build_detect(a, probe(a.clip), size, cats)
    from censor.live import LiveReplay
    rec = Recorder()
    replay = LiveReplay(a.clip, detect, height=a.height, delay_s=a.delay, on_emit=rec, cats=cats)
    for p in (a.out, a.report):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    proc: Optional[subprocess.Popen] = None
    failure, rc = None, None
    try:
        proc = subprocess.Popen(_ffmpeg_cmd(replay.size, replay.fps, a.out), stdin=subprocess.PIPE)
        replay.start()
        _record(replay, rec, proc, replay.size, a.seconds)
    except (OSError, RuntimeError) as e:  # a dead ffmpeg pipe, an ffmpeg that will not start, a stalled engine
        failure = f"{type(e).__name__}: {e}"
    finally:  # whatever happened, the frames collected so far are the evidence: never lose a 30 s GPU run
        replay.stop()
        if proc is not None:
            rc = _close_ffmpeg(proc)
        stats = _write_report(a, replay, rec)
    if failure is not None:
        print(f"run failed: {failure}", file=sys.stderr)
    d = stats["detector"]
    print(f"{len(rec.snapshot())} frames emitted in {a.seconds:g} s of stream, ffmpeg exit {rc}\n"
          f"blind share {stats['blind_share']:.3f} ({stats['blind']} frames), dropped {stats['dropped_out']}\n"
          f"detector p50 {d['p50_ms']} ms, p95 {d['p95_ms']} ms over {d['passes']} passes")
    return 0 if failure is None and rc == 0 else 2


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
    r.add_argument("--cats", default=",".join(CATS), metavar="LIST",
                   help='what to mask: comma-separated subset of "face" and "license plate" (default both)')
    r.add_argument("--stub", action="store_true",
                   help="no GPU: replay <clip>.tracks.json if present, else detect faces with YuNet (--yunet / CENSOR_YUNET)")
    r.add_argument("--yunet", metavar="ONNX", help="with --stub: YuNet model for a clip without a sidecar "
                   "(default: $CENSOR_YUNET; from opencv_zoo, face_detection_yunet_2023mar.onnx)")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("score", help="check a run report against ground-truth labels "
                                     "(exit 1 below the recall floors or above the blind ceiling)")
    s.add_argument("report")
    s.add_argument("labels")
    s.add_argument("--min-face", type=float, default=MIN_RECALL["face"])
    s.add_argument("--min-plate", type=float, default=MIN_RECALL["license plate"])
    s.add_argument("--max-blind", type=float, default=MAX_BLIND, help="fail above this share of blind emitted frames")
    s.set_defaults(fn=cmd_score)
    return ap


def main(argv: Optional[list[str]] = None) -> None:
    a = build_parser().parse_args(argv)
    raise SystemExit(a.fn(a))


if __name__ == "__main__":
    main()
