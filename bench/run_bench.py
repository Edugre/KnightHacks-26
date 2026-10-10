"""Benchmark: every number in IMPLEMENTATION.md section 6 comes from here, never by hand.

For each clip in bench/clips/manifest.yaml, with labels in bench/labels/<id>.json:
1. Coverage pass: Session.find_objects for every labeled category, mask everything found, and score the
   rects the renderer would actually draw (FrameMasker boxes through _padded). A labeled box counts as
   covered when >= COVER_MIN of its area is inside the union of mask rects on that frame.
2. Requests from bench/requests.yaml, each on a fresh op log over the coverage pass's cached tracks, run
   three ways: ours (Agent: LLM when CENSOR_MODEL_URL is set, else the template fallback); keyword
   (baseline: nouns from the request go straight to the detector, every hit is masked, no attribute check,
   no times); fail_open (ours with fail-closed off).
3. Latency per stage (detect, agent request, render), offline speed and peak GPU memory.

fail_open is measurement only: it sets the renderer's hold-after-loss to 0 frames, stops doubling padding
on uncertain keyframes and reads VLM "unknown" as "no", all by patching inside this process. Nothing
here changes what ships.

Run:  python -m bench.run_bench [--clip ID] [--no-render] [--no-baselines] [--out-dir bench]
      CENSOR_STUB=1 picks the stub detector; CENSOR_MODEL_URL=http://127.0.0.1:8100 the real agent + VLM.
Writes <out-dir>/report.json and <out-dir>/report.md.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # also runnable as `python bench/run_bench.py`
from bench.gt import LabelSet  # noqa: E402
from censor.agent import agent as agent_mod  # noqa: E402
from censor.agent import intents  # noqa: E402
from censor.agent.tools import Session, ToolError  # noqa: E402
from censor.core.timeline import Timeline  # noqa: E402
from censor.render import render as render_mod  # noqa: E402
from censor.vision import vlm as vlm_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
CLIPS_DIR = BENCH / "clips"
LABELS_DIR = BENCH / "labels"

COVER_MIN = 0.90    # a labeled box is covered when this share of its area is masked
REQ_MASKED = 0.90   # must_mask objects: covered on at least this share of their labeled frames
REQ_VISIBLE = 0.10  # must_not_mask objects: covered on at most this share
VARIANTS = ("ours", "keyword", "fail_open")

Rect = tuple[int, int, int, int]


class Clip(BaseModel):
    id: str
    file: str  # under bench/clips/
    source: str
    consent: str
    duration: float
    notes: str = ""
    synthetic: bool = False


class BenchRequest(BaseModel):
    clip: str
    text: str
    must_mask: list[str] = Field(default_factory=list)
    must_not_mask: list[str] = Field(default_factory=list)
    window: Optional[tuple[float, float]] = None  # seconds; only labeled frames inside it are checked
    before: list[str] = Field(default_factory=list)  # applied first, not scored


def _yaml_list(path: Path, key: str) -> list:
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get(key) or []


def load_manifest(path: Path = CLIPS_DIR / "manifest.yaml") -> list[Clip]:
    return [Clip.model_validate(c) for c in _yaml_list(path, "clips")]


def load_requests(path: Path = BENCH / "requests.yaml") -> list[BenchRequest]:
    return [BenchRequest.model_validate(r) for r in _yaml_list(path, "requests")]


def check_request(req: BenchRequest, labels: LabelSet) -> None:
    present = {b.gt_id for fr in labels.frames_in(req.window) for b in fr.boxes}
    bad = [g for g in req.must_mask + req.must_not_mask if g not in present]
    if bad:
        where = f" inside window {list(req.window)}" if req.window else ""
        raise ValueError(f"{req.clip}: request {req.text!r} names {bad}, which have no labeled box{where}")


# ---- what the renderer would mask ---------------------------------------------------------------

@contextlib.contextmanager
def _patched(obj, name: str, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _without_uncertain(tl: Timeline) -> Timeline:
    tl = tl.model_copy(deep=True)
    for t in tl.targets.values():
        t.keyframes = [k.model_copy(update={"state": "detected"}) if k.state == "uncertain" else k for k in t.keyframes]
    return tl


def mask_rects(tl: Timeline, fail_closed: bool = True) -> dict[int, list[Rect]]:
    """Per-frame pixel rects exactly as the renderer masks them (FrameMasker, then _padded)."""
    if not fail_closed:
        tl = _without_uncertain(tl)
    with _patched(render_mod, "HOLD_FRAMES", render_mod.HOLD_FRAMES if fail_closed else 0):
        masker = render_mod.FrameMasker(tl)
    w, h = tl.video.width, tl.video.height
    return {f: [render_mod._padded(box, pad, w, h) for box, pad, _ in items] for f, items in masker.per_frame.items()}


def fail_closed_rate(tl: Timeline, fail_closed: bool = True) -> dict:
    """Share of 'uncertain' keyframes rendered with doubled padding. Fail-closed makes this 100% by rule."""
    view = tl if fail_closed else _without_uncertain(tl)
    n = doubled = 0
    for tid, t in tl.targets.items():
        unc = [k.f for k in t.keyframes if k.state == "uncertain"]
        if unc:
            per = render_mod.expand_target(view.targets[tid], tl.video.n_frames)
            n += len(unc)
            doubled += sum(f in per and per[f][1] >= 2 * t.padding - 1e-9 for f in unc)
    return {"uncertain_keyframes": n, "doubled": doubled, "rate": doubled / n if n else None}


def _vlm_unknown_is_no():
    real = vlm_mod.check_attribute

    def check(video, tracks, question):
        return [dataclasses.replace(a, answer="no") if a.answer == "unknown" else a
                for a in real(video, tracks, question)]
    return _patched(vlm_mod, "check_attribute", check)


# ---- scoring ------------------------------------------------------------------------------------

def _mean(xs) -> Optional[float]:
    return float(np.mean(xs)) if len(xs) else None


def _clip_box(box, w: int, h: int) -> Optional[Rect]:
    x1, y1 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x2, y2 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def coverage(box: Rect, rects: list[Rect]) -> float:
    """Share of `box` inside the union of `rects`. A union, so overlapping masks aren't counted twice."""
    x1, y1, x2, y2 = box
    m = np.zeros((y2 - y1, x2 - x1), bool)
    for rx1, ry1, rx2, ry2 in rects:
        ax1, ay1, ax2, ay2 = max(rx1, x1), max(ry1, y1), min(rx2, x2), min(ry2, y2)
        if ax2 > ax1 and ay2 > ay1:
            m[ay1 - y1:ay2 - y1, ax1 - x1:ax2 - x1] = True
    return float(m.mean())


def over_blur(rects: list[Rect], boxes: list[Rect], w: int, h: int) -> float:
    """Share of the frame that is masked but outside every labeled box."""
    m = np.zeros((h, w), bool)
    for x1, y1, x2, y2 in rects:
        m[y1:y2, x1:x2] = True
    for x1, y1, x2, y2 in boxes:
        m[y1:y2, x1:x2] = False
    return float(m.mean())


def score_frames(labels: LabelSet, rects: dict[int, list[Rect]],
                 window: Optional[tuple[float, float]] = None) -> tuple[dict[str, list[bool]], list[float]]:
    """Per object, covered or not on each labeled frame it appears in (in order); over-blur per labeled frame."""
    w, h = labels.video.width, labels.video.height
    hits: dict[str, list[bool]] = {}
    over = []
    for fr in labels.frames_in(window):
        frame_rects = rects.get(fr.f, [])
        boxes = []
        for b in fr.boxes:
            box = _clip_box(b.box, w, h)
            if box is not None:
                hits.setdefault(b.gt_id, []).append(coverage(box, frame_rects) >= COVER_MIN)
                boxes.append(box)
        over.append(over_blur(frame_rects, boxes, w, h))
    return hits, over


def longest_miss(hits: list[bool]) -> int:
    """Longest run of consecutive labeled frames where the object was not covered."""
    run = best = 0
    for ok in hits:
        run = 0 if ok else run + 1
        best = max(best, run)
    return best


def coverage_report(labels: LabelSet, rects: dict[int, list[Rect]]) -> dict:
    hits, over = score_frames(labels, rects)
    by_cat: dict[str, list[bool]] = {}
    for gid, hs in hits.items():
        by_cat.setdefault(labels.objects[gid].category, []).extend(hs)
    return {"recall": {c: _mean(v) for c, v in sorted(by_cat.items())},
            "boxes": {c: len(v) for c, v in sorted(by_cat.items())},
            "hits": {c: int(sum(v)) for c, v in sorted(by_cat.items())},
            "objects": {g: {"category": labels.objects[g].category, "covered": _mean(hs), "longest_miss": longest_miss(hs)}
                        for g, hs in hits.items()},
            "over_blur": _mean(over)}


def verdict(req: BenchRequest, hits: dict[str, list[bool]]) -> dict:
    covered = {g: _mean(hits.get(g, [])) or 0.0 for g in req.must_mask + req.must_not_mask}
    missed = [g for g in req.must_mask if covered[g] < REQ_MASKED]
    leaked = [g for g in req.must_not_mask if covered[g] > REQ_VISIBLE]
    return {"ok": not missed and not leaked, "missed": missed, "leaked": leaked, "covered": covered}


# ---- running ------------------------------------------------------------------------------------

def _apply(s: Session, text: str, variant: str) -> dict:
    if variant != "keyword":
        return agent_mod.Agent(s).run(text)
    plan = intents.parse(text)  # nouns only: attributes, times, unblur and undo are ignored on purpose
    for q in plan.queries if plan else []:
        ids = [o["track_id"] for o in s.find_objects(q)]
        if ids:
            s.add_masks(ids, request_id="keyword")
    return {"path": "keyword"}


def run_request(video: Path, tracks_json: Path, labels: LabelSet, req: BenchRequest, workdir: Path,
                variant: str) -> dict:
    workdir.mkdir(parents=True)
    shutil.copy(tracks_json, workdir / "tracks.json")  # reuse the coverage pass's detections
    s = Session(video, workdir)
    with _vlm_unknown_is_no() if variant == "fail_open" else contextlib.nullcontext():
        for text in req.before:
            _apply(s, text, variant)
        t0 = time.perf_counter()
        try:
            res = _apply(s, req.text, variant)
        except ToolError as e:
            res = {"path": "error", "message": str(e)}
        seconds = time.perf_counter() - t0
    hits, _ = score_frames(labels, mask_rects(s.timeline, fail_closed=variant != "fail_open"), req.window)
    return {**verdict(req, hits), "path": res.get("path"), "seconds": round(seconds, 3),
            "tool_calls": len(res.get("calls", [])), "message": res.get("message")}


def bench_clip(video: Path, labels: LabelSet, reqs: list[BenchRequest], render: bool = True,
               variants: tuple[str, ...] = VARIANTS) -> dict:
    for r in reqs:
        check_request(r, labels)
    v = labels.video
    out: dict = {"clip": labels.clip, "file": video.name, "resolution": f"{v.width}x{v.height}",
                 "frames": v.n_frames, "duration": round(v.n_frames / v.fps, 2), "errors": []}
    if labels.n_drafts:
        out["errors"].append(f"{labels.n_drafts} of {len(labels.frames)} labeled frames are unconfirmed detector drafts "
                             "and were not scored (confirm them in bench/label_server.py)")
    with tempfile.TemporaryDirectory(prefix=f"bench_{labels.clip}_", ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        s = Session(video, tmp / "coverage")
        out["detector"] = type(s.detector).__name__
        ids: list[str] = []
        t0 = time.perf_counter()
        for cat in sorted({o.category for o in labels.objects.values()}):
            try:
                ids += [o["track_id"] for o in s.find_objects(cat)]
            except ToolError as e:
                out["errors"].append(str(e))
        detect_s = time.perf_counter() - t0
        if ids:
            s.add_masks(ids, request_id="bench_coverage")
        s.save()
        tl = s.timeline
        out["coverage"] = {"ours": coverage_report(labels, mask_rects(tl)),
                           "fail_open": coverage_report(labels, mask_rects(tl, fail_closed=False))}
        out["fail_closed"] = {"ours": fail_closed_rate(tl), "fail_open": fail_closed_rate(tl, fail_closed=False)}

        render_s = None
        if render and shutil.which(render_mod.FFMPEG):
            t0 = time.perf_counter()
            render_mod.render_video(str(video), tl, str(tmp / "render.mp4"))
            render_s = time.perf_counter() - t0
        elif render:
            out["errors"].append(f"render skipped: {render_mod.FFMPEG!r} not found (set CENSOR_FFMPEG)")
        out["latency"] = {"detect_s": round(detect_s, 3), "detect_ms_per_frame": 1000 * detect_s / v.n_frames,
                          "render_s": None if render_s is None else round(render_s, 3),
                          "render_ms_per_frame": None if render_s is None else 1000 * render_s / v.n_frames,
                          "offline_speed": None if render_s is None else out["duration"] / (detect_s + render_s)}

        out["requests"] = []
        for i, r in enumerate(reqs):
            row = {"text": r.text, "before": r.before, "window": r.window}
            for var in variants:
                print(f"[bench] {labels.clip}: {var}: {r.text}", file=sys.stderr)
                row[var] = run_request(video, s.workdir / "tracks.json", labels, r, tmp / f"req{i}_{var}", var)
            out["requests"].append(row)
    return out


def gpu_memory() -> dict:
    """Peak GPU memory of this process (the detector runs here) plus what the model server has loaded."""
    out: dict = {"bench_process_gb": None, "model_server_gb": None}
    if importlib.util.find_spec("torch"):
        import torch
        if torch.cuda.is_available():
            out["bench_process_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if vlm_mod.MODEL_URL:
        try:
            with urllib.request.urlopen(f"{vlm_mod.MODEL_URL}/health", timeout=3) as r:
                out["model_server_gb"] = json.load(r).get("gpu_mem_gb")
        except Exception:
            pass
    known = [x for x in out.values() if x is not None]
    out["total_gb"] = round(sum(known), 2) if known else None
    return out


def make_synthetic(video: Path) -> None:
    print(f"[bench] generating {video.relative_to(ROOT)}", file=sys.stderr)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_clip.py"), str(video)], check=True)


# ---- report -------------------------------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def summarize(clips: list[dict], variants: tuple[str, ...]) -> dict:
    def pooled(var: str) -> tuple[dict, dict]:
        boxes: dict[str, int] = {}
        hits: dict[str, int] = {}
        for c in clips:
            cov = c["coverage"][var]
            for cat, n in cov["boxes"].items():
                boxes[cat] = boxes.get(cat, 0) + n
                hits[cat] = hits.get(cat, 0) + cov["hits"][cat]
        return boxes, hits

    boxes, hits = pooled("ours")
    _, hits_open = pooled("fail_open")
    st_boxes = sum(boxes.get(c, 0) for c in ("screen", "text"))
    reqs = [r for c in clips for r in c["requests"]]
    unc = sum(c["fail_closed"]["ours"]["uncertain_keyframes"] for c in clips)
    doubled = sum(c["fail_closed"]["ours"]["doubled"] for c in clips)
    lat = [c["latency"] for c in clips]
    rendered = bool(lat) and all(x["render_s"] is not None for x in lat)
    detect_s, render_s = sum(x["detect_s"] for x in lat), sum(x["render_s"] or 0 for x in lat)
    frames, duration = sum(c["frames"] for c in clips), sum(c["duration"] for c in clips)
    agent_s = [r["ours"]["seconds"] for r in reqs]
    return {
        "boxes": boxes,
        "recall": {"ours": {c: hits[c] / n for c, n in boxes.items()},
                   "fail_open": {c: hits_open[c] / n for c, n in boxes.items()}},
        "recall_screen_text": sum(hits.get(c, 0) for c in ("screen", "text")) / st_boxes if st_boxes else None,
        "longest_miss": max((o["longest_miss"] for c in clips for o in c["coverage"]["ours"]["objects"].values()), default=None),
        "over_blur": {v: _mean([c["coverage"][v]["over_blur"] for c in clips if c["coverage"][v]["over_blur"] is not None])
                      for v in ("ours", "fail_open")},
        "fail_closed": {"uncertain_keyframes": unc, "doubled": doubled, "rate": doubled / unc if unc else None},
        "requests": {"n": len(reqs), "success": {v: sum(r[v]["ok"] for r in reqs) for v in variants}},
        "latency": {"detect_ms_per_frame": 1000 * detect_s / frames if frames else None,
                    "render_ms_per_frame": 1000 * render_s / frames if frames and rendered else None,
                    "offline_speed": duration / (detect_s + render_s) if rendered and detect_s + render_s > 0 else None,
                    "agent_s_mean": _mean(agent_s), "agent_s_max": max(agent_s, default=None)},
        "gpu": gpu_memory(),
    }


def _row(metric: str, target: str, value, ok: Callable, fmt: Callable = "{:.2f}".format) -> dict:
    return {"metric": metric, "target": target, "value": value,
            "shown": "n/a" if value is None else fmt(value), "pass": None if value is None else bool(ok(value))}


def targets(s: dict) -> list[dict]:
    """IMPLEMENTATION.md section 6. Request success is scaled when there aren't 20 requests yet."""
    rec, n, ok = s["recall"]["ours"], s["requests"]["n"], s["requests"]["success"].get("ours", 0)
    pct = "{:.1%}".format
    return [
        _row("Recall: faces", ">= 0.95", rec.get("face"), lambda v: v >= 0.95),
        _row("Recall: plates", ">= 0.90", rec.get("license plate"), lambda v: v >= 0.90),
        _row("Recall: screens/text", ">= 0.85", s["recall_screen_text"], lambda v: v >= 0.85),
        _row("Over-blur (mean masked non-target area)", "<= 10% of frame", s["over_blur"]["ours"], lambda v: v <= 0.10, pct),
        _row("Request success", ">= 15 / 20 (75%)", ok / n if n else None, lambda v: v >= 0.75,
             lambda v: f"{ok} / {n}" + (f" (only {n} requests so far)" if n < 20 else "")),
        _row("Fail-closed rate on uncertain keyframes", "100%", s["fail_closed"]["rate"], lambda v: v >= 1.0, pct),
        _row("Offline speed (target is for 1080p)", ">= 0.5x real time", s["latency"]["offline_speed"], lambda v: v >= 0.5,
             "{:.1f}x".format),
        _row("Peak GPU memory, all models loaded", "<= 28 GB", s["gpu"]["total_gb"], lambda v: v <= 28, "{:.1f} GB".format),
        _row("Longest unmasked run of a labeled target", "<= 1 labeled frame", s["longest_miss"], lambda v: v <= 1, str),
    ]


def build_report(clips: list[dict], skipped: list[dict], variants: tuple[str, ...]) -> dict:
    s = summarize(clips, variants)
    dirty = _git("status", "--porcelain", "--", ".", ":!bench/report.md", ":!bench/report.json")
    meta = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": _git("rev-parse", "--short", "HEAD"),
            "dirty": bool(dirty), "n_clips": len(clips), "skipped": skipped, "variants": list(variants),
            "detectors": sorted({c["detector"] for c in clips}),
            "agent": "LLM via CENSOR_MODEL_URL, template fallback" if agent_mod.MODEL_URL else "template only (CENSOR_MODEL_URL unset)",
            "vlm": "Qwen3-VL via the model server" if vlm_mod.MODEL_URL else "stub, every answer is 'unknown' (fail closed)",
            "thresholds": {"cover_min": COVER_MIN, "req_masked": REQ_MASKED, "req_visible": REQ_VISIBLE}}
    return {"meta": meta, "summary": s, "targets": targets(s), "clips": clips}


def _f(v, fmt: str = "{:.2f}") -> str:
    return "n/a" if v is None else fmt.format(v)


def to_markdown(rep: dict) -> str:
    m, s, clips = rep["meta"], rep["summary"], rep["clips"]
    names = {"ours": "Ours", "keyword": "Keyword baseline", "fail_open": "Fail-closed off"}
    variants = m["variants"]
    L = ["# Benchmark report", "",
         f"Generated {m['generated']} by `bench/run_bench.py` at commit `{m['commit'] or 'unknown'}`"
         f"{' plus uncommitted changes' if m['dirty'] else ''}. Do not edit by hand.", "",
         f"- Clips: {m['n_clips']} ({', '.join(c['clip'] for c in clips) or 'none'})"
         + (f"; skipped: {'; '.join(x['clip'] + ': ' + x['reason'] for x in m['skipped'])}" if m["skipped"] else ""),
         f"- Detector: {', '.join(m['detectors']) or 'n/a'}. Agent: {m['agent']}. VLM: {m['vlm']}.",
         f"- A labeled box is covered when at least {COVER_MIN:.0%} of its area is masked. A request passes when each "
         f"must-mask object is covered on at least {REQ_MASKED:.0%} of its labeled frames and each must-stay-visible "
         f"object on at most {REQ_VISIBLE:.0%}.",
         "- Keyword baseline: request nouns go straight to the detector and every hit is masked (no attribute check, "
         "no times). Fail-closed off: no hold-after-loss, no extra padding when uncertain, VLM 'unknown' read as 'no'."]
    if "StubDetector" in m["detectors"]:
        L.append("- Stub detector: it replays ground truth, so recall here checks the plumbing, not detection quality.")

    L += ["", "## Targets (IMPLEMENTATION.md section 6)", "", "| Metric | Target | Measured | Pass |", "| --- | --- | --- | --- |"]
    L += [f"| {t['metric']} | {t['target']} | {t['shown']} | {'n/a' if t['pass'] is None else 'pass' if t['pass'] else 'FAIL'} |"
          for t in rep["targets"]]

    L += ["", "## Recall by category", "", "| Category | Labeled boxes | Ours | Fail-closed off |", "| --- | --- | --- | --- |"]
    L += [f"| {c} | {n} | {_f(s['recall']['ours'][c])} | {_f(s['recall']['fail_open'][c])} |" for c, n in sorted(s["boxes"].items())]

    L += ["", "## Objects", "", "| Clip | Object | Category | Covered frames | Longest miss (labeled frames) |",
          "| --- | --- | --- | --- | --- |"]
    L += [f"| {c['clip']} | {g} | {o['category']} | {_f(o['covered'], '{:.0%}')} | {o['longest_miss']} |"
          for c in clips for g, o in c["coverage"]["ours"]["objects"].items()]

    L += ["", "## Over-blur and fail-closed", "",
          "| Clip | Over-blur | Over-blur, fail-closed off | Uncertain keyframes | Doubled padding |", "| --- | --- | --- | --- | --- |"]
    L += [f"| {c['clip']} | {_f(c['coverage']['ours']['over_blur'], '{:.1%}')} | {_f(c['coverage']['fail_open']['over_blur'], '{:.1%}')} "
          f"| {c['fail_closed']['ours']['uncertain_keyframes']} | {_f(c['fail_closed']['ours']['rate'], '{:.0%}')} |" for c in clips]

    ok = s["requests"]["success"]
    L += ["", "## Requests", "", ", ".join(f"{names[v]}: {ok[v]} / {s['requests']['n']}" for v in variants) + ".", "",
          "| Clip | Request | Path | " + " | ".join(names[v] for v in variants) + " | Seconds | Why ours failed |",
          "| --- | --- | --- | " + " | ".join("---" for _ in variants) + " | --- | --- |"]
    for c in clips:
        for r in c["requests"]:
            o = r["ours"]
            why = "; ".join(x for x in (f"missed {', '.join(o['missed'])}" if o["missed"] else "",
                                         f"masked {', '.join(o['leaked'])}" if o["leaked"] else "") if x)
            text = r["text"].replace("|", "/") + (f" (after: {'; '.join(r['before'])})" if r["before"] else "")
            L.append(f"| {c['clip']} | {text} | {o['path']} | " + " | ".join("pass" if r[v]["ok"] else "FAIL" for v in variants)
                     + f" | {o['seconds']:.2f} | {why} |")

    L += ["", "## Latency", "", "| Clip | Resolution | Frames | Detect ms/frame | Render ms/frame | Offline speed | Agent s, mean / max |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in clips:
        x, secs = c["latency"], [r["ours"]["seconds"] for r in c["requests"]]
        L.append(f"| {c['clip']} | {c['resolution']} | {c['frames']} | {_f(x['detect_ms_per_frame'], '{:.1f}')} "
                 f"| {_f(x['render_ms_per_frame'], '{:.1f}')} | {_f(x['offline_speed'], '{:.1f}x')} "
                 f"| {_f(_mean(secs))} / {_f(max(secs, default=None))} |")
    g = s["gpu"]
    L += ["", f"GPU memory: bench process (detector) {_f(g['bench_process_gb'], '{:.2f} GB')}, "
              f"model server {_f(g['model_server_gb'], '{:.1f} GB')}."]
    errors = [f"{c['clip']}: {e}" for c in clips for e in c["errors"]]
    if errors:
        L += ["", "## Errors", ""] + [f"- {e}" for e in errors]
    return "\n".join(L) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run the benchmark; writes report.json and report.md.")
    ap.add_argument("--clip", action="append", help="only this clip id (repeatable)")
    ap.add_argument("--no-render", action="store_true", help="skip render timing (no ffmpeg needed)")
    ap.add_argument("--no-baselines", action="store_true", help="score requests with our pipeline only")
    ap.add_argument("--out-dir", default=str(BENCH))
    args = ap.parse_args(argv)

    clips, reqs = load_manifest(), load_requests()
    unknown = sorted({r.clip for r in reqs} - {c.id for c in clips})
    if unknown:
        sys.exit(f"bench/requests.yaml names clips missing from the manifest: {unknown}")
    variants = ("ours",) if args.no_baselines else VARIANTS
    results, skipped = [], []
    for clip in clips:
        if args.clip and clip.id not in args.clip:
            continue
        video, label_path = CLIPS_DIR / clip.file, LABELS_DIR / f"{clip.id}.json"
        if not video.exists() and clip.synthetic:
            make_synthetic(video)
        if not video.exists():
            skipped.append({"clip": clip.id, "reason": f"no video at bench/clips/{clip.file}"})
        elif not label_path.exists():
            skipped.append({"clip": clip.id, "reason": f"no labels at bench/labels/{clip.id}.json"})
        else:
            print(f"[bench] {clip.id}", file=sys.stderr)
            results.append(bench_clip(video, LabelSet.load(label_path), [r for r in reqs if r.clip == clip.id],
                                      render=not args.no_render, variants=variants))

    rep = build_report(results, skipped, variants)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    md = to_markdown(rep)
    (out / "report.md").write_text(md, encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
