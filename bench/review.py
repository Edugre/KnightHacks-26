"""Review what EgoBlur sees on a clip, what the app masks, and how much of each real plate (or face) stays visible.

Detection runs once per clip, on every frame at a low threshold, and is cached; the app's rule (every 3rd
frame, threshold, linking, the confirmation rule, hold-after-loss) is then replayed from the cache, so a rule
change is scored in milliseconds without the model. Runs where EgoBlur runs: the R9700, or a laptop with CUDA.

  python -m bench.review detect clip.mp4                    # raw detections -> data/review/<clip>/raw.json
  python -m bench.review candidates clip.mp4 --cat plate    # crop of every candidate track -> candidates_plate.jpg
  python -m bench.review label clip.mp4 --cat plate --real 3 7 12   # those candidates are real -> bench/labels/
  python -m bench.review score clip.mp4 --cat plate         # per real plate: frames masked, longest visible run
  python -m bench.review frames clip.mp4 --frames 0 60 120  # overlay: raw detections, masks, labels

Labels are boxes per frame (bench/labels/<clip>.<cat>.json), so they don't depend on candidate ids.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from censor.core.timeline import Target  # noqa: E402
from censor.core.tracks import Keyframe  # noqa: E402
from censor.core.video import frames, probe, read_frame  # noqa: E402
from censor.render.render import _padded, expand_target  # noqa: E402
from censor.vision import specialist as sp  # noqa: E402

RAW_CONF = 0.05   # below the app's 0.1, so the review also shows what the app never sees
COVERED = 0.9     # a labeled box counts as masked when masks cover this much of it (as bench/run_bench.py)
CATS = {"plate": "license plate", "face": "face"}
Box = tuple[float, float, float, float]


TAG = ""  # set by --tag: keeps one cache per model setup (raw.json, raw_gen2.json, ...)


def out_dir(clip: str) -> Path:
    d = ROOT / "data" / "review" / Path(clip).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def label_path(clip: str, cat: str) -> Path:
    return ROOT / "bench" / "labels" / f"{Path(clip).stem}.{cat}.json"


# ---- detection (the only part that needs the model) --------------------------------------------------------

def detect(clip: str) -> dict:
    infer = sp.load_egoblur()
    raw = {c: {} for c in sp.EGOBLUR_FILES}
    for f, img in frames(clip, prefetch=8):
        for c, fn in infer.items():
            raw[c][f] = [[*map(float, b), s] for b, s in fn(img) if s >= RAW_CONF]
        if f % 50 == 0:
            print(f"frame {f}", file=sys.stderr)
    info = probe(clip)
    data = {"clip": clip, "width": info.width, "height": info.height, "fps": info.fps, "n_frames": info.n_frames,
            "raw_conf": RAW_CONF, "dets": raw}
    (out_dir(clip) / f"raw{'_' + TAG if TAG else ''}.json").write_text(json.dumps(data))
    return data


def load_raw(clip: str) -> dict:
    p = out_dir(clip) / f"raw{'_' + TAG if TAG else ''}.json"
    if not p.exists():
        sys.exit(f"no detections cached for {clip}: run `python -m bench.review detect {clip}"
                 f"{' --tag ' + TAG if TAG else ''}` first")
    d = json.loads(p.read_text())
    d["dets"] = {c: {int(f): v for f, v in per.items()} for c, per in d["dets"].items()}
    return d


def dets_at(raw: dict, cat: str, f: int, conf: float) -> list[tuple[Box, float]]:
    out = [(tuple(x[:4]), x[4]) for x in raw["dets"][cat].get(f, []) if x[4] >= conf]
    if cat == "face":
        out = [(sp._expand(b, sp.FACE_EXPAND), s) for b, s in out]
    return out


# ---- the app's rule, replayed ------------------------------------------------------------------------------

def app_tracks(raw: dict, cat: str, every: int = sp.EGOBLUR_EVERY, conf: float = sp.EGOBLUR_CONF,
               rule: Optional[dict] = None) -> tuple[list[list[Keyframe]], list[list[Keyframe]]]:
    """(kept, dropped) tracks, exactly as EgoBlurDetector builds them from these detections."""
    rule = sp.CONFIRM.get(cat) if rule is None else rule
    seq = [(f, dets_at(raw, cat, f, conf)) for f in range(0, raw["n_frames"], every)]
    return sp.build_tracks(seq, rule, raw["height"])


def mask_rects(raw: dict, tracks: list[list[Keyframe]], padding: float = 0.15) -> dict[int, list[tuple]]:
    """Per frame, the rectangles the renderer would blur for these tracks."""
    per: dict[int, list[tuple]] = {}
    for kfs in tracks:
        t = Target(label="x", source="detector", padding=padding, keyframes=kfs)
        for f, (box, pad) in expand_target(t, raw["n_frames"]).items():
            per.setdefault(f, []).append(_padded(box, pad, raw["width"], raw["height"]))
    return per


# ---- candidates and labels ---------------------------------------------------------------------------------

def candidates(raw: dict, cat: str, conf: float = sp.EGOBLUR_CONF) -> list[list[Keyframe]]:
    """Every-frame tracks of everything detected at >= conf: what a labeler picks real objects from."""
    seq = [(f, dets_at(raw, cat, f, conf)) for f in range(raw["n_frames"])]
    tracks = [t for t in sp.link(seq, 0.35)]
    return sorted(tracks, key=lambda t: (t[0].f, -max(k.conf for k in t)))


def candidate_sheet(clip: str, raw: dict, cat: str, out: Path, min_len: int = 1) -> int:
    cands = [(i, t) for i, t in enumerate(candidates(raw, cat)) if len(t) >= min_len]  # numbers as `label` reads them
    kept, _ = app_tracks(raw, cat)
    kept_rects = mask_rects(raw, kept)
    tiles = []
    for i, t in cands:
        best = max(t, key=lambda k: k.conf)
        img = read_frame(clip, best.f)
        x1, y1, x2, y2 = best.box
        w, h = x2 - x1, y2 - y1
        cx1, cy1 = int(max(0, x1 - 1.5 * w)), int(max(0, y1 - 1.5 * h))
        cx2, cy2 = int(min(raw["width"], x2 + 1.5 * w)), int(min(raw["height"], y2 + 1.5 * h))
        crop = img[cy1:cy2, cx1:cx2].copy()
        cv2.rectangle(crop, (int(x1 - cx1), int(y1 - cy1)), (int(x2 - cx1), int(y2 - cy1)), (0, 255, 255), 1)
        crop = cv2.resize(crop, (240, 160))
        masked = sum(1 for k in t if _coverage(k.box, kept_rects.get(k.f, [])) >= COVERED)
        fps = raw["fps"]
        lines = [f"#{i}  {t[0].f / fps:.1f}-{t[-1].f / fps:.1f}s  n={len(t)}",
                 f"max {best.conf:.2f}  {w:.0f}x{h:.0f}  masked {masked}/{len(t)}"]
        bar = np.zeros((40, 240, 3), np.uint8)
        for j, s in enumerate(lines):
            cv2.putText(bar, s, (4, 15 + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(np.vstack([crop, bar]))
    if not tiles:
        return 0
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    for part, start in enumerate(range(0, len(rows), 6)):  # 36 candidates per image, so each stays readable
        cv2.imwrite(str(out.with_name(f"{out.stem}_{part}{out.suffix}")), np.vstack(rows[start:start + 6]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    return len(cands)


def write_labels(clip: str, raw: dict, short: str, real: list[int]) -> Path:
    cands = candidates(raw, CATS[short])
    objs = [{"keyframes": [{"f": k.f, "box": list(k.box)} for k in cands[i]]} for i in real]
    p = label_path(clip, short)
    p.write_text(json.dumps({"clip": Path(clip).name, "category": CATS[short], "source": "bench/review.py candidates, "
                             "checked by eye", "objects": objs}, indent=1))
    return p


def load_labels(clip: str, short: str) -> list[dict[int, Box]]:
    """Each labeled object as frame -> box, with gaps of up to 15 frames between detections interpolated."""
    p = label_path(clip, short)
    if not p.exists():
        sys.exit(f"no labels at {p}: run `candidates`, look at the sheet, then `label --real ...`")
    objs = []
    for o in json.loads(p.read_text())["objects"]:
        kfs = sorted(o["keyframes"], key=lambda k: k["f"])
        boxes: dict[int, Box] = {}
        for a, b in zip(kfs, kfs[1:] + [None]):
            boxes[a["f"]] = tuple(a["box"])
            if b and b["f"] - a["f"] <= 15:
                for f in range(a["f"] + 1, b["f"]):
                    t = (f - a["f"]) / (b["f"] - a["f"])
                    boxes[f] = tuple(a["box"][i] + (b["box"][i] - a["box"][i]) * t for i in range(4))
        objs.append(boxes)
    return objs


# ---- scoring -----------------------------------------------------------------------------------------------

def _coverage(box: Box, rects: list[tuple]) -> float:
    """Fraction of box covered by the union of rects (a pixel grid over the box)."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    if x2 <= x1 or y2 <= y1:
        return 1.0
    grid = np.zeros((y2 - y1, x2 - x1), bool)
    for rx1, ry1, rx2, ry2 in rects:
        ax1, ay1, ax2, ay2 = max(rx1, x1), max(ry1, y1), min(rx2, x2), min(ry2, y2)
        if ax2 > ax1 and ay2 > ay1:
            grid[ay1 - y1:ay2 - y1, ax1 - x1:ax2 - x1] = True
    return float(grid.mean())


def score(raw: dict, cat: str, labels: list[dict[int, Box]], **rule_kw) -> dict:
    """Per labeled object: frames it's on screen, frames masked, longest visible run. Plus false masks: mask
    rectangles that touch no labeled object, per frame."""
    kept, dropped = app_tracks(raw, cat, **rule_kw)
    rects = mask_rects(raw, kept)
    fps, per_obj = raw["fps"], []
    for boxes in labels:
        fs = sorted(boxes)
        masked = [_coverage(boxes[f], rects.get(f, [])) >= COVERED for f in fs]
        run = best = 0
        for m in masked:
            run = 0 if m else run + 1
            best = max(best, run)
        per_obj.append({"from_s": round(fs[0] / fps, 2), "to_s": round(fs[-1] / fps, 2), "frames": len(fs),
                        "masked": round(sum(masked) / len(fs), 3), "longest_visible_s": round(best / fps, 2)})
    false = 0
    over = []  # per frame: masked area outside every labeled box, as a fraction of the frame (on a 1/8 grid)
    gw, gh = raw["width"] // 8, raw["height"] // 8
    for f in range(raw["n_frames"]):
        rs = rects.get(f, [])
        false += sum(1 for r in rs if not any(_overlap(r, b[f]) for b in labels if f in b))
        m = np.zeros((gh, gw), bool)
        for x1, y1, x2, y2 in rs:
            m[y1 // 8:-(-y2 // 8), x1 // 8:-(-x2 // 8)] = True
        for b in labels:
            if f in b:
                x1, y1, x2, y2 = (int(v) for v in b[f])
                m[max(0, y1) // 8:-(-y2 // 8), max(0, x1) // 8:-(-x2 // 8)] = False
        over.append(m.mean())
    total = sum(o["frames"] for o in per_obj)
    return {"objects": per_obj, "frames_masked": round(sum(o["masked"] * o["frames"] for o in per_obj) / total, 3)
            if total else None, "false_mask_frames": false, "over_blur_mean": round(float(np.mean(over)), 4),
            "over_blur_max": round(float(np.max(over)), 3), "tracks_kept": len(kept), "tracks_dropped": len(dropped)}


def _overlap(r: tuple, b: Box) -> bool:
    return min(r[2], b[2]) > max(r[0], b[0]) and min(r[3], b[3]) > max(r[1], b[1])


# ---- frame overlays ----------------------------------------------------------------------------------------

def frame_sheet(clip: str, raw: dict, short: str, fs: list[int], out: Path) -> None:
    cat = CATS[short]
    kept, _ = app_tracks(raw, cat)
    rects = mask_rects(raw, kept)
    labels = load_labels(clip, short) if label_path(clip, short).exists() else []
    tiles = []
    for f in fs:
        img = read_frame(clip, f)
        over = img.copy()
        for r in rects.get(f, []):
            cv2.rectangle(over, r[:2], r[2:], (255, 120, 0), -1)
        img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
        for b, s in dets_at(raw, cat, f, RAW_CONF):
            color = (0, 200, 0) if s >= 0.5 else (0, 165, 255) if s >= sp.EGOBLUR_CONF else (0, 0, 255)
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
            cv2.putText(img, f"{s:.2f}", (int(b[0]), int(b[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for o in labels:
            if f in o:
                b = o[f]
                cv2.rectangle(img, (int(b[0]) - 3, int(b[1]) - 3), (int(b[2]) + 3, int(b[3]) + 3), (255, 255, 255), 1)
        cv2.putText(img, f"frame {f} ({f / raw['fps']:.1f}s)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        tiles.append(cv2.resize(img, (960, int(960 * img.shape[0] / img.shape[1]))))
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("detect", "candidates", "label", "score", "frames"))
    ap.add_argument("clip")
    ap.add_argument("--cat", choices=tuple(CATS), default="plate")
    ap.add_argument("--real", type=int, nargs="+", help="label: candidate numbers that are real objects")
    ap.add_argument("--frames", type=int, nargs="+", help="frames: which frames to draw")
    ap.add_argument("--min-len", type=int, default=1, help="candidates: only tracks seen in this many frames")
    ap.add_argument("--tag", default="", help="a separate detection cache, e.g. gen2 with CENSOR_EGOBLUR_LP set")
    a = ap.parse_args()
    global TAG
    TAG = a.tag
    cat = CATS[a.cat]
    if a.cmd == "detect":
        d = detect(a.clip)
        print({c: sum(len(v) for v in per.values()) for c, per in d["dets"].items()}, "detections")
        return
    raw = load_raw(a.clip)
    if a.cmd == "candidates":
        n = candidate_sheet(a.clip, raw, cat, out_dir(a.clip) / f"candidates_{a.cat}.jpg", a.min_len)
        print(f"{n} candidates -> {out_dir(a.clip)}/candidates_{a.cat}_*.jpg")
    elif a.cmd == "label":
        print(f"wrote {write_labels(a.clip, raw, a.cat, a.real or [])}")
    elif a.cmd == "score":
        print(json.dumps(score(raw, cat, load_labels(a.clip, a.cat)), indent=1))
    elif a.cmd == "frames":
        out = out_dir(a.clip) / f"frames_{a.cat}.jpg"
        frame_sheet(a.clip, raw, a.cat, a.frames or [0], out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
