"""PP4AV image benchmark: face and license-plate recall on public labels, no hand labeling.

PP4AV (Trinh et al., "PP4AV: A Benchmarking Dataset for Privacy-Preserving Autonomous Driving", WACV 2023,
https://github.com/khaclinh/pp4av) has 3,447 driving images from European cities, day and night, with face
(class 0) and plate (class 1) boxes in YOLO format. License: CC BY-NC-ND 4.0. Using it to evaluate this
non-commercial project is fine, but the images, the labels and any converted copy stay out of git: they live
in ~/datasets/pp4av/ (outside the repo; CENSOR_PP4AV_DIR overrides). Only this script and its numbers are committed.

Frames are scored one by one with what the renderer would draw for a single detected frame: each detection
padded by 0.15 (doubled below the uncertain threshold), then a labeled box is covered when >= 90% of it is
inside the union of those rects, the same rule as bench/run_bench.py. Tracking isn't tested here; the
video clips cover that.

GPU server:
  python3 -m bench.pp4av download                     # 4.5 GB from Hugging Face into ~/datasets/pp4av/ (needs network)
  python3 -m bench.pp4av run [--limit 300] [--subsets paris zurich] [--imgsz 1280]
Writes bench/report_pp4av.md and bench/report_pp4av.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.run_bench import COVER_MIN, _clip_box, _git, _mean, coverage, over_blur  # noqa: E402
from censor.render.render import _padded  # noqa: E402
from censor.vision import detect as detect_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# One copy per machine, outside any clone, so everyone sharing the server account uses the same 4.5 GB.
DATA = Path(os.environ.get("CENSOR_PP4AV_DIR", Path.home() / "datasets" / "pp4av"))
REPO = "khaclinh/pp4av"
CLASSES = {0: "face", 1: "license plate"}
PADDING = 0.15  # Op default in censor/core/timeline.py
SIZE_BUCKETS = ((0, 16), (16, 32), (32, 64), (64, 10**6))  # by labeled box height in pixels
IMAGE_EXT = (".png", ".jpg", ".jpeg")

Box = tuple[float, float, float, float]
Detector = Callable[[np.ndarray], list[tuple[str, Box, float]]]  # image -> [(label, xyxy, conf)]


def download(dest: Path = DATA) -> None:
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        sys.exit("HF_HUB_OFFLINE=1 is set; run the download outside ./run.sh")
    from huggingface_hub import hf_hub_download
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("annotations.zip", "images.zip"):
        z = hf_hub_download(REPO, f"data/{name}", repo_type="dataset")
        print(f"[pp4av] extracting {name}", file=sys.stderr)
        with zipfile.ZipFile(z) as f:
            f.extractall(dest / name.removesuffix(".zip"))
    print(f"[pp4av] ready in {dest}")


def load_samples(root: Path = DATA) -> list[dict]:
    """Pair every label file with its image by subset folder and file stem."""
    images = {(p.parent.name, p.stem): p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXT}
    samples = []
    for txt in sorted((root / "annotations").rglob("*.txt")):
        img = images.get((txt.parent.name, txt.stem))
        if img is not None:
            rows = [line.split() for line in txt.read_text().splitlines() if line.strip()]
            samples.append({"subset": txt.parent.name, "image": img,
                            "labels": [(int(r[0]), *map(float, r[1:5])) for r in rows]})
    return samples


def to_pixels(label: tuple, w: int, h: int) -> tuple[str, Box]:
    cls, cx, cy, bw, bh = label
    return CLASSES[cls], ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)


def yoloe_detector(prompts: dict[str, list[str]], imgsz: int = detect_mod.DETECT_IMGSZ,
                   conf: float = detect_mod.MIN_CONF, device=0) -> Detector:
    """YOLOE loaded once. `prompts` maps each category to the text prompts that count as it."""
    from ultralytics import YOLOE
    queries = [q for qs in prompts.values() for q in qs]
    category = {q: c for c, qs in prompts.items() for q in qs}
    model = YOLOE("yoloe-11s-seg.pt")
    model.set_classes(queries, model.get_text_pe(queries))

    def detect(img: np.ndarray) -> list[tuple[str, Box, float]]:
        r = model.predict(img, device=device, imgsz=imgsz, conf=conf, verbose=False)[0]
        return [(category[queries[int(c)]], tuple(b), float(s))
                for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
    return detect


EGOBLUR_DIR = Path(os.environ.get("CENSOR_EGOBLUR_DIR", Path.home() / "datasets" / "egoblur"))


def egoblur_detector(conf: float, model_dir: Path = EGOBLUR_DIR, device: str = "cuda", nms_iou: float = 0.3) -> Detector:
    """Meta's EgoBlur Gen1 face and plate models (Apache 2.0), called the way its demo script does:
    a BGR uint8 CHW tensor in, (boxes, labels, scores, dims) out, then NMS and a score threshold."""
    import torch
    import torchvision
    models = {}
    for cat, name in (("face", "ego_blur_face_gen1.jit"), ("license plate", "ego_blur_lp_gen1.jit")):
        m = torch.jit.load(str(model_dir / name), map_location="cpu").to(device)
        m.eval()
        models[cat] = m

    def detect(img: np.ndarray) -> list[tuple[str, Box, float]]:
        t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).to(device)
        out = []
        with torch.no_grad():
            for cat, m in models.items():
                boxes, _, scores, _ = m(t)
                keep = torchvision.ops.nms(boxes, scores, nms_iou)
                out += [(cat, tuple(b), float(s)) for b, s in zip(boxes[keep].tolist(), scores[keep].tolist()) if s >= conf]
        return out
    return detect


def write_debug(samples: list[dict], detector: Detector, out: Path, n: int) -> None:
    """Overlay labeled boxes (green) and detections (red, with confidence) on the n most crowded images."""
    out.mkdir(parents=True, exist_ok=True)
    for s in sorted(samples, key=lambda s: -len(s["labels"]))[:n]:
        img = cv2.imread(str(s["image"]))
        h, w = img.shape[:2]
        dets = detector(img)
        for label in s["labels"]:
            _, (x1, y1, x2, y2) = to_pixels(label, w, h)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        for cat, (x1, y1, x2, y2), c in dets:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(img, f"{cat} {c:.2f}", (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(out / f"{s['subset']}_{s['image'].stem}.jpg"), img)
    print(f"[pp4av] wrote {min(n, len(samples))} debug images to {out}", file=sys.stderr)


def mask_rects(dets: Iterable[tuple[str, Box, float]], w: int, h: int,
               padding: Optional[dict[str, float]] = None) -> list[tuple[int, int, int, int]]:
    """What the renderer draws for one frame: padding per category, doubled for uncertain detections (fail closed)."""
    pad = lambda cat: (padding or {}).get(cat, PADDING)  # noqa: E731
    return [_padded(box, pad(cat) * (2 if conf < detect_mod.UNCERTAIN_BELOW else 1), w, h) for cat, box, conf in dets]


def evaluate(samples: list[dict], detector: Detector, padding: Optional[dict[str, float]] = None) -> dict:
    hits: dict[str, list[bool]] = {}
    by_subset: dict[str, dict[str, list[bool]]] = {}
    by_size: dict[str, dict[str, list[bool]]] = {}
    cover: dict[str, list[float]] = {}  # raw coverage per labeled box: tells a missed object from a box that's too tight
    over, seconds = [], []
    for s in samples:
        img = cv2.imread(str(s["image"]))
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        rects = mask_rects(detector(img), w, h, padding)
        seconds.append(time.perf_counter() - t0)
        boxes = []
        for label in s["labels"]:
            cat, pixels = to_pixels(label, w, h)
            box = _clip_box(pixels, w, h)
            if box is None:
                continue
            frac = coverage(box, rects)
            cover.setdefault(cat, []).append(frac)
            ok = frac >= COVER_MIN
            bucket = next(f"{lo}-{hi}px" if hi < 10**6 else f">={lo}px" for lo, hi in SIZE_BUCKETS if lo <= box[3] - box[1] < hi)
            for d in (hits, by_subset.setdefault(s["subset"], {}), by_size.setdefault(bucket, {})):
                d.setdefault(cat, []).append(ok)
            boxes.append(box)
        over.append(over_blur(rects, boxes, w, h))
    summary = lambda d: {c: {"recall": _mean(v), "boxes": len(v)} for c, v in sorted(d.items())}  # noqa: E731
    return {"images": len(samples), "recall": summary(hits),
            "by_subset": {k: summary(v) for k, v in sorted(by_subset.items())},
            "by_size": {k: summary(v) for k, v in by_size.items()},
            "partial": {c: {"half_covered": _mean([v >= 0.5 for v in vs]), "touched": _mean([v > 0 for v in vs])}
                        for c, vs in sorted(cover.items())},
            "over_blur": _mean(over), "ms_per_image": 1000 * (_mean(seconds) or 0)}


def to_markdown(rep: dict) -> str:
    f = lambda v, fmt="{:.2f}": "n/a" if v is None else fmt.format(v)  # noqa: E731
    cell = lambda d, c: f"{f(d[c]['recall'])} ({d[c]['boxes']})" if c in d else "n/a"  # noqa: E731
    m = rep["meta"]
    L = ["# PP4AV face and plate recall", "",
         f"Generated {m['generated']} by `bench/pp4av.py` at commit `{m['commit'] or 'unknown'}`. Do not edit by hand.", "",
         f"- Data: PP4AV (Trinh et al., WACV 2023, CC BY-NC-ND 4.0), {rep['images']} images. Not redistributed here.",
         f"- Detector: {m['detector']}, min conf {m['min_conf']}; padding {PADDING} (faces {m.get('face_padding', PADDING)}), doubled below conf {m['uncertain_below']}.",
         f"- A labeled box is covered when at least {COVER_MIN:.0%} of it is masked. Cells show recall (labeled boxes).", "",
         "| Category | Recall (boxes) | Target | At least 50% covered | Any overlap |", "| --- | --- | --- | --- | --- |"]
    targets = {"face": ">= 0.95", "license plate": ">= 0.90"}
    part = rep.get("partial", {})
    L += [f"| {c} | {cell(rep['recall'], c)} | {targets.get(c, '')} | {f(part.get(c, {}).get('half_covered'))} | "
          f"{f(part.get(c, {}).get('touched'))} |" for c in CLASSES.values()]
    L += ["", f"Over-blur: {f(rep['over_blur'], '{:.1%}')} of the image on average. Detector time: {rep['ms_per_image']:.1f} ms per image.",
          "", "## By subset", "", "| Subset | Faces | Plates |", "| --- | --- | --- |"]
    L += [f"| {k} | {cell(v, 'face')} | {cell(v, 'license plate')} |" for k, v in rep["by_subset"].items()]
    L += ["", "## By labeled box height", "", "| Height | Faces | Plates |", "| --- | --- | --- |"]
    order = [f"{lo}-{hi}px" if hi < 10**6 else f">={lo}px" for lo, hi in SIZE_BUCKETS]
    L += [f"| {k} | {cell(rep['by_size'][k], 'face')} | {cell(rep['by_size'][k], 'license plate')} |"
          for k in order if k in rep["by_size"]]
    return "\n".join(L) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="fetch PP4AV (4.5 GB) into ~/datasets/pp4av/")
    p = sub.add_parser("run", help="score the detector on PP4AV")
    p.add_argument("--limit", type=int, help="score every Nth image so about this many are used")
    p.add_argument("--subsets", nargs="+", help="only these subset folders, e.g. paris zurich")
    p.add_argument("--detector", choices=("yoloe", "egoblur"), default="yoloe",
                   help="egoblur: Meta's Gen1 face + plate models from ~/datasets/egoblur (CENSOR_EGOBLUR_DIR)")
    p.add_argument("--imgsz", type=int, default=detect_mod.DETECT_IMGSZ)
    p.add_argument("--conf", type=float, default=detect_mod.MIN_CONF, help="minimum detection confidence")
    p.add_argument("--face-prompts", nargs="+", default=["face"], help="YOLOE text prompts that count as faces")
    p.add_argument("--plate-prompts", nargs="+", default=["license plate"], help="YOLOE text prompts that count as plates")
    p.add_argument("--face-padding", type=float, default=PADDING,
                   help="padding per side around face detections (labels often box the whole head)")
    p.add_argument("--debug", type=int, default=0, metavar="N", help="also write N overlay images to data/pp4av_debug/")
    p.add_argument("--out-dir", default=str(ROOT / "bench"))
    a = ap.parse_args(argv)
    if a.cmd == "download":
        download()
        return 0

    from censor import netguard
    netguard.install_from_env()
    samples = load_samples()
    if not samples:
        sys.exit(f"no PP4AV images with labels under {DATA}; run: python3 -m bench.pp4av download")
    if a.subsets:
        samples = [s for s in samples if s["subset"] in a.subsets]
    if a.limit and len(samples) > a.limit:
        samples = samples[::max(1, len(samples) // a.limit)]
    print(f"[pp4av] scoring {len(samples)} images", file=sys.stderr)
    prompts = {"face": a.face_prompts, "license plate": a.plate_prompts}
    if a.detector == "egoblur":
        detector, described = egoblur_detector(a.conf), f"EgoBlur Gen1 face + plate ({EGOBLUR_DIR})"
    else:
        detector, described = yoloe_detector(prompts, a.imgsz, a.conf), f"YOLOE-11s-seg, prompts {prompts}, imgsz {a.imgsz}"
    if a.debug:
        write_debug(samples, detector, ROOT / "data" / "pp4av_debug", a.debug)
    rep = evaluate(samples, detector, {"face": a.face_padding})
    rep["meta"] = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": _git("rev-parse", "--short", "HEAD"),
                   "detector": described, "imgsz": a.imgsz,
                   "min_conf": a.conf, "face_padding": a.face_padding, "uncertain_below": detect_mod.UNCERTAIN_BELOW, "subsets": a.subsets, "limit": a.limit}
    out = Path(a.out_dir)
    (out / "report_pp4av.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    md = to_markdown(rep)
    (out / "report_pp4av.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
