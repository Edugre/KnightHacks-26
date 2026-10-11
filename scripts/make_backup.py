"""Backup demo material: blurred exports of the real clips, made with the real detectors on this machine.

  python scripts/make_backup.py                       # the default set below -> data/backup/
  python scripts/make_backup.py --only street_plates

For the demo's fallback (docs/DEMO.md): if the live app or the server fails, show these. Each run loads the
clip, sends the request through the agent (the template path: no language model, so attribute questions answer
"unsure" and those people stay masked), renders the export, and writes:
  data/backup/<clip>__<request>.mp4        the blurred video
  data/backup/<clip>__<request>.jpg        before / after at three moments
  data/backup/README.md                    what each file is, with the agent's message and timings
Runs wherever the detectors run (a CUDA GPU or the R9700; slower on CPU). data/ is gitignored: share the folder
by Drive, not git.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from censor.agent.agent import Agent  # noqa: E402
from censor.agent.tools import Session  # noqa: E402
from censor.core.video import read_frame  # noqa: E402
from censor.render.render import render_video  # noqa: E402

DEFAULT = [
    ("street_plates", "blur all faces and plates"),
    ("street_walk", "blur all faces and plates"),
    ("street_walk", "pixelate everyone"),
    ("street_talk", "blur all faces"),
]
OUT = ROOT / "data" / "backup"


def side_by_side(src: str, out: str, n_frames: int, path: Path) -> None:
    rows = []
    for f in (n_frames // 6, n_frames // 2, 5 * n_frames // 6):
        a, b = read_frame(src, f), read_frame(out, f)
        w = 800
        a, b = (cv2.resize(x, (w, int(w * x.shape[0] / x.shape[1]))) for x in (a, b))
        for img, text in ((a, f"original, frame {f}"), (b, "blurred export")):
            cv2.putText(img, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        rows.append(np.hstack([a, np.full((a.shape[0], 6, 3), 255, np.uint8), b]))
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", help="clip ids to run, e.g. street_plates")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Backup demo material", "",
             "Made by `scripts/make_backup.py` with the real detectors (EgoBlur for faces and plates, YOLOE for people) "
             "and the template agent (no language model: person questions answer \"unsure\", so those people stay masked).", ""]
    for clip, text in DEFAULT:
        if a.only and clip not in a.only:
            continue
        src = ROOT / "bench" / "clips" / f"{clip}.mp4"
        if not src.exists():
            print(f"skip {clip}: {src} missing (bash scripts/fetch_clips.sh)", file=sys.stderr)
            continue
        name = f"{clip}__{re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')}"
        work = OUT / "work" / name
        shutil.rmtree(work, ignore_errors=True)
        t0 = time.perf_counter()
        s = Session(src, work)
        r = Agent(s).run(text)
        t1 = time.perf_counter()
        out = OUT / f"{name}.mp4"
        stats = render_video(str(src), s.timeline, str(out))
        t2 = time.perf_counter()
        side_by_side(str(src), str(out), s.info.n_frames, OUT / f"{name}.jpg")
        print(f"{name}: {r['message']} request {t1 - t0:.0f} s, render {t2 - t1:.0f} s")
        lines += [f"## {clip}: \"{text}\"", "",
                  f"- `{name}.mp4`, before/after stills in `{name}.jpg`",
                  f"- Agent: {r['message']} ({r['path']} path)",
                  f"- Masked: {r['summary']['by_label']}; {stats.masked_boxes} masked boxes over {stats.frames} frames, "
                  f"{len(stats.failed_frames)} failed",
                  f"- Time on this machine: scan + request {t1 - t0:.0f} s, render {t2 - t1:.0f} s", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
