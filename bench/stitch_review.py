"""Try the person-track stitcher on cached tracks and show every join it makes, to check by eye.

  python -m bench.compare_people <clips> --models yoloe-11s-seg.pt --no-half --save   # once: caches tracks
  python -m bench.stitch_review bench/clips/street_walk.mp4 [more clips]

For each clip: person tracks and pieces under 1 s, before and after stitching; and a sheet
data/review/people/joins_<clip>.jpg with one tile per join: the end of piece A next to the start of piece B,
the gap in frames and the cost. A join of two different people is the failure to look for.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from censor.core.tracks import Keyframe  # noqa: E402
from censor.core.video import probe, read_frame  # noqa: E402
from censor.vision import stitch as st  # noqa: E402

PEOPLE = ROOT / "data" / "review" / "people"


def load(clip: str) -> list[list[Keyframe]]:
    p = PEOPLE / f"{Path(clip).stem}_yoloe-11s-seg.json"
    if not p.exists():
        sys.exit(f"no cached tracks for {clip}: run bench.compare_people with --save first")
    raw = json.loads(p.read_text())
    return [[Keyframe(f=f, box=tuple(b), conf=min(c, 1.0)) for f, b, c in v] for v in raw.values()]


def ends_for(clip: str, tracks: list[list[Keyframe]], n: int = 3) -> list[st.Ends]:
    frames: dict[int, np.ndarray] = {}

    def img(f):
        if f not in frames:
            frames[f] = read_frame(clip, f)
        return frames[f]

    def mean_hist(kfs):
        return np.mean([st.color_hist(img(k.f), k.box) for k in kfs], axis=0).astype(np.float32)
    return [st.Ends(start=mean_hist(t[:n]), end=mean_hist(t[-n:])) for t in tracks]


def tile(clip, a_kf, b_kf, text):
    def crop(k):
        x1, y1, x2, y2 = (int(max(0, v)) for v in k.box)
        c = read_frame(clip, k.f)[y1:y2, x1:x2]
        t = np.zeros((200, 100, 3), np.uint8)
        if c.size:
            s = min(100 / c.shape[1], 200 / c.shape[0])
            c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))))
            t[:c.shape[0], :c.shape[1]] = c
        return t
    t = np.hstack([crop(a_kf), np.full((200, 4, 3), 255, np.uint8), crop(b_kf)])
    bar = np.zeros((20, t.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return np.vstack([t, bar])


def main() -> None:
    for clip in sys.argv[1:]:
        tracks = load(clip)
        ends = ends_for(clip, tracks)
        fps = probe(clip).fps
        out = st.stitch(tracks, ends)
        short = lambda ts: sum(1 for t in ts if (t[-1].f - t[0].f + 1) / fps < 1.0)  # noqa: E731
        # every join: piece j continues piece i
        joins = []
        for i, a in enumerate(tracks):
            for j, b in enumerate(tracks):
                if i != j:
                    c = st.link_cost(a, b, ends[i], ends[j])
                    if c is not None:
                        joins.append((c, i, j))
        nxt, prv, picked = {}, {}, []
        for c, i, j in sorted(joins):
            if i in nxt or j in prv or st._same_chain(i, j, nxt, prv):
                continue
            nxt[i], prv[j] = j, i
            picked.append((c, i, j))
        print(json.dumps({"clip": Path(clip).stem, "tracks before": len(tracks), "under 1 s before": short(tracks),
                          "tracks after": len(out), "under 1 s after": short(out), "joins": len(picked)}))
        tiles = [tile(clip, tracks[i][-1], tracks[j][0], f"{i}->{j} gap {tracks[j][0].f - tracks[i][-1].f} cost {c:.2f}")
                 for c, i, j in picked]
        if tiles:
            cols = 8
            while len(tiles) % cols:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(PEOPLE / f"joins_{Path(clip).stem}.jpg"),
                        np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]))


if __name__ == "__main__":
    main()
