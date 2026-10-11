"""Live replay CLI: the recall scorer (pure, no GPU) and `run --stub` on a synthetic clip."""
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from censor.live_cli import main, score_report

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = os.environ.get("CENSOR_FFMPEG", "ffmpeg")

LIVE_W, LIVE_H = 1280, 720
SRC_H = 1080                 # label file is in 1080p pixels: scoring must scale by LIVE_H / SRC_H
SCALE = LIVE_H / SRC_H
FACE = [300.0, 150.0, 420.0, 300.0]
PLATE = [900.0, 800.0, 1050.0, 860.0]
LABELED = (0, 10, 20, 30)


def _labels(path: Path, extra_draft: bool = False) -> Path:
    frames = [{"f": f, "boxes": [{"gt_id": "face_1", "box": FACE}, {"gt_id": "license_plate_1", "box": PLATE},
                                 {"gt_id": "person_1", "box": [0.0, 0.0, 1900.0, 1070.0]}]} for f in LABELED]
    if extra_draft:  # an unreviewed draft frame must never be scored
        frames.append({"f": 40, "boxes": [{"gt_id": "face_1", "box": FACE}], "reviewed": False})
    doc = {"clip": "t", "video": {"fps": 30.0, "width": 1920, "height": SRC_H, "n_frames": 60}, "every": 10,
           "objects": {"face_1": {"category": "face", "desc": ""},
                       "license_plate_1": {"category": "license plate", "desc": ""},
                       "person_1": {"category": "person", "desc": ""}},
           "frames": frames}
    path.write_text(json.dumps(doc))
    return path


def _scaled(box, pad=2) -> list[int]:
    x1, y1, x2, y2 = (v * SCALE for v in box)
    return [int(x1) - pad, int(y1) - pad, math.ceil(x2) + pad, math.ceil(y2) + pad]


def _frame(f, rects=None, blind=False, clip_f=None) -> dict:
    if rects is None:
        rects = [] if blind else [_scaled(FACE), _scaled(PLATE)]
    return {"f": f, "clip_f": f if clip_f is None else clip_f, "blind": blind, "rects": rects}


def _report(path: Path, frames: list[dict]) -> Path:
    path.write_text(json.dumps({"clip": "t", "size": [LIVE_W, LIVE_H], "fps": 30.0, "n_frames": 60,
                                "delay_s": 1.0, "stats": {}, "frames": frames}))
    return path


def _full(n=40, **overrides) -> list[dict]:
    out = [_frame(f) for f in range(n)]
    for f, fr in overrides.items():
        out[int(f[1:])] = fr
    return out


@pytest.fixture()
def labels(tmp_path):
    return _labels(tmp_path / "labels.json", extra_draft=True)


# ---- score_report (pure) ------------------------------------------------------------------------

def test_score_report_counts_boxes_and_scales_label_boxes(tmp_path, labels):
    from bench.gt import LabelSet
    res = score_report(json.loads(_report(tmp_path / "r.json", _full()).read_text()), LabelSet.load(labels), LIVE_H)
    assert res["face"]["boxes"] == 4 and res["face"]["covered"] == 4 and res["face"]["recall"] == 1.0
    assert res["license plate"]["recall"] == 1.0
    assert res["blind_share"] == 0.0
    assert "person" not in res  # only the live categories are scored


def test_score_report_longest_uncovered_run_in_seconds_and_first_bare_frame(tmp_path, labels):
    from bench.gt import LabelSet
    frames = _full(f10=_frame(10, rects=[_scaled(PLATE)]), f20=_frame(20, rects=[_scaled(PLATE)]))
    res = score_report(json.loads(_report(tmp_path / "r.json", frames).read_text()), LabelSet.load(labels), LIVE_H)
    face = res["face"]
    assert face["covered"] == 2 and face["recall"] == 0.5
    assert face["failing"]["face_1"]["clip_f"] == 10
    assert face["longest_run_s"]["face_1"] == pytest.approx(20 / 30)  # frames 10..20 bare + one label step


def test_score_report_recall_is_none_without_boxes(tmp_path, labels):
    from bench.gt import LabelSet
    rep = json.loads(_report(tmp_path / "r.json", [_frame(5)]).read_text())  # no labeled frame emitted
    res = score_report(rep, LabelSet.load(labels), LIVE_H)
    assert res["face"]["boxes"] == 0 and res["face"]["recall"] is None


# ---- score (CLI) --------------------------------------------------------------------------------

def test_score_passes_and_prints_per_category_recall(tmp_path, labels, capsys):
    rep = _report(tmp_path / "r.json", _full())
    with pytest.raises(SystemExit) as e:
        main(["score", str(rep), str(labels)])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "face" in out and "license plate" in out and "recall 1.000" in out


def test_score_fails_and_names_the_bare_object_and_frame(tmp_path, labels, capsys):
    frames = _full(f20=_frame(20, rects=[_scaled(PLATE)]))  # face_1 bare on a vouched frame
    rep = _report(tmp_path / "r.json", frames)
    with pytest.raises(SystemExit) as e:
        main(["score", str(rep), str(labels)])
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "face_1" in out and "20" in out and "recall 0.750" in out


def test_score_threshold_flags_are_honoured(tmp_path, labels):
    frames = _full(f20=_frame(20, rects=[_scaled(FACE)]))  # plate recall 0.75
    rep = _report(tmp_path / "r.json", frames)
    with pytest.raises(SystemExit) as e:
        main(["score", str(rep), str(labels)])
    assert e.value.code == 1
    with pytest.raises(SystemExit) as e:
        main(["score", str(rep), str(labels), "--min-plate", "0.70"])
    assert e.value.code == 0


def test_score_exempts_blind_frames(tmp_path, labels, capsys):
    rep = _report(tmp_path / "r.json", _full(f20=_frame(20, blind=True)))
    with pytest.raises(SystemExit) as e:
        main(["score", str(rep), str(labels)])
    assert e.value.code == 0
    assert "blind" in capsys.readouterr().out


def test_score_skips_labeled_frames_that_were_never_emitted(tmp_path, labels):
    frames = [fr for fr in _full() if fr["clip_f"] != 20]  # labeled frame 20 was never emitted
    rep = _report(tmp_path / "r.json", frames)
    with pytest.raises(SystemExit) as e:
        main(["score", str(rep), str(labels)])
    assert e.value.code == 0


def test_score_ignores_unreviewed_draft_frames(tmp_path, labels):
    frames = _full(n=45, f40=_frame(40, rects=[]))  # bare, but label frame 40 is a draft
    rep = _report(tmp_path / "r.json", frames)
    with pytest.raises(SystemExit) as e:
        main(["score", str(rep), str(labels)])
    assert e.value.code == 0


def test_score_rejects_missing_files_with_exit_2(tmp_path, labels):
    with pytest.raises(SystemExit) as e:
        main(["score", str(tmp_path / "nope.json"), str(labels)])
    assert e.value.code == 2


# ---- run --stub ---------------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which(FFMPEG) is None, reason="ffmpeg not on PATH")
def test_run_stub_writes_playable_mp4_and_report(tmp_path):
    pytest.importorskip("censor.live")
    clip = tmp_path / "synthetic.mp4"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_clip.py"), str(clip), "3"], check=True)
    out, rep = tmp_path / "live.mp4", tmp_path / "live.json"
    with pytest.raises(SystemExit) as e:
        main(["run", str(clip), "--stub", "--seconds", "1.5", "--delay", "0.3",
              "--out", str(out), "--report", str(rep)])
    assert e.value.code == 0
    from censor.core.video import probe
    info = probe(str(out))
    assert info.n_frames > 0 and (info.width, info.height) == (LIVE_W, LIVE_H)
    doc = json.loads(rep.read_text())
    assert {"clip", "size", "fps", "n_frames", "delay_s", "stats", "frames"} <= set(doc)
    assert doc["size"] == [LIVE_W, LIVE_H] and doc["frames"]
    assert set(doc["frames"][0]) == {"f", "clip_f", "blind", "rects"}
    assert [fr["f"] for fr in doc["frames"]] == sorted(fr["f"] for fr in doc["frames"])


def test_run_without_egoblur_models_exits_2(tmp_path, monkeypatch, capsys):
    import censor.live_cli as cli
    monkeypatch.setattr(cli, "egoblur_available", lambda: False)
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"")
    with pytest.raises(SystemExit) as e:
        main(["run", str(clip), "--out", str(tmp_path / "o.mp4"), "--report", str(tmp_path / "o.json")])
    assert e.value.code == 2
    assert "EgoBlur" in capsys.readouterr().err
