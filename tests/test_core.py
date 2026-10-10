"""Core tests: schema examples, fold/undo, fail-closed renderer rules, intents. No GPU needed."""
from pathlib import Path

import numpy as np
import pytest

from censor.agent import intents
from censor.core.timeline import Op, OpLog, Timeline, VideoInfo, fold
from censor.core.tracks import Keyframe, Track, TrackStore
from censor.render.render import HOLD_FRAMES, MAX_GAP, FrameMasker, expand_target

ROOT = Path(__file__).resolve().parents[1]
VIDEO = VideoInfo(fps=30, width=1280, height=720, n_frames=180)


def store_with(*tracks):
    s = TrackStore()
    for t in tracks:
        s.add(t)
    return s


def trk(tid, frames, label="face", state="detected"):
    return Track(track_id=tid, label=label, keyframes=[Keyframe(f=f, box=(100, 100, 200, 200), conf=0.9, state=state) for f in frames])


@pytest.mark.parametrize("name", ["plate_moving.json", "face_and_box.json"])
def test_examples_validate(name):
    tl = Timeline.load(ROOT / "schema" / "examples" / name)
    assert tl.version == 1 and tl.targets


def test_ops_example_validates():
    log = OpLog.model_validate_json((ROOT / "schema" / "examples" / "ops.json").read_text())
    assert [o.op_id for o in log.ops] == ["op_1", "op_2", "op_3"]


def test_empty_targets_rejected():
    with pytest.raises(ValueError):
        Op(op_id="op_1", kind="add_targets", targets=[])


def test_fold_add_remove_undo_and_time_range():
    s = store_with(trk("trk_1", range(0, 90)), trk("trk_2", range(0, 90), label="license plate"))
    log = OpLog()
    log.append(Op(op_id="op_1", kind="add_targets", targets=["trk_1"]))
    log.append(Op(op_id="op_2", kind="add_targets", targets=["trk_2"], time_range=(1.0, 2.0), effect="pixelate"))
    tl = fold(log.ops, s, VIDEO)
    assert set(tl.targets) == {"trk_1", "trk_2"}
    assert [k.f for k in tl.targets["trk_2"].keyframes] == list(range(30, 61))
    log.append(Op(op_id="op_3", kind="remove_targets", targets=["trk_1"]))
    assert set(fold(log.ops, s, VIDEO).targets) == {"trk_2"}
    log.undo()
    assert set(fold(log.ops, s, VIDEO).targets) == {"trk_1", "trk_2"}


def test_fold_unknown_track_raises():
    with pytest.raises(KeyError):
        fold([Op(op_id="op_1", kind="add_targets", targets=["nope"])], TrackStore(), VIDEO)


def test_interpolation_and_hold_after_loss():
    tl = fold([Op(op_id="op_1", kind="add_targets", targets=["t"])], store_with(trk("t", [0, 10, 100])), VIDEO)
    per = expand_target(tl.targets["t"], VIDEO.n_frames)
    assert all(f in per for f in range(0, 11))                 # gap 10 <= MAX_GAP: interpolated
    assert 10 + HOLD_FRAMES in per and 10 + HOLD_FRAMES + 1 not in per  # big gap: hold, then nothing
    assert 100 + HOLD_FRAMES in per                             # hold after the last keyframe too
    assert per[12][1] > per[11][1]                              # held boxes grow
    assert MAX_GAP < 90


def test_uncertain_doubles_padding():
    t = fold([Op(op_id="op_1", kind="add_targets", targets=["t"], padding=0.2)],
             store_with(trk("t", [5], state="uncertain")), VIDEO).targets["t"]
    assert expand_target(t, VIDEO.n_frames)[5][1] == pytest.approx(0.4)


def test_masker_changes_pixels_inside_box_only():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    orig = frame.copy()
    tl = fold([Op(op_id="op_1", kind="add_targets", targets=["t"], padding=0.0)], store_with(trk("t", [0])), VIDEO)
    FrameMasker(tl).apply(frame, 0)
    assert not np.array_equal(frame[100:200, 100:200], orig[100:200, 100:200])
    assert np.array_equal(frame[300:, 300:], orig[300:, 300:])


def test_masker_fails_closed(monkeypatch):
    import censor.render.render as r
    frame = np.random.default_rng(1).integers(0, 255, (72, 128, 3), dtype=np.uint8)
    orig = frame.copy()
    tl = fold([Op(op_id="op_1", kind="add_targets", targets=["t"])], store_with(trk("t", [0])), VIDEO)
    calls = {"n": 0}
    real = r.apply_effect

    def boom(fr, rect, effect):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated failure")
        return real(fr, rect, effect)

    monkeypatch.setattr(r, "apply_effect", boom)
    stats = r.RenderStats()
    FrameMasker(tl).apply(frame, 0, stats)
    assert stats.failed_frames == [0]
    assert not np.array_equal(frame[60:, 120:], orig[60:, 120:])  # whole frame blurred, even far from the box


@pytest.mark.parametrize("text,queries,effect,attr,time", [
    ("blur all plates", ["license plate"], "blur", None, None),
    ("Pixelate every face and screen", ["face", "screen"], "pixelate", None, None),
    ("blur all plates and the person in the red shirt around halfway", ["license plate", "person"], "blur",
     "Is this person wearing red shirt?", "halfway"),
    ("black box the screens in the first 10 seconds", ["screen"], "box", None, "first 10 seconds"),
])
def test_intents(text, queries, effect, attr, time):
    p = intents.parse(text)
    assert p.action == "mask" and p.queries == queries and p.effect == effect
    assert p.attribute_question == attr and p.time_phrase == time


def test_intents_remove_undo_unknown():
    assert intents.parse("unblur the faces").action == "remove"
    assert intents.parse("undo").action == "undo"
    assert intents.parse("make it look cinematic") is None


def test_detector_failure_is_reported_not_raised(tmp_path):
    import subprocess, sys
    from censor.agent.agent import Agent
    from censor.agent.tools import Session
    clip = tmp_path / "c.mp4"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_clip.py"), str(clip), "1"], check=True)

    class Broken:
        def run(self, *a, **k):
            raise ModuleNotFoundError("No module named 'lap'")

    s = Session(clip, tmp_path / "w", detector=Broken())
    r = Agent(s).run("blur all plates")
    assert "Detection failed" in r["message"] and "lap" in r["message"]
    assert s.summarize()["n_targets"] == 0 and s.tracks.queries == []


def test_find_objects_rejects_compound_queries(tmp_path):
    import subprocess, sys
    from censor.agent.tools import Session, ToolError
    from censor.vision.detect import StubDetector
    clip = tmp_path / "c.mp4"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_clip.py"), str(clip), "1"], check=True)
    s = Session(clip, tmp_path / "w", detector=StubDetector())
    with pytest.raises(ToolError, match="check_attribute"):
        s.find_objects("person wearing a dark jacket")
    assert s.tracks.queries == []
    assert len(s.find_objects("license plate")) == 1
