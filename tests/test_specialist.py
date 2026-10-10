"""Router and EgoBlur tracking with fake models standing in for the 420 MB TorchScript files. No GPU."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from censor.core.tracks import TrackStore
from censor.vision import specialist as sp
from censor.vision.detect import StubDetector

ROOT = Path(__file__).resolve().parents[1]
needs_ffmpeg = pytest.mark.skipif(shutil.which(os.environ.get("CENSOR_FFMPEG", "ffmpeg")) is None,
                                  reason="ffmpeg not on PATH")


def test_link_follows_overlapping_boxes_and_splits_distant_ones():
    a, b = (100, 100, 140, 150), (600, 100, 640, 150)
    tracks = sp.link([(0, [(a, 0.9), (b, 0.2)]), (3, [((104, 102, 144, 152), 0.8), (b, 0.9)])], uncertain_below=0.35)
    assert len(tracks) == 2 and all(len(t) == 2 for t in tracks)
    low = next(t for t in tracks if t[0].box == b)
    assert low[0].state == "uncertain" and low[1].state == "detected"  # fail closed on the weak detection


def test_link_starts_a_new_track_after_a_long_gap():
    a = (100, 100, 140, 150)
    assert len(sp.link([(0, [(a, 0.9)]), (sp.LINK_GAP + 1, [(a, 0.9)])], 0.35)) == 2


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("router") / "c.mp4"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_clip.py"), str(out), "1"], check=True)
    return out


def fake_egoblur(calls):
    def face(img):
        calls.append("face")
        return [((855, 90, 945, 210), 0.9), ((10, 10, 20, 20), 0.05)]  # second one is below the threshold
    def plate(img):
        calls.append("plate")
        return []
    return {"face": face, "license plate": plate}


@needs_ffmpeg
def test_egoblur_detector_tracks_widened_faces(clip):
    calls = []
    store = TrackStore()
    found = sp.EgoBlurDetector(infer=fake_egoblur(calls), every=3).run(str(clip), ["faces"], store)
    assert len(found) == 1 and found[0].label == "faces"
    assert [k.f for k in found[0].keyframes] == list(range(0, 30, 3))
    x1, y1, x2, y2 = found[0].keyframes[0].box
    assert x2 - x1 > 90 and y2 - y1 > 120  # widened past EgoBlur's tight face box
    assert set(calls) == {"face"}  # the plate model never runs for a face query


@needs_ffmpeg
def test_router_sends_faces_to_egoblur_and_the_rest_to_yoloe(clip):
    calls = []
    router = sp.RoutedDetector(StubDetector(), sp.EgoBlurDetector(infer=fake_egoblur(calls), every=3))
    store = TrackStore()
    found = router.run(str(clip), ["face", "person"], store)
    by_label = {}
    for t in found:
        by_label.setdefault(t.label, []).append(t)
    assert len(by_label["face"]) == 1 and by_label["face"][0].keyframes[0].f == 0 and calls  # from EgoBlur
    assert len(by_label["person"]) == 2  # from the general detector (stub here)


def test_routed_or_falls_back_without_model_files(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "EGOBLUR_DIR", tmp_path)
    general = StubDetector()
    assert sp.routed_or(general) is general
    for name in sp.EGOBLUR_FILES.values():
        (tmp_path / name).write_bytes(b"")
    assert isinstance(sp.routed_or(general), sp.RoutedDetector)


def test_confirmation_rule():
    from censor.core.tracks import Keyframe
    kf = lambda f, c: Keyframe(f=f, box=(0, 0, 10, 10), conf=c)  # noqa: E731
    rule = (0.5, 3)
    assert not sp.confirmed([kf(0, 0.2)], rule)                      # one weak blip
    assert sp.confirmed([kf(0, 0.9)], rule)                          # one strong detection
    assert sp.confirmed([kf(0, 0.2), kf(3, 0.15), kf(6, 0.3)], rule)  # weak but steady
    assert sp.confirmed([kf(0, 0.2)], None)                          # no rule (faces)


@needs_ffmpeg
def test_weak_plate_blips_are_dropped_but_steady_and_strong_plates_stay(clip):
    calls = []

    def plate(img):
        calls.append(1)
        dets = [((100, 500, 180, 530), 0.9),     # a real plate, strongly detected every time
                ((700, 500, 760, 520), 0.15)]    # a distant plate, weak but steady
        if len(calls) == 1:
            dets.append(((1000, 100, 1040, 115), 0.2))  # a sign mistaken for a plate, once
        return dets

    det = sp.EgoBlurDetector(infer={"face": lambda img: [], "license plate": plate}, every=3)
    found = det.run(str(clip), ["license plate"], TrackStore())
    kept = sorted(round(t.keyframes[0].box[0]) for t in found)
    assert kept == [100, 700] and det.dropped == {"license plate": 1}
    weak = next(t for t in found if t.keyframes[0].box[0] == 700)
    assert all(k.state == "uncertain" for k in weak.keyframes)  # kept, and still padded extra
