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
    rule = {"min_conf": 0.5, "min_frames": 3}
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
                ((700, 500, 760, 520), 0.32),    # a distant plate, weak but steady
                ((1000, 300, 1060, 320), 0.15)]  # a plate-shaped sign, faint in every frame
        if len(calls) == 1:
            dets.append(((1000, 100, 1040, 115), 0.2))  # a sign mistaken for a plate, once
        return dets

    det = sp.EgoBlurDetector(infer={"face": lambda img: [], "license plate": plate}, every=3)
    found = det.run(str(clip), ["license plate"], TrackStore())
    kept = sorted(round(t.keyframes[0].box[0]) for t in found)
    assert kept == [100, 700] and det.dropped == {"license plate": 2}
    weak = next(t for t in found if t.keyframes[0].box[0] == 700)
    assert all(k.state == "uncertain" for k in weak.keyframes)  # kept, and still padded extra


def test_weak_persistent_tracks_must_be_plate_shaped():
    from censor.core.tracks import Keyframe
    rule = sp.CONFIRM["license plate"]
    track = lambda box, c=0.32: [Keyframe(f=f, box=box, conf=c) for f in (0, 3, 6)]  # noqa: E731
    assert sp.confirmed(track((100, 500, 260, 540)), rule, 720)        # weak, steady, 4:1: a distant plate
    assert not sp.confirmed(track((100, 500, 260, 540), 0.2), rule, 720)  # faint all along (the "7 7" sign)
    assert not sp.confirmed(track((565, 56, 1150, 1009)), rule, 1080)  # weak, steady, taller than wide (a dress)
    assert not sp.confirmed(track((1051, 705, 1097, 809)), rule, 1080) # weak, steady, 0.44:1 (a sign post)
    assert not sp.confirmed(track((0, 0, 900, 400)), rule, 720)        # wide but over a quarter of the frame tall
    assert not sp.confirmed(track((565, 56, 1150, 1009), 0.9), rule, 1080, 1920)  # confident but taller than wide
    assert sp.plate_shaped((0, 0, 120, 100), 720, 1.2) and not sp.plate_shaped((0, 0, 110, 100), 720, 1.2)


def test_warm_up_runs_each_model_on_each_size():
    """The app warms EgoBlur up at startup so the first upload doesn't pay for the first slow calls."""
    seen = []
    fake = {c: (lambda img, c=c: seen.append((c, img.shape)) or []) for c in ("face", "license plate")}
    sp.EgoBlurDetector(infer=fake).warm_up(sizes=((72, 128),))
    assert sorted(seen) == [("face", (72, 128, 3))] * 2 + [("license plate", (72, 128, 3))] * 2


def test_one_confident_blip_is_not_enough():
    """On street_plates most false plate tracks were one confident detection on a tail light or sign."""
    from censor.core.tracks import Keyframe
    rule = sp.CONFIRM["license plate"]
    kf = lambda f, c: Keyframe(f=f, box=(100, 500, 260, 540), conf=c)  # noqa: E731
    assert not sp.confirmed([kf(0, 0.9)], rule, 720)
    assert sp.confirmed([kf(0, 0.9), kf(3, 0.8)], rule, 720)


def test_link_follows_a_small_fast_plate():
    """A 13 px plate moving 10 px between checked frames barely overlaps itself; it is still one track."""
    dets = [(f, [((100 + 3.3 * f, 424, 113 + 3.3 * f, 434), 0.6)]) for f in (27, 30, 33, 36)]
    assert len(sp.link(dets, 0.35)) == 1
    far = [(0, [((100, 424, 113, 434), 0.6)]), (3, [((140, 424, 153, 434), 0.6)])]  # 3 box sizes apart
    assert len(sp.link(far, 0.35)) == 2


def test_absorb_folds_a_fragment_into_the_track_it_precedes():
    from censor.core.tracks import Keyframe
    k = lambda f, x, c=0.6: Keyframe(f=f, box=(x, 0, x + 40, 20), conf=c)  # noqa: E731
    kept = [[k(9, 100), k(12, 102), k(15, 104)]]
    left = sp.absorb(kept, [[k(6, 98, 0.2)], [k(6, 600, 0.2)]])  # one fragment touches the track, one doesn't
    assert [kf.f for kf in kept[0]] == [6, 9, 12, 15] and len(left) == 1


def test_big_faint_faces_are_dropped():
    """EgoBlur's false faces were car backs: big boxes at low confidence. Real faces that big score ~1.0."""
    from censor.core.tracks import Keyframe
    rule = sp.CONFIRM["face"]
    track = lambda box, c: [Keyframe(f=f, box=box, conf=c) for f in (0, 3, 6)]  # noqa: E731
    assert not sp.confirmed(track((0, 100, 700, 900), 0.47), rule, 1080)  # 800 px "face" at 0.47: a car back
    assert sp.confirmed(track((0, 100, 300, 460), 1.0), rule, 720)        # a close-up face
    assert sp.confirmed(track((500, 300, 545, 360), 0.15), rule, 1080)    # a small faint face in a crowd stays


def test_confident_plates_must_be_wider_than_tall_except_at_the_edge_or_tiny():
    """The taxi's round sticker was a confident 131x123 "plate"; real plate tracks were 1.69:1 or wider."""
    from censor.core.tracks import Keyframe
    rule = sp.CONFIRM["license plate"]
    track = lambda box: [Keyframe(f=f, box=box, conf=0.92) for f in (0, 3, 6)]  # noqa: E731
    assert not sp.confirmed(track((339, 751, 470, 874)), rule, 1080, 1920)  # the round sticker, 1.06:1
    assert sp.confirmed(track((911, 640, 1167, 706)), rule, 1080, 1920)     # FL 320, 3.9:1
    assert sp.confirmed(track((1890, 525, 1920, 569)), rule, 1080, 1920)    # FL 320 cut off by the right edge
    assert sp.confirmed(track((102, 424, 115, 434)), rule, 720, 1280)       # a 13x10 px plate: too small to judge


def test_a_tiny_plate_seen_once_is_checked_in_the_frames_around_it():
    """An 11x7 px plate was there on 4 frames in a row but on one checked frame, so it was dropped as a blip."""
    from censor.core.tracks import Keyframe
    rule = sp.CONFIRM["license plate"]
    blip = [[Keyframe(f=27, box=(125, 395, 137, 404), conf=0.43, state="uncertain")]]
    there = {25: 0.49, 26: 0.44, 28: 0.22}
    detect_at = lambda f: [((125, 395, 137, 404), there[f])] if f in there else []  # noqa: E731
    kept, checked = sp.recheck_tiny(blip, detect_at, rule, 720, 1280, 221)
    assert checked == 4 and len(kept) == 1 and [k.f for k in kept[0]] == [25, 26, 27, 28]
    big = [[Keyframe(f=27, box=(100, 300, 220, 360), conf=0.45, state="detected")]]  # a 60 px tall blip: not rechecked
    assert sp.recheck_tiny(big, lambda f: [((100, 300, 220, 360), 0.45)], rule, 720, 1280, 221) == ([], 0)
