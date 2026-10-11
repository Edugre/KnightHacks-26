"""censor/live.py: the live replay engine. No GPU: detectors are scripted or replayed from a sidecar.

The pure masking rules (mask_boxes, BoxCache, shape rules) are deterministic. The three thread tests run the
real threads in real time for a second or two and assert invariants, never counts.
"""
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bench.run_bench import COVER_MIN, _clip_box, coverage
from censor.core.timeline import Target
from censor.core.tracks import Keyframe, Track, TrackStore
from censor.live import (EGOBLUR_CONF, FACE_EXPAND, LIVE_CATS, WINDOW_BACK, BoxCache, Detection, LiveReplay,
                         egoblur_detect, mask_boxes, plate_ok, sidecar_detect, yunet_detect)
from censor.render.render import HOLD_FRAMES, LEAD_FRAMES, expand_target

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = os.environ.get("CENSOR_FFMPEG", "ffmpeg")
W, H = 1280, 720
FACE = (600.0, 300.0, 640.0, 340.0)  # a 40 px face: passes the shape rules


def det(f: int, face=None, plate=None) -> Detection:
    dets = {}
    if face is not None:
        dets["face"] = [(face, 0.9)]
    if plate is not None:
        dets["license plate"] = [(plate, 0.9)]
    return Detection(f=f, dets=dets, took_s=0.1)


def contains(outer, inner) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def test_mask_boxes_is_none_without_a_detection_after_f():
    ready = [det(0, face=FACE), det(5, face=FACE)]
    assert mask_boxes(7, ready, W, H) is None
    assert mask_boxes(0, [], W, H) is None


def test_mask_boxes_is_none_without_a_detection_before_f():
    ready = [det(5, face=FACE), det(10, face=FACE)]
    assert mask_boxes(3, ready, W, H) is None


def test_a_gap_longer_than_link_gap_still_covers_the_frames_between_via_the_second_tracks_lead():
    # 22 frames apart: two tracks (LINK_GAP is 9). The first holds 1..15; the second leads back 21 frames.
    # hold=30 lifts the vouch limit so the lead-in is what's under test here.
    ready = [det(0, face=FACE), det(22, face=FACE)]
    for f in range(1, 22):
        boxes = mask_boxes(f, ready, W, H, hold=30)
        assert boxes and any(contains(b, FACE) for b, _, _ in boxes), f


def test_a_gap_beyond_hold_plus_one_is_not_vouched_even_with_detections_on_both_sides():
    ready = [det(0, face=FACE), det(22)]
    # the held box ends at 15, so 16..21 would be bare although both sides exist; nothing in 1..21 is vouched
    for f in range(16, 22):
        assert mask_boxes(f, ready, W, H) is None, f


def test_a_gap_of_exactly_hold_plus_one_is_vouched_and_the_held_box_covers_all_of_it():
    ready = [det(0, face=FACE), det(HOLD_FRAMES + 1)]
    for f in range(1, HOLD_FRAMES + 1):
        boxes = mask_boxes(f, ready, W, H)
        assert boxes and any(contains(b, FACE) for b, _, _ in boxes), f


def test_an_object_first_seen_at_b_is_masked_back_to_the_previous_detection():
    ready = [det(0), det(5, face=FACE)]  # no speed to extrapolate: the lead is the first box (and a union past 3)
    for f in range(1, 5):
        boxes = mask_boxes(f, ready, W, H)
        assert boxes and any(contains(b, FACE) for b, _, _ in boxes), f
    assert mask_boxes(0, ready, W, H) == []  # the previous detection itself saw nothing


def test_interpolated_frames_match_expand_target_on_the_same_target():
    a, b = (100.0, 300.0, 140.0, 340.0), (110.0, 300.0, 150.0, 340.0)  # one track: boxes overlap
    ready = [det(0, face=a), det(5, face=b)]
    target = Target(label="face", source="rule", effect="blur", padding=0.15,
                    keyframes=[Keyframe(f=0, box=a, conf=0.9), Keyframe(f=5, box=b, conf=0.9)])
    want = expand_target(target, n_frames=2 + HOLD_FRAMES + 1, lead_frames=LEAD_FRAMES)[2]
    assert mask_boxes(2, ready, W, H) == [(want[0], want[1], "blur")]


def test_held_box_covers_frames_after_the_last_sighting_until_hold_runs_out():
    ready = [det(0, face=FACE), det(5), det(20)]
    held = mask_boxes(10, ready, W, H)
    assert held and contains(held[0][0], FACE)
    assert mask_boxes(HOLD_FRAMES + 1, ready, W, H) == []  # vouched (5 and 20), nothing left to hold


def test_egoblur_detect_expands_faces_and_drops_scores_below_the_threshold():
    face, plate = (100.0, 100.0, 140.0, 160.0), (10.0, 10.0, 70.0, 30.0)
    infer = {"face": lambda img: [(face, 0.9), ((0.0, 0.0, 9.0, 9.0), EGOBLUR_CONF - 0.01)],
             "license plate": lambda img: [(plate, EGOBLUR_CONF), ((5.0, 5.0, 6.0, 6.0), 0.01)]}
    out = egoblur_detect(infer)(0, None)
    assert out["face"] == [((100 - 40 * FACE_EXPAND, 100 - 60 * FACE_EXPAND, 140 + 40 * FACE_EXPAND,
                             160 + 60 * FACE_EXPAND), 0.9)]
    assert out["license plate"] == [(plate, EGOBLUR_CONF)]


def test_sidecar_detect_scales_boxes_to_the_live_height_and_keeps_only_faces_and_plates(tmp_path):
    store = TrackStore()
    for label, box in (("face", (300.0, 300.0, 330.0, 330.0)), ("license plate", (600.0, 600.0, 660.0, 630.0)),
                       ("person", (0.0, 0.0, 99.0, 99.0))):
        store.add(Track(track_id=store.next_id(), label=label, keyframes=[Keyframe(f=0, box=box, conf=0.8)]))
    store.save(tmp_path / "t.json")
    out = sidecar_detect(tmp_path / "t.json", src_h=1080, live_h=720, n_frames=10)(0, None)
    assert set(out) == set(LIVE_CATS)
    (fbox, fscore), = out["face"]
    assert fscore == 0.8 and fbox[0] == pytest.approx((300 - 30 * FACE_EXPAND) * 2 / 3)
    (pbox, pscore), = out["license plate"]
    assert pbox == pytest.approx((400.0, 400.0, 440.0, 420.0)) and pscore == 0.8
    assert sidecar_detect(tmp_path / "t.json", 1080, 720, 10)(10, None) == out  # clip_f wraps


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    if shutil.which(FFMPEG) is None:
        pytest.skip("ffmpeg not on PATH")
    out = tmp_path_factory.mktemp("clips") / "synthetic.mp4"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_clip.py"), str(out), "3"], check=True)
    return out


def sidecar_boxes(clip: Path, clip_f: int) -> list:
    store = TrackStore.load(str(clip) + ".tracks.json")
    return [k.box for t in store.tracks.values() if t.label in LIVE_CATS for k in t.keyframes if k.f == clip_f]


def run_engine(eng: LiveReplay, seconds: float) -> None:
    eng.start()
    try:
        time.sleep(seconds)
    finally:
        eng.stop()


def test_every_emitted_frame_is_blind_or_covers_every_face_and_plate_of_its_clip_frame(clip):
    inner = sidecar_detect(Path(str(clip) + ".tracks.json"), src_h=H, live_h=H, n_frames=90)
    emitted, lock = [], threading.Lock()

    def slow_detect(clip_f, frame):
        time.sleep(0.15)  # a pass of ~4.5 frames at 30 fps
        return inner(clip_f, frame)

    def record(f, clip_f, blind, rects):
        with lock:
            emitted.append((f, clip_f, blind, rects))

    eng = LiveReplay(str(clip), slow_detect, delay_s=0.6, on_emit=record)
    run_engine(eng, 2.5)
    assert emitted and [e[0] for e in emitted] == sorted({e[0] for e in emitted})
    assert any(not blind for _, _, blind, _ in emitted), "nothing was ever vouched: the test would be vacuous"
    for f, clip_f, blind, rects in emitted:
        if blind:
            continue
        for box in sidecar_boxes(clip, clip_f):
            assert coverage(_clip_box(box, W, H), rects) >= COVER_MIN, (f, clip_f, box)


def test_a_detector_that_raises_makes_every_frame_blind_and_the_engine_keeps_emitting(clip):
    emitted = []

    def broken(clip_f, frame):
        raise RuntimeError("boom")

    eng = LiveReplay(str(clip), broken, delay_s=0.2, on_emit=lambda f, c, blind, r: emitted.append((blind, r)))
    run_engine(eng, 1.2)
    s = eng.stats()
    assert s["detector"]["errors"] > 0 and s["detector"]["passes"] == 0
    assert emitted and all(blind and rects == [] for blind, rects in emitted)
    assert s["frames_out"] == s["blind"] == len(emitted) and s["blind_share"] == 1.0


def sidecar_for(clip: Path):
    return sidecar_detect(Path(str(clip) + ".tracks.json"), src_h=H, live_h=H, n_frames=90)


def sharpness(jpeg_or_img) -> float:
    img = cv2.imdecode(np.frombuffer(jpeg_or_img, np.uint8), cv2.IMREAD_GRAYSCALE) \
        if isinstance(jpeg_or_img, bytes) else cv2.cvtColor(jpeg_or_img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def test_a_masking_error_blurs_the_whole_frame(clip, monkeypatch):
    def boom(img, rect, effect):
        raise RuntimeError("effect failed")

    monkeypatch.setattr("censor.live.apply_effect", boom)  # full_frame_blur uses render's own name: unaffected
    emitted = []
    eng = LiveReplay(str(clip), sidecar_for(clip), delay_s=0.3,
                     on_emit=lambda f, c, blind, rects: emitted.append((blind, rects)))
    eng.start()
    try:
        time.sleep(1.5)
        shown = eng.latest(-1, 1.0)
    finally:
        eng.stop()
    sighted = [rects for blind, rects in emitted if not blind]
    assert sighted, "nothing was ever vouched: the test would be vacuous"
    assert all(rects == [(0, 0, W, H)] for rects in sighted)
    assert eng.stats()["emit_errors"] >= len(sighted)
    cap = cv2.VideoCapture(str(clip))
    ok, plain = cap.read()
    cap.release()
    assert ok and shown is not None and sharpness(shown[1]) < sharpness(plain) / 10  # blurred, not just reported


def test_blind_frames_after_a_detector_death_freeze_the_last_vouched_picture(clip, monkeypatch):
    inner, t0, last_ok = sidecar_for(clip), [None], [-1]

    def dies_after_a_while(clip_f, frame):
        t0[0] = t0[0] or time.monotonic()
        if time.monotonic() - t0[0] > 0.8:
            raise RuntimeError("detector gone")
        last_ok[0] = clip_f
        return inner(clip_f, frame)

    emitted = []
    eng = LiveReplay(str(clip), dies_after_a_while, delay_s=0.3,
                     on_emit=lambda f, c, blind, rects: emitted.append((f, blind, rects)))
    eng.start()
    try:
        time.sleep(2.0)
        newest = eng.latest(-1, 1.0)
        following = eng.latest(newest[0], 1.0)  # the stream goes on
        still_running = eng.running
    finally:
        eng.stop()
    after = [(blind, rects) for f, blind, rects in emitted if f > last_ok[0]]
    assert last_ok[0] > 0 and len(after) > 5, "the detector never worked, or the run was too short"
    assert all(blind and rects == [] for blind, rects in after)
    assert still_running and eng.stats()["detector"]["errors"] > 0  # an error is caught per pass: not a thread death
    assert newest[0] > last_ok[0] and following[0] > newest[0]
    assert following[1] == newest[1]  # the same frozen picture, not a fresh unmasked one
    badge = cv2.imdecode(np.frombuffer(newest[1], np.uint8), cv2.IMREAD_COLOR)
    assert abs(float(badge[-14:-10, 6:10].mean()) - 110) < 20  # the grey DETECTOR LATE plate, bottom left
    thread_death_stops_the_engine(clip, monkeypatch, "_record_pass")


def thread_death_stops_the_engine(clip, monkeypatch, name: str) -> None:
    def die(*args, **kwargs):
        raise RuntimeError(f"{name} died")

    monkeypatch.setattr(LiveReplay, name, die)
    eng = LiveReplay(str(clip), sidecar_for(clip), delay_s=0.2)
    eng.start()
    try:
        deadline = time.monotonic() + 2.0
        while eng.running and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not eng.running and eng.stats()["phase"] == "idle", f"{name}: a dead thread left the engine 'running'"
    finally:
        eng.stop()


@pytest.mark.parametrize("name", ["_pump", "_finish"])  # a dead source thread, a dead emitter thread
def test_a_dead_source_or_emitter_thread_makes_running_false(clip, monkeypatch, name):
    thread_death_stops_the_engine(clip, monkeypatch, name)


def test_a_slow_on_emit_evicts_frames_and_counts_dropped_out(clip):
    emitted = []

    def slow_emit(f, clip_f, blind, rects):
        emitted.append(f)
        time.sleep(0.15)  # 4.5 frames of the 30 fps clip per emit: the ring (23 deep) must evict

    eng = LiveReplay(str(clip), sidecar_for(clip), delay_s=0.2, on_emit=slow_emit)
    eng.start()
    try:
        time.sleep(2.0)
        still_running = eng.running
    finally:
        eng.stop()
    s = eng.stats()
    assert still_running and s["frames_out"] > 0 and s["dropped_out"] > 0 and s["emit_errors"] == 0
    assert emitted == sorted(set(emitted))
    assert s["frames_out"] + s["dropped_out"] == emitted[-1] + 1  # every frame up to the last sent is sent or dropped


def test_stop_returns_promptly_and_wakes_a_pending_latest(clip):
    eng = LiveReplay(str(clip), lambda c, f: {}, delay_s=5.0)  # nothing is emitted for 5 s
    result = []
    eng.start()
    waiter = threading.Thread(target=lambda: result.append(eng.latest(-1, 10.0)))
    waiter.start()
    time.sleep(0.2)
    t0 = time.monotonic()
    eng.stop()
    waiter.join(timeout=2)
    assert time.monotonic() - t0 < 2.0 and not waiter.is_alive()
    assert result == [None]
    assert eng.stats()["phase"] == "idle" and eng.stats()["running"] is False


def test_latest_returns_the_first_newer_frame_none_on_timeout_and_the_newest_when_the_client_is_ahead(clip):
    eng = LiveReplay(str(clip), lambda c, f: {})  # never started: frames are published by hand
    assert eng.latest(-1, 0.05) is None
    eng._publish(7, b"seven")
    eng._publish(8, b"eight")
    assert eng.latest(-1, 0.2) == (7, b"seven")
    assert eng.latest(7, 0.2) == (8, b"eight")
    assert eng.latest(8, 0.2) is None
    assert eng.latest(10 ** 9, 0.2) == (8, b"eight")


def test_stats_before_any_pass_has_the_documented_keys_with_nulls_for_percentiles(clip):
    eng = LiveReplay(str(clip), lambda c, f: {})
    s = eng.stats()
    assert set(s) == {"phase", "running", "warming", "run", "clip", "size", "fps", "n_frames", "delay_s",
                      "frames_out", "blind", "blind_share", "dropped_out", "emit_errors", "detector", "emit",
                      "detected_share", "gap_frames", "hold_frames", "frame_age_ms", "input_fps", "gaps_in", "mode", "cats"}
    assert s["phase"] == "idle" and s["size"] == [W, H] and s["hold_frames"] == HOLD_FRAMES
    assert s["mode"] == "clip" and s["input_fps"] is None and s["gaps_in"] == 0
    assert s["detector"] == {"passes": 0, "p50_ms": None, "p95_ms": None, "baseline_p95_ms": None, "errors": 0}
    assert s["emit"] == {"p50_ms": None, "p95_ms": None}
    assert s["gap_frames"] == {"mean": None, "max": None} and s["frame_age_ms"] == {"p50": None}
    assert s["blind_share"] == 0.0 and s["detected_share"] == 0.0


def test_box_cache_matches_mask_boxes_for_every_frame_including_nones():
    script = [det(0, face=FACE), det(4, face=(110.0, 300.0, 150.0, 340.0), plate=(200.0, 500.0, 290.0, 530.0)),
              det(9, plate=(210.0, 500.0, 300.0, 530.0)), det(14), det(40, face=FACE), det(47)]
    cache = BoxCache()
    for f in range(-2, 70):
        ready = [d for d in script if f - WINDOW_BACK <= d.f <= f + HOLD_FRAMES]
        assert cache.boxes(f, ready, W, H) == mask_boxes(f, ready, W, H), f
    assert any(mask_boxes(f, [d for d in script if f - WINDOW_BACK <= d.f <= f + HOLD_FRAMES], W, H) is None
               for f in range(-2, 70))


def pair(box):
    """Two keyframes of one object, as plate_ok sees a short track."""
    return [Keyframe(f=0, box=box, conf=0.9), Keyframe(f=3, box=box, conf=0.9)]


def face_kept(box, conf) -> bool:
    ready = [Detection(0, {"face": [(box, conf)]}, 0.1), Detection(3, {"face": [(box, conf)]}, 0.1)]
    return bool(mask_boxes(1, ready, W, H))


def test_a_big_faint_face_is_dropped_and_a_big_confident_face_is_kept():
    big = (300.0, 100.0, 540.0, 388.0)  # 40% of the frame height
    assert not face_kept(big, 0.3)
    assert face_kept(big, 0.95)


def test_a_wide_plate_is_kept_and_a_square_one_is_dropped_unless_it_touches_the_frame_edge():
    assert plate_ok(pair((300.0, 400.0, 420.0, 440.0)), W, H)       # 3:1
    assert not plate_ok(pair((300.0, 400.0, 360.0, 460.0)), W, H)  # 1:1, 60 px
    assert plate_ok(pair((0.0, 400.0, 60.0, 460.0)), W, H)         # cut off by the left edge
    assert plate_ok(pair((300.0, 400.0, 310.0, 410.0)), W, H)       # tiny: shape not judged


# ---- camera / push mode ---------------------------------------------------------------------------------------

def jpeg_of(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def flat_jpeg(w: int, h: int, grey: int = 90) -> bytes:
    return jpeg_of(np.full((h, w, 3), grey, np.uint8))


def shape_of(jpeg: bytes) -> tuple[int, int]:
    h, w = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape[:2]
    return w, h


@pytest.fixture(scope="module")
def jpegs(clip):
    """Every frame of the fixture clip as a camera upload would send it."""
    cap, out = cv2.VideoCapture(str(clip)), []
    ok, img = cap.read()
    while ok:
        out.append(jpeg_of(img))
        ok, img = cap.read()
    cap.release()
    assert len(out) == 90
    return out


def camera_detect(clip: Path, idx_of: dict):
    """The sidecar replay for a pushed stream: stream frame f carries clip frame idx_of[f]. The test thread fills
    idx_of right after push() returns f, so the detector may be a moment ahead of it: wait for the entry."""
    inner = sidecar_for(clip)

    def detect(f, frame):
        deadline = time.monotonic() + 1.0
        while f not in idx_of and time.monotonic() < deadline:
            time.sleep(0.0005)
        return inner(idx_of[f], frame)
    return detect


def push_at_30fps(eng: LiveReplay, jpegs: list[bytes], seconds: float, idx_of: dict) -> None:
    t0 = time.monotonic()
    for i in range(int(seconds * 30)):
        time.sleep(max(0.0, t0 + i / 30 - time.monotonic()))
        idx_of[eng.push(jpegs[i % len(jpegs)])] = i % len(jpegs)


def test_push_mode_end_to_end_every_emitted_frame_is_blind_or_covers_every_face_and_plate(clip, jpegs):
    idx_of, emitted, lock = {}, [], threading.Lock()

    def record(f, clip_f, blind, rects):
        with lock:
            emitted.append((f, clip_f, blind, rects))

    eng = LiveReplay(None, camera_detect(clip, idx_of), height=H, delay_s=0.4, push=True, on_emit=record)
    eng.start()
    try:
        push_at_30fps(eng, jpegs, 1.5, idx_of)
        pushing = eng.stats()  # read while the pushes are inside the 2 s input window
        time.sleep(0.8)        # let the delay buffer drain
    finally:
        eng.stop()
    s = eng.stats()
    assert emitted and [e[0] for e in emitted] == sorted({e[0] for e in emitted})
    assert any(not blind for _, _, blind, _ in emitted), "nothing was ever vouched: the test would be vacuous"
    assert s["mode"] == "push" and s["clip"] == "camera" and s["n_frames"] == 0
    assert eng.size == (1280, 720) and s["size"] == [1280, 720] and eng.fps == 30.0
    assert 20 <= pushing["input_fps"] <= 35, pushing["input_fps"]
    assert s["detector"]["passes"] > 0 and s["emit_errors"] == 0 and s["dropped_out"] == 0
    # every number below the last emitted one either arrived and was sent, or never arrived
    assert s["frames_out"] + s["gaps_in"] == emitted[-1][0] + 1
    for f, clip_f, blind, rects in emitted:
        assert clip_f == f  # no clip in push mode: the clip frame is the stream frame
        if blind:
            continue
        for box in sidecar_boxes(clip, idx_of[f]):
            assert coverage(_clip_box(box, W, H), rects) >= COVER_MIN, (f, idx_of[f], box)


def test_push_rejects_bytes_that_are_not_a_jpeg_and_counts_nothing():
    eng = LiveReplay(None, lambda f, frame: {}, height=H, push=True)
    eng.start()
    try:
        for junk in (b"junk", b"", b"\xff\xd8\xff truncated"):
            with pytest.raises(ValueError, match="JPEG"):
                eng.push(junk)
        assert eng._n["frames_in"] == 0 and eng.size == (0, 0) and eng.stats()["input_fps"] == 0.0
        assert eng.push(flat_jpeg(640, 360)) == 0  # and the engine still takes a good one
        assert eng._n["frames_in"] == 1
    finally:
        eng.stop()


def patch_sof(jpeg: bytes, w: int, h: int) -> bytes:
    i = jpeg.index(b"\xff\xc0") if b"\xff\xc0" in jpeg else jpeg.index(b"\xff\xc2")
    return jpeg[:i + 5] + h.to_bytes(2, "big") + w.to_bytes(2, "big") + jpeg[i + 9:]


def test_push_refuses_a_jpeg_that_declares_a_huge_frame_before_decoding_it(monkeypatch):
    good = flat_jpeg(64, 48)
    bomb = patch_sof(good, 20000, 20000)
    assert len(bomb) == len(good) and len(bomb) < 2000
    decoded = []
    real = cv2.imdecode
    monkeypatch.setattr("censor.live.cv2.imdecode", lambda *a: decoded.append(1) or real(*a))
    eng = LiveReplay(None, lambda f, frame: {}, height=360, push=True)
    eng.start()
    try:
        with pytest.raises(ValueError, match="too large or malformed"):
            eng.push(bomb)
        with pytest.raises(ValueError, match="too large or malformed"):
            eng.push(patch_sof(good, 0, 0))
        assert decoded == [] and eng._n["frames_in"] == 0 and eng.size == (0, 0)
        assert eng.push(good) == 0 and decoded == [1]  # a real small JPEG passes
    finally:
        eng.stop()


def test_push_needs_a_running_push_engine(clip):
    eng = LiveReplay(None, lambda f, frame: {}, height=H, push=True)
    with pytest.raises(RuntimeError):
        eng.push(flat_jpeg(640, 360))  # not started
    eng.start()
    assert eng.push(flat_jpeg(640, 360)) == 0
    eng.stop()
    with pytest.raises(RuntimeError):
        eng.push(flat_jpeg(640, 360))  # stopped
    assert eng._n["frames_in"] == 1
    with pytest.raises(RuntimeError):
        LiveReplay(str(clip), lambda f, frame: {}).push(flat_jpeg(640, 360))  # a clip engine has no camera


def test_a_clip_is_required_unless_the_engine_is_in_push_mode():
    with pytest.raises(ValueError):
        LiveReplay(None, lambda f, frame: {})


def test_a_push_engine_before_its_first_frame_has_size_zero_and_stats_and_latest_do_not_crash():
    eng = LiveReplay(None, lambda f, frame: {}, height=H, push=True)
    s = eng.stats()
    assert eng.size == (0, 0) and s["size"] == [0, 0] and s["mode"] == "push" and s["clip"] == "camera"
    assert s["fps"] == 30.0 and s["n_frames"] == 0 and s["input_fps"] == 0.0 and s["gaps_in"] == 0
    assert s["frames_out"] == 0 and s["detected_share"] == 0.0 and s["phase"] == "idle"
    assert eng.latest(-1, 0.2) is None
    eng.start()
    try:
        assert eng.latest(-1, 0.2) is None  # running, nothing pushed yet
        assert eng.stats()["size"] == [0, 0]
    finally:
        eng.stop()


def test_the_first_frame_sets_the_size_from_its_aspect_even_width_and_later_frames_conform_to_it():
    eng = LiveReplay(None, lambda f, frame: {}, height=540, out_w=2000, delay_s=0.1, push=True)
    eng.start()
    try:
        eng.push(flat_jpeg(641, 360))  # 641 * 540 / 360 = 961.5: rounds to the even 962
        assert eng.size == (962, 540) and eng.stats()["size"] == [962, 540]
        first = eng.latest(-1, 3.0)
        eng.push(flat_jpeg(400, 400))  # another aspect later (camera switched): stretched to the live size
        second = eng.latest(first[0], 3.0)
    finally:
        eng.stop()
    assert shape_of(first[1]) == (962, 540) and shape_of(second[1]) == (962, 540)  # a 641 px upload is upscaled


def test_the_display_size_follows_the_first_frame_not_a_clip():
    eng = LiveReplay(None, lambda f, frame: {}, height=H, out_w=960, delay_s=0.1, push=True)
    eng.start()
    try:
        eng.push(flat_jpeg(1280, 720))
        shown = eng.latest(-1, 3.0)
    finally:
        eng.stop()
    assert eng.size == (1280, 720) and shape_of(shown[1]) == (960, 540)  # downscaled for the wire, masked at 720p


def test_frame_numbers_follow_elapsed_time_so_a_stall_is_a_gap_in_the_numbers_not_a_neighbour():
    emitted = []
    eng = LiveReplay(None, lambda f, frame: {}, height=360, delay_s=0.3, push=True,
                     on_emit=lambda f, c, blind, rects: emitted.append(f))
    jpeg = flat_jpeg(640, 360)
    eng.start()
    try:
        a, b = eng.push(jpeg), eng.push(jpeg)  # a burst: consecutive numbers
        time.sleep(2.0)                          # a Wi-Fi stall
        c, d = eng.push(jpeg), eng.push(jpeg)
        time.sleep(0.8)
    finally:
        eng.stop()
    s = eng.stats()
    assert (a, b) == (0, 1) and d == c + 1
    assert 57 <= c - b <= 62, (b, c)        # 2 s at the nominal 30 fps, not 1
    assert emitted == [a, b, c, d]
    assert s["gaps_in"] == c - b - 1 and s["dropped_out"] == 0 and s["frames_out"] == 4
    assert s["frames_out"] + s["gaps_in"] == d + 1
    assert s["detector"]["passes"] >= 2 and s["gap_frames"]["max"] >= 55  # detections on both sides of the stall: far apart
    assert s["gap_frames"]["max"] > HOLD_FRAMES + 1  # so the vouch rule will not bridge them


def test_a_full_ring_counts_true_evictions_as_dropped_out_and_never_arrived_numbers_as_gaps_in():
    release = threading.Event()

    def stuck_emit(f, clip_f, blind, rects):
        release.wait(5.0)  # the emitter stalls on its first frame; pushes keep coming

    eng = LiveReplay(None, lambda f, frame: {}, height=360, delay_s=0.05, push=True, on_emit=stuck_emit)
    jpeg = flat_jpeg(640, 360)
    depth = eng._ring.maxlen
    eng.start()
    try:
        assert eng.push(jpeg) == 0
        time.sleep(0.4)  # the emitter has claimed frame 0 and is stuck inside on_emit with it; ~12 numbers go by
        fs = [eng.push(jpeg) for _ in range(depth + 10)]  # back to back: consecutive numbers, the ring keeps `depth`
        stuck = eng.stats()
        release.set()
        time.sleep(0.6)  # the emitter catches up on what the ring kept
        drained = eng.stats()
    finally:
        release.set()
        eng.stop()
    assert fs == list(range(fs[0], fs[0] + depth + 10)) and fs[0] >= 10
    # depth + 11 pushed, `depth` kept: 11 left the ring; frame 0 was claimed (and went out), the next 10 were not
    assert stuck["dropped_out"] == 10 and drained["dropped_out"] == 10, (stuck, drained)
    assert drained["gaps_in"] == fs[0] - 1  # numbers 1..fs[0]-1 (the 0.4 s of silence) never arrived
    assert drained["frames_out"] == 1 + depth
    assert drained["frames_out"] + drained["dropped_out"] + drained["gaps_in"] == fs[-1] + 1  # every number accounted


def test_input_fps_counts_the_pushes_of_the_last_two_seconds():
    eng = LiveReplay(None, lambda f, frame: {}, height=360, delay_s=0.1, push=True)
    jpeg = flat_jpeg(640, 360)
    eng.start()
    try:
        for _ in range(20):
            eng.push(jpeg)
            time.sleep(0.02)
        assert eng.stats()["input_fps"] == pytest.approx(10.0)  # 20 pushes / 2 s
    finally:
        eng.stop()


# ---- yunet_detect ---------------------------------------------------------------------------------------------

def test_yunet_detect_without_the_model_file_names_where_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="opencv_zoo") as e:
        yunet_detect(tmp_path / "nope.onnx")
    assert "face_detection_yunet_2023mar.onnx" in str(e.value)


def test_yunet_detect_maps_columns_resizes_the_input_only_on_a_new_size_and_widens_faces(tmp_path, monkeypatch):
    model = tmp_path / "yunet.onnx"
    model.write_bytes(b"not a real model: the detector is faked")
    seen = {"sizes": []}

    class FakeDetector:
        def setInputSize(self, size):
            seen["sizes"].append(size)

        def detect(self, frame):
            if frame.shape[1] != 640:
                return 1, None  # no face: OpenCV answers None, not an empty array
            return 1, np.array([[100, 50, 40, 60, *range(10), 0.93],      # x, y, w, h, 5 landmarks, score
                                [300, 80, 20, 20, *range(10), 0.71]], np.float32)

    def create(path, config, size, **kw):
        seen.update(path=path, config=config, size=size, kw=kw)
        return FakeDetector()

    monkeypatch.setattr("censor.live.cv2", SimpleNamespace(FaceDetectorYN=SimpleNamespace(create=create)))
    detect = yunet_detect(model, score=0.7)
    assert seen["path"] == str(model) and seen["config"] == "" and seen["size"] == (320, 320)
    assert seen["kw"] == {"score_threshold": 0.7, "nms_threshold": 0.3, "top_k": 50}

    frame = np.zeros((360, 640, 3), np.uint8)
    out = detect(0, frame)
    assert set(out) == {"face", "license plate"} and out["license plate"] == []
    (box1, s1), (box2, s2) = out["face"]
    assert box1 == pytest.approx((100 - 40 * FACE_EXPAND, 50 - 60 * FACE_EXPAND, 140 + 40 * FACE_EXPAND,
                                 110 + 60 * FACE_EXPAND)) and s1 == pytest.approx(0.93)
    assert box2 == pytest.approx((300 - 20 * FACE_EXPAND, 80 - 20 * FACE_EXPAND, 320 + 20 * FACE_EXPAND,
                                 100 + 20 * FACE_EXPAND)) and s2 == pytest.approx(0.71)
    detect(1, frame)
    assert seen["sizes"] == [(640, 360)]  # set once, not per call
    assert detect(2, np.zeros((480, 800, 3), np.uint8)) == {"face": [], "license plate": []}
    assert seen["sizes"] == [(640, 360), (800, 480)]


@pytest.mark.skipif(not os.environ.get("CENSOR_YUNET"), reason="set CENSOR_YUNET to the YuNet ONNX file to run this")
def test_yunet_detect_runs_the_real_model_on_a_synthetic_frame():
    detect = yunet_detect(os.environ["CENSOR_YUNET"])
    frame = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    out = detect(0, frame)
    assert set(out) == {"face", "license plate"} and out["license plate"] == []
    assert all(len(box) == 4 and 0.0 <= score <= 1.0 for box, score in out["face"])


def test_camera_mode_keeps_a_big_faint_face_that_the_clip_rule_drops():
    w, h = 960, 540
    big = (330.0, 120.0, 630.0, 420.0)  # 300 px tall on a 540 px frame: over the rule's 20% height at a score of 0.6
    ready = [Detection(0, {"face": [(big, 0.6)]}, 0.1), Detection(3, {"face": [(big, 0.6)]}, 0.1)]
    assert mask_boxes(1, ready, w, h) == []                      # the scan's rule: a faint big "face" is a car back
    kept = mask_boxes(1, ready, w, h, face_rule=None)
    assert kept and contains(kept[0][0], big)
    cache = BoxCache()
    assert cache.boxes(1, ready, w, h, face_rule=None) == kept   # the cache key sees the rule
    assert cache.boxes(1, ready, w, h) == []


def test_a_push_engine_masks_with_no_face_rule_and_a_clip_engine_keeps_it(clip):
    assert LiveReplay(None, lambda f, fr: {}, push=True)._face_rule is None
    assert LiveReplay(str(clip), lambda f, fr: {})._face_rule is not None


# ---- per-run categories ---------------------------------------------------------------------------------------

PLATE = (100.0, 400.0, 160.0, 420.0)  # a wide 60 x 20 box: passes plate_ok


def test_mask_boxes_cats_face_only_ignores_plate_detections():
    ready = [det(0, plate=PLATE), det(5, plate=PLATE)]
    both = mask_boxes(2, ready, W, H)
    assert both and any(contains(b, PLATE) for b, _, _ in both)
    assert mask_boxes(2, ready, W, H, cats=("face",)) == []


def test_mask_boxes_cats_plate_only_ignores_faces():
    ready = [det(0, face=FACE, plate=PLATE), det(5, face=FACE, plate=PLATE)]
    boxes = mask_boxes(2, ready, W, H, cats=("license plate",))
    assert boxes and all(not contains(b, FACE) for b, _, _ in boxes)


def test_box_cache_keys_on_cats():
    ready = [det(0, face=FACE, plate=PLATE), det(5, face=FACE, plate=PLATE)]
    cache = BoxCache()
    assert len(cache.boxes(2, ready, W, H)) == 2
    assert len(cache.boxes(2, ready, W, H, cats=("face",))) == 1
    assert len(cache.boxes(2, ready, W, H)) == 2


def test_egoblur_detect_cats_face_never_calls_the_plate_model():
    calls = []

    def plate_model(img):
        calls.append("plate")
        return []
    infer = {"face": lambda img: [((100.0, 100.0, 140.0, 160.0), 0.9)], "license plate": plate_model}
    out = egoblur_detect(infer, cats=("face",))(0, None)
    assert set(out) == {"face"} and out["face"] and not calls


def test_sidecar_and_yunet_return_only_requested_cats(tmp_path):
    store = TrackStore()
    for label, box in (("face", (300.0, 300.0, 330.0, 330.0)), ("license plate", (600.0, 600.0, 660.0, 630.0))):
        store.add(Track(track_id=store.next_id(), label=label, keyframes=[Keyframe(f=0, box=box, conf=0.8)]))
    store.save(tmp_path / "t.json")
    out = sidecar_detect(tmp_path / "t.json", 1080, 720, 10, cats=("face",))(0, None)
    assert set(out) == {"face"} and out["face"]
    assert set(sidecar_detect(tmp_path / "t.json", 1080, 720, 10, cats=("face",))(5, None)) == {"face"}


@pytest.mark.parametrize("cats", [(), ("plate",), ("face", "plate"), ("face", "face", "x")])
def test_live_replay_rejects_bad_cats(cats):
    with pytest.raises(ValueError):
        LiveReplay(None, lambda f, fr: {}, push=True, cats=cats)


def test_live_replay_stats_report_cats():
    assert LiveReplay(None, lambda f, fr: {}, push=True).stats()["cats"] == list(LIVE_CATS)
    r = LiveReplay(None, lambda f, fr: {}, push=True, cats=("face",))
    assert r.cats == ("face",) and r.stats()["cats"] == ["face"]
