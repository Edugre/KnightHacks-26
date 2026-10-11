"""bench/live_speed.py: the stride and delay it plans, and the paced run's latest-frame-wins loop, with a fake detector."""
import time

import numpy as np

from bench.live_speed import paced, plan, stats_ms
from censor.render.render import HOLD_FRAMES, LEAD_FRAMES


def test_stats_are_in_ms_with_the_tail():
    s = stats_ms([0.01] * 98 + [0.05, 0.2])
    assert s == {"p50": 10.0, "p95": 10.0, "p99": 50.0, "max": 200.0}


def test_plan_detects_every_frame_when_a_pass_fits_the_budget():
    p = plan([0.02] * 50, fps=30)
    assert p["every_n_frames"] == 1 and p["keeps_up_every_frame"]
    assert p["delay_ms"] == round((1 + LEAD_FRAMES) * 1000 / 30 + 20)


def test_plan_strides_on_the_p95_pass():
    # median 50 ms, p95 110 ms: at 30 fps (33.3 ms a frame) that's every 4th frame, not every 2nd
    p = plan([0.05] * 90 + [0.11] * 10, fps=30)
    assert p["every_n_frames"] == 4 and not p["keeps_up_every_frame"]
    assert p["detections_per_s"] == 20.0


def test_paced_run_drops_frames_the_detector_cant_reach():
    pool = [np.zeros((4, 4, 3), np.uint8)]
    r = paced(lambda img: time.sleep(0.025), pool, fps=100, seconds=0.6)  # a pass takes 2.5 frames
    assert r["frames"] == 60
    assert 10 <= r["detected"] < 40
    assert r["gap_frames"]["max"] >= 2
    assert r["max_gap_within_hold"] == (r["gap_frames"]["max"] <= HOLD_FRAMES)
    assert r["staleness_ms"]["p50"] >= 25


def test_paced_run_sees_every_frame_when_the_detector_is_fast():
    r = paced(lambda img: None, [np.zeros((4, 4, 3), np.uint8)], fps=50, seconds=0.4)
    assert r["detected"] == r["frames"] == 20
    assert r["gap_frames"]["max"] == 1
