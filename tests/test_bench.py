"""Benchmark harness tests. No GPU: stub detector, and either the stub VLM or a color check standing in for a working one."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bench import run_bench as rb
from bench.gt import GTBox, GTObject, LabeledFrame, LabelSet, from_sidecar
from censor.agent import agent as agent_mod
from censor.core.timeline import VideoInfo
from censor.vision import vlm as vlm_mod

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "bench" / "labels" / "synthetic.json"
needs_ffmpeg = pytest.mark.skipif(shutil.which(os.environ.get("CENSOR_FFMPEG", "ffmpeg")) is None,
                                  reason="ffmpeg not on PATH")


def test_coverage_is_a_union():
    box = (0, 0, 100, 100)
    assert rb.coverage(box, [(0, 0, 60, 100), (40, 0, 100, 100)]) == 1.0
    assert rb.coverage(box, [(0, 0, 50, 100), (0, 0, 50, 100)]) == 0.5  # overlap isn't counted twice
    assert rb.coverage(box, []) == 0.0


def test_longest_miss():
    assert rb.longest_miss([True, False, False, True, False]) == 2
    assert rb.longest_miss([True, True]) == 0


def test_labels_reject_unknown_ids():
    with pytest.raises(ValueError, match="nope"):
        LabelSet(clip="c", video=VideoInfo(fps=10, width=10, height=10, n_frames=10), objects={},
                 frames=[LabeledFrame(f=0, boxes=[GTBox(gt_id="nope", box=(0, 0, 1, 1))])])


def _red_and_blue():
    red, blue = (10, 10, 50, 90), (110, 10, 150, 90)
    labels = LabelSet(clip="t", video=VideoInfo(fps=10, width=200, height=100, n_frames=30),
                      objects={"person_1": GTObject(category="person"), "person_2": GTObject(category="person")},
                      frames=[LabeledFrame(f=f, boxes=[GTBox(gt_id="person_1", box=red), GTBox(gt_id="person_2", box=blue)])
                              for f in (0, 10, 20)])
    req = rb.BenchRequest(clip="t", text="blur the person in the red shirt", must_mask=["person_1"], must_not_mask=["person_2"])
    return labels, req, red, blue


def test_verdict_red_not_blue():
    labels, req, red, blue = _red_and_blue()
    score = lambda rects: rb.verdict(req, rb.score_frames(labels, {f: rects for f in (0, 10, 20)})[0])  # noqa: E731
    assert score([red])["ok"]
    both = score([red, blue])
    assert not both["ok"] and both["leaked"] == ["person_2"] and both["missed"] == []
    neither = score([])
    assert not neither["ok"] and neither["missed"] == ["person_1"]


def test_over_blur_ignores_labeled_area():
    labels, _, red, _ = _red_and_blue()
    _, over = rb.score_frames(labels, {0: [red], 10: [(0, 0, 200, 100)]})
    assert over[0] == 0.0 and 0.0 < over[1] < 1.0 and over[2] == 0.0


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench") / "synthetic.mp4"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_clip.py"), str(out)], check=True)
    return out


@pytest.fixture()
def stub(monkeypatch):
    monkeypatch.setenv("CENSOR_STUB", "1")
    monkeypatch.setattr(agent_mod, "MODEL_URL", None)
    monkeypatch.setattr(vlm_mod, "MODEL_URL", None)


@needs_ffmpeg
def test_committed_labels_match_the_generator(synthetic):
    committed, fresh = LabelSet.load(LABELS), from_sidecar(synthetic, clip="synthetic")
    assert fresh.video == committed.video and fresh.frames == committed.frames
    assert {g: o.category for g, o in fresh.objects.items()} == {g: o.category for g, o in committed.objects.items()}


@needs_ffmpeg
def test_synthetic_stub_recall_and_requests(synthetic, stub):
    reqs = [r for r in rb.load_requests() if r.clip == "synthetic"]
    res = rb.bench_clip(synthetic, LabelSet.load(LABELS), reqs, render=False)
    cov = res["coverage"]["ours"]
    assert cov["recall"] == {"face": 1.0, "license plate": 1.0, "person": 1.0, "screen": 1.0}
    assert all(o["longest_miss"] == 0 for o in cov["objects"].values())
    assert res["errors"] == []

    red = next(r for r in res["requests"] if "red shirt" in r["text"])
    # The stub VLM answers "unknown" for both people. Fail closed keeps both, so blue is masked and the request fails.
    assert not red["ours"]["ok"] and red["ours"]["leaked"] == ["person_2"] and red["ours"]["missed"] == []
    # With fail-closed off, "unknown" reads as "no" and red is missed instead.
    assert not red["fail_open"]["ok"] and red["fail_open"]["missed"] == ["person_1"]
    failing = [(r["text"], r["ours"]) for r in res["requests"] if r is not red and not r["ours"]["ok"]]
    assert failing == []


@needs_ffmpeg
def test_red_request_passes_when_the_attribute_check_works(synthetic, stub, monkeypatch):
    def by_color(video, tracks, question):  # stands in for a VLM that gets the shirt color right
        answers = []
        for t in tracks:
            b, g, r = np.mean([c.reshape(-1, 3).mean(0) for c in vlm_mod.track_crops(video, t)], axis=0)
            answers.append(vlm_mod.AttributeAnswer(t.track_id, "yes" if r > 1.5 * max(b, g) else "no", 0.9))
        return answers

    monkeypatch.setattr(vlm_mod, "check_attribute", by_color)
    req = next(r for r in rb.load_requests() if "red shirt" in r.text)
    row = rb.bench_clip(synthetic, LabelSet.load(LABELS), [req], render=False, variants=("ours", "keyword"))["requests"][0]
    assert row["ours"]["ok"]
    assert not row["keyword"]["ok"] and row["keyword"]["leaked"] == ["person_2"]  # no attribute check: blue gets masked too
