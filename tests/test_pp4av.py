"""PP4AV image benchmark on a tiny fake copy of the dataset layout. No GPU, no download."""
import cv2
import numpy as np
import pytest

from bench import pp4av


@pytest.fixture()
def fake_root(tmp_path):
    (tmp_path / "annotations" / "paris").mkdir(parents=True)
    (tmp_path / "images" / "paris").mkdir(parents=True)
    # 200x100 image: a face at (40..60, 20..60) and a plate at (120..180, 70..90), YOLO-normalized
    (tmp_path / "annotations" / "paris" / "frame_000001.txt").write_text(
        "0 0.25 0.4 0.1 0.4\n1 0.75 0.8 0.3 0.2\n")
    (tmp_path / "annotations" / "paris" / "frame_000002.txt").write_text("0 0.5 0.5 0.1 0.1\n")  # no image: skipped
    cv2.imwrite(str(tmp_path / "images" / "paris" / "frame_000001.png"), np.zeros((100, 200, 3), np.uint8))
    return tmp_path


def oracle(conf):
    def detect(img):
        h, w = img.shape[:2]
        return [(cat, box, conf) for cat, box in (pp4av.to_pixels((0, .25, .4, .1, .4), w, h),
                                                    pp4av.to_pixels((1, .75, .8, .3, .2), w, h))]
    return detect


def test_load_pairs_labels_with_images(fake_root):
    samples = pp4av.load_samples(fake_root)
    assert len(samples) == 1 and samples[0]["subset"] == "paris"
    assert samples[0]["labels"] == [(0, 0.25, 0.4, 0.1, 0.4), (1, 0.75, 0.8, 0.3, 0.2)]


def test_to_pixels():
    assert pp4av.to_pixels((1, 0.75, 0.8, 0.3, 0.2), 200, 100) == ("license plate", pytest.approx((120, 70, 180, 90)))


def test_recall_with_a_perfect_and_an_empty_detector(fake_root):
    samples = pp4av.load_samples(fake_root)
    perfect = pp4av.evaluate(samples, oracle(0.9))
    assert perfect["recall"] == {"face": {"recall": 1.0, "boxes": 1}, "license plate": {"recall": 1.0, "boxes": 1}}
    assert perfect["by_subset"]["paris"]["face"]["recall"] == 1.0 and 0 < perfect["over_blur"] < 0.1
    blind = pp4av.evaluate(samples, lambda img: [])
    assert blind["recall"]["face"]["recall"] == 0.0 and blind["over_blur"] == 0.0


def test_uncertain_detections_get_double_padding():
    sure, unsure = pp4av.mask_rects([("face", (100, 100, 200, 200), 0.9), ("face", (100, 100, 200, 200), 0.2)], 1000, 1000)
    assert (sure[2] - sure[0]) < (unsure[2] - unsure[0])


def test_report_renders(fake_root):
    rep = pp4av.evaluate(pp4av.load_samples(fake_root), oracle(0.9))
    rep["meta"] = {"generated": "now", "commit": "abc", "detector": "oracle", "imgsz": 1280, "min_conf": 0.15, "uncertain_below": 0.35}
    md = pp4av.to_markdown(rep)
    assert "| face | 1.00 (1) | >= 0.95 | 1.00 | 1.00 |" in md and "CC BY-NC-ND 4.0" in md


def test_face_padding_only_widens_faces():
    dets = [("face", (100, 100, 200, 200), 0.9), ("license plate", (300, 300, 400, 340), 0.9)]
    base, wide = pp4av.mask_rects(dets, 1000, 1000), pp4av.mask_rects(dets, 1000, 1000, {"face": 0.5})
    assert wide[0][2] - wide[0][0] > base[0][2] - base[0][0] and wide[1] == base[1]
