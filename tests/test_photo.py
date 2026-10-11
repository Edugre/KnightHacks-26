"""Photos as one-frame videos: the readers, the renderer and a whole request on a photo. No GPU or ffmpeg needed."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from censor.core.timeline import Op, fold
from censor.core.tracks import Keyframe, Track, TrackStore
from censor.core.video import frames, is_image, probe, read_frame
from censor.render.render import expand_target, render_image

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def photo(tmp_path_factory):
    out = tmp_path_factory.mktemp("photos") / "synthetic.png"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_clip.py"), "--image", str(out)], check=True)
    return out


def noise(path: Path, w=320, h=200) -> np.ndarray:
    img = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return img


def one_box(box=(40, 40, 120, 100), label="face") -> TrackStore:
    s = TrackStore()
    s.add(Track(track_id="t", label=label, keyframes=[Keyframe(f=0, box=box, conf=0.9, state="detected")]))
    return s


def test_a_photo_reads_as_one_frame(tmp_path):
    img = noise(tmp_path / "a.png")
    assert is_image(tmp_path / "a.PNG") and not is_image("clip.mp4")
    info = probe(str(tmp_path / "a.png"))
    assert (info.kind, info.n_frames, info.fps, info.width, info.height) == ("image", 1, 1.0, 320, 200)
    assert [(f, fr.shape) for f, fr in frames(str(tmp_path / "a.png"), every=5, prefetch=4)] == [(0, (200, 320, 3))]
    assert np.array_equal(read_frame(str(tmp_path / "a.png"), 0), img)
    with pytest.raises(IndexError):
        read_frame(str(tmp_path / "a.png"), 1)


def test_an_unreadable_photo_is_an_ioerror(tmp_path):
    (tmp_path / "bad.jpg").write_bytes(b"not a jpeg")
    with pytest.raises(IOError):
        probe(str(tmp_path / "bad.jpg"))


def test_old_sessions_without_kind_are_videos():
    from censor.core.timeline import VideoInfo
    assert VideoInfo(fps=30, width=1, height=1, n_frames=1).kind == "video"


def test_one_keyframe_masks_only_frame_0(tmp_path):
    noise(tmp_path / "a.png")
    info = probe(str(tmp_path / "a.png"))
    tl = fold([Op(op_id="op_1", kind="add_targets", targets=["t"])], one_box(), info)
    assert list(expand_target(tl.targets["t"], info.n_frames)) == [0]  # no lead-in or hold past the one frame


@pytest.mark.parametrize("ext", [".png", ".jpg", ".webp"])
def test_render_image_masks_inside_the_box_only(tmp_path, ext):
    src = tmp_path / f"a{ext}"
    noise(src)
    orig = read_frame(str(src), 0)  # as decoded, so JPEG loss on the source doesn't count
    tl = fold([Op(op_id="op_1", kind="add_targets", targets=["t"], padding=0.0)], one_box(), probe(str(src)))
    dst = tmp_path / f"out{ext}"
    stats = render_image(str(src), tl, str(dst))
    assert (stats.frames, stats.masked_boxes, stats.failed_frames) == (1, 1, [])
    out = cv2.imread(str(dst))
    assert out.shape == orig.shape
    inside = np.abs(out[45:95, 45:115].astype(int) - orig[45:95, 45:115].astype(int)).mean()
    outside = np.abs(out[130:, 160:].astype(int) - orig[130:, 160:].astype(int)).mean()
    assert inside > 30 and outside < (12 if ext != ".png" else 0.01)


def test_render_image_drops_the_source_metadata(tmp_path):
    """A photo's EXIF can hold where it was taken; the export must not carry it."""
    noise(tmp_path / "plain.jpg")
    data = (tmp_path / "plain.jpg").read_bytes()
    # An APP1 Exif segment right after SOI: a big-endian TIFF header, an empty IFD, and a marker string
    tiff = b"MM\x00\x2a\x00\x00\x00\x08" + b"\x00\x00" + b"\x00\x00\x00\x00" + b"GPS 28.6024 -81.2001"
    app1 = b"Exif\x00\x00" + tiff
    src = tmp_path / "exif.jpg"
    src.write_bytes(data[:2] + b"\xff\xe1" + (len(app1) + 2).to_bytes(2, "big") + app1 + data[2:])
    assert b"GPS 28.6024" in src.read_bytes()
    tl = fold([Op(op_id="op_1", kind="add_targets", targets=["t"])], one_box(), probe(str(src)))
    render_image(str(src), tl, str(tmp_path / "out.jpg"))
    out = (tmp_path / "out.jpg").read_bytes()
    assert b"Exif" not in out and b"GPS 28.6024" not in out


def test_render_image_fails_closed(tmp_path, monkeypatch):
    import censor.render.render as r
    src = tmp_path / "a.png"
    orig = noise(src)
    tl = fold([Op(op_id="op_1", kind="add_targets", targets=["t"])], one_box(), probe(str(src)))
    real, calls = r.apply_effect, {"n": 0}

    def boom(fr, rect, effect):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return real(fr, rect, effect)
    monkeypatch.setattr(r, "apply_effect", boom)
    stats = render_image(str(src), tl, str(tmp_path / "out.png"))
    assert stats.failed_frames == [0]
    out = cv2.imread(str(tmp_path / "out.png"))
    assert np.abs(out[130:, 160:].astype(int) - orig[130:, 160:].astype(int)).mean() > 30  # the whole photo blurred


def test_a_request_on_a_photo_masks_and_exports(photo, tmp_path):
    from censor.agent.agent import Agent
    from censor.agent.tools import Session
    from censor.vision.detect import StubDetector

    s = Session(photo, tmp_path / "w", detector=StubDetector())
    assert s.video_info()["n_frames"] == 1 and s.info.kind == "image"
    r = Agent(s).run("blur all plates and the screen")
    assert r["path"] == "template" and r["summary"]["by_label"] == {"license plate": 1, "screen": 1}

    drawn = s.track_box(0, [855, 90, 945, 210])  # a drawn box on a photo is that box, nothing to follow
    assert (drawn["start"], drawn["end"]) == (0.0, 0.0)
    assert [k.f for k in s.tracks.get(drawn["track_id"]).keyframes] == [0]
    s.add_masks([drawn["track_id"]], request_id="ui_box")

    out = tmp_path / "out.png"
    stats = render_image(str(photo), s.timeline, str(out))
    assert stats.masked_boxes == 3
    a, b = cv2.imread(str(photo)), cv2.imread(str(out))
    detail = lambda img: np.abs(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F)).mean()  # noqa: E731
    for label in ("license plate", "screen"):  # "ABC 1234" and "PASSWORD" are no longer legible
        x1, y1, x2, y2 = (int(v) for v in s.tracks.by_label(label)[0].keyframes[0].box)
        assert detail(b[y1:y2, x1:x2]) < 0.2 * detail(a[y1:y2, x1:x2]), label
    assert np.abs(a[0:80, 0:400].astype(int) - b[0:80, 0:400].astype(int)).mean() < 1  # the sky, far from every target
