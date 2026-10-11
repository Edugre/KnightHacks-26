"""Photos as one-frame videos: the readers, the renderer and a whole request on a photo. No GPU or ffmpeg needed."""
import shutil
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


# ---- detectors and agent on a photo ---------------------------------------------------------------------------

def test_egoblur_keeps_one_frame_plates_on_a_photo(photo):
    """Video rules want a plate seen twice confidently or on 3 frames; a photo has one frame, so that would drop
    every plate. One detection counts, still judged by shape, and weak ones are kept as uncertain."""
    from censor.vision import specialist as sp
    seen = []

    def plate(img):
        seen.append(img.shape)
        return [((100, 500, 260, 540), 0.9),    # a plate, confident
                ((700, 500, 780, 520), 0.42),   # a distant plate, under min_conf: kept, uncertain
                ((900, 400, 960, 420), 0.15),   # faint but plate-shaped: kept, uncertain (fail closed)
                ((300, 100, 360, 200), 0.3)]    # faint and taller than wide: not a plate
    det = sp.EgoBlurDetector(infer={"face": lambda img: [], "license plate": plate}, every=3)
    found = det.run(str(photo), ["license plate"], TrackStore())
    assert len(seen) == 1  # one pass, no recheck of frames a photo doesn't have
    states = {round(t.keyframes[0].box[0]): t.keyframes[0].state for t in found}
    assert states == {100: "detected", 700: "uncertain", 900: "uncertain"} and det.dropped == {"license plate": 1}
    assert all(len(t.keyframes) == 1 and t.keyframes[0].f == 0 for t in found)


def test_yoloe_detects_a_photo_once_without_the_tracker(photo):
    from types import SimpleNamespace as NS
    from censor.vision import detect as d
    got = {}

    class FakeYoloe:
        predictor = None

        def set_classes(self, names, pe):
            got["classes"] = names

        def get_text_pe(self, names):
            return None

        def track(self, **kw):
            raise AssertionError("a photo has nothing to track")

        def predict(self, source, **kw):
            got["shape"] = source.shape  # the decoded pixels, rotated as read_image reads them
            rows = [((10, 20, 110, 220), 0.8, 0), ((300, 50, 340, 90), 0.2, 1)]
            return [NS(boxes=NS(xyxy=NS(tolist=lambda: [list(b) for b, _, _ in rows]),
                                conf=NS(tolist=lambda: [c for _, c, _ in rows]),
                                cls=NS(tolist=lambda: [float(k) for _, _, k in rows])))]

    det = object.__new__(d.YoloeDetector)
    det.weights, det.device = "fake.pt", "cpu"
    d.YoloeDetector._models["fake.pt"] = FakeYoloe()
    try:
        found = det.run(str(photo), ["car", "dog"], TrackStore())
    finally:
        d.YoloeDetector._models.pop("fake.pt")
    assert got == {"classes": ["car", "dog"], "shape": (720, 1280, 3)}
    assert [(t.label, len(t.keyframes), t.keyframes[0].f, t.keyframes[0].state) for t in found] == \
        [("car", 1, 0, "detected"), ("dog", 1, 0, "uncertain")]

def test_a_time_phrase_on_a_photo_is_ignored_with_a_note(photo, tmp_path):
    from censor.agent.agent import Agent
    from censor.agent.tools import Session
    from censor.vision.detect import StubDetector
    s = Session(photo, tmp_path / "w", detector=StubDetector())
    assert s.resolve_time("last 5 seconds") == [0.0, 1.0]
    r = Agent(s).run("blur the screen in the last 5 seconds")
    assert r["summary"]["by_label"] == {"screen": 1}
    assert "A photo has no time" in r["message"]
    assert all(o.time_range is None for o in s.log.ops)


# ---- the server ------------------------------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CENSOR_STUB", "1")
    monkeypatch.setenv("CENSOR_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("CENSOR_MODEL_URL", raising=False)
    import importlib
    import censor.server.app as app_mod
    importlib.reload(app_mod)
    from fastapi.testclient import TestClient
    return TestClient(app_mod.app), app_mod


def test_upload_request_export_a_photo(photo, client):
    c, _ = client
    with open(photo, "rb") as f:
        j = c.post("/api/videos", files={"file": ("Holiday.PNG", f, "image/png")}).json()
    vid = j["video_id"]
    assert j["info"]["kind"] == "image" and j["info"]["n_frames"] == 1
    with open(str(photo) + ".tracks.json", "rb") as f:
        assert c.post(f"/api/videos/{vid}/tracks-sidecar", files={"file": ("t.json", f)}).json()["ok"]

    r = c.post(f"/api/videos/{vid}/request", json={"text": "blur all plates and faces"}).json()
    assert r["summary"]["by_label"] == {"license plate": 1, "face": 1}
    assert c.post(f"/api/videos/{vid}/box", json={"frame": 0, "box": [1050, 300, 1230, 420]}).json()["track"]["end"] == 0
    masks = c.get(f"/api/videos/{vid}/masks").json()
    assert masks["n_frames"] == 1 and all(f[0] == 0 for m in masks["targets"].values() for f in m["frames"])
    assert c.get(f"/api/videos/{vid}/preview/0.jpg").status_code == 200
    assert c.get(f"/api/videos/{vid}/preview/1.jpg").status_code == 404

    e = c.post(f"/api/videos/{vid}/export").json()
    assert e["kind"] == "image" and e["url"].startswith(f"api/videos/{vid}/export.png?v=")
    assert e["stats"] == {"frames": 1, "masked_boxes": 3, "failed_frames": 0}
    got = c.get("/" + e["url"])
    assert got.status_code == 200 and got.headers["content-type"] == "image/png"
    out = cv2.imdecode(np.frombuffer(got.content, np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == (720, 1280, 3)
    assert c.get(f"/api/videos/{vid}/export.mp4").status_code == 404
    assert c.get(f"/api/videos/{vid}/export.exe").status_code == 404


def test_photos_the_server_cant_read_are_refused(client, tmp_path):
    c, app_mod = client
    r = c.post("/api/videos", files={"file": ("IMG_0001.HEIC", b"\x00" * 64, "image/heic")})
    assert r.status_code == 415 and "JPEG" in r.json()["detail"]
    r = c.post("/api/videos", files={"file": ("broken.jpg", b"not a jpeg", "image/jpeg")})
    assert r.status_code == 400
    assert not any(app_mod.DATA_DIR.iterdir())  # no half-made session left behind


def test_sample_photos_load_without_a_file_picker(photo, client, tmp_path, monkeypatch):
    c, app_mod = client
    photos = tmp_path / "photos"
    photos.mkdir()
    shutil.copyfile(photo, photos / "street.png")
    shutil.copyfile(str(photo) + ".tracks.json", photos / "street.png.tracks.json")
    (photos / "notes.txt").write_text("not a photo")
    monkeypatch.setattr(app_mod, "PHOTOS_DIR", photos)
    monkeypatch.setattr(app_mod, "SAMPLES_DIR", tmp_path / "no-clips")
    listed = c.get("/api/samples?kind=image").json()
    assert [(p["id"], p["width"], p["height"], p["sidecar"]) for p in listed] == [("street", 1280, 720, True)]
    assert c.get("/api/samples").json() == []  # the clip list stays clips
    thumb = c.get("/" + listed[0]["url"])
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/png"
    assert c.get("/api/samples/../photo").status_code == 404

    j = c.post("/api/samples/street?kind=image").json()
    assert j["info"]["kind"] == "image"
    r = c.post(f"/api/videos/{j['video_id']}/request", json={"text": "blur all plates"}).json()
    assert r["summary"]["by_label"] == {"license plate": 1}  # the sidecar came along
    assert c.post("/api/samples/nope?kind=image").status_code == 404
