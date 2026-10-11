"""The UI's live preview layers: GET /masks must give exactly the rectangles the renderer blurs."""
import shutil

import pytest

from censor.render.render import FrameMasker, _padded
from tests.test_e2e_stub import FFMPEG, clip, client  # noqa: F401  (pytest fixtures)

pytestmark = pytest.mark.skipif(shutil.which(FFMPEG) is None, reason="ffmpeg not on PATH")


def _load(client, clip):
    c, app_mod = client
    vid = c.post("/api/videos", files={"file": ("x.mp4", clip.read_bytes(), "video/mp4")}).json()["video_id"]
    c.post(f"/api/videos/{vid}/tracks-sidecar",
           files={"file": ("t.json", (clip.parent / f"{clip.name}.tracks.json").read_bytes(), "application/json")})
    return c, app_mod, vid


def test_masks_match_the_renderer(clip, client):
    c, app_mod, vid = _load(client, clip)
    assert c.get(f"/api/videos/{vid}/masks").json()["targets"] == {}
    c.post(f"/api/videos/{vid}/request", json={"text": "blur all plates"})
    c.post(f"/api/videos/{vid}/request", json={"text": "pixelate faces in the first 1 seconds"})
    m = c.get(f"/api/videos/{vid}/masks").json()
    tl = app_mod.session(vid).timeline
    assert set(m["targets"]) == set(tl.targets)
    expected = {(f, *_padded(box, pad, tl.video.width, tl.video.height))
                for f, items in FrameMasker(tl).per_frame.items() for box, pad, _ in items}
    got = {tuple(r) for t in m["targets"].values() for r in t["frames"]}
    assert got == expected
    effects = {tl.targets[t].label: v["effect"] for t, v in m["targets"].items()}
    assert effects == {"license plate": "blur", "face": "pixelate"}


def test_time_limited_mask_stops_after_its_window_plus_hold(clip, client):
    c, app_mod, vid = _load(client, clip)
    c.post(f"/api/videos/{vid}/request", json={"text": "pixelate faces in the first 1 seconds"})
    m = c.get(f"/api/videos/{vid}/masks").json()
    (face,) = m["targets"].values()
    frames = [r[0] for r in face["frames"]]
    fps = m["fps"]
    assert min(frames) == 0
    # the window ends at 1 s; the renderer keeps masking a lost track for up to HOLD_FRAMES (15) more
    assert round(fps) <= max(frames) <= round(fps) + 15
