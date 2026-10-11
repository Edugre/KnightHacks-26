"""The editor's Redo, and requests that point at a target ("blur everyone except this person")."""
import shutil

import pytest

from tests.test_e2e_stub import FFMPEG, clip, client  # noqa: F401  (pytest fixtures)
from tests.test_preview_routes import _load

pytestmark = pytest.mark.skipif(shutil.which(FFMPEG) is None, reason="ffmpeg not on PATH")

BLUE = [650, 260, 740, 480]  # the blue person in scripts/make_test_clip.py, standing still


def _targets(c, vid):
    return c.get(f"/api/videos/{vid}/timeline").json()["targets"]


def _people(c, vid):
    return {t["track_id"]: t for t in c.get(f"/api/videos/{vid}/tracks").json()["tracks"].values() if t["label"] == "person"}


def _blue(c, vid):
    return next(i for i, t in _people(c, vid).items() if list(t["keyframes"][0]["box"]) == BLUE)


def test_redo_puts_back_what_undo_took(clip, client):
    c, _, vid = _load(client, clip)
    c.post(f"/api/videos/{vid}/request", json={"text": "blur all people"})
    one = sorted(_targets(c, vid))[0]
    c.post(f"/api/videos/{vid}/targets", json={"action": "remove", "track_ids": [one]})
    after = _targets(c, vid)
    r = c.post(f"/api/videos/{vid}/undo").json()
    assert r["summary"]["n_redo"] == 1 and one in _targets(c, vid)
    r = c.post(f"/api/videos/{vid}/redo").json()
    assert r["redone"] and r["summary"]["n_redo"] == 0 and _targets(c, vid) == after
    assert c.post(f"/api/videos/{vid}/redo").json()["redone"] is None  # nothing left to redo


def test_a_new_edit_clears_redo(clip, client):
    c, _, vid = _load(client, clip)
    c.post(f"/api/videos/{vid}/request", json={"text": "blur all people"})
    c.post(f"/api/videos/{vid}/undo")
    assert c.get(f"/api/videos/{vid}").json()["summary"]["n_redo"] == 1
    c.post(f"/api/videos/{vid}/request", json={"text": "blur all plates"})
    assert c.get(f"/api/videos/{vid}").json()["summary"]["n_redo"] == 0
    assert c.post(f"/api/videos/{vid}/redo").json()["redone"] is None


def test_everyone_except_a_drawn_box(clip, client):
    c, _, vid = _load(client, clip)
    ref = {"frame": 5, "box": [BLUE[0] - 6, BLUE[1] + 4, BLUE[2] + 3, BLUE[3] - 8]}  # drawn roughly around them
    r = c.post(f"/api/videos/{vid}/request", json={"text": "blur everyone except this person", "ref": ref}).json()
    assert r["path"] == "reference"
    resolved = next(x["result"] for x in r["calls"] if x["tool"] == "resolve_ref")
    blue = _blue(c, vid)
    assert resolved["track_id"] == blue and resolved["how"] == "matched"
    masked = _targets(c, vid)
    assert set(masked) == set(_people(c, vid)) - {blue}


def test_only_a_picked_target(clip, client):
    c, _, vid = _load(client, clip)
    c.post(f"/api/videos/{vid}/request", json={"text": "unblur the people"})  # scans people, masks nothing
    blue = _blue(c, vid)
    r = c.post(f"/api/videos/{vid}/request", json={"text": "pixelate this person", "ref": {"track_id": blue}}).json()
    assert r["path"] == "reference"
    assert {i: t["effect"] for i, t in _targets(c, vid).items()} == {blue: "pixelate"}


def test_a_reference_box_on_nothing_masks_the_box(clip, client):
    c, _, vid = _load(client, clip)
    r = c.post(f"/api/videos/{vid}/request", json={"text": "blur this", "ref": {"frame": 10, "box": [20, 20, 120, 120]}}).json()
    assert next(x["result"] for x in r["calls"] if x["tool"] == "resolve_ref")["how"] == "tracked"
    assert [t["label"] for t in _targets(c, vid).values()] == ["user box"]


@pytest.mark.parametrize("ref", [{"track_id": "trk_999"}, {"frame": 10}, {"frame": 10, "box": [50, 50, 10, 10]}])
def test_a_bad_reference_changes_nothing(clip, client, ref):
    c, _, vid = _load(client, clip)
    r = c.post(f"/api/videos/{vid}/request", json={"text": "blur everyone except this person", "ref": ref})
    assert r.status_code == 200 and "Couldn't use the reference" in r.json()["message"]
    assert _targets(c, vid) == {}
