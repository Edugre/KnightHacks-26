"""Hand edits on chosen targets (POST /targets): mask, unmask and restyle one track; each is one undoable op."""
import shutil

import pytest

from tests.test_e2e_stub import FFMPEG, clip, client  # noqa: F401  (pytest fixtures)
from tests.test_preview_routes import _load

pytestmark = pytest.mark.skipif(shutil.which(FFMPEG) is None, reason="ffmpeg not on PATH")


def _targets(c, vid):
    return c.get(f"/api/videos/{vid}/timeline").json()["targets"]


def test_add_remove_and_restyle_one_track(clip, client):
    c, _, vid = _load(client, clip)
    c.post(f"/api/videos/{vid}/request", json={"text": "blur all people"})
    people = sorted(_targets(c, vid))
    assert len(people) == 2
    one, other = people

    r = c.post(f"/api/videos/{vid}/targets", json={"action": "effect", "track_ids": [one], "effect": "pixelate"})
    assert r.status_code == 200
    assert _targets(c, vid)[one]["effect"] == "pixelate" and _targets(c, vid)[other]["effect"] == "blur"

    c.post(f"/api/videos/{vid}/targets", json={"action": "remove", "track_ids": [one]})
    assert set(_targets(c, vid)) == {other}

    # the plate was found by no request yet: scan it, then mask that one track by hand
    plates = c.post(f"/api/videos/{vid}/request", json={"text": "unblur the plates"}).json()
    plate = next(x["result"][0]["track_id"] for x in plates["calls"] if x["tool"] == "find_objects")
    r = c.post(f"/api/videos/{vid}/targets", json={"action": "add", "track_ids": [plate], "effect": "box"})
    assert r.status_code == 200 and r.json()["op_id"]
    assert _targets(c, vid)[plate]["effect"] == "box"


def test_each_hand_edit_is_one_undo(clip, client):
    c, _, vid = _load(client, clip)
    c.post(f"/api/videos/{vid}/request", json={"text": "blur all people"})
    before = _targets(c, vid)
    one = sorted(before)[0]
    c.post(f"/api/videos/{vid}/targets", json={"action": "remove", "track_ids": [one]})
    assert one not in _targets(c, vid)
    assert c.post(f"/api/videos/{vid}/undo").json()["undone"]
    assert _targets(c, vid) == before


@pytest.mark.parametrize("body", [
    {"action": "remove", "track_ids": ["trk_999"]},
    {"action": "remove", "track_ids": []},
    {"action": "effect", "track_ids": ["trk_1"], "effect": "sparkles"},
])
def test_bad_hand_edits_are_400(clip, client, body):
    c, _, vid = _load(client, clip)
    c.post(f"/api/videos/{vid}/request", json={"text": "blur all plates"})
    assert c.post(f"/api/videos/{vid}/targets", json=body).status_code == 400


def test_unknown_action_is_rejected(clip, client):
    c, _, vid = _load(client, clip)
    assert c.post(f"/api/videos/{vid}/targets", json={"action": "move", "track_ids": ["trk_1"]}).status_code == 422
