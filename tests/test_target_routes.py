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
    assert c.post(f"/api/videos/{vid}/targets", json={"action": "rotate", "track_ids": ["trk_1"]}).status_code == 422


def test_move_retracks_from_the_corrected_box(clip, client):
    c, app_mod, vid = _load(client, clip)
    c.post(f"/api/videos/{vid}/request", json={"text": "blur all plates"})
    (plate,) = _targets(c, vid)
    f = 30
    old_box = next(k["box"] for k in _targets(c, vid)[plate]["keyframes"] if k["f"] == f)
    box = [old_box[0] + 20, old_box[1] - 10, old_box[2] + 20, old_box[3] - 10]
    r = c.post(f"/api/videos/{vid}/targets", json={"action": "move", "track_ids": [plate], "frame": f, "box": box})
    assert r.status_code == 200, r.text
    new = r.json()["track"]["track_id"]
    assert len(r.json()["op_ids"]) == 2
    tl = _targets(c, vid)
    end = max(k["f"] for k in tl[new]["keyframes"])
    assert min(k["f"] for k in tl[new]["keyframes"]) == f                # the new one starts there
    assert not any(f <= k["f"] <= end for k in tl[plate]["keyframes"])  # the old box is off while it runs
    assert max(k["f"] for k in tl[plate]["keyframes"] if k["f"] < f) == f - 1
    assert tl[new]["keyframes"][0]["box"] == box and tl[new]["label"] == "license plate"
    assert tl[new]["effect"] == tl[plate]["effect"]
    # both ops share a request id, so the UI undoes them together; two undos restore the old mask
    ops = c.get(f"/api/videos/{vid}").json()["ops"]
    assert ops[-1]["request_id"] == ops[-2]["request_id"] != ops[-3]["request_id"]
    c.post(f"/api/videos/{vid}/undo"); c.post(f"/api/videos/{vid}/undo")
    assert set(_targets(c, vid)) == {plate} and len(_targets(c, vid)[plate]["keyframes"]) > f


def test_move_keeps_the_old_mask_where_the_new_track_stops(clip, client, monkeypatch):
    """The box tracker can lose the object early; past that point the old mask must come back (fail closed)."""
    c, app_mod, vid = _load(client, clip)
    c.post(f"/api/videos/{vid}/request", json={"text": "blur all plates"})
    (plate,) = _targets(c, vid)
    import censor.agent.tools as tools_mod
    real = tools_mod.track_mod.track_box

    def short(video, f, box, store, label):  # a track that gives up after 10 frames
        t = real(video, f, box, store, label)
        t.keyframes = t.keyframes[:11]
        return t
    monkeypatch.setattr(tools_mod.track_mod, "track_box", short)
    f = 30
    old_box = next(k["box"] for k in _targets(c, vid)[plate]["keyframes"] if k["f"] == f)
    r = c.post(f"/api/videos/{vid}/targets", json={"action": "move", "track_ids": [plate], "frame": f, "box": old_box})
    assert r.status_code == 200, r.text
    tl = _targets(c, vid)
    new = r.json()["track"]["track_id"]
    assert max(k["f"] for k in tl[new]["keyframes"]) == f + 10
    old_frames = {k["f"] for k in tl[plate]["keyframes"]}
    assert not old_frames & set(range(f, f + 11)) and f + 11 in old_frames  # back right after the new track ends


@pytest.mark.parametrize("body", [
    {"action": "move", "track_ids": ["trk_1"], "frame": 10},                          # no box
    {"action": "move", "track_ids": ["trk_1", "trk_2"], "frame": 10, "box": [0, 0, 50, 50]},
    {"action": "move", "track_ids": ["trk_1"], "frame": 99999, "box": [0, 0, 50, 50]},
    {"action": "move", "track_ids": ["trk_2"], "frame": 10, "box": [0, 0, 50, 50]},   # found but not masked
])
def test_bad_moves_are_400(clip, client, body):
    c, _, vid = _load(client, clip)
    c.post(f"/api/videos/{vid}/request", json={"text": "blur all plates"})
    c.post(f"/api/videos/{vid}/request", json={"text": "unblur the faces"})  # scans faces without masking them
    assert c.post(f"/api/videos/{vid}/targets", json=body).status_code == 400
