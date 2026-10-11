"""The person-track stitcher: join pieces of one person, never two people."""
import numpy as np

from censor.core.tracks import Keyframe
from censor.vision import stitch as st


def piece(f0, f1, x0, vx=2.0, y=100.0, w=30.0, h=80.0):
    """A person box moving right at vx px per frame, from frame f0 to f1."""
    return [Keyframe(f=f, box=(x0 + vx * (f - f0), y, x0 + vx * (f - f0) + w, y + h), conf=0.8) for f in range(f0, f1 + 1)]


RED, BLUE = np.zeros(32, np.float32), np.zeros(32, np.float32)
RED[0], BLUE[20] = 1.0, 1.0


def test_a_person_lost_for_a_few_frames_is_one_track_again():
    a = piece(0, 20, 100)
    b = piece(26, 50, 100 + 2 * 26)  # where a was heading
    out = st.stitch([a, b], [st.Ends(RED, RED), st.Ends(RED, RED)])
    assert len(out) == 1 and [k.f for k in out[0]][:3] == [0, 1, 2] and out[0][-1].f == 50


def test_different_people_stay_apart():
    a = piece(0, 20, 100)
    far = piece(26, 50, 400)                 # appears 300 px away
    big = piece(26, 50, 152, h=200)          # where a was heading, but much taller: someone nearer the camera
    other = piece(26, 50, 152)               # right place and size, other colors
    late = piece(80, 100, 220)               # too long after
    for b, eb in ((far, RED), (big, RED), (other, BLUE), (late, RED)):
        assert len(st.stitch([a, b], [st.Ends(RED, RED), st.Ends(eb, eb)])) == 2


def test_each_piece_is_joined_once_best_match_first():
    a = piece(0, 20, 100)
    close = piece(23, 40, 146)   # 3 frames later, right on the predicted spot
    also = piece(28, 40, 157)    # 8 frames later, also plausible
    out = st.stitch([a, close, also], [st.Ends(RED, RED)] * 3)
    assert len(out) == 2 and any(t[0].f == 0 and t[-1].f == 40 and 23 in [k.f for k in t] for t in out)


def test_a_duplicate_on_top_of_the_person_is_folded_in_and_chains_work():
    a = piece(0, 20, 100)
    dup = piece(18, 30, 136)     # starts while a still runs, on top of it
    c = piece(36, 50, 172)       # continues dup after a short gap
    out = st.stitch([a, dup, c], [st.Ends(RED, RED)] * 3)
    assert len(out) == 1 and out[0][0].f == 0 and out[0][-1].f == 50
    assert len({k.f for k in out[0]}) == len(out[0])  # no frame twice


def test_color_hist_ignores_the_box_edges():
    img = np.zeros((100, 100, 3), np.uint8)
    img[:, 40:60] = (0, 0, 255)  # a red stripe down the middle, black around it
    h = st.color_hist(img, (20, 0, 80, 100))
    assert h.sum() > 0.99 and h.argmax() != 0  # mostly the red middle, not the black edges
