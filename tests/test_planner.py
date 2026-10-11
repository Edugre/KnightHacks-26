"""The planner's plan checking and the executor's only/except rules, without a model."""
from censor.agent import planner


def test_parse_plan_normalizes_what_the_llm_writes():
    p = planner.parse_plan('Sure! {"action": "mask", "effect": "pixelate", "targets": ['
                           '{"object": "woman", "only": null, "except": "Is this a child?", "time": "first 5 seconds"},'
                           '{"object": "Plates"}]}')
    assert p.effect == "pixelate"
    assert [(t.object, t.only, t.exclude, t.time) for t in p.targets] == [
        ("person", ["Is this a woman?"], ["Is this a child?"], "first 5 seconds"),  # a kind of person -> a question
        ("license plate", [], [], None)]


def test_parse_plan_rejects_plans_it_cannot_run():
    import pytest
    for bad in ("no json here", '{"action": "mask", "targets": []}', '{"action": "explode", "targets": []}'):
        with pytest.raises(ValueError):
            planner.parse_plan(bad)
    assert planner.parse_plan('{"action": "clarify", "message": "What should I hide?"}').message == "What should I hide?"


class FakeSession:
    def __init__(self, owner=None):
        self.timeline = type("T", (), {"targets": {}})()
        self._owner = owner or {}

    def person_for(self, ids):
        return {i: self._owner.get(i) for i in ids}


def run(plan, answers, owner=None):
    """Execute with fake tools: find_objects returns t1..t3, check_attribute answers from `answers`."""
    calls = []

    def call(name, args):
        calls.append((name, args))
        if name == "find_objects":
            return [{"track_id": t} for t in ("t1", "t2", "t3")]
        if name == "check_attribute":
            return [{"track_id": i, "answer": answers[args["question"]].get(i, "no")} for i in args["track_ids"]]
        return "op"
    msg = planner.execute(plan, FakeSession(owner), call)
    masked = [a["track_ids"] for n, a in calls if n == "add_masks"]
    return (masked[0] if masked else []), calls, msg


def test_only_keeps_yes_and_unknown_except_drops_only_yes():
    only = planner.Plan(targets=[planner.Target("person", only=["Is this a woman?"])])
    assert run(only, {"Is this a woman?": {"t1": "yes", "t2": "unknown"}})[0] == ["t1", "t2"]
    exc = planner.Plan(targets=[planner.Target("person", exclude=["Is this a man wearing red?"])])
    # t2 is unsure, so it stays masked: an unsure answer means more blur, never less
    assert run(exc, {"Is this a man wearing red?": {"t1": "yes", "t2": "unknown"}})[0] == ["t2", "t3"]


def test_face_questions_are_asked_about_the_person_the_face_belongs_to():
    plan = planner.Plan(targets=[planner.Target("face", exclude=["Is this the face of a man wearing red?"])])
    masked, calls, _ = run(plan, {"Is this a man wearing red?": {"p1": "yes"}}, owner={"t1": "p1", "t2": "p2"})
    asked = [a for n, a in calls if n == "check_attribute"][0]
    assert asked["question"] == "Is this a man wearing red?" and sorted(asked["track_ids"]) == ["p1", "p2", "t3"]
    assert masked == ["t2", "t3"]  # t1 is the face of the man in red; t3 has no person, so its face was asked


def test_the_template_plan_runs_through_the_same_executor():
    from censor.agent import intents
    plan = planner.from_template(intents.parse("blur all plates and the woman around halfway"))
    assert [(t.object, t.only, t.time) for t in plan.targets] == [
        ("license plate", [], None), ("person", ["Is this a woman?"], "halfway")]


def test_except_one_person_spares_only_the_best_match():
    """"Everyone but the lady": blurry passers-by the VLM thought might be women must stay blurred."""
    calls = []

    def call(name, args):
        calls.append(name)
        if name == "find_objects":
            return [{"track_id": t} for t in ("t1", "t2", "t3")]
        if name == "check_attribute":
            p = {"t1": 0.99, "t2": 0.7, "t3": 0.1}
            return [{"track_id": i, "answer": "yes" if p[i] > 0.65 else "no", "p_yes": p[i]} for i in args["track_ids"]]
        return "op"
    masked = {}
    for one in (True, False):
        plan = planner.Plan(targets=[planner.Target("person", exclude=["Is this a woman?"], one=one)])
        planner.execute(plan, FakeSession(), lambda n, a: masked.setdefault(one, a["track_ids"]) if n == "add_masks" else call(n, a))
    assert masked[True] == ["t2", "t3"]   # one lady: only the surest match stays visible
    assert masked[False] == ["t3"]        # "but the women": every match stays visible


def test_except_one_person_breaks_a_tie_by_screen_presence():
    """street_talk, "everyone except the woman": she and a passer-by seen for 1 s both answered 1.0, and the passer-by
    was spared while she was blurred. A tie goes to the track that fills more of the video."""
    from types import SimpleNamespace as NS
    kf = lambda n, size: [NS(f=f, box=(0, 0, size, size)) for f in range(n)]  # noqa: E731
    tracks = {"t1": NS(keyframes=kf(25, 80)), "t2": NS(keyframes=kf(500, 400)), "t3": NS(keyframes=kf(500, 400))}
    s = FakeSession()
    s.tracks = NS(get=tracks.get)
    masked = []

    def call(name, args):
        if name == "find_objects":
            return [{"track_id": t} for t in ("t1", "t2", "t3")]
        if name == "check_attribute":
            p = {"t1": 1.0, "t2": 0.997, "t3": 0.0}
            return [{"track_id": i, "answer": "yes" if p[i] > 0.65 else "no", "p_yes": p[i]} for i in args["track_ids"]]
        if name == "add_masks":
            masked.extend(args["track_ids"])
        return "op"
    planner.execute(planner.Plan(targets=[planner.Target("person", exclude=["Is this a woman?"], one=True)]), s, call)
    assert masked == ["t1", "t3"]  # the woman on screen the whole clip (t2) stays visible, not the 1 s passer-by


def test_a_time_phrase_after_in_is_not_clothing():
    from censor.agent import intents
    assert intents.parse("blur the woman in the first 3 seconds").attribute_questions == ["Is this a woman?"]


def test_one_person_or_a_group_is_read_from_the_words():
    from censor.agent import intents
    assert intents.one_person("hide everybody but the lady") is True
    assert intents.one_person("blur everyone except the guy in the red hoodie") is True
    assert intents.one_person("pixelate all faces but mine") is True
    assert intents.one_person("blur everyone but the men") is False
    assert intents.one_person("blur everyone except people in red") is False
    assert intents.one_person("blur all plates") is None
