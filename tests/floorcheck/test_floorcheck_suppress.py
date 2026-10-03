"""A named check's rejection suppressed: the check is still evaluated, and a plan
it would reject continues to the checks below it, which is the counterfactual
the paper reports with the proportion check's rejection suppressed. Without the
parameter the ladder is the one section 7.1 specifies."""

import pytest

from floorcheck.ladder import NOT_REACHED, NOT_RUN, PASSED, REJECTED, SUPPRESSED, grade
from floorcheck.model import plan_from_dict
from floorcheck.report import check


def rect(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def plan(rooms):
    return plan_from_dict({"source": "rplan", "rooms": rooms, "entry": [0.0, 0.5]})


def room(category, ring):
    return {"category": category, "ring": [list(point) for point in ring]}


def verdicts(graded):
    return {result.name: result.verdict for result in graded.rungs}


# A 4 by 1.6 m bedroom: squareness 0.40, under every plan-level bedroom floor,
# with a Polsby-Popper score of 0.64, over every plan-level compactness floor.
NARROW_BEDROOM = rect(0, 0, 4, 1.6)
LIVING = rect(4, 0, 8, 3)


def test_a_plan_failing_only_the_suppressed_check_continues_and_passes():
    rings = [NARROW_BEDROOM, LIVING]
    rooms = [room("bedroom", rings[0]), room("livingroom", rings[1])]

    plain = grade(plan(rooms), list(rings))
    assert plain.stopped_at == "roomProportions"
    assert not plain.passed

    graded = grade(plan(rooms), list(rings), suppress={"roomProportions"})
    assert graded.passed
    assert graded.stopped_at is None
    assert verdicts(graded)["roomProportions"] == SUPPRESSED
    proportions = next(r for r in graded.rungs if r.name == "roomProportions")
    assert proportions.reason == next(r for r in plain.rungs if r.name == "roomProportions").reason
    assert NOT_REACHED not in verdicts(graded).values()


def test_a_plan_failing_the_suppressed_check_and_a_later_one_is_reported_at_the_later():
    # 30 by 1 m: fails compactness and, below it, proportions.
    rings = [rect(0, 0, 30, 1), rect(30, 0, 34, 3)]
    rooms = [room("bathroom", rings[0]), room("livingroom", rings[1])]

    assert grade(plan(rooms), list(rings)).stopped_at == "roomCompactness"

    graded = grade(plan(rooms), list(rings), suppress={"roomCompactness"})
    assert graded.stopped_at == "roomProportions"
    assert verdicts(graded)["roomCompactness"] == SUPPRESSED
    assert verdicts(graded)["roomProportions"] == REJECTED
    assert verdicts(graded)["roomAppendices"] == NOT_REACHED


def test_a_right_angle_stop_after_a_suppressed_rejection_is_a_stop_without_a_verdict():
    # Squareness on the unrotated 4.3 by 1.6 m box is 0.37, under the bedroom
    # floor, and the east wall is 0.3 m off both axes.
    skewed = ((0.0, 0.0), (4.0, 0.0), (4.3, 1.6), (0.0, 1.6))
    rings = [skewed, rect(4.3, 0, 8, 3)]
    rooms = [room("bedroom", rings[0]), room("livingroom", rings[1])]

    assert grade(plan(rooms), list(rings)).stopped_at == "roomProportions"

    graded = grade(plan(rooms), list(rings), suppress={"roomProportions"})
    assert graded.stopped_at is None
    assert graded.not_run == "roomAppendices"
    assert not graded.passed
    assert verdicts(graded)["roomProportions"] == SUPPRESSED
    assert verdicts(graded)["roomAppendices"] == NOT_RUN
    assert verdicts(graded)["suiteContainer"] == NOT_REACHED


def test_a_check_that_passes_is_reported_passed_under_suppression():
    rings = [rect(0, 0, 4, 3), LIVING]
    rooms = [room("bedroom", rings[0]), room("livingroom", rings[1])]
    graded = grade(plan(rooms), list(rings), suppress={"roomProportions"})
    assert verdicts(graded)["roomProportions"] == PASSED
    assert graded.passed


def test_suppressing_a_check_the_ladder_does_not_have_is_an_error():
    rings = [NARROW_BEDROOM, LIVING]
    rooms = [room("bedroom", rings[0]), room("livingroom", rings[1])]
    with pytest.raises(ValueError, match="roomProportion"):
        grade(plan(rooms), list(rings), suppress={"roomProportion"})


def test_the_report_names_the_suppressed_check_only_when_one_is():
    rings = [NARROW_BEDROOM, LIVING]
    rooms = [room("bedroom", rings[0]), room("livingroom", rings[1])]
    assert "suppressed" not in check(plan(rooms)).to_dict()["header"]
    report = check(plan(rooms), suppress={"roomProportions"}).to_dict()
    assert report["header"]["suppressed"] == ["roomProportions"]
    assert report["verdict"] == "passed every transferable rung"
    assert report["ladder"][2]["verdict"] == SUPPRESSED
