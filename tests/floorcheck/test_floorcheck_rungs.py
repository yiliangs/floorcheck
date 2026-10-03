"""Slice 4: the ladder of section 7 and the rungs of section 8.

Section 8.2 supplies three exact anchors that this file uses as its ground truth,
because they are computed geometry rather than measurements: "a square scores pi
over four, 0.7854; a two-to-one rectangle scores 0.6981; a three-to-one rectangle
scores 0.5890."
"""

from __future__ import annotations

import math

import pytest

from floorcheck.constants import ABS_TOL_M
from floorcheck.ladder import (
    BEDROOM_PROP_CAP,
    NOT_REACHED,
    NOT_RUN,
    PASSED,
    REJECTED,
    grade,
    rung_room_appendices,
    rung_room_compactness,
    rung_room_outlines,
    rung_room_proportions,
)
from floorcheck.measure import polsby_popper, squareness
from floorcheck.model import plan_from_dict
from floorcheck.report import check
from floorcheck.requirements import Requirement
from floorcheck.scope import NOT_TRANSFERABLE


def rect(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def plan(rooms, **extra):
    payload = {"source": "rplan", "rooms": rooms}
    payload.update(extra)
    return plan_from_dict(payload)


def room(category, ring):
    return {"category": category, "ring": [list(point) for point in ring]}


# --- the measure, against section 8.2's exact anchors ----------------------


def test_polsby_popper_matches_the_three_exact_anchors_of_section_8_2():
    """Section 8.2: "a square scores pi over four, 0.7854; a two-to-one rectangle
    scores 0.6981; a three-to-one rectangle scores 0.5890"."""
    assert polsby_popper(rect(0, 0, 1, 1)) == pytest.approx(math.pi / 4.0)
    assert polsby_popper(rect(0, 0, 1, 1)) == pytest.approx(0.7854, abs=5e-5)
    assert polsby_popper(rect(0, 0, 2, 1)) == pytest.approx(2 * math.pi / 9.0)
    assert polsby_popper(rect(0, 0, 2, 1)) == pytest.approx(0.6981, abs=5e-5)
    assert polsby_popper(rect(0, 0, 3, 1)) == pytest.approx(3 * math.pi / 16.0)
    assert polsby_popper(rect(0, 0, 3, 1)) == pytest.approx(0.5890, abs=5e-5)


def test_squareness_is_the_short_side_over_the_long_not_the_other_way_round():
    """Section 8.3: "the check rejects when a room's squareness, the short side
    over the long side of its unrotated bounding box, is below its floor. A
    description of it as long over short is backwards and inverts any value taken
    from it." Section 4.4
    records the run that caught the inversion: "The first `dataset-p05` run
    rejected 100 per cent of every group at the proportions rung, because
    `propReq` had been filled from the tabled aspect ratio, long side over
    short"."""
    assert squareness(rect(0, 0, 4, 1)) == pytest.approx(0.25)
    assert squareness(rect(0, 0, 1, 4)) == pytest.approx(0.25)
    assert squareness(rect(0, 0, 2, 2)) == pytest.approx(1.0)
    # In the half-open interval from zero to one for every room.
    assert 0.0 < squareness(rect(0, 0, 9, 1)) <= 1.0


def test_the_dominant_direction_correction_is_applied_to_a_rotated_building():
    """Section 2.5: "A building rotated a few degrees off the world axes therefore
    makes every one of its otherwise rectangular rooms register as
    non-axis-aligned. The correction, applied per plan, is a dominant-direction
    alignment ... A spot check on room 0 of `graph_out/0.pickle` finds a
    parallelogram whose four edges are each about 10.3 degrees off the nearest
    axis, consistent with an internally rectangular room in a rotated building
    rather than a skewed room"."""
    from floorcheck.measure import dominant_direction

    angle = math.radians(10.3)
    cosine, sine = math.cos(angle), math.sin(angle)
    turned = tuple(
        (x * cosine - y * sine, x * sine + y * cosine) for x, y in rect(0, 0, 4, 1)
    )
    found = dominant_direction([turned])
    # Without the correction the world-axis box of a rotated 4 by 1 room is far
    # squarer than the room is.
    assert squareness(turned) > squareness(turned, found)
    assert squareness(turned, found) == pytest.approx(0.25, abs=1e-9)


# --- rung 1, section 8.1 ---------------------------------------------------


def test_rung_one_reads_geometry_alone_and_passes_a_sound_ring():
    """Section 8.1: "Reads geometry alone. No request constant and no per-token
    requirement is associated with it in the audit"."""
    assert rung_room_outlines([rect(0, 0, 4, 3)]).verdict == PASSED


def test_rung_one_rejects_a_ring_with_no_area():
    """Section 8.1: "A plan that reaches this rung and fails it has a geometric
    defect the ingest refusals of section 6 did not catch". A collapsed ring is
    such a defect: section 6 measures only how rings cover and overlap the floor."""
    result = rung_room_outlines([((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))])
    assert result.verdict == REJECTED


def test_rung_one_rejects_a_self_intersecting_outline():
    bowtie = ((0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0))
    assert rung_room_outlines([bowtie]).verdict == REJECTED


# --- rung 2, section 8.2 ---------------------------------------------------


def test_rung_two_takes_the_plans_worst_room():
    """Section 8.2: "The rung takes each plan's worst room"."""
    rooms = [room("livingroom", rect(0, 0, 4, 4)), room("bedroom", rect(4, 0, 14, 1))]
    result = rung_room_compactness(plan(rooms), [rect(0, 0, 4, 4), rect(4, 0, 14, 1)], 0.4618)
    assert result.verdict == REJECTED
    assert result.worst["room"] == 1


def test_rung_two_excludes_hallways_as_the_lowest_score_reduction_does():
    """Section 8.2: "excluding hallways as the solver's own lowest-score reduction
    does". MSD's `corridor` category maps to the `Hallway` token (section 4.5)."""
    rings = [rect(0, 0, 4, 4), rect(4, 0, 24, 1)]
    rooms = [room("livingroom", rings[0]), room("corridor", rings[1])]
    result = rung_room_compactness(plan(rooms), rings, 0.4618)
    assert result.verdict == PASSED
    assert result.worst["room"] == 0


def test_rung_two_compares_against_the_corpus_floor_of_section_8_2():
    """Section 8.2: the `dataset-p05` values are "0.4618 for RPLAN and 0.4702 for
    MSD"."""
    # A 4:1 room scores 4 pi 4 / 100 = 0.5027, above RPLAN's floor.
    rings = [rect(0, 0, 4, 1)]
    rooms = [room("bedroom", rings[0])]
    assert rung_room_compactness(plan(rooms), rings, 0.4618).verdict == PASSED
    # A 6:1 room scores 4 pi 6 / 196 = 0.3847, below it.
    rings = [rect(0, 0, 6, 1)]
    rooms = [room("bedroom", rings[0])]
    assert rung_room_compactness(plan(rooms), rings, 0.4618).verdict == REJECTED


def test_rung_two_applies_the_floor_independently_of_any_token_scalar():
    """Section 8.2: "a dimensionless floor in the half-open interval from zero to
    one, applied independently of any token scalar"."""
    rings = [rect(0, 0, 6, 1)]
    for category in ("bedroom", "livingroom", "bathroom", "balcony"):
        rooms = [room(category, rings[0])]
        assert rung_room_compactness(plan(rooms), rings, 0.4618).verdict == REJECTED


# --- rung 3, section 8.3 ---------------------------------------------------


def _reqs(**by_tag):
    return {
        tag: Requirement(
            category=tag,
            area_req=0.0,
            min_width=0.0,
            prop_req=value,
            area_reason="",
            width_reason="",
            prop_reason="",
        )
        for tag, value in by_tag.items()
    }


def test_rung_three_compares_every_room_against_its_own_tokens_floor():
    """Section 8.3: "The floor is a vector rather than a number: every room is
    compared against its own token's `propReq`, indexed by category, where
    compactness compares a plan's worst room against one builder constant"."""
    rings = [rect(0, 0, 4, 1), rect(4, 0, 8, 1)]
    rooms = [room("bathroom", rings[0]), room("kitchen", rings[1])]
    # Both rooms are 0.25 square. The bathroom's floor lets it through and the
    # kitchen's does not.
    result = rung_room_proportions(plan(rooms), rings, _reqs(Bathroom=0.2, Kitchen=0.4))
    assert result.verdict == REJECTED
    assert result.worst["tag"] == "Kitchen"


def test_rung_three_is_skipped_for_a_token_whose_floor_is_zero_or_below():
    """Section 1.4: a `propReq` of 0.0 or below disables the rung, because "the
    proportions check skips a room whose requirement is not positive". Section 8.3: under
    `standards` the value is "disabled, spelled 0.0, for every category", and "no
    `stand-in` or `standards` run document carries a `roomProportions` rejection
    row at all"."""
    rings = [rect(0, 0, 20, 1)]
    rooms = [room("bathroom", rings[0])]
    assert rung_room_proportions(plan(rooms), rings, _reqs(Bathroom=0.0)).verdict == PASSED
    assert rung_room_proportions(plan(rooms), rings, _reqs(Bathroom=-1.0)).verdict == PASSED


def test_rung_three_caps_a_bedrooms_floor_at_sixty_hundredths():
    """Section 8.3: "The solver caps the enforced floor at 0.60 for a `Bedroom` or
    a `PrimaryBedroom`, so a bedroom is never held above that value however strict
    its requirement is"."""
    assert BEDROOM_PROP_CAP == 0.60
    rings = [rect(0, 0, 10, 7)]  # squareness 0.70
    for tag, category in (("Bedroom", "bedroom"), ("PrimaryBedroom", "masterroom")):
        rooms = [room(category, rings[0])]
        # A token floor of 0.9 is capped to 0.6, so a 0.7 room passes.
        result = rung_room_proportions(plan(rooms), rings, _reqs(**{tag: 0.9}))
        assert result.verdict == PASSED
        assert result.worst["propReq"] == pytest.approx(0.60)
    # No cap for any other class: a kitchen at 0.9 is held to 0.9.
    rooms = [room("kitchen", rings[0])]
    result = rung_room_proportions(plan(rooms), rings, _reqs(Kitchen=0.9))
    assert result.verdict == REJECTED


def test_rung_three_scores_rooms_tagged_hallway_which_compactness_skips():
    """Section 8.3: "The rung also scores rooms tagged Hallway, which the
    compactness reduction skips"."""
    rings = [rect(0, 0, 20, 1)]
    rooms = [room("corridor", rings[0])]
    result = rung_room_proportions(plan(rooms), rings, _reqs(Hallway=0.5))
    assert result.verdict == REJECTED
    assert result.worst["tag"] == "Hallway"


def test_rung_three_admits_a_room_within_abstol_of_its_floor():
    """Section 8.3: "A room within `abstol`, one millimetre, of its floor is admitted"."""
    rings = [rect(0, 0, 2.0, 1.0)]  # squareness exactly 0.5
    rooms = [room("kitchen", rings[0])]
    just_over = 0.5 + ABS_TOL_M / 2.0
    assert rung_room_proportions(plan(rooms), rings, _reqs(Kitchen=just_over)).verdict == PASSED
    well_over = 0.5 + ABS_TOL_M * 10
    assert rung_room_proportions(plan(rooms), rings, _reqs(Kitchen=well_over)).verdict == REJECTED


def test_rung_three_gives_no_requirement_to_a_category_that_falls_to_the_unset_tag():
    """Section 8.7: "Any category not in that table takes the unset tag and no
    requirement at all, and the adapter counts it rather than failing on it.
    RPLAN `studyroom` and MSD `stairs` are the two named instances"."""
    rings = [rect(0, 0, 20, 1)]
    for category in ("studyroom", "stairs"):
        rooms = [room(category, rings[0])]
        result = rung_room_proportions(plan(rooms), rings, _reqs(Kitchen=0.9))
        assert result.verdict == PASSED


def test_rung_three_measures_the_unrotated_box_not_the_dominant_direction_frame(tmp_path):
    """Section 8.3: the check rejects when "a room's squareness, the short side
    over the long side of its unrotated bounding box, is below its floor", so it
    reads the solver's unrotated, axis-aligned box. Nothing in section 8.3 rotates the room into
    the plan's dominant wall direction first; that correction is section 2.5's,
    read by rung 4's orthogonality precondition, not by this rung.

    A 4-by-1 kitchen, rotated 10 degrees inside a building of rectangles rotated
    the same 10 degrees, is 0.25 square on its own true axes but about 0.4083
    square measured on the unrotated, world-axis box. A floor of 0.35 sits
    between the two, so the two measures disagree on the verdict: the rung must
    follow the unrotated box, not the dominant-direction-corrected one.
    """
    angle = math.radians(10.0)
    cosine, sine = math.cos(angle), math.sin(angle)

    def spin(ring):
        return tuple((x * cosine - y * sine, x * sine + y * cosine) for x, y in ring)

    kitchen_ring = spin(rect(0, 0, 4, 1))
    filler_a = spin(tuple((x + 10.0, y + 10.0) for x, y in rect(0, 0, 3, 3)))
    filler_b = spin(tuple((x - 10.0, y - 10.0) for x, y in rect(0, 0, 2, 2)))
    rings = [kitchen_ring, filler_a, filler_b]
    rooms = [
        room("kitchen", kitchen_ring),
        room("studyroom", filler_a),
        room("studyroom", filler_b),
    ]

    # squareness(kitchen_ring) with no angle is the unrotated box: about 0.4083.
    # squareness(kitchen_ring, dominant_direction(rings)) is the true 0.25.
    assert squareness(kitchen_ring) == pytest.approx(0.4083, abs=1e-3)

    stats = tmp_path / "category-statistics.md"
    stats.write_text(
        "## rplan\n\n"
        "| RoomTag | count | area p5 | p25 | p50 | p75 | p95 | width p5 | p25 |"
        " p50 | aspect p5 | p25 | p50 | p75 | p95 |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        "| Kitchen | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |"
        " 1.0 | 1.0 | 1.0 | 2.857143 |\n\n"
        "| p01 | p05 | p25 | p50 | p75 |\n"
        "|---|---|---|---|---|\n"
        "| 0.20 | 0.30 | 0.50 | 0.60 | 0.70 |\n",
        encoding="utf-8",
    )

    graded = grade(plan(rooms), list(rings), statistics=str(stats))
    proportions = next(r for r in graded.rungs if r.name == "roomProportions")
    assert proportions.verdict == PASSED, proportions.reason
    assert proportions.worst["squareness"] == pytest.approx(0.4083, abs=1e-3)


# --- rung 4, section 8.4 ---------------------------------------------------


def test_rung_four_faults_a_room_with_a_genuinely_diagonal_wall_segment():
    """Section 8.4: the check reaches the appendix policy only after it verifies
    each room's wall loop, which refuses a segment aligned with neither axis to
    within `abstol`. A corner shoved 1 m sideways is not a near miss.
    Section 9.4: a precondition fault is not a verdict, so the rung is not run,
    and the reason carries the solver's message verbatim."""
    diagonal = ((4.0, 0.0), (8.0, 0.0), (9.0, 3.0), (4.0, 3.0))
    rings = [rect(0, 0, 4, 3), diagonal]
    rooms = [room("livingroom", rings[0]), room("studyroom", rings[1])]
    result = rung_room_appendices(plan(rooms), rings)
    assert result.verdict == NOT_RUN
    assert result.reason.startswith(
        "precondition fault: Room boundary contains a non-orthogonal segment"
    )
    assert result.worst["room"] == 1


def test_rung_four_admits_a_segment_off_axis_by_less_than_abs_tol_m():
    """The tolerance is real, not decorative: a segment within `ABS_TOL_M` of an
    axis is still orthogonal."""
    delta = ABS_TOL_M / 2.0
    near_miss = ((4.0, 0.0), (8.0, 0.0), (8.0 + delta, 3.0), (4.0, 3.0))
    rings = [near_miss]
    rooms = [room("studyroom", rings[0])]
    result = rung_room_appendices(plan(rooms), rings)
    assert result.verdict == NOT_TRANSFERABLE


def test_rung_four_faults_a_segment_off_axis_by_more_than_abs_tol_m():
    """The boundary sits exactly where `ABS_TOL_M` says it does."""
    delta = ABS_TOL_M * 10
    over = ((4.0, 0.0), (8.0, 0.0), (8.0 + delta, 3.0), (4.0, 3.0))
    rings = [over]
    rooms = [room("studyroom", rings[0])]
    result = rung_room_appendices(plan(rooms), rings)
    assert result.verdict == NOT_RUN
    assert result.worst["offAxisM"] == pytest.approx(delta)


def test_rung_four_does_not_reject_a_hallway_with_a_diagonal_segment():
    """Section 8.4: the check skips a room tagged Hallway, and MSD's `corridor` category
    maps to it (section 4.5)."""
    diagonal = ((4.0, 0.0), (8.0, 0.0), (9.0, 3.0), (4.0, 3.0))
    rings = [diagonal]
    rooms = [room("corridor", rings[0])]
    result = rung_room_appendices(plan(rooms), rings)
    assert result.verdict == NOT_TRANSFERABLE


# --- rung 6, section 8.6 ---------------------------------------------------


def test_rung_six_is_not_transferable_and_a_no_entry_plan_still_reaches_a_verdict():
    """Section 8.6: the solver's finalAdjacency
    only re-checks a requested adjacency program, the audit sends none, and a null
    program passes, so the rung is not transferable on the public path. It is
    reported as skipped, never as a pass, a rejection or a stop, and no rung reads
    the entry, so a plan without one is graded like any other."""
    rings = [rect(0, 0, 4, 3), rect(4, 0, 8, 3)]
    rooms = [room("livingroom", rings[0]), room("bedroom", rings[1])]
    graded = grade(plan(rooms), list(rings))
    rung_six = graded.rungs[5]
    assert rung_six.number == 6
    assert rung_six.name == "finalAdjacency"
    assert rung_six.verdict == NOT_TRANSFERABLE
    assert rung_six.reason
    assert graded.passed is True
    assert graded.stopped_at is None
    assert graded.not_run is None

    report = check(plan(rooms)).to_dict()
    assert report["ladder"][5]["verdict"] == NOT_TRANSFERABLE


# --- the ladder, section 7 -------------------------------------------------


def test_the_ladder_runs_the_six_rungs_in_the_order_of_section_7_1():
    """Section 7.1's table: roomOutline, roomCompactness, roomProportions,
    roomAppendices, suiteContainer, finalAdjacency."""
    rings = [rect(0, 0, 4, 3), rect(4, 0, 8, 3)]
    rooms = [room("livingroom", rings[0]), room("bedroom", rings[1])]
    graded = grade(plan(rooms, entry=[0.0, 1.5]), list(rings))
    assert [result.name for result in graded.rungs] == [
        "roomOutline",
        "roomCompactness",
        "roomProportions",
        "roomAppendices",
        "suiteContainer",
        "finalAdjacency",
    ]
    assert [result.number for result in graded.rungs] == [1, 2, 3, 4, 5, 6]


def test_a_rung_that_rejects_stops_the_ladder_and_the_rungs_below_never_run():
    """Section 7.1: "A rung that rejects returns immediately, so a plan is reported
    at the first rung it fails and the rungs below it never run. This is why a
    histogram over stages is a statement about where a corpus stops rather than a
    count of everything wrong with it"."""
    rings = [rect(0, 0, 30, 1)]  # fails compactness
    rooms = [room("bedroom", rings[0])]
    graded = grade(plan(rooms), list(rings))
    assert graded.stopped_at == "roomCompactness"
    assert graded.rungs[0].verdict == PASSED
    assert graded.rungs[1].verdict == REJECTED
    assert all(result.verdict == NOT_REACHED for result in graded.rungs[2:])


def test_rungs_four_five_and_six_are_not_transferable_and_do_not_stop_the_ladder_for_a_clean_plan():
    """Sections 8.4, 8.5 and 8.6 all rule the policy not transferable. Section 7.2
    records that skipping a rung "never changes the sequence ... The surviving
    rungs keep their relative position", so each skipped rung is still reported
    in its place and none stops the ladder, for a plan whose walls all clear
    rung 4's own orthogonality precondition."""
    rings = [rect(0, 0, 4, 3), rect(4, 0, 8, 3)]
    rooms = [room("livingroom", rings[0]), room("bedroom", rings[1])]
    graded = grade(plan(rooms, entry=[0.0, 1.5]), list(rings))
    assert graded.rungs[3].verdict == NOT_TRANSFERABLE
    assert graded.rungs[4].verdict == NOT_TRANSFERABLE
    assert graded.rungs[5].verdict == NOT_TRANSFERABLE
    assert graded.rungs[3].reason and graded.rungs[4].reason and graded.rungs[5].reason
    assert graded.passed is True


def test_a_rung_four_precondition_fault_stops_the_ladder_with_no_verdict():
    """Section 9.4: a plan a rung stops on a precondition fault "was admitted and
    received no verdict". The ladder stops there, so rungs 5 and 6 are not
    reached, and no rung is named as the one the plan was stopped at."""
    diagonal = ((4.0, 0.0), (8.0, 0.0), (9.0, 3.0), (4.0, 3.0))
    rings = [rect(0, 0, 4, 3), diagonal]
    rooms = [room("livingroom", rings[0]), room("studyroom", rings[1])]
    graded = grade(plan(rooms, entry=[0.0, 1.5]), list(rings))
    assert graded.stopped_at is None
    assert graded.not_run == "roomAppendices"
    assert not graded.passed
    assert graded.rungs[3].verdict == NOT_RUN
    assert graded.rungs[4].verdict == NOT_REACHED
    assert graded.rungs[5].verdict == NOT_REACHED


def test_a_not_transferable_rung_is_neither_a_pass_nor_a_failure():
    rings = [rect(0, 0, 4, 3)]
    rooms = [room("livingroom", rings[0])]
    graded = grade(plan(rooms), list(rings))
    for result in graded.rungs[3:6]:
        assert result.verdict not in (PASSED, REJECTED)


def test_the_standards_source_never_rejects_at_the_proportions_rung():
    """Section 8.3: "`standards` disables `propReq` for every category ... no
    `stand-in` or `standards` run document carries a `roomProportions` rejection
    row at all. The `dataset-p05` calibration is the only reason this rung has ever
    rejected a plan"."""
    rings = [rect(0, 0, 20, 1)]
    rooms = [room("bathroom", rings[0])]
    graded = grade(plan(rooms), list(rings), source="standards")
    proportions = next(r for r in graded.rungs if r.name == "roomProportions")
    assert proportions.verdict != REJECTED
