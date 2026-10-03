"""Slice 2b: the tiling rule of specification section 6.

Fixtures are hand written from the specification's own examples. No dataset file
is read.
"""

from __future__ import annotations

import pytest

from floorcheck.constants import CLOSED_SEAM_SHARE, JOIN_TOL_M, OVERLAP_SHARE
from floorcheck.hygiene import quantise_point
from floorcheck.ingest import ingest
from floorcheck.model import plan_from_dict

RPLAN_PIXEL = 18.0 / 256.0  # section 2.3, 0.0703125 m


def q(value: float) -> float:
    """Snap one coordinate onto the quantisation lattice of section 2.2.

    Fixtures are built on the lattice so that an expected area is the area the
    ingest actually sees. Section 2.2 quantises every ring "before anything
    geometric is asserted", so a fixture written at round metres is silently
    moved by up to half a step and its nominal area is not the area under test.
    Log finding 7 records what that does to a source whose own grid is not a
    multiple of this step.
    """
    return quantise_point((value, 0.0))[0]


def square(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    x0, y0, x1, y1 = q(x0), q(y0), q(x1), q(y1)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def plan(rooms, source="rplan", **extra):
    payload = {"source": source, "rooms": rooms}
    payload.update(extra)
    return plan_from_dict(payload)


def room(category, ring):
    return {"category": category, "ring": ring}


def two_rooms_that_tile():
    return [
        room("livingroom", square(0.0, 0.0, 4.0, 3.0)),
        room("bedroom", square(4.0, 0.0, 7.0, 3.0)),
    ]


# --- the happy path --------------------------------------------------------


def test_rooms_that_tile_exactly_are_accepted_with_no_seam_closed():
    """Section 9.1: "RPLAN is also the control on the width rule, because its
    rooms touch exactly: not one RPLAN plan has a seam closed into a room"."""
    result = ingest(plan(two_rooms_that_tile()))
    assert not result.refused
    assert result.seams_closed == 0
    assert result.measurements["uncoveredArea"] == pytest.approx(0.0, abs=1e-9)
    assert result.measurements["twiceClaimedArea"] == pytest.approx(0.0, abs=1e-9)


def test_the_root_outline_is_the_union_of_the_rooms_and_never_a_supplied_polygon():
    """Section 6.1: "The root outline is the union of the room rings the adapter
    emits, taken after the metre conversion and the quantisation, and never a
    boundary polygon the dataset supplied." Section 6.1 again: "That is a
    statement about the root the solver receives and not about the floor extent of
    section 6.3, which is a different object and does not become the root"."""
    result = ingest(
        plan(two_rooms_that_tile(), floorExtent=square(-5.0, -5.0, 20.0, 20.0))
    )
    assert not result.refused
    # The union of the two rooms, not the 625 square metres the extent covers.
    assert result.outline.area == pytest.approx(q(7.0) * q(3.0), rel=1e-9)


# --- the uncovered half, section 6.3 --------------------------------------


def test_a_hole_exactly_one_source_pixel_wide_is_closed_rather_than_refused():
    """Section 6.3: "At most one pixel, not narrower than one pixel. The
    distinction decides whether the rule does anything at all. The test is whether
    the hole survives being eroded by half the tolerance, so a hole exactly one
    pixel wide is closed rather than refused"."""
    gap = RPLAN_PIXEL
    rooms = [
        room("livingroom", square(0.0, 0.0, 4.0, 3.0)),
        room("bedroom", square(4.0 + gap, 0.0, 7.0, 3.0)),
        room("kitchen", square(0.0, 3.0, 7.0, 5.0)),
        room("bathroom", square(0.0, -2.0, 7.0, 0.0)),
    ]
    result = ingest(plan(rooms))
    assert not result.refused, result.detail
    assert result.seams_closed >= 1
    assert result.measurements["closedSeamArea"] > 0.0


def test_a_hole_wider_than_one_source_pixel_refuses_the_plan():
    """Section 6.3: "The plan is then refused if any uncovered floor survives,
    measured as section 6.2 measures it and not by a second enumeration here ...
    There is no share allowance on the surviving area: it refuses above zero"."""
    gap = RPLAN_PIXEL * 6
    rooms = [
        room("livingroom", square(0.0, 0.0, 4.0, 3.0)),
        room("bedroom", square(4.0 + gap, 0.0, 7.0, 3.0)),
        room("kitchen", square(0.0, 3.0, 7.0, 5.0)),
        room("bathroom", square(0.0, -2.0, 7.0, 0.0)),
    ]
    result = ingest(plan(rooms))
    assert result.refusal == "untiled-rooms"
    assert result.measurements["uncoveredArea"] > 0.0


def test_the_uncovered_half_refuses_above_zero_with_no_share_allowance():
    """Section 6.3: "Any uncovered area that survives the closing refuses the
    plan; there is no share allowance left on this half." A courtyard of one
    square metre in a very large floor is still a refusal."""
    rooms = [
        room("livingroom", square(0.0, 0.0, 30.0, 12.0)),
        room("bedroom", square(0.0, 13.0, 30.0, 25.0)),
        room("kitchen", square(0.0, 12.0, 12.0, 13.0)),
        room("bathroom", square(13.0, 12.0, 30.0, 13.0)),
    ]
    result = ingest(plan(rooms))
    assert result.refusal == "untiled-rooms"


def test_an_outlying_part_further_than_one_pixel_from_the_mass_is_uncovered_floor():
    """Section 6.2: "Measured as the area of the union's holes plus the area of
    its outlying parts"."""
    rooms = [
        room("livingroom", square(0.0, 0.0, 4.0, 3.0)),
        room("bedroom", square(4.0, 0.0, 7.0, 3.0)),
        room("balcony", square(20.0, 20.0, 22.0, 22.0)),
    ]
    result = ingest(plan(rooms))
    assert result.refusal == "untiled-rooms"
    assert result.measurements["outlyingArea"] == pytest.approx(
        (q(22.0) - q(20.0)) ** 2, rel=1e-9
    )


def test_a_channel_within_one_pixel_is_bridged_for_every_source():
    """Section 9.2, the second ruling of 2026-09-16: "such a gap is bridged, for
    every source". Section 6.3: "It is closed whether it is a hole in the union or
    a channel between two parts of it, which is the symmetric form the rule took
    on 2026-09-16"."""
    gap = RPLAN_PIXEL * 0.9
    rooms = [
        room("livingroom", square(0.0, 0.0, 4.0, 3.0)),
        room("bedroom", square(4.0 + gap, 0.0, 7.0, 3.0)),
    ]
    result = ingest(plan(rooms))
    assert not result.refused, result.detail
    assert result.seams_closed >= 1


def test_a_one_pixel_channel_with_a_bend_is_bridged_rather_than_refused():
    """Section 6.1: a channel "has no bounded width to erode, so the gap is read
    off a morphological closing of the union at the tolerance instead". Section
    6.3 counts the residue a bridged channel leaves at a right-angled corner, "a
    pixel and a half across its diagonal", as already accepted. An L-shaped
    channel one House-GAN++ pixel wide is judged by the closing alone, as
    `planaudit.geometry._bridge_seams` judges it, and not refused for its corner
    surviving erosion."""
    pixel = 0.28125
    rooms = [
        room("bedroom", [[0.0, 0.0], [18.0, 0.0], [18.0, 18.0], [0.0, 18.0]]),
        room(
            "livingroom",
            [
                [18.0 + pixel, 0.0],
                [20.25, 0.0],
                [20.25, 20.25],
                [0.0, 20.25],
                [0.0, 18.0 + pixel],
                [18.0 + pixel, 18.0 + pixel],
            ],
        ),
    ]
    result = ingest(plan(rooms, source="housegan++"))
    assert not result.refused, result.detail
    assert result.seams_closed >= 1
    assert result.measurements["uncoveredArea"] == pytest.approx(0.0, abs=1e-6)


def test_a_self_crossing_ring_keeps_its_largest_lobe_and_discards_the_rest():
    """Section 2.1: "the larger lobe is the room, and every lobe dropped is
    discarded". The union is taken over the repaired rooms, as
    `planaudit.geometry.tile` takes it (tests/test_tiling.py reads the same
    figure eight as 16 square metres), so a lobe on the edge of the plan lies
    outside the outline and is not measured as uncovered floor."""
    a, b = q(4.0), q(4.5)
    figure_eight = [
        [0.0, 0.0], [a, 0.0], [a, a], [b, a],
        [b, b], [a, b], [a, a], [0.0, a],
    ]
    result = ingest(plan([room("livingroom", figure_eight)]))
    assert not result.refused, result.detail
    assert len(result.rings) == 1
    assert result.outline.area == pytest.approx(a * a), "the largest lobe alone"
    assert result.measurements["uncoveredArea"] == pytest.approx(0.0, abs=1e-9)


def test_the_tolerance_is_the_source_pixel_and_differs_between_sources():
    """Section 6.3's table: House-GAN++ at 18/64 m, HouseDiffusion at 18/256 m.
    The same geometry is a bridged seam on one source and a refusal on the other.

    The floor is deliberately large. Section 6.3: "What the share of floor
    measures instead is a pixel times the plan's interior wall length over its
    area, which rises with the number of rooms and falls with the size of the
    plan", so a two-room fixture of 21 square metres trips the backstop on a
    seam the width rule is happy with."""
    gap = 18.0 / 64.0  # one House-GAN++ pixel, four HouseDiffusion pixels
    rooms = [
        room("livingroom", square(0.0, 0.0, 12.0, 9.0)),
        room("bedroom", square(12.0 + gap, 0.0, 24.0, 9.0)),
    ]
    assert not ingest(plan(rooms, source="housegan++")).refused
    assert ingest(plan(rooms, source="gsdiff")).refusal == "untiled-rooms"


def test_the_synthetic_fixture_is_held_to_the_quantisation_step_not_to_exact_tiling():
    """Section 6.3's table gives the synthetic fixture a source pixel of 0.0,
    "stated, and held to one quantisation step", which agrees with the
    subsection's widening rule: "The adapter widens that length by its own
    quantisation step before applying it ... because it gives a plan with no
    source frame of its own the floating point slack the union arithmetic needs
    and nothing more."

    Section 2.2 has already snapped every coordinate onto the step, so the
    smallest gap a plan can express is exactly one step, and a tolerance of
    nought widened by one step closes it. Log finding 11 is why the table cell
    reads that way rather than "held to tiling exactly": no source is held to
    tiling exactly, and the strictest any source can be is a gap of two
    quantisation steps."""
    one_step = [
        room("livingroom", square(0.0, 0.0, 4.0, 3.0)),
        room("bedroom", square(q(4.0) + JOIN_TOL_M, 0.0, 7.0, 3.0)),
    ]
    assert not ingest(plan(one_step, source="synthetic")).refused

    two_steps = [
        room("livingroom", square(0.0, 0.0, 4.0, 3.0)),
        room("bedroom", square(q(4.0) + 2 * JOIN_TOL_M, 0.0, 7.0, 3.0)),
    ]
    assert ingest(plan(two_steps, source="synthetic")).refusal == "untiled-rooms"
    # The same geometry is a bridged seam on RPLAN, whose pixel is two steps.
    assert not ingest(plan(two_steps, source="rplan")).refused


def test_a_source_that_states_no_pixel_is_refused_rather_than_guessed_at():
    """Section 6.3: "The pixel is stated by the reader, not inferred"."""
    result = ingest(plan(two_rooms_that_tile(), source="some-new-generator"))
    assert result.refusal == "unknown-source-pixel"


def test_a_seam_is_clipped_to_the_floor_extent_where_the_plan_carries_one():
    """Section 6.3: "a channel left open between two MSD rooms may lie outside the
    floor the partition ruled was floor, and closing it would not be expressing a
    wall the source could not draw but handing a room a slot the building does not
    have ... Every region either repair would fill is therefore clipped to the
    plan's floor extent where the plan carries one"."""
    gap = 0.3  # under one MSD pixel of 0.38 m
    rooms = [
        room("livingroom", square(0.0, 0.0, 12.0, 9.0)),
        room("bedroom", square(12.0 + gap, 0.0, 24.0, 9.0)),
    ]
    # An extent covering only the lower half of the channel, standing for the
    # floor the partition ruled was floor.
    extent = square(0.0, 0.0, 24.0, 4.5)
    clipped = ingest(plan(rooms, source="msd", floorExtent=extent))
    unclipped = ingest(plan(rooms, source="msd"))
    assert unclipped.measurements["closedSeamArea"] > clipped.measurements["closedSeamArea"]
    # Roughly the half of the seam the extent covers, which is the point of the
    # clause: section 6.3 measures that "the share of the channel lying inside the
    # extent runs from 1.7 to 68.9 per cent".
    assert clipped.measurements["closedSeamArea"] == pytest.approx(
        unclipped.measurements["closedSeamArea"] / 2.0, rel=0.1
    )
    # What survives the clip is a notch open at one end, which is neither a hole
    # nor an outlying part, so section 6.2's measurement does not see it and the
    # plan is not refused. That is coherent rather than a gap, because section 6.1
    # derives the root outline from the leaves: a notch is the floor's shape, not
    # floor no room covers. Log finding 12 records the wording that obscures it.
    assert not clipped.refused
    assert clipped.measurements["uncoveredArea"] == pytest.approx(0.0, abs=1e-9)


def test_a_plan_with_no_extent_is_not_clipped_at_all():
    """Section 6.3: "a plan that carries none is not clipped at all and behaves
    as before". Log finding 10 is why the section reads that way rather than
    "clipped to nothing", which read as its own opposite."""
    gap = 0.3
    rooms = [
        room("livingroom", square(0.0, 0.0, 12.0, 9.0)),
        room("bedroom", square(12.0 + gap, 0.0, 24.0, 9.0)),
    ]
    result = ingest(plan(rooms, source="msd"))
    assert not result.refused
    width = q(12.0 + gap) - q(12.0)
    height = q(9.0)
    # Section 6.3 records the residue a closing leaves at a right-angled corner,
    # "0.0017 square metres, at the corners, where a right-angled band is a pixel
    # and a half across its diagonal", so the closed area is the seam less that.
    assert result.measurements["closedSeamArea"] == pytest.approx(
        width * height, rel=0.05
    )
    assert result.measurements["closedSeamArea"] <= width * height


def test_the_closed_seam_share_backstop_sits_at_four_per_cent():
    """Section 6.3: "the cap at four per cent of the floor is the backstop under
    it ... the largest closed share carried by a plan with no uncovered floor and
    no twice-claimed floor is 3.44 per cent"."""
    assert CLOSED_SEAM_SHARE == 0.04


def test_the_backstop_refuses_a_plan_that_is_a_mesh_of_unexpressed_walls():
    """Section 6.3: "The width test is the rule; the cap ... is the backstop under
    it, against a plan that is not one floor with a few unexpressed walls but a
    mesh of them"."""
    # Many narrow rooms on a coarse frame: every wall is a bridged seam, so the
    # closed share climbs past the cap while every gap stays under one pixel.
    pixel = 18.0 / 32.0  # House-GAN
    gap = pixel * 0.9
    rooms = []
    x = 0.0
    for index in range(12):
        rooms.append(room("bedroom", square(x, 0.0, x + 1.0, 3.0)))
        x += 1.0 + gap
    result = ingest(plan(rooms, source="housegan"))
    assert result.refusal == "closed-seam-share", result.detail
    assert result.measurements["closedSeamShare"] > CLOSED_SEAM_SHARE


# --- the twice-claimed half, section 6.4 ----------------------------------


def test_a_room_wholly_inside_another_is_refused_and_the_pair_is_named():
    """Section 6.4: "Above it a room genuinely contains another ... the honest
    answer is an ingest refusal naming the area". And: "the rooms it is made of
    are named, by finding every pair where more than 99 per cent of one ring's
    area lies inside another's"."""
    rooms = [
        room("livingroom", square(0.0, 0.0, 10.0, 10.0)),
        room("stairs", square(3.0, 3.0, 5.0, 5.0)),
    ]
    result = ingest(plan(rooms))
    assert result.refusal == "rooms-claim-the-same-floor"
    assert (1, 0) in result.swallowed
    assert result.measurements["twiceClaimedArea"] == pytest.approx(
        (q(5.0) - q(3.0)) ** 2, rel=1e-9
    )


def test_the_union_is_blind_to_containment_which_is_why_it_is_measured_separately():
    """Section 6.2: "Where ring A contains ring B, A carries no wall face along
    B's boundary ... The union is blind to this: A alone already covers the
    region, so the outline has no hole and the uncovered area is zero. This is why
    the second half had to be measured separately rather than inferred from the
    first"."""
    rooms = [
        room("livingroom", square(0.0, 0.0, 10.0, 10.0)),
        room("stairs", square(3.0, 3.0, 5.0, 5.0)),
    ]
    result = ingest(plan(rooms))
    assert result.measurements["uncoveredArea"] == pytest.approx(0.0, abs=1e-9)
    assert result.measurements["twiceClaimedArea"] > 0.0


def test_quantisation_residue_below_one_part_per_million_is_not_a_refusal():
    """Section 6.4: "below one part per million the overlap is quantisation
    residue: a ring that misses its neighbour by a rounding step leaves a sliver
    with no meaningful area"."""
    assert OVERLAP_SHARE == 1e-6
    # A sliver of 1e-8 m2 against a floor of 21 m2 is a share of 5e-10.
    sliver = 1e-8 / 3.0
    rooms = [
        room("livingroom", square(0.0, 0.0, 4.0, 3.0)),
        room("bedroom", square(4.0 - sliver, 0.0, 7.0, 3.0)),
    ]
    result = ingest(plan(rooms))
    assert not result.refused, result.detail


def test_the_annulus_case_is_refused_and_the_wrapped_room_is_not_absorbed():
    """Section 5.5 and section 6.4, ruled 2026-09-16: "the refusal stands on its
    own basis and a wrapped room is not absorbed", and declining to absorb even a
    stair core, because "absorbing a stair core still enlarges the measured area
    of the room that wraps it ... and area and compactness are graded rungs"."""
    rooms = [
        room("corridor", square(0.0, 0.0, 10.0, 10.0)),
        room("stairs", square(4.0, 4.0, 6.0, 6.0)),
    ]
    result = ingest(plan(rooms, source="msd"))
    assert result.refusal == "rooms-claim-the-same-floor"
    assert (1, 0) in result.swallowed


# --- refusals are outcomes, not rungs -------------------------------------


def test_a_refusal_carries_its_measurements_rather_than_only_prose():
    """Section 6.2: "The refusal carries its measurements on the exception rather
    than in its message text, so that a tally never has to parse prose back into
    numbers, and so that it can be counted as one cause. Grouping by message text
    does not work at corpus scale, because the message names the plan and every
    plan then reads as its own distinct cause"."""
    rooms = [
        room("livingroom", square(0.0, 0.0, 10.0, 10.0)),
        room("stairs", square(3.0, 3.0, 5.0, 5.0)),
    ]
    result = ingest(plan(rooms))
    assert result.refusal == "rooms-claim-the-same-floor"
    assert "twiceClaimedArea" in result.measurements
    assert "twiceClaimedShare" in result.measurements
    assert result.detail  # prose beside the numbers, never instead of them


def test_the_uncovered_half_is_measured_before_the_twice_claimed_half():
    """Section 6.3 governs the uncovered half and refuses above zero; section 6.4
    says its own half "is measured after, on the rings the solver will actually
    see". A plan failing both is reported under the first."""
    rooms = [
        room("livingroom", square(0.0, 0.0, 10.0, 10.0)),
        room("stairs", square(3.0, 3.0, 5.0, 5.0)),
        room("balcony", square(30.0, 30.0, 32.0, 32.0)),
    ]
    assert ingest(plan(rooms)).refusal == "untiled-rooms"


# --- section 6.3, which room takes a closed region ----------------------------

# The probe: a 0.2109 by 0.28125 m void bounded along one 0.28125 m edge
# by a thick livingroom and along the other three edges (0.703125 m in all) by a
# thin bedroom. The livingroom shares more grown area with the void; the bedroom
# bounds more of its boundary.
_RANK_SPLIT = {
    "name": "t18-rank-split-asis",
    "source": "housegan++",
    "rooms": [
        {"category": "livingroom", "ring": [[0.2109375, -0.73828125], [1.6171875, -0.73828125], [1.6171875, 1.01953125], [0.2109375, 1.01953125]]},
        {"category": "bedroom", "ring": [[-0.10546875, -0.03515625], [0.2109375, -0.03515625], [0.2109375, 0.0], [0.0, 0.0], [0.0, 0.28125], [0.2109375, 0.28125], [0.2109375, 0.31640625], [-0.10546875, 0.31640625]]},
    ],
}


def test_the_room_bounding_most_of_a_region_by_length_takes_it_first():
    from shapely.geometry import Polygon, box

    from floorcheck.ingest import _bounding_rooms

    polygons = [Polygon(room["ring"]) for room in _RANK_SPLIT["rooms"]]
    void = box(0.0, 0.0, 0.2109375, 0.28125)
    assert _bounding_rooms(void, polygons, 0.31640625) == [1, 0]


def test_the_rank_split_void_goes_to_the_bedroom_and_no_rung_rejects():
    # Handed to the livingroom, the bedroom kept its thin C shape and failed
    # rung 2.
    from floorcheck.report import check

    report = check(plan_from_dict(_RANK_SPLIT)).to_dict()
    assert report["ingest"]["seamsClosed"] == 1
    assert [r["verdict"] for r in report["ladder"]].count("rejected") == 0, report["ladder"]
