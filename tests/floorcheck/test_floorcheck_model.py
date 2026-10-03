"""Slice 1: the input model of specification section 2.

Written from the specification before the module, per the item's plan. Every
assertion cites the sentence it tests.
"""

from __future__ import annotations

import json

import pytest

from floorcheck.model import (
    PlanRejected,
    RPLAN_PIXEL_M,
    plan_from_dict,
    plan_from_json,
)


def square(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def minimal_payload() -> dict:
    return {
        "source": "rplan",
        "rooms": [
            {"category": "livingroom", "ring": square(0.0, 0.0, 4.0, 3.0)},
            {"category": "bedroom", "ring": square(4.0, 0.0, 7.0, 3.0)},
        ],
    }


def test_a_plan_is_rooms_with_one_ring_and_one_category_each():
    """Section 2.1: "a set of rooms, each carrying exactly one closed ring of
    coordinates and one dataset category"."""
    plan = plan_from_dict(minimal_payload())
    assert len(plan.rooms) == 2
    assert [room.category for room in plan.rooms] == ["livingroom", "bedroom"]
    assert all(len(room.ring) == 4 for room in plan.rooms)


def test_a_closing_repeat_of_the_first_vertex_is_accepted_and_dropped():
    """Section 2.2: "every ring is closed, deduplicated". A ring given closed and
    a ring given open must describe the same room."""
    payload = minimal_payload()
    payload["rooms"][0]["ring"] = square(0.0, 0.0, 4.0, 3.0) + [[0.0, 0.0]]
    plan = plan_from_dict(payload)
    assert len(plan.rooms[0].ring) == 4


def test_the_entry_the_extent_and_the_adjacency_are_optional():
    """Section 2.1 makes the entrance part of a plan; section 5.2 makes the floor
    extent something that "travels on the plan". Neither is present on every
    source, so both are optional inputs."""
    plan = plan_from_dict(minimal_payload())
    assert plan.entry is None
    assert plan.floor_extent is None
    assert plan.adjacency == ()

    payload = minimal_payload()
    payload["entry"] = [0.0, 1.5]
    payload["floorExtent"] = square(0.0, 0.0, 7.0, 3.0)
    payload["adjacency"] = [[1, 0]]
    plan = plan_from_dict(payload)
    assert plan.entry == (0.0, 1.5)
    assert len(plan.floor_extent) == 4
    # Stored low index first so a pair is one edge however it was written.
    assert plan.adjacency == ((0, 1),)


def test_a_plan_carries_the_scale_and_frame_of_its_source():
    """Section 2.3: "one RPLAN pixel is 0.0703125 m, and the 256 pixel frame spans
    exactly 18 m"."""
    plan = plan_from_dict(minimal_payload())
    assert plan.source == "rplan"
    assert plan.source_pixel_m == pytest.approx(0.0703125)
    assert RPLAN_PIXEL_M == pytest.approx(18.0 / 256.0)


def test_msds_source_pixel_is_its_partition_grid_step_not_its_coordinate_unit():
    """Section 2.5 says MSD "publishes room geometry as vector polygons with
    coordinates already in metres, so it imposes no raster floor", but section
    6.3 tables MSD's source pixel as "its own partition grid step", "0.38 m by
    default", and says the pixel "is not the same fact as the plan's
    `units_per_meter` ... MSD is exactly that case: its coordinates are metres
    while its partition resolves the floor onto a grid"."""
    payload = minimal_payload()
    payload["source"] = "msd"
    plan = plan_from_dict(payload)
    assert plan.source_pixel_m == pytest.approx(0.38)


def test_the_synthetic_fixture_states_a_zero_pixel():
    """Section 6.3's table: "the synthetic fixture | exact geometry | 0.0 |
    stated, and held to one quantisation step"."""
    payload = minimal_payload()
    payload["source"] = "synthetic"
    assert plan_from_dict(payload).source_pixel_m == 0.0


def test_a_plan_may_state_its_own_source_pixel():
    """Section 6.3: "The pixel is stated by the reader, not inferred. Every plan
    carries the smallest length its source can tell apart, in metres, as a field
    of its own"."""
    payload = minimal_payload()
    payload["meta"] = {"sourcePixelM": 0.25}
    assert plan_from_dict(payload).source_pixel_m == pytest.approx(0.25)


def test_an_unknown_source_has_no_source_pixel_rather_than_a_guessed_one():
    payload = minimal_payload()
    payload["source"] = "some-new-generator"
    assert plan_from_dict(payload).source_pixel_m is None


def test_housegan_and_gsdiff_frames_both_span_eighteen_metres():
    """Section 2.4: House-GAN++ at a 64 unit frame, HouseDiffusion at 256, both
    spanning the same 18 m."""
    for source, expected in (("housegan++", 18.0 / 64.0), ("gsdiff", 18.0 / 256.0)):
        payload = minimal_payload()
        payload["source"] = source
        assert plan_from_dict(payload).source_pixel_m == pytest.approx(expected)


# --- what section 2 says cannot be expressed -------------------------------


def test_a_room_with_a_hole_is_refused_not_repaired():
    """Section 2.1: "One room is one ring ... A checker that models a room as a
    polygon with holes will accept plans this specification refuses"."""
    payload = minimal_payload()
    payload["rooms"][0]["ring"] = [square(0.0, 0.0, 4.0, 3.0), square(1.0, 1.0, 2.0, 2.0)]
    with pytest.raises(PlanRejected) as caught:
        plan_from_dict(payload)
    assert caught.value.reason == "room-with-hole"


def test_a_room_declaring_interior_rings_by_name_is_refused():
    payload = minimal_payload()
    payload["rooms"][0]["holes"] = [square(1.0, 1.0, 2.0, 2.0)]
    with pytest.raises(PlanRejected) as caught:
        plan_from_dict(payload)
    assert caught.value.reason == "room-with-hole"


def test_a_polygon_object_rather_than_a_ring_is_refused_as_a_hole_bearing_shape():
    payload = minimal_payload()
    payload["rooms"][0]["ring"] = {"exterior": square(0.0, 0.0, 4.0, 3.0), "interiors": []}
    with pytest.raises(PlanRejected) as caught:
        plan_from_dict(payload)
    assert caught.value.reason == "room-with-hole"


def test_a_degenerate_ring_is_refused():
    """Fewer than three distinct vertices is not a ring."""
    payload = minimal_payload()
    payload["rooms"][0]["ring"] = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
    with pytest.raises(PlanRejected) as caught:
        plan_from_dict(payload)
    assert caught.value.reason == "malformed-ring"


def test_a_room_without_a_category_is_refused():
    """Section 2.1 gives every room "one dataset category"; without one there is
    no token to grade it against."""
    payload = minimal_payload()
    del payload["rooms"][0]["category"]
    with pytest.raises(PlanRejected) as caught:
        plan_from_dict(payload)
    assert caught.value.reason == "missing-category"


def test_a_plan_without_a_source_is_refused():
    """Section 2.1: a plan carries "the scale and frame of the source it came
    from", and section 6.3's width rule reads it."""
    payload = minimal_payload()
    del payload["source"]
    with pytest.raises(PlanRejected) as caught:
        plan_from_dict(payload)
    assert caught.value.reason == "missing-source"


def test_a_plan_with_no_rooms_is_refused():
    with pytest.raises(PlanRejected) as caught:
        plan_from_dict({"source": "rplan", "rooms": []})
    assert caught.value.reason == "no-rooms"


def test_a_ring_with_a_non_finite_coordinate_is_refused_by_name():
    """Section 2.6 lists "when a coordinate is not a finite number" among the
    named refusals, because the corpus produces it: RPLAN stores a handful of
    multi-part rooms with a literal NaN separator row, and `rplan/42921` and
    `rplan/77154` of the 4,000 the seventh run graded carry one into the schema.

    `float()` accepts NaN without raising, so before this the NaN travelled to
    the rungs, where every length and ratio derived from it is NaN and every
    comparison against a floor is false.
    """
    payload = minimal_payload()
    payload["rooms"][0]["ring"] = square(0.0, 0.0, 4.0, 3.0) + [[float("nan"), 0.0]]
    with pytest.raises(PlanRejected) as caught:
        plan_from_dict(payload)
    assert caught.value.reason == "non-finite-coordinate"
    # Section 2.6: "Each refusal names its reason and its room."
    assert "room 0" in caught.value.detail


def test_an_infinite_coordinate_is_refused_the_same_way():
    """`float("inf")` is not a ValueError either, and an infinite extent poisons
    the same quantities. One refusal covers both."""
    payload = minimal_payload()
    payload["entry"] = [0.0, float("inf")]
    with pytest.raises(PlanRejected) as caught:
        plan_from_dict(payload)
    assert caught.value.reason == "non-finite-coordinate"


def test_a_nan_plan_reaches_the_report_as_a_reject_reason_and_not_an_exception():
    """The checker's contract: invalid generator output is
    graded "with a reject reason rather than an exception". The D5 agreement
    harness found the two RPLAN plans that broke this, so the end of the path is pinned and
    not only the model layer."""
    from floorcheck.report import check_json

    payload = minimal_payload()
    payload["rooms"][1]["ring"] = square(4.0, 0.0, 7.0, 3.0) + [[4.0, float("nan")]]
    report = check_json(json.dumps(payload), name="rplan/42921")
    assert report["ingest"]["outcome"] == "non-finite-coordinate"
    assert report["ladder"] is None


def test_adjacency_indices_must_name_rooms():
    payload = minimal_payload()
    payload["adjacency"] = [[0, 9]]
    with pytest.raises(PlanRejected) as caught:
        plan_from_dict(payload)
    assert caught.value.reason == "malformed-adjacency"


def test_malformed_json_is_a_reject_reason_not_a_traceback():
    """D3-public-checker.md: the checker must "grade any plan ... with a reject
    reason rather than an exception"."""
    with pytest.raises(PlanRejected) as caught:
        plan_from_json("{not json")
    assert caught.value.reason == "malformed-json"


def test_every_refusal_carries_a_named_reason():
    """The reject-reason vocabulary is machine readable, so every refusal must
    carry a token rather than only a message."""
    payload = minimal_payload()
    payload["rooms"] = "not a list"
    with pytest.raises(PlanRejected) as caught:
        plan_from_dict(payload)
    assert isinstance(caught.value.reason, str) and caught.value.reason
    assert " " not in caught.value.reason
