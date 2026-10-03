"""A room that the quantisation of section 2.2 collapses ends in a named outcome.

Section 2.6: "Every refusal is a named outcome and never an exception." Section
2.2 quantises every ring "before anything geometric is asserted", so a room whose
schema-valid ring collapses under it reaches the checker with no area, and
section 8.1 names what happens then: "the rung rejects the plan at the first room
whose outline is not" simple and of positive area.
"""

from __future__ import annotations

import json
from pathlib import Path

from floorcheck.ladder import REJECTED, rung_room_outlines
from floorcheck.measure import bounding_sides
from floorcheck.model import plan_from_dict
from floorcheck.report import check

EXAMPLE = Path(__file__).resolve().parents[2] / "floorcheck" / "examples" / "rplan-apartment.json"


def _sliver_plan():
    """A sliver probe: the kitchen ends at x = 8.99 and a storage room fills
    the 0.01 m strip, which the quantisation collapses to a line."""
    payload = json.loads(EXAMPLE.read_text())
    kitchen = payload["rooms"][3]
    kitchen["ring"] = [[6.75, 3.375], [8.99, 3.375], [8.99, 6.75], [6.75, 6.75]]
    payload["rooms"].append(
        {"category": "storage", "ring": [[8.99, 3.375], [9.0, 3.375], [9.0, 6.75], [8.99, 6.75]]}
    )
    return plan_from_dict(payload)


def _degenerate_triangle_plan():
    """A degenerate-triangle probe: the second room keeps three coordinates under the
    quantisation but encloses no area."""
    return plan_from_dict(
        {
            "rooms": [
                {"category": "livingroom", "ring": [[3.0, 0.0], [3.0, 0.5], [0.0, 0.5]]},
                {
                    "category": "livingroom",
                    "ring": [
                        [0.0, 0.5],
                        [0.07734375, 0.50734375],
                        [0.007343749999999996, 0.50734375],
                        [0.0, 0.57734375],
                    ],
                },
            ],
            "source": "rplan",
        }
    )


def test_a_room_collapsed_to_a_line_is_rejected_at_rung_one_not_raised():
    """Ingest must not raise "A linearring requires at least 4 coordinates"."""
    report = check(_sliver_plan())
    assert not report.ingest.refused
    first = report.graded.rungs[0]
    assert (first.name, first.verdict) == ("roomOutline", REJECTED)
    assert first.reason == "room 4 is not a ring"


def test_a_room_with_no_area_is_rejected_at_rung_one_and_its_width_is_zero():
    """The width measure must not raise "max() arg is an empty sequence"."""
    report = check(_degenerate_triangle_plan(), source="standards")
    first = report.graded.rungs[0]
    assert (first.name, first.verdict) == ("roomOutline", REJECTED)
    assert first.reason == "room 1 is not a ring"
    widths = {row["room"]: row["minWidthM"] for row in report.observations}
    assert widths.get(1, 0.0) == 0.0


def test_rung_one_rejects_a_two_point_ring_rather_than_raising():
    """Section 8.1's predicate, read on a ring shorter than a ring."""
    result = rung_room_outlines([((0.0, 0.0), (1.0, 0.0))])
    assert (result.verdict, result.reason) == (REJECTED, "room 0 is not a ring")


def test_an_empty_ring_has_zero_extent():
    assert bounding_sides(()) == (0.0, 0.0)
