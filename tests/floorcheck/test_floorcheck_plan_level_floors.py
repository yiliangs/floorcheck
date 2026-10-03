"""Sections 8.2 and 8.3: rungs 2 and 3 grade against the plan-level floors of the plan's kind.

An MSD plan stating more than one apartment (`Room.unit`) is a whole floor; an
MSD plan stating one apartment or none is graded as an apartment; RPLAN and
generated plans take the RPLAN floors. The values here are a fixture table, so
no committed floor is restated in this file.
"""

from __future__ import annotations

import json

import pytest

from floorcheck.ladder import grade
from floorcheck.model import plan_from_dict
from floorcheck.requirements import FLOOR_KINDS, floor_kind, plan_level_floors

FIXTURE = {
    "level": 0.05,
    "provenance": {"command": "fixture", "commit": "fixture", "input": "fixture"},
    "kinds": {
        "rplan": {"compactness": 0.101, "propReq": {"Kitchen": 0.201}},
        "msd-unit": {"compactness": 0.102, "propReq": {"Kitchen": 0.202}},
        "msd-floor": {"compactness": 0.103, "propReq": {"Kitchen": 0.203}},
    },
}


def _rings():
    return [
        [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)],
        [(3.0, 0.0), (6.0, 0.0), (6.0, 3.0), (3.0, 3.0)],
    ]


def _plan(source, units):
    rooms = []
    for ring, unit in zip(_rings(), units):
        entry = {"category": "kitchen", "ring": [list(point) for point in ring]}
        if unit is not None:
            entry["unit"] = unit
        rooms.append(entry)
    return plan_from_dict({"source": source, "rooms": rooms})


@pytest.fixture
def floors(tmp_path):
    path = tmp_path / "plan-level-floors.json"
    path.write_text(json.dumps(FIXTURE), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "source,corpus,units,kind",
    [
        ("msd", None, ("u0", "u1"), "msd-floor"),
        ("msd", None, ("u3", "u3"), "msd-unit"),
        ("msd", None, (None, None), "msd-unit"),
        ("msd", None, ("u0", None), "msd-unit"),
        ("rplan", None, (None, None), "rplan"),
        ("rplan", "rplan", (None, None), "rplan"),
    ],
)
def test_rungs_two_and_three_read_the_floors_of_the_plans_kind(floors, source, corpus, units, kind):
    graded = grade(_plan(source, units), _rings(), corpus=corpus, floors=floors)
    assert graded.floor_kind == kind
    expected = FIXTURE["kinds"][kind]
    assert graded.rungs[1].worst["minPassingPPScore"] == expected["compactness"]
    assert graded.rungs[2].worst["propReq"] == expected["propReq"]["Kitchen"]


def test_a_generated_plan_answers_to_the_rplan_floors(floors):
    """A generated sample is graded against RPLAN's corpus (`checker_outcomes`
    passes the corpus from the export), so it takes the RPLAN floors."""
    graded = grade(_plan("rplan", (None, None)), _rings(), corpus="rplan", floors=floors)
    assert graded.floor_kind == "rplan"
    assert graded.rungs[1].worst["minPassingPPScore"] == FIXTURE["kinds"]["rplan"]["compactness"]


def test_the_kind_rule():
    assert floor_kind("msd", 2) == "msd-floor"
    assert floor_kind("msd", 1) == "msd-unit"
    assert floor_kind("msd", 0) == "msd-unit"
    assert floor_kind("rplan", 5) == "rplan"
    with pytest.raises(ValueError):
        floor_kind("bogus", 1)


def test_the_committed_table_carries_every_kind_and_the_ladder_reads_it():
    """Read at test time, never restated: the default grade uses the committed
    table's floors for the plan's kind."""
    for kind in FLOOR_KINDS:
        floors = plan_level_floors(kind)
        assert 0.0 < floors.compactness < 1.0
        assert floors.prop_req and all(0.0 < v <= 1.0 for v in floors.prop_req.values())
    graded = grade(_plan("msd", ("u0", "u1")), _rings())
    assert graded.rungs[1].worst["minPassingPPScore"] == plan_level_floors("msd-floor").compactness
