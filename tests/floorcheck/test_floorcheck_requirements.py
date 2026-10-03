"""Slice 2: the two requirement sources of specification section 1.3.

Written from the specification before the module, per the item's plan. Every
assertion cites the sentence it tests.
"""

from __future__ import annotations

import pytest

from floorcheck.requirements import (
    SOURCE_NAMES,
    FLOOR_KINDS,
    compactness_floor,
    dataset_p05_requirements,
    plan_level_floors,
    requirements,
    standards_requirements,
)


def test_source_names_are_exactly_standards_and_dataset_p05():
    """Section 1.3: "D3 may implement `standards` and `dataset-p05`."."""
    assert SOURCE_NAMES == ("standards", "dataset-p05")


def test_stand_in_is_refused():
    """Section 1.3: "It may not implement `stand-in`."."""
    with pytest.raises(ValueError):
        requirements("stand-in", "rplan")


def test_standards_names_exactly_eleven_categories_split_six_three_two():
    """Section 8.7: "Under `standards`, six categories carry both an area and
    a width, three carry a width only, and two carry neither."."""
    table = standards_requirements()
    assert len(table) == 11

    both = [r for r in table.values() if r.area_reason == "" and r.width_reason == ""]
    width_only = [r for r in table.values() if r.area_reason != "" and r.width_reason == ""]
    neither = [r for r in table.values() if r.area_reason != "" and r.width_reason != ""]
    assert len(both) == 6
    assert len(width_only) == 3
    assert len(neither) == 2


def test_standards_spot_check_against_the_clauses():
    """Table 3 of the provenance table, spot-checked cell by cell."""
    table = standards_requirements()

    living_room = table["LivingRoom"]
    assert living_room.area_req == pytest.approx(6.5)
    assert living_room.min_width == pytest.approx(2.134)
    assert living_room.area_reason == ""
    assert living_room.width_reason == ""

    primary_bedroom = table["PrimaryBedroom"]
    assert primary_bedroom.min_width == pytest.approx(2.75)

    bathroom = table["Bathroom"]
    assert bathroom.min_width == pytest.approx(0.762)
    assert bathroom.area_req == 0.0
    assert bathroom.area_reason != ""

    balcony = table["Balcony"]
    assert balcony.area_req == pytest.approx(5.0)
    assert balcony.min_width == pytest.approx(1.5)


def test_standards_prop_req_is_always_zero_with_a_reason():
    """Section 8.3: "standards | disabled, spelled 0.0, for every category.",
    and "Every `propReq` cell of Table 3 of the provenance table reads
    'panel'."."""
    table = standards_requirements()
    assert len(table) == 11
    for requirement in table.values():
        assert requirement.prop_req == 0.0
        assert requirement.prop_reason != ""


def test_every_disabled_standards_cell_carries_a_reason():
    """Section 1.4: "no requirement" has to be spelled as a value, and this
    implementation carries the disabling reason on the record rather than in
    a log."""
    table = standards_requirements()
    disabled_area = [r for r in table.values() if r.area_reason != ""]
    disabled_width = [r for r in table.values() if r.width_reason != ""]
    assert len(disabled_area) == 5  # Bathroom, Foyer, Hallway, Kitchen, Closet
    assert len(disabled_width) == 2  # Kitchen, Closet
    for requirement in disabled_area:
        assert requirement.area_req == 0.0
    for requirement in disabled_width:
        assert requirement.min_width == 0.0


def test_dataset_p05_category_counts_per_corpus():
    """Section 8.7: MSD reaches three fewer rows than RPLAN, "so those three
    rows are exercised on RPLAN only."."""
    rplan = dataset_p05_requirements("rplan")
    assert len(rplan) == 10
    for kind in ("msd-unit", "msd-floor"):
        assert len(dataset_p05_requirements(kind)) == 8


def test_dataset_p05_msd_misses_primary_bedroom_guest_room_and_foyer():
    """Section 8.7: "MSD reaches no primary bedroom, no guest room and no
    foyer."."""
    for kind in ("msd-unit", "msd-floor"):
        msd = dataset_p05_requirements(kind)
        assert "PrimaryBedroom" not in msd
        assert "GuestRoom" not in msd
        assert "Foyer" not in msd


def test_dataset_p05_prop_req_is_the_plan_level_floor_of_the_kind():
    """Section 8.3: propReq is the kind's plan-level floor from
    plan-level-floors.json, stored before the bedroom cap, read here at test
    time rather than restated."""
    for kind in FLOOR_KINDS:
        floors = plan_level_floors(kind).prop_req
        for tag, requirement in dataset_p05_requirements(kind).items():
            assert requirement.prop_req == floors[tag]
            assert requirement.prop_reason == ""


def test_dataset_p05_prop_req_is_a_ratio_in_the_half_open_unit_interval():
    """Section 8.3: propReq is "a dimensionless ratio in the half-open
    interval from zero to one."."""
    for kind in FLOOR_KINDS:
        for requirement in dataset_p05_requirements(kind).values():
            assert 0.0 < requirement.prop_req <= 1.0


def test_dataset_p05_area_and_width_come_from_the_p5_columns():
    """Section 8.3: `area_req` and `min_width` are the corpus fifth percentile."""
    rplan = dataset_p05_requirements("rplan")
    assert rplan["LivingRoom"].area_req == pytest.approx(25.5251)
    assert rplan["LivingRoom"].min_width == pytest.approx(4.0078)

    for kind in ("msd-unit", "msd-floor"):
        msd = dataset_p05_requirements(kind)
        assert msd["Kitchen"].area_req == pytest.approx(4.8957)
        assert msd["Kitchen"].min_width == pytest.approx(1.6984)


def test_compactness_floor_dataset_p05_and_standards():
    """Section 8.2: under dataset-p05 the floor is the kind's plan-level floor
    from plan-level-floors.json; under standards it is the 0.589 anchor."""
    for kind in FLOOR_KINDS:
        assert compactness_floor("dataset-p05", kind) == plan_level_floors(kind).compactness
        assert 0.0 < compactness_floor("dataset-p05", kind) < 1.0
        assert compactness_floor("standards", kind) == pytest.approx(0.589)


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        requirements("bogus-source", "rplan")


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        requirements("dataset-p05", "msd")
    with pytest.raises(ValueError):
        dataset_p05_requirements("bogus-kind")
    with pytest.raises(ValueError):
        compactness_floor("dataset-p05", "bogus-kind")
