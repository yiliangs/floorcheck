"""The end-to-end path: `floorcheck.report.check_file`, `floorcheck.report.check_json`
and `floorcheck.__main__.main` against the three example plans.

The interface under test is `floorcheck <plan.json>`, reading the neutral plan
schema of floorcheck/SCHEMA.md, with output a verdict per rung plus the
reject-reason vocabulary, machine-readable and human-readable. The same
document's clean-room rule requires the checker to "grade any plan in RPLAN or MSD
format, including invalid generator output (overlaps, rooms outside the boundary,
missing entries), with a reject reason rather than an exception," which is the property
most of this file exercises: nothing here ever expects an exception to escape.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import floorcheck
from floorcheck.__main__ import main
from floorcheck.ladder import NOT_REACHED, PASSED
from floorcheck.report import check_file, check_json
from floorcheck.scope import NOT_TRANSFERABLE

_PACKAGE_DIR = pathlib.Path(floorcheck.__file__).resolve().parent
EXAMPLES_DIR = _PACKAGE_DIR / "examples"
RPLAN_APARTMENT = EXAMPLES_DIR / "rplan-apartment.json"
GSDIFF_SEAMED = EXAMPLES_DIR / "gsdiff-seamed.json"
MSD_WRAPPED_CORE = EXAMPLES_DIR / "msd-wrapped-core.json"


# --- the three example plans, end to end ------------------------------------


def test_rplan_apartment_ingests_clean_and_passes_every_transferable_rung():
    """Section 9.1: "RPLAN is also the control on the width rule, because its
    rooms touch exactly: not one RPLAN plan has a seam closed into a room."
    Sections 8.4, 8.5 and 8.6 rule rungs 4, 5 and 6 not transferable, so this
    example's verdict rests on rungs 1, 2 and 3 alone.
    """
    report = check_file(str(RPLAN_APARTMENT))
    assert report["ingest"]["outcome"] == "accepted"
    assert report["ingest"]["seamsClosed"] == 0
    assert report["verdict"] == "passed every transferable rung"
    rungs = {row["rung"]: row["verdict"] for row in report["ladder"]}
    assert rungs[1] == PASSED
    assert rungs[2] == PASSED
    assert rungs[3] == PASSED
    assert rungs[4] == NOT_TRANSFERABLE
    assert rungs[5] == NOT_TRANSFERABLE
    assert rungs[6] == NOT_TRANSFERABLE


def test_gsdiff_seamed_reproduces_the_bridged_compactness_stop_of_section_9_2():
    """Section 9.2: "The admission is bought by handing each admitted plan a
    median of 1.561 square metres of floor its generator never emitted, which
    the area, compactness and proportions rungs then grade as though it had,
    and 963 of the 1,025 that reach a verdict are stopped at compactness,
    almost nineteen in twenty." This example reproduces that finding: a seam
    is closed to admit the plan, and it is then stopped at roomCompactness.
    """
    report = check_file(str(GSDIFF_SEAMED))
    assert report["ingest"]["outcome"] == "accepted"
    assert report["ingest"]["seamsClosed"] >= 1
    assert report["verdict"] == "stopped at roomCompactness"
    rungs = {row["rung"]: row["verdict"] for row in report["ladder"]}
    assert rungs[4] == NOT_REACHED
    assert rungs[5] == NOT_REACHED
    assert rungs[6] == NOT_REACHED


def test_msd_wrapped_core_is_refused_as_the_same_floor_claimed_twice():
    """Sections 5.5 / 6.4 rule that this refusal stands on its own basis and
    that a wrapped room is not absorbed into the room that wraps it. This example's stairs room is wrapped by
    its corridor room, so it is refused at ingest rather than graded.
    """
    report = check_file(str(MSD_WRAPPED_CORE))
    assert report["ingest"]["outcome"] == "rooms-claim-the-same-floor"
    assert report["ladder"] is None
    assert report["verdict"].startswith("no verdict")


def test_every_reports_header_names_the_checker_version_source_and_corpus():
    """Section 9.4: "Two conditions travel with those numbers and neither may
    be dropped." A verdict printed without its requirement source is a number
    nobody can place, so every header carries it and the corpus alongside the
    checker's own name and version.
    """
    for path in (RPLAN_APARTMENT, GSDIFF_SEAMED, MSD_WRAPPED_CORE):
        report = check_file(str(path))
        header = report["header"]
        assert header["checker"] == "floorcheck"
        assert header["version"]
        assert header["requirementSource"]
        assert header["corpus"]


# --- requirement sources, through the CLI -----------------------------------


def test_both_public_requirement_sources_work_through_the_cli_and_a_third_is_refused(
    capsys,
):
    """Section 1.3: "D3 may implement `standards` and `dataset-p05`. It may not
    implement `stand-in`."
    """
    for source in ("standards", "dataset-p05"):
        exit_code = main([str(RPLAN_APARTMENT), "--requirements", source])
        assert exit_code == 0
        captured = capsys.readouterr()
        report = json.loads(captured.out)
        assert report["header"]["requirementSource"] == source

    with pytest.raises(SystemExit):
        main([str(RPLAN_APARTMENT), "--requirements", "stand-in"])


_CUSTOM_LIVING_ROOM_TABLE = (
    "## rplan\n\n"
    "| RoomTag | count | area p5 | p25 | p50 | p75 | p95 | width p5 | p25 | p50 |"
    " aspect p5 | p25 | p50 | p75 | p95 |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    "| LivingRoom | 1 | 25.5251 | 30.0 | 34.0 | 39.0 | 48.0 | 100.0000 | 5.0 | 5.9 |"
    " 1.5 | 1.6 | 1.7 | 1.8 | 2.2548 |\n\n"
    "Pooled room compactness.\n\n"
    "| p01 | p05 | p25 | p50 | p75 |\n"
    "|---|---|---|---|---|\n"
    "| 0.40 | 0.4618 | 0.5 | 0.6 | 0.7 |\n"
)


def test_the_category_statistics_flag_moves_the_requirement_seen_at_grade(tmp_path, capsys):
    """`--category-statistics` has to reach `report.check` -> `ladder.grade` ->
    `requirements.dataset_p05_requirements`, not stop at the CLI's own argument
    parsing. A table naming an absurd LivingRoom width, distinct from the
    committed table's, proves it reached the grade: the minWidth observation
    the checker reports for the room changes and its own floor stops meeting it.
    """
    table = tmp_path / "category-statistics.md"
    table.write_text(_CUSTOM_LIVING_ROOM_TABLE, encoding="utf-8")

    default_report = check_file(str(RPLAN_APARTMENT))
    default_living = next(
        row for row in default_report["observations"]["minWidth"] if row["tag"] == "LivingRoom"
    )

    exit_code = main([str(RPLAN_APARTMENT), "--category-statistics", str(table)])
    assert exit_code == 0
    captured = capsys.readouterr()
    overridden_report = json.loads(captured.out)
    overridden_living = next(
        row for row in overridden_report["observations"]["minWidth"] if row["tag"] == "LivingRoom"
    )

    assert overridden_living["requiredM"] == pytest.approx(100.0)
    assert overridden_living["requiredM"] != default_living["requiredM"]
    assert overridden_living["meets"] is False


# --- malformed and unexpressible input, per the clean-room rule -------------


def test_check_json_reports_malformed_json_rather_than_raising():
    """The checker's contract: it "must grade any
    plan in RPLAN or MSD format, including invalid generator output (overlaps,
    rooms outside the boundary, missing entries), with a reject reason rather
    than an exception".
    """
    report = check_json("{not valid json")
    assert report["ingest"]["outcome"] == "malformed-json"
    assert report["ladder"] is None


def test_check_json_reports_a_room_with_a_hole_rather_than_raising():
    """Section 2.1: "One room is one ring ... A checker that models a room as a
    polygon with holes will accept plans this specification refuses, and the
    two will disagree on exactly the population section 9 counts." A plan that
    hands the checker a hole is refused by name instead of silently accepted.
    """
    payload = {
        "name": "hole-plan",
        "source": "rplan",
        "rooms": [
            {
                "category": "bedroom",
                "ring": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
                "holes": [[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]],
            }
        ],
    }
    report = check_json(json.dumps(payload))
    assert report["ingest"]["outcome"] == "room-with-hole"


# --- the CLI never crashes on a refusal --------------------------------------


def test_main_returns_zero_for_every_example_in_both_formats(capsys):
    """The checker's contract requires a reject reason
    "rather than an exception" for invalid input; a plan refused at ingest is
    such a reject reason, not a crash, so `main` exits 0 for it exactly as it
    does for a plan that reaches a verdict.
    """
    for path in (RPLAN_APARTMENT, GSDIFF_SEAMED, MSD_WRAPPED_CORE):
        for fmt in ("json", "text"):
            exit_code = main([str(path), "--format", fmt])
            assert exit_code == 0
            captured = capsys.readouterr()
            if fmt == "json":
                json.loads(captured.out)


def test_main_reports_unreadable_file_rather_than_crashing(capsys):
    """The checker's contract: it "must grade any
    plan in RPLAN or MSD format, including invalid generator output ..., with
    a reject reason rather than an exception"; a path that names no file at
    all gets the same treatment as malformed generator output.
    """
    missing = EXAMPLES_DIR / "does-not-exist.json"
    exit_code = main([str(missing)])
    assert exit_code == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["ingest"]["outcome"] == "unreadable-file"


def test_every_report_round_trips_through_json_dumps():
    """The checker's contract: output must be "machine-readable
    and human-readable." Round-tripping through `json.dumps` without a
    `TypeError` is what makes the machine-readable half true.
    """
    for path in (RPLAN_APARTMENT, GSDIFF_SEAMED, MSD_WRAPPED_CORE):
        report = check_file(str(path))
        dumped = json.dumps(report)
        assert json.loads(dumped) == report


def test_a_null_source_pixel_is_a_named_refusal_not_an_exception(tmp_path, capsys):
    """`meta.sourcePixelM: null` raised TypeError out of `float()` and exited 2.
    Section 2.6 makes every refusal a named outcome; SCHEMA.md admits only a number."""
    payload = json.loads(RPLAN_APARTMENT.read_text(encoding="utf-8"))
    payload["meta"] = {"sourcePixelM": None}
    path = tmp_path / "null-pixel.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert main([str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ingest"]["outcome"] == "malformed-plan"
    assert report["ladder"] is None
