"""The two public requirement sources of specification section 1.3.

Section 1.3: "the per-token requirements, `areaReq`, `minWidth`, `propReq` and
the appendix policy. These are not layered. One source is selected by name for
a whole run, from the three that the private reference implementation declares
as `SOURCE_NAMES`: `stand-in`, a recorded fixture of the private solver, which
is private and is the D1 path only; `standards`, the public derivations of Table 3 of the
provenance table ...; `dataset-p05`, the fifth percentile of the category's own
whole-corpus statistics ... D3 may implement `standards` and `dataset-p05`. It
may not implement `stand-in`."

Under `dataset-p05`, `areaReq` and `minWidth` are the corpus fifth percentiles
of `category-statistics.md`. `propReq` and the rung 2 floor `minPassingPPScore`
are plan-level instead (specification sections 8.2 and 8.3): each rung judges a plan by its worst room,
so a room-level percentile refuses more plans the more rooms a plan holds.
`plan-level-floors.json` holds, for each of three kinds of real plan (RPLAN
apartments, MSD apartments, MSD whole floors), the floors at which each rung
alone refuses a stated share of that kind's real plans, fitted by
`tools/plan_level_floor.py --emit`. `floor_kind` picks a plan's kind; the
private audit reads the same table through the same rule, so the checker and
the solver grade rungs 2 and 3 against identical floors.

Section 1.4 gives the sentinel a disabled field carries, because "every field
of the solver's `TokenRecord` is a plain `f64` with no serde default, so 'no
requirement' has to be spelled as a value":

| Field       | Disabling value      | Why it is vacuous |
|-------------|-----------------------|--------------------|
| `minWidth`  | 0.0                   | the width test rejects only a span shorter than the requirement less `abstol`, which no span is |
| `propReq`   | 0.0 or below          | the proportions check skips a room whose requirement is not positive |
| `areaReq`   | no sentinel exists    | see below |

`areaReq` has no disabling sentinel because it is not a floor: "The solver reads `areaReq`
as a proportional weight" (section 1.4), and "on the
check-only path this audit uses ... the area band never gates" (section 1.4).
This module spells a disabled `areaReq` cell as 0.0 and carries the reason
string separately, matching the comment at the assignment below: no sentinel
exists for `areaReq`, `0.0` is chosen because as a proportional weight it means
no share, and section 1.4 states the area band never gates on the check-only
path so the choice is unobservable at the ladder. Nothing is logged here; the
log entry belongs to the parent item, not to this clean-room implementation.
"""

from __future__ import annotations

import pathlib
import json
import os
import re
from dataclasses import dataclass

# Section 1.3: the two sources D3 may implement, in the order the specification
# names them. "stand-in" is deliberately absent: "It may not implement
# `stand-in`."
SOURCE_NAMES = ("standards", "dataset-p05")

# Section 2, "rplan" or "msd" throughout, and section 8.7: "MSD reaches no
# primary bedroom, no guest room and no foyer" is a statement about which of
# these two corpora a category is exercised on, not a third corpus.
_CORPORA = ("rplan", "msd")

# The three kinds of real plan the rung 2 and rung 3 floors are fitted on
# (sections 8.2 and 8.3): RPLAN apartments, MSD apartments and MSD whole floors. Each
# kind's floors are plan-level: at the fitted level, each rung alone refuses
# that share of the kind's real plans. A generated plan keeps the RPLAN floors.
FLOOR_KINDS = ("rplan", "msd-unit", "msd-floor")

# The corpus whose category statistics give a kind its `areaReq` and
# `minWidth`; only `propReq` and the compactness floor are plan-level.
KIND_CORPUS = {"rplan": "rplan", "msd-unit": "msd", "msd-floor": "msd"}

# The category-statistics table and the plan-level floors table are package
# data: they sit beside this module, so an installed checker grades against the
# same tables a source checkout does.
_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_CATEGORY_STATISTICS_PATH = _PACKAGE_DIR / "category-statistics.md"
DEFAULT_PLAN_LEVEL_FLOORS_PATH = _PACKAGE_DIR / "plan-level-floors.json"

# Section 8.3: "Every `propReq` cell of Table 3 of the provenance table reads
# 'panel'", and the private audit's requirements module spells the reason on
# every standards record as 'no public source; D4 panel'.
_STANDARDS_PROP_REASON = "no public source; D4 panel"

# Table 1 of the provenance table, `minPassingPPScore` row: "Value implied:
# 0.589 if the 3:1 box is the limit", status "provisional, panel calibration
# in D4". Kept at three digits rather than the four-digit exact geometry value
# of a 3:1 rectangle, 3*pi/16 = 0.5890, per section 10.3: "a four-digit anchor
# is the exact geometry and a three-digit one is a candidate floor named after
# it," and 0.589 is the candidate floor named after the 3:1 anchor, not the
# exact rectangle score.
STANDARDS_COMPACTNESS_FLOOR = 0.589


@dataclass(frozen=True)
class Requirement:
    """One category's per-token requirement, under one requirement source.

    ``area_reason``, ``width_reason`` and ``prop_reason`` are ``""`` for a real
    value and otherwise state why the field is disabled, per section 1.4's
    sentinel table.
    """

    category: str  # the RoomTag spelling
    area_req: float
    min_width: float
    prop_req: float
    area_reason: str
    width_reason: str
    prop_reason: str


def _validate_corpus(corpus: str) -> None:
    if corpus not in _CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; section 2 names only {_CORPORA!r}")


def _validate_source(source: str) -> None:
    if source not in SOURCE_NAMES:
        raise ValueError(
            f"unknown requirement source {source!r}; section 1.3: D3 may implement "
            f"only {SOURCE_NAMES!r} and may not implement 'stand-in'"
        )


def _validate_kind(kind: str) -> None:
    if kind not in FLOOR_KINDS:
        raise ValueError(f"unknown floor kind {kind!r}; the floors table names only {FLOOR_KINDS!r}")


def floor_kind(corpus: str, units: int) -> str:
    """The kind of plan whose rung 2 and rung 3 floors grade a plan.

    ``corpus`` is the plan's grading corpus and ``units`` the number of distinct
    apartments it states. An MSD plan with more than one apartment is a whole
    floor; an MSD plan with one apartment or none is graded as an apartment;
    every other plan, RPLAN or generated, takes the RPLAN floors. This is the
    one selection rule: the checker calls it with `scope.unit_count(plan)` and
    the oracle side with the apartments `planaudit.units.assign` derives, which
    are the ones the exporter writes as `Room.unit`.
    """
    _validate_corpus(corpus)
    if corpus != "msd":
        return "rplan"
    return "msd-floor" if units > 1 else "msd-unit"


@dataclass(frozen=True)
class PlanLevelFloors:
    """One kind's rung 2 and rung 3 floors from the plan-level floors table.

    ``prop_req`` is keyed by RoomTag and holds each category's floor before the
    bedroom cap, which the rung applies (`ladder.BEDROOM_PROP_CAP`) as the solver
    does (specification section 8.3).
    """

    kind: str
    compactness: float
    prop_req: dict[str, float]


def plan_level_floors(kind: str, path: str | os.PathLike | None = None) -> PlanLevelFloors:
    """``kind``'s floors, read from `plan-level-floors.json` (or ``path``)."""
    _validate_kind(kind)
    floors_path = pathlib.Path(path) if path is not None else DEFAULT_PLAN_LEVEL_FLOORS_PATH
    table = json.loads(floors_path.read_text(encoding="utf-8"))
    entry = table["kinds"].get(kind)
    if entry is None:
        raise ValueError(f"{floors_path.name} carries no floors for kind {kind!r}")
    return PlanLevelFloors(
        kind=kind,
        compactness=float(entry["compactness"]),
        prop_req={tag: float(value) for tag, value in entry["propReq"].items()},
    )


# --- "standards": Table 3 of the provenance table, transcribed verbatim ----
#
# Section 8.7: "Under `standards`, six categories carry both an area and a
# width, three carry a width only, and two carry neither." The six are
# LivingRoom, PrimaryBedroom, Bedroom, GuestRoom, DiningRoom and Balcony; the
# three width-only are Bathroom, Foyer and Hallway; the two with neither are
# Kitchen and Closet, per Table 3's closing paragraph: "'standards' carries
# only cells whose source names a published clause for the whole value ...
# Kitchen and Closet get neither, because the kitchen width mixes a guideline
# with a 600 mm convention and the closet depth is convention alone."
#
# `propReq` is 0.0 for every one of the 11 categories: section 8.3, "standards
# | disabled, spelled 0.0, for every category."


def standards_requirements() -> dict[str, Requirement]:
    """Table 3 of the provenance table, the `standards` requirement source."""
    entries: dict[str, Requirement] = {}

    def add(
        category: str,
        area_req: float,
        area_reason: str,
        min_width: float,
        width_reason: str,
    ) -> None:
        entries[category] = Requirement(
            category=category,
            area_req=area_req,
            min_width=min_width,
            prop_req=0.0,
            area_reason=area_reason,
            width_reason=width_reason,
            prop_reason=_STANDARDS_PROP_REASON,
        )

    # Table 3: area 6.5, "IRC 2021 R304.1, habitable room not less than 70 sq
    # ft"; width 2.134, "IRC 2021 R304.2, 7 ft in any horizontal dimension".
    add("LivingRoom", 6.5, "", 2.134, "")

    # Table 3: area 11.5, "UK NDSS 2015, double or twin bedroom at least 11.5
    # sq m"; width 2.75, "UK NDSS 2015, one double or twin bedroom at least
    # 2.75 m wide".
    add("PrimaryBedroom", 11.5, "", 2.75, "")

    # Table 3: area 11.5, "UK NDSS 2015, double or twin bedroom"; width 2.55,
    # "UK NDSS 2015, every other double or twin bedroom at least 2.55 m wide".
    add("Bedroom", 11.5, "", 2.55, "")

    # Table 3: area 7.5, "UK NDSS 2015, single bedroom at least 7.5 sq m";
    # width 2.15, "UK NDSS 2015, single bedroom at least 2.15 m wide".
    add("GuestRoom", 7.5, "", 2.15, "")

    # Table 3: area 6.5, "IRC 2021 R304.1"; width 2.134, "IRC 2021 R304.2".
    add("DiningRoom", 6.5, "", 2.134, "")

    # Table 3: area 5.0, "London Plan 2021 Policy D6 and Housing Design
    # Standards LPG 2023, 5 sq m for a 1 to 2 person dwelling plus 1 sq m per
    # additional occupant"; width 1.5, "Same, minimum depth and width of all
    # balconies 1500 mm".
    add("Balcony", 5.0, "", 1.5, "")

    # Table 3: area disabled, "No code sets a bathroom area; the statistic is
    # the only non-circular source"; width 0.762, "IRC 2021 R307.1, 15 in from
    # water closet centreline to any obstruction on each side".
    add(
        "Bathroom",
        0.0,
        "no code sets a bathroom area; the statistic is the only non-circular source",
        0.762,
        "",
    )

    # Table 3: area disabled, "No numeric residential standard for foyer or
    # vestibule area exists; IRC excludes halls and foyers from the
    # habitable-space minimum"; width 0.914, "IRC 2021 R311.6, borrowed as the
    # circulation floor a foyer must also meet".
    add(
        "Foyer",
        0.0,
        "no numeric residential standard for foyer or vestibule area exists; "
        "IRC excludes halls and foyers from the habitable-space minimum",
        0.914,
        "",
    )

    # Table 3: area disabled, "No code sets a hallway area; IRC excludes halls
    # from the habitable-space minimum"; width 0.914, "IRC 2021 R311.6, not
    # less than 3 ft (914 mm)".
    add(
        "Hallway",
        0.0,
        "no code sets a hallway area; IRC excludes halls from the habitable-space minimum",
        0.914,
        "",
    )

    # Table 3: area disabled, "IRC R304 exempts kitchens from both minimums
    # and NDSS sets none, so the dataset category statistic is the only
    # non-circular source"; width disabled, per Table 3's closing paragraph,
    # "the kitchen width mixes a guideline with a 600 mm convention, so it is
    # not a published clause for the whole value".
    add(
        "Kitchen",
        0.0,
        "IRC R304 exempts kitchens from both minimums and NDSS sets none, so "
        "the dataset category statistic is the only non-circular source",
        0.0,
        "the kitchen width mixes a guideline with a 600 mm convention, so it "
        "is not a published clause for the whole value",
    )

    # Table 3: area disabled, "No code source. The commonly cited IRC R301.6
    # is in fact 'Roof load', so that citation is false"; width disabled, "24
    # in depth is a trade convention with no enforceable clause behind it".
    add(
        "Closet",
        0.0,
        "no code source; the commonly cited IRC R301.6 is in fact \"Roof load\", "
        "so that citation is false",
        0.0,
        "24 in depth is a trade convention with no enforceable clause behind it",
    )

    return entries


# --- "dataset-p05": parsed at runtime from category-statistics.md ----------


def _split_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells if cell)


def _markdown_tables(text: str) -> list[list[list[str]]]:
    """Every markdown table in ``text``, as a list of rows of cells.

    A table is a run of consecutive lines that start and end with ``|``.
    Tolerates surrounding whitespace on each line.
    """
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 1:
            current.append(_split_cells(stripped))
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def _corpus_section(text: str, corpus: str) -> str:
    """The slice of ``text`` under the ``## <corpus>`` heading, up to the next
    ``##`` heading or the end of the file. Section 8.7's corpus names,
    "rplan" and "msd", are the two headings `category-statistics.md` uses.
    """
    heading = re.compile(rf"^##\s+{re.escape(corpus)}\s*$", re.MULTILINE)
    match = heading.search(text)
    if match is None:
        raise ValueError(f"category-statistics.md has no '## {corpus}' section")
    start = match.end()
    next_heading = re.search(r"^##\s+\S", text[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(text)
    return text[start:end]


def _data_rows(table: list[list[str]]) -> list[list[str]]:
    """A table's rows after the header, skipping the ``|---|---|`` separator."""
    return [row for row in table[1:] if not _is_separator_row(row)]


def _find_table(tables: list[list[list[str]]], header_first_cell: str) -> list[list[str]]:
    for table in tables:
        if table and table[0] and table[0][0].strip() == header_first_cell:
            return table
    raise ValueError(f"category-statistics.md has no table headed {header_first_cell!r}")


def _to_float(cell: str) -> float | None:
    try:
        return float(cell.replace(",", ""))
    except ValueError:
        return None


# The per-category table's header is
# "| RoomTag | count | area p5 | p25 | p50 | p75 | p95 | width p5 | p25 | p50 |
#  aspect p5 | p25 | p50 | p75 | p95 |", so these are fixed column positions
# rather than names, since several columns share a bare percentile label. Only
# the area and width floors are read from it: `propReq` and the compactness
# floor are plan-level and come from `plan-level-floors.json`.
_CATEGORY_COL = 0
_AREA_P5_COL = 2
_WIDTH_P5_COL = 7


def dataset_p05_requirements(
    kind: str,
    statistics: str | os.PathLike | None = None,
    floors: str | os.PathLike | None = None,
) -> dict[str, Requirement]:
    """The `dataset-p05` requirement source for one floor kind.

    Section 8.3: `area_req` and `min_width` come from the category's own
    corpus 5th percentile (`area p5`, `width p5`) in `category-statistics.md`.
    `prop_req` is the kind's plan-level floor from `plan-level-floors.json`
    (specification section 8.3): the per-category squareness floor at which rung 3 alone
    refuses the fitted share of that kind's real plans, stored before the
    bedroom cap. A category the floors table does not name carries the
    disabling sentinel of section 1.4, 0.0, with a reason.
    """
    _validate_kind(kind)
    corpus = KIND_CORPUS[kind]
    stats_path = (
        pathlib.Path(statistics) if statistics is not None else DEFAULT_CATEGORY_STATISTICS_PATH
    )
    text = stats_path.read_text(encoding="utf-8")
    section = _corpus_section(text, corpus)
    table = _find_table(_markdown_tables(section), "RoomTag")
    prop_floors = plan_level_floors(kind, floors).prop_req

    entries: dict[str, Requirement] = {}
    for row in _data_rows(table):
        if len(row) <= _WIDTH_P5_COL:
            continue
        category = row[_CATEGORY_COL]
        if not category:
            continue
        area_p5 = _to_float(row[_AREA_P5_COL])
        width_p5 = _to_float(row[_WIDTH_P5_COL])
        prop_req = prop_floors.get(category)

        entries[category] = Requirement(
            category=category,
            area_req=area_p5 if area_p5 is not None else 0.0,
            min_width=width_p5 if width_p5 is not None else 0.0,
            prop_req=prop_req if prop_req is not None else 0.0,
            area_reason="" if area_p5 is not None else f"category-statistics.md carries no area p5 for {category!r}",
            width_reason="" if width_p5 is not None else f"category-statistics.md carries no width p5 for {category!r}",
            prop_reason=(
                ""
                if prop_req is not None
                else f"the plan-level floors table carries no propReq for {category!r} on {kind}"
            ),
        )

    return entries


def requirements(
    source: str,
    kind: str,
    statistics: str | os.PathLike | None = None,
    floors: str | os.PathLike | None = None,
) -> dict[str, Requirement]:
    """Dispatch to whichever of the two implemented sources ``source`` names,
    for a plan of floor kind ``kind`` (see :func:`floor_kind`).

    Section 1.3: "D3 may implement `standards` and `dataset-p05`. It may not
    implement `stand-in`."
    """
    _validate_source(source)
    _validate_kind(kind)
    if source == "standards":
        return standards_requirements()
    return dataset_p05_requirements(kind, statistics, floors)


def compactness_floor(
    source: str, kind: str, floors: str | os.PathLike | None = None
) -> float:
    """The `minPassingPPScore` floor of section 8.2, under one requirement source.

    For `dataset-p05` this is the kind's plan-level floor from
    `plan-level-floors.json` (specification section 8.2): the worst-room Polsby-Popper score
    below which rung 2 alone refuses the fitted share of that kind's real
    plans. For `standards` it is :data:`STANDARDS_COMPACTNESS_FLOOR`, 0.589,
    the three-to-one-box candidate of Table 1 of the provenance table.
    """
    _validate_source(source)
    _validate_kind(kind)
    if source == "standards":
        return STANDARDS_COMPACTNESS_FLOOR
    return plan_level_floors(kind, floors).compactness
