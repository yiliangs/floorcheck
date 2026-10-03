"""The gate ladder of specification sections 7 and 8.

Section 7.1 fixes the order and the stopping rule:

    "The order is load-bearing: each check assumes what the one above it proved.
    A check that rejects returns immediately, so a plan is reported at the first
    check it fails and the checks below it never run. A histogram over stages is
    therefore a statement about where a corpus stops rather than a count of
    everything wrong with it, and it is the single most important thing for a
    reader of section 9 to hold on to."

| # | Rung | Wire name | D3 |
|---|---|---|---|
| 1 | Room outlines | `roomOutline` | implemented, section 8.1 |
| 2 | Room compactness | `roomCompactness` | implemented, section 8.2 |
| 3 | Room proportions | `roomProportions` | implemented, section 8.3 |
| 4 | Room appendices | `roomAppendices` | precondition implemented, policy not transferable, section 8.4 |
| 5 | Suite container | `suiteContainer` | not transferable, section 8.5 |
| 6 | Final adjacency constraints | `finalAdjacency` | not transferable, section 8.6 |

A rung this checker cannot run does not stop the ladder. Section 7.2 records the
same behaviour for a profile that skips rungs: "A profile changes which rungs run
and what a failure does. It never changes the sequence, so one ordering serves
every profile. ... The surviving rungs keep their relative position." A rung ruled
not transferable is skipped in exactly that sense, so rung 6 is still reached
and reported as not transferable.

Rung 4 is a partial case rather than a pure skip. Specification section 8.4: the
check reaches its appendix-policy ruling only after a precondition, the
verification of each room's wall loop, that reads no policy at all and rejects on its own. That precondition is implemented
here and can stop the ladder like any other rung; a plan that clears it still
reports rung 4 as not transferable, because the policy behind it remains out of
reach, and the ladder continues exactly as before.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field

from .constants import ABS_TOL_M
from .measure import (
    area,
    bounding_sides,
    min_width,
    polsby_popper,
    squareness,
)
from .model import Plan, Ring
from .requirements import Requirement, compactness_floor, floor_kind, requirements
from .scope import FINAL_ADJACENCY, ROOM_APPENDICES, SUITE_CONTAINER, Refusal, unit_count
from .tags import is_hallway, tag_for

PASSED = "passed"
REJECTED = "rejected"
NOT_REACHED = "not reached"
# Section 8.6, predicate 2: a plan carrying no entry "has this test reported as
# not run rather than silently passed". A rung that is not run leaves the plan
# with no verdict, as an ingest refusal does, rather than a pass.
NOT_RUN = "not run"
# Section 9.4: a precondition fault "is not a verdict"; the rung is not run and
# the plan, though admitted, receives none. The solver's message, quoted in
# section 8.4.
NON_ORTHOGONAL_SEGMENT = "Room boundary contains a non-orthogonal segment"
# Not a verdict the specification knows: a check named to `grade(suppress=...)`
# that would have rejected the plan. It was evaluated, its reason is kept, and
# the plan went on to the checks below. Only a counterfactual run produces it.
SUPPRESSED = "suppressed"

# Section 8.3: "The solver caps the enforced floor at 0.60 for a `Bedroom` or a
# `PrimaryBedroom`, so a bedroom is never held above that value however strict
# its requirement is", that is, the enforced floor is `min(propReq, 0.60)`.
BEDROOM_PROP_CAP = 0.60
BEDROOM_CLASS = frozenset({"Bedroom", "PrimaryBedroom"})


@dataclass
class RungResult:
    number: int
    name: str
    verdict: str
    reason: str = ""
    worst: dict[str, float | str] = field(default_factory=dict)


@dataclass
class Graded:
    rungs: list[RungResult]
    requirement_source: str
    corpus: str
    floor_kind: str
    stopped_at: str | None = None
    # The rungs whose rejection this grading suppressed; see `grade`.
    suppressed: frozenset[str] = frozenset()

    @property
    def not_run(self) -> str | None:
        """The rung the ladder reached but could not run, which leaves no verdict."""
        return next((r.name for r in self.rungs if r.verdict == NOT_RUN), None)

    @property
    def passed(self) -> bool:
        return self.stopped_at is None and self.not_run is None


def _tags(plan: Plan) -> list[str]:
    return [tag_for(room.category) for room in plan.rooms]


# --- rung 1, section 8.1 ---------------------------------------------------


def rung_room_outlines(rings: list[Ring]) -> RungResult:
    """Section 8.1: "Reads geometry alone. No request constant and no per-token
    requirement is associated with it in the audit, and no ruling has been
    recorded against it. A plan that reaches this rung and fails it has a
    geometric defect the ingest refusals of section 6 did not catch; the audit
    records the stage and does not interpret it further. State: no ruling."

    The specification never says what the rung tests, only that it reads geometry
    and that what it catches is a defect section 6 did not (log finding 19). The
    narrowest reading that satisfies both sentences is taken: a room outline is
    sound when it is a simple ring of positive area, which is a property of
    geometry alone and is not among the things section 6 measures, since section 6
    measures only how rings cover and overlap the floor and never whether one ring
    is itself well formed.
    """
    from shapely.geometry import Polygon

    for index, ring in enumerate(rings):
        if len(ring) < 3:
            return RungResult(1, "roomOutline", REJECTED, f"room {index} is not a ring")
        polygon = Polygon(ring)
        if polygon.area <= 0.0:
            return RungResult(1, "roomOutline", REJECTED, f"room {index} has no area")
        if not polygon.is_simple:
            return RungResult(
                1, "roomOutline", REJECTED, f"room {index} has a self-intersecting outline"
            )
        if not polygon.is_valid:
            return RungResult(1, "roomOutline", REJECTED, f"room {index} is not a valid ring")
    return RungResult(1, "roomOutline", PASSED)


# --- rung 2, section 8.2 ---------------------------------------------------


def rung_room_compactness(plan: Plan, rings: list[Ring], floor: float) -> RungResult:
    """Section 8.2: "The request constant `minPassingPPScore`, a dimensionless
    floor in the half-open interval from zero to one, applied independently of any
    token scalar. A room's score is the Polsby-Popper ratio, four pi times area
    over perimeter squared, reported on the wire as `ppScore`. The rung takes each
    plan's worst room, excluding hallways as the solver's own lowest-score
    reduction does."
    """
    worst_score = None
    worst_index = -1
    for index, (room, ring) in enumerate(zip(plan.rooms, rings)):
        if is_hallway(room.category):
            continue
        score = polsby_popper(ring)
        if worst_score is None or score < worst_score:
            worst_score, worst_index = score, index

    if worst_score is None:
        # Every room is a hallway, so the reduction has nothing to take a
        # minimum over. The specification does not name this case; taking it as a
        # pass follows the reduction's own shape, a minimum over an empty set
        # binding nothing. Log finding 20.
        return RungResult(2, "roomCompactness", PASSED, "every room is a hallway")

    result = RungResult(
        2,
        "roomCompactness",
        PASSED,
        worst={
            "room": worst_index,
            "category": plan.rooms[worst_index].category,
            "ppScore": worst_score,
            "minPassingPPScore": floor,
        },
    )
    if worst_score < floor:
        result.verdict = REJECTED
        result.reason = (
            f"room {worst_index} ({plan.rooms[worst_index].category}) scores "
            f"{worst_score:.4f} against a floor of {floor:.4f}"
        )
    return result


# --- rung 3, section 8.3 ---------------------------------------------------


def rung_room_proportions(
    plan: Plan, rings: list[Ring], reqs: dict[str, Requirement]
) -> RungResult:
    """Section 8.3: "The per-room requirement `propReq`, a dimensionless ratio in
    the half-open interval from zero to one. It is a minimum squareness, not a
    limit on elongation: the check rejects when a room's squareness, the short
    side over the long side of its unrotated bounding box, is below its floor."

    Section 8.3 names no rotation: the check reads the solver's unrotated,
    axis-aligned bounding box of the room, not a box turned into the
    plan's dominant wall direction. That correction is section 2.5's, and it is
    for rung 4's orthogonality precondition, which reads a wall's angle against
    the nearest axis; this rung reads a box's short side over its long side, and
    the box section 8.3 means is the one the solver never rotates. So `actual`
    here is ``squareness(ring)`` at its default angle of zero, not
    ``squareness(ring, dominant_direction(rings))``.

    Three further qualifications from the same subsection are honoured here.
    "The floor is a vector rather than a number: every room is compared against
    its own token's `propReq`, indexed by category". The bedroom cap. And
    "A room within `abstol`, one millimetre, of its floor is admitted."

    Section 1.4 disables the rung per room with a `propReq` of 0.0 or below: "the
    proportions check skips a room whose requirement is not positive".
    """
    worst = None
    for index, (room, ring) in enumerate(zip(plan.rooms, rings)):
        tag = tag_for(room.category)
        requirement = reqs.get(tag)
        if requirement is None:
            # Section 8.7: a category that falls to the unset tag "takes no
            # requirement at all".
            continue
        floor = requirement.prop_req
        if floor <= 0.0:
            continue
        if tag in BEDROOM_CLASS:
            floor = min(floor, BEDROOM_PROP_CAP)
        # Unrotated: section 8.3 reads the room's axis-aligned bounding box.
        actual = squareness(ring)
        margin = actual - floor
        if worst is None or margin < worst[0]:
            worst = (margin, index, actual, floor, tag)

    if worst is None:
        return RungResult(
            3, "roomProportions", PASSED, "no token carries a proportions requirement"
        )

    margin, index, actual, floor, tag = worst
    result = RungResult(
        3,
        "roomProportions",
        PASSED,
        worst={
            "room": index,
            "category": plan.rooms[index].category,
            "tag": tag,
            "squareness": actual,
            "propReq": floor,
        },
    )
    # Section 8.3: "A room within abstol, one millimetre, of its floor is admitted."
    if actual < floor - ABS_TOL_M:
        result.verdict = REJECTED
        result.reason = (
            f"room {index} ({plan.rooms[index].category}, {tag}) has squareness "
            f"{actual:.4f} against a floor of {floor:.4f}"
        )
    return result


# --- rung 4, section 8.4 ---------------------------------------------------


def rung_room_appendices(plan: Plan, rings: list[Ring]) -> RungResult:
    """Section 8.4 rules the appendix policy not transferable (`ROOM_APPENDICES`
    in `scope.py`), but the check reaches that ruling only after a
    precondition the solver enforces regardless of policy: it verifies each
    room's wall loop before it loads the policy, and that verification rejects a
    boundary that is not closed, is not a polyline, or carries a segment aligned
    with neither axis to within `abstol`. The precondition applies to a room that
    is internal and not tagged Hallway (section 8.4).

    floorcheck carries a hallway tag, `tags.is_hallway`, so that half of the
    exclusion is honoured the same way `rung_room_compactness` honours it.
    floorcheck has no internal/external notion to read (nothing in `model.py` or
    `tags.py` names one), so the precondition is applied to every room but a
    hallway, which is the widest reading the audit's own input model supports.

    A ring here already passed the closed, deduplicated ring that ingest
    guarantees (section 2.2), so only the third refusal, a non-orthogonal
    segment, is reachable from this side. A ring is stored open (`_as_ring` in
    `model.py`), so the loop's own segments close from the last point back to the
    first: `ring[i]` to `ring[(i + 1) % len(ring)]`. A segment is orthogonal when
    its endpoints share a y or share an x within `ABS_TOL_M`; the fault is
    reported through the smaller of the two axis deltas, which is how far short
    of either axis the segment falls.

    A room that clears every segment has still not been measured against the
    appendix policy, which section 8.4 rules not transferable on its own terms,
    so a clean plan reports the same not-transferable result the rung reported
    before this precondition existed.
    """
    for index, (room, ring) in enumerate(zip(plan.rooms, rings)):
        if is_hallway(room.category):
            continue
        count = len(ring)
        for i in range(count):
            start = ring[i]
            end = ring[(i + 1) % count]
            along_x = abs(start[1] - end[1]) <= ABS_TOL_M
            along_y = abs(start[0] - end[0]) <= ABS_TOL_M
            if along_x or along_y:
                continue
            deviation = min(abs(start[1] - end[1]), abs(start[0] - end[0]))
            return RungResult(
                4,
                "roomAppendices",
                NOT_RUN,
                (
                    f"precondition fault: {NON_ORTHOGONAL_SEGMENT}: room {index} "
                    f"({room.category}) has a segment {deviation:.4f} m off both "
                    f"axes, past the {ABS_TOL_M} m specification section 8.4 allows, "
                    "so the appendix test did not run"
                ),
                worst={"room": index, "category": room.category, "offAxisM": deviation},
            )
    return _skipped(ROOM_APPENDICES, 4)


# --- the ladder ------------------------------------------------------------


def _skipped(refusal: Refusal, number: int) -> RungResult:
    return RungResult(number, refusal.name, refusal.kind, refusal.reason)


def grade(
    plan: Plan,
    rings: list[Ring],
    source: str = "dataset-p05",
    corpus: str | None = None,
    statistics: str | os.PathLike | None = None,
    floors: str | os.PathLike | None = None,
    suppress: Iterable[str] = (),
) -> Graded:
    """Run the six rungs in the order of section 7.1, stopping at the first
    rejection.

    ``suppress`` names rungs, by wire name, whose rejection is suppressed for a
    counterfactual: such a rung is still evaluated, a rejection is recorded as
    `SUPPRESSED` with its reason, and the plan continues to the rungs below.
    Nothing else changes: a later rejection stops the plan there, and a
    precondition fault below still stops it with no verdict. Empty, the default,
    is the ladder of section 7.1.

    The rung 2 and rung 3 floors are the plan-level floors of the plan's kind,
    `floor_kind(corpus, unit_count(plan))`: an MSD plan stating more than one
    apartment is a whole floor, any other MSD plan an apartment, and every
    other plan takes the RPLAN floors.

    ``statistics`` is the `dataset-p05` category-statistics table the area and
    width requirements are read from, and ``floors`` the plan-level floors
    table; ``None`` keeps each committed default beside `requirements.py`.
    """
    suppressed = frozenset(suppress)
    unknown = sorted(suppressed - set(_WIRE_NAMES.values()))
    if unknown:
        raise ValueError(f"no rung named {', '.join(unknown)} on the ladder")
    corpus = corpus or ("msd" if plan.source == "msd" else "rplan")
    kind = floor_kind(corpus, unit_count(plan))
    reqs = requirements(source, kind, statistics, floors)
    floor = compactness_floor(source, kind, floors)

    graded = Graded(
        rungs=[], requirement_source=source, corpus=corpus, floor_kind=kind, suppressed=suppressed
    )

    ordered = [
        lambda: rung_room_outlines(rings),
        lambda: rung_room_compactness(plan, rings, floor),
        lambda: rung_room_proportions(plan, rings, reqs),
        lambda: rung_room_appendices(plan, rings),
        lambda: _skipped(SUITE_CONTAINER, 5),
        lambda: _skipped(FINAL_ADJACENCY, 6),
    ]

    stopped = False
    for number, run in enumerate(ordered, start=1):
        if stopped:
            graded.rungs.append(RungResult(number, _WIRE_NAMES[number], NOT_REACHED))
            continue
        result = run()
        if result.verdict == REJECTED and result.name in suppressed:
            result.verdict = SUPPRESSED
        graded.rungs.append(result)
        if result.verdict == REJECTED:
            graded.stopped_at = result.name
        # Section 9.4: a rung not run for a precondition fault stops the ladder
        # too, but names no rung the plan was stopped at: it has no verdict.
        if result.verdict in (REJECTED, NOT_RUN):
            stopped = True

    return graded


_WIRE_NAMES = {
    1: "roomOutline",
    2: "roomCompactness",
    3: "roomProportions",
    4: "roomAppendices",
    5: "suiteContainer",
    6: "finalAdjacency",
}


def construction_observations(
    plan: Plan, rings: list[Ring], reqs: dict[str, Requirement], angle: float
) -> list[dict[str, float | str]]:
    """`minWidth`, reported beside the ladder and never on it.

    Section 7.3: "Two further requirements are not on this ladder at all, and a
    checker that puts them there will disagree with the solver systematically. The
    minimum width requirement is applied while the solver constructs a layout, not
    as a check on a finished plan (section 8.7). The area program does not gate on
    the check-only path, for the reason given in section 1.4."

    Section 1.4 disables it with a zero: "the width test rejects only a span
    shorter than the requirement less a tolerance, which no span is".
    """
    out: list[dict[str, float | str]] = []
    for index, (room, ring) in enumerate(zip(plan.rooms, rings)):
        tag = tag_for(room.category)
        requirement = reqs.get(tag)
        if requirement is None or requirement.min_width <= 0.0:
            continue
        width = min_width(ring, angle)
        out.append(
            {
                "room": index,
                "category": room.category,
                "tag": tag,
                "minWidthM": width,
                "requiredM": requirement.min_width,
                "meets": width >= requirement.min_width - ABS_TOL_M,
            }
        )
    return out
