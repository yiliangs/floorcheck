"""What D3 refuses to speak about, and why, from sections 4, 8 and 9.

A refusal here is not a verdict. Section 9 opens: "This section states what the
audit cannot speak about. It is the section a reviewer should read first, because
every margin in section 8 is conditional on it." The same holds for the checker:
a rung reported as not transferable has not passed and has not failed, and a
reader who counts it as either is reading a number the specification says does not
exist.

Two kinds are distinguished, because they have different force.

*Not transferable* is a ruling of the specification. The rung could not be
implemented from public material at all, and no future plan will change that
without new public material. Rungs 4, 5 and 6 are all of this kind.

*Refused* is conditional on the input. The rung is implementable, but this plan
does not carry what it reads, so grading it would be reading a fiction. Corridor
and demising exposure on a single-apartment input are of this kind.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Plan

NOT_TRANSFERABLE = "not transferable"
REFUSED = "refused"


@dataclass(frozen=True)
class Refusal:
    name: str
    kind: str
    reason: str


# Section 8.4, rung 4. "State: RULED not transferable, rung disabled. All four
# fields are zero under every public source, which makes both conjunctions
# vacuous."
ROOM_APPENDICES = Refusal(
    name="roomAppendices",
    kind=NOT_TRANSFERABLE,
    reason=(
        "section 8.4: both tests are conjunctions and Table 2 of the provenance "
        "table finds a public derivation for exactly one member of each pair, the "
        "RPLAN raster cell for minArea and the door clear width for minThickness, "
        "and none at all for minAreaFraction or minAspect, so a publicly derived "
        "threshold cannot act without a companion that has no public source; "
        "neutralising the companion would require the range of two solver "
        "internals and would import a production assumption by the back door. "
        "What this rules not transferable is the appendix policy itself. The "
        "rung's own precondition, the orthogonal wall loop of specification "
        "section 8.4, reads no policy and is reached before this "
        "ruling ever applies, so it is implemented in `rung_room_appendices` and "
        "does reject a room whose wall loop is not orthogonal."
    ),
)

# Section 8.5, rung 5. "State: RULED not transferable, out of scope."
SUITE_CONTAINER = Refusal(
    name="suiteContainer",
    kind=NOT_TRANSFERABLE,
    reason=(
        "section 8.5: suites are named out of scope for the public checker, "
        "alongside the production-constant "
        "appendix policy, and a plan that is one apartment or one floor carries no "
        "suite structure, so the rung is reached with nothing to contain"
    ),
)

# Section 8.6, rung 6: "Not transferable on the public path."
FINAL_ADJACENCY = Refusal(
    name="finalAdjacency",
    kind=NOT_TRANSFERABLE,
    reason=(
        "section 8.6: the rung re-checks a requested adjacency program against the "
        "final geometry, and a request without one passes; the public path carries "
        "no adjacency program (ruling 8, 2026-09-16), so the rung reads nothing a "
        "public plan states, and it has no reachability or dead-end test to stand "
        "in for it"
    ),
)

# Section 4.2. On a single-unit input, cases 3 and 4 of the exposure
# classification are unreachable.
CORRIDOR_EXPOSURE = Refusal(
    name="corridorExposure",
    kind=REFUSED,
    reason=(
        "section 4.2: cases 3 and 4 are reachable only from a wall on the unit's "
        "own outline, so on a single-unit floor every outline wall falls through "
        "to exterior except the entry run, and that single-unit fallback is the D1 "
        "placeholder path, in which every boundary segment is exterior except a "
        "synthetic entry door run at corridor; any rung reading demising or "
        "hallway exposure on that path is reading a fiction"
    ),
)

DEMISING_EXPOSURE = Refusal(
    name="demisingExposure",
    kind=REFUSED,
    reason=(
        "section 4.1: demising has no referent until a unit is named, because it "
        "means a wall to a different unit, which is a relation between one unit "
        "and the rest of the floor rather than a property of the floor; section "
        "4.2 records that on RPLAN the absence of a second unit is forced, so D2 "
        "records corridor and demising exposure on RPLAN as disabled rungs"
    ),
)

# Section 8.7 puts both off the ladder, and section 7.3 warns what happens if a
# checker puts them on it: "Two further checks are not on this ladder at all, and
# a checker that puts them there will disagree with the oracle systematically."
AREA_PROGRAM = Refusal(
    name="areaProgram",
    kind=NOT_TRANSFERABLE,
    reason=(
        "sections 1.4 and 7.3: areaReq is read as a proportional weight rather "
        "than an area floor, and on the check-only path this audit uses, where a "
        "flat snapshot reports zero non-exact cuts and the mode is not the suite "
        "double gate, the area band never gates; a specification that describes "
        "the area rung as an area floor is wrong"
    ),
)


def unit_count(plan: Plan) -> int:
    """How many apartments this plan states.

    Section 3.2 derives units from access edges, which the section 2 input model
    does not carry (log finding 4), so a plan either labels its rooms with a unit
    or is the single-unit case.
    """
    units = {room.unit for room in plan.rooms if room.unit is not None}
    return len(units) if units else 1


def exposure_refusals(plan: Plan) -> list[Refusal]:
    """The exposure rungs this plan cannot be graded on.

    A plan with two or more units carries what section 4.2's cases 3 and 4 read,
    so neither refusal applies to it. A single-unit plan carries neither.
    """
    if unit_count(plan) > 1:
        return []
    return [CORRIDOR_EXPOSURE, DEMISING_EXPOSURE]
