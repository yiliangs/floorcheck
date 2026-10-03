"""What lies on the other side of each wall of one apartment.

The solver reads a segment's exposure as a set of codes. A code of zero or more
indexes the request's token list, and a negative code is one of four structural
tokens: -1 exterior, -2 corridor, -3 demising, -4 hallway. Only the first three
are read by a build, which registers one zero-area room beyond each distinct run
of them; -4 is carried for the record and for the histograms, because the
circulation it names is a room of the program on a ground-truth plan rather than
something the solver has yet to carve.

The request is one apartment, not one floor. Three reasons, and the first is
decisive. A request carries one program, one circulation seed and one entry door;
a floor with five apartments has five of each, and the union of five programs is
not a program any architect writes. Second, -3 demising has no referent until a
unit is named: it means a wall to a DIFFERENT unit, so it is a relation between
one unit and the rest of the floor, not a property of the floor. Third, the rungs
that matter here grade a dwelling, and a traversal from a bedroom to a bathroom
across five apartments is not a question. So the floor is context: every other
unit on it becomes demising, the shared core becomes corridor, and each unit is
graded on its own.

Classification of one wall run of a unit, in the order the codes are tried:

- a room of this same unit that is circulation, -4 hallway;
- any other room of this same unit, that room's token index;
- a room of another unit on this floor, -3 demising;
- a room of no unit at all, the shared core, -2 corridor;
- nothing within the wall tolerance, -1 exterior.

The third and fourth cases are only reachable from a wall on the unit's own
outline, which is where a floor with several units differs from a floor with one.
On a single-unit floor every outline wall falls through to exterior except the
entry run, which is the placeholder path the adapter keeps for comparison.

One caution to carry into the transfer specification. Case four takes EVERY room
outside every unit as corridor, which is what the solver's vocabulary can say: a
cellar compartment and a shared balcony are graded as corridor beside the real
stair core. The adapter counts corridor-facing wall length separately for
circulation rooms and for the rest, so the size of that approximation is a number
rather than an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from planaudit import geometry as geo
from planaudit import units as units_mod

EXTERIOR = -1
CORRIDOR = -2
DEMISING = -3
HALLWAY = -4

CODE_NAMES = {EXTERIOR: "exterior", CORRIDOR: "corridor", DEMISING: "demising", HALLWAY: "hallway"}


@dataclass(frozen=True)
class ExposureContext:
    """Everything one unit's walls can face, plus its own circulation rooms.

    `outside` pairs a ring on the floor but outside this unit with the code its
    walls take. `circulating` names this unit's own rooms that are circulation, by
    position in the unit's own room list, so the adapter can turn a token index
    into -4. `shared_circulation` is the subset of `outside` that is genuine
    circulation, kept so the corridor approximation can be measured.
    """

    outside: tuple[tuple[geo.Ring, int], ...]
    circulating: frozenset[int]
    shared_circulation: frozenset[int]


def context_for(view: units_mod.UnitView) -> ExposureContext:
    """Build the context one unit is graded against, from its floor."""
    floor, assignment, index = view.floor, view.units, view.index
    outside: list[tuple[geo.Ring, int]] = []
    circulation: set[int] = set()
    for room_index, room in enumerate(floor.rooms):
        unit = assignment.unit_of[room_index]
        if unit == index:
            continue
        if unit is None and room.category in units_mod.CIRCULATION:
            circulation.add(len(outside))
        outside.append((room.ring, DEMISING if unit is not None else CORRIDOR))
    circulating = frozenset(
        position
        for position, floor_index in enumerate(view.keeps)
        if floor.rooms[floor_index].category in units_mod.CIRCULATION
    )
    return ExposureContext(tuple(outside), circulating, frozenset(circulation))


def encode(owner: int, circulating: frozenset[int]) -> list[int]:
    """The exposure list one wall run carries, from the room on its other side.

    A run facing this unit's own circulation carries the hallway code beside the
    room's token index rather than instead of it. Dropping the index would erase
    the only statement in the request that the two rooms share a wall, and the
    hallway code alone is read by no gate; carrying both keeps the adjacency and
    still records the classification.
    """
    if owner < 0:
        return [owner]
    return [owner, HALLWAY] if owner in circulating else [owner]


def code_of(exposure: Sequence[int]) -> int:
    """The single class one segment belongs to, for a histogram.

    A run carrying both a token index and the hallway code is hallway; a run
    carrying a token index alone is a room of the same unit; otherwise the
    structural code is the class.
    """
    values = list(exposure)
    if HALLWAY in values:
        return HALLWAY
    negative = [v for v in values if v < 0]
    return negative[0] if negative else 0


def histogram(request: dict[str, Any]) -> dict[str, float]:
    """Wall length in meters per exposure class over a whole request.

    Class `room` is a wall to another room of the same unit, which carries a
    token index. Every segment of every node is counted once, the outline
    included, so the totals are wall length seen from both sides.
    """
    out: dict[str, float] = {name: 0.0 for name in CODE_NAMES.values()}
    out["room"] = 0.0
    for node in request.get("snapshot", []):
        for segment in node.get("segments", []):
            (ax, ay), (bx, by) = segment["a"], segment["b"]
            length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
            out[CODE_NAMES.get(code_of(segment["exposure"]), "room")] += length
    return out


def counts(request: dict[str, Any]) -> dict[str, int]:
    """Segment counts per exposure class, the companion to `histogram`."""
    out: dict[str, int] = {name: 0 for name in CODE_NAMES.values()}
    out["room"] = 0
    for node in request.get("snapshot", []):
        for segment in node.get("segments", []):
            out[CODE_NAMES.get(code_of(segment["exposure"]), "room")] += 1
    return out


def unclassified(request: dict[str, Any], token_count: int) -> list[dict[str, Any]]:
    """Segments whose exposure says nothing the solver can read.

    An empty list, or a code that is neither one of the four structural tokens
    nor an index into the request's token list, is a segment the build cannot
    resolve. The acceptance run asserts this comes back empty.
    """
    bad: list[dict[str, Any]] = []
    for node in request.get("snapshot", []):
        for segment in node.get("segments", []):
            values = list(segment.get("exposure", []))
            if not values or any(v < HALLWAY or v >= token_count for v in values):
                bad.append(segment)
    return bad
