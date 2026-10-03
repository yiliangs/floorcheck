"""The input model of specification section 2.

A plan is "a set of rooms, each carrying exactly one closed ring of coordinates
and one dataset category, plus an entrance, plus the scale and frame of the
source it came from. It carries no boundary polygon." (section 2.1)

The specification states that model in prose and never fixes a wire format, so
the JSON spelling below is this implementation's, documented in
``floorcheck/SCHEMA.md``. Only the field meanings are the
specification's; the field names are not.

The load-bearing constraint of this module is section 2.1's first rule:

    "One room is one ring. This is true in the audit, in the oracle's snapshot
    node, and in the partition output, and it is the constraint that makes a room
    wrapping another inexpressible rather than merely awkward. A checker that
    models a room as a polygon with holes will accept plans this specification
    refuses, and the two will disagree on exactly the population section 9
    counts."

So a ring with a hole is refused at parse time rather than repaired. Refusals
here are ``PlanRejected``, which the CLI reports as an ingest outcome; they are
never exceptions that escape, per the D3 clean-room rule's requirement that the
checker "grade any plan ... including invalid generator output (overlaps, rooms
outside the boundary, missing entries), with a reject reason rather than an
exception".
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

Point = tuple[float, float]
Ring = tuple[Point, ...]


class PlanRejected(Exception):
    """A plan the input model of section 2 cannot express.

    Carries a stable reason token so the CLI can emit a named reject reason
    rather than a traceback.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


# Sections 2.3, 2.4 and 6.3 fix the frames and the source pixels; they live in
# `floorcheck.constants` so that every constant in the package has one home.
from .constants import RPLAN_CELL_M as RPLAN_PIXEL_M  # noqa: E402
from .constants import JOIN_TOL_M, SOURCE_PIXEL_M  # noqa: E402

# Section 3.2 step 3, quoting the specification's own vocabulary: "with
# `corridor` and `stairs` as circulation, `bedroom`, `livingroom`, `kitchen` and
# `dining` as habitable, and `bathroom` as the wet room the test requires."
CIRCULATION_CATEGORIES = frozenset({"corridor", "stairs"})
HABITABLE_CATEGORIES = frozenset({"bedroom", "livingroom", "kitchen", "dining"})
WET_CATEGORY = "bathroom"


@dataclass(frozen=True)
class Room:
    """One room: exactly one closed ring and exactly one dataset category."""

    category: str
    ring: Ring
    # Section 3.2 derives an apartment on MSD. A plan that already knows its
    # units may say so; a plan that does not leaves this None and section 4.2's
    # single-unit fallback applies.
    unit: str | None = None

    @property
    def is_circulation(self) -> bool:
        return self.category in CIRCULATION_CATEGORIES

    @property
    def is_habitable(self) -> bool:
        return self.category in HABITABLE_CATEGORIES


@dataclass(frozen=True)
class Plan:
    """One plan in the input model of section 2."""

    rooms: tuple[Room, ...]
    source: str
    # Section 2.1: "It carries no boundary polygon. The outline is derived".
    # Section 5.2, however, says the grid partition's recovered outline "is the
    # plan's floor extent, it travels on the plan, and it is what bounds the seam
    # repair of section 6.3 later, because it is the one thing this path decides
    # that the room set it returns cannot express." So the extent is an optional
    # input, distinct from the derived outline.
    floor_extent: Ring | None = None
    # Section 2.3: RPLAN's entrance is read from the raw record. Section 3.2 step
    # 5 gives MSD's. Either way it reaches this model as a point or not at all.
    entry: Point | None = None
    # What the source states about its own plan, carried and never graded.
    # Section 2.6: "provenance travelling with the plan, never a requirement",
    # because section 8.6's rung 6 reads the adjacency it derives from the
    # geometry and the audit path never sends a stated list at all. Kept so the
    # schema can still record what a source asserted, which is the only way to
    # ask why RPLAN's `rEdge` is a strict superset of wall adjacency. Log
    # findings 21 and 25.
    adjacency: tuple[tuple[int, int], ...] = ()
    name: str = "plan"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def source_pixel_m(self) -> float | None:
        """The smallest length this plan's source can tell apart, in metres.

        Section 6.3: "The pixel is stated by the reader, not inferred. Every plan
        carries the smallest length its source can tell apart, in metres, as a
        field of its own ... It is not the same fact as the plan's
        `units_per_meter`, which converts that plan's own coordinates, and the
        two part company wherever a source states metric geometry that was
        resolved at a coarser step. MSD is exactly that case: its coordinates are
        metres while its partition resolves the floor onto a grid."

        So a plan may state its own, and a plan that does not falls back to the
        table of section 6.3. ``None`` means no length is known for the source,
        which the ingest reports rather than guesses at.
        """
        if "sourcePixelM" in self.meta:
            return float(self.meta["sourcePixelM"])
        return SOURCE_PIXEL_M.get(self.source)

    @property
    def seam_tolerance_m(self) -> float | None:
        """The source pixel widened by the quantisation step, or ``None``.

        Section 6.3: "The adapter widens that length by its own quantisation step
        before applying it". Ingest closes seams at this width, and section 8.6
        reads whether the entry lies on a room at the same width.
        """
        pixel = self.source_pixel_m
        return None if pixel is None else pixel + JOIN_TOL_M


def _as_point(value: Any, where: str) -> Point:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise PlanRejected("malformed-point", f"{where} is not a pair of numbers")
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        raise PlanRejected("malformed-point", f"{where} is not a pair of numbers") from None
    if not math.isfinite(x) or not math.isfinite(y):
        # `float()` accepts NaN and both infinities without raising, and a NaN
        # that reaches the rungs is worse than a refusal: every length, area and
        # ratio computed from it is NaN, every comparison against a floor is
        # false, and the plan passes the ladder silently. Section 2.6 lists this
        # among the named refusals for that reason. The corpus does produce it:
        # RPLAN stores a handful of multi-part rooms with a literal NaN separator
        # row, and two plans of the 4,000 the seventh run graded, `rplan/42921`
        # and `rplan/77154`, carry one into the schema.
        raise PlanRejected("non-finite-coordinate", f"{where} is not a finite number")
    return (x, y)


def _as_ring(value: Any, where: str) -> Ring:
    """Read one closed ring.

    Section 2.2: "every ring is closed, deduplicated and oriented
    counter-clockwise". Closing and orienting happen in ingest; this function
    only refuses what is not a ring at all.
    """
    if isinstance(value, dict):
        # A GeoJSON-style polygon with an interior ring list is exactly the shape
        # section 2.1 refuses, and saying so by name is more useful than
        # "malformed ring".
        raise PlanRejected("room-with-hole", f"{where} is a polygon object, not a ring")
    if not isinstance(value, (list, tuple)):
        raise PlanRejected("malformed-ring", f"{where} is not a list of points")
    if value and isinstance(value[0], (list, tuple)) and value[0] and isinstance(
        value[0][0], (list, tuple)
    ):
        # [[exterior...], [hole...]] is the other spelling of a polygon with
        # holes. Section 2.1: one room is one ring.
        raise PlanRejected("room-with-hole", f"{where} carries more than one ring")
    points = tuple(_as_point(p, f"{where}[{i}]") for i, p in enumerate(value))
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        raise PlanRejected("malformed-ring", f"{where} has fewer than three distinct vertices")
    return points


def plan_from_dict(payload: Any, name: str = "plan") -> Plan:
    """Build a :class:`Plan` from parsed JSON, refusing what section 2 cannot express."""
    if not isinstance(payload, dict):
        raise PlanRejected("malformed-plan", "the top level is not an object")

    raw_rooms = payload.get("rooms")
    if not isinstance(raw_rooms, list) or not raw_rooms:
        raise PlanRejected("no-rooms", "a plan carries at least one room")

    rooms: list[Room] = []
    for index, raw in enumerate(raw_rooms):
        if not isinstance(raw, dict):
            raise PlanRejected("malformed-room", f"room {index} is not an object")
        category = raw.get("category")
        if not isinstance(category, str) or not category:
            # Section 2.1: each room carries "one dataset category". A room
            # without one cannot be given a token, so it is not expressible.
            raise PlanRejected("missing-category", f"room {index} carries no category")
        if "holes" in raw or "interiors" in raw:
            raise PlanRejected("room-with-hole", f"room {index} declares interior rings")
        ring = _as_ring(raw.get("ring"), f"room {index} ring")
        unit = raw.get("unit")
        if unit is not None and not isinstance(unit, str):
            raise PlanRejected("malformed-room", f"room {index} unit is not a string")
        rooms.append(Room(category=category, ring=ring, unit=unit))

    source = payload.get("source")
    if not isinstance(source, str) or not source:
        # Section 2.1 makes "the scale and frame of the source it came from" part
        # of a plan, and section 6.3's width rule reads it.
        raise PlanRejected("missing-source", "a plan names the source it came from")
    source = source.lower()

    extent_raw = payload.get("floorExtent")
    floor_extent = _as_ring(extent_raw, "floorExtent") if extent_raw is not None else None

    entry_raw = payload.get("entry")
    entry = _as_point(entry_raw, "entry") if entry_raw is not None else None

    adjacency: list[tuple[int, int]] = []
    for pair in payload.get("adjacency", ()) or ():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise PlanRejected("malformed-adjacency", "an adjacency entry is not a pair")
        try:
            a, b = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            raise PlanRejected("malformed-adjacency", "an adjacency entry is not a pair of indices") from None
        if not (0 <= a < len(rooms)) or not (0 <= b < len(rooms)):
            raise PlanRejected("malformed-adjacency", f"adjacency ({a}, {b}) indexes no room")
        if a == b:
            raise PlanRejected("malformed-adjacency", f"adjacency ({a}, {b}) joins a room to itself")
        adjacency.append((a, b) if a < b else (b, a))

    meta = payload.get("meta")
    if meta is not None and not isinstance(meta, dict):
        raise PlanRejected("malformed-plan", "meta is not an object")
    if meta and "sourcePixelM" in meta:
        # `source_pixel_m` reads a stated pixel with float(); a value it cannot
        # read (null, a word) is refused here, since section 2.6 makes every
        # refusal a named outcome and never an exception.
        try:
            float(meta["sourcePixelM"])
        except (TypeError, ValueError):
            raise PlanRejected("malformed-plan", "meta.sourcePixelM is not a number") from None

    return Plan(
        rooms=tuple(rooms),
        source=source,
        floor_extent=floor_extent,
        entry=entry,
        adjacency=tuple(dict.fromkeys(adjacency)),
        name=str(payload.get("name", name)),
        meta=dict(meta or {}),
    )


def plan_from_json(text: str, name: str = "plan") -> Plan:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlanRejected("malformed-json", str(exc)) from None
    return plan_from_dict(payload, name=name)


def load_plan(path: str) -> Plan:
    with open(path, "r", encoding="utf-8") as handle:
        return plan_from_json(handle.read(), name=path)
