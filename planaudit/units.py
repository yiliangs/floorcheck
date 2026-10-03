"""Apartment-unit membership on a multi-unit floor, derived rather than read.

MSD carries no unit field. `graph_out` gives a `room_type` per room and a
`connectivity` per access edge, and nothing anywhere says which apartment a room
belongs to. The rule below derives it. It is part of the transfer specification,
because every corridor and demising ruling on MSD rests on it.

The rule, in plain words.

1. Cut every access edge whose connectivity is `entrance`. An entrance edge is an
   apartment door or a core door, never an internal one: over the 4,167 train
   floors, 14,205 of the 16,517 entrance edges join two circulation rooms, which
   is a unit's own entry hall meeting the shared stair or corridor.
2. Take the connected components of what is left, over the `door` and `passage`
   edges. A component is a set of rooms reachable from one another without
   passing an apartment door.
3. Keep a component as an apartment only if it is a dwelling, which is taken in
   the ordinary sense a building code takes it: it holds a bathroom and at least
   one habitable room, a bedroom, a living room, a kitchen or a dining room.
   Everything else is context and not a unit: the shared stair and corridor core,
   a cellar compartment, a lone bay off the landing, a communal kitchen, a shared
   balcony.
4. Every room of every discarded component is shared context. Such a room never
   enters any unit's program, and the walls of unit rooms that face it carry
   corridor exposure rather than a token.
5. A unit's entry seed is the first entrance edge joining one of its rooms to a
   room outside it. A unit the floor gave no entrance edge falls back to a door
   placed on its longest corridor-facing boundary run.

Why the components alone will not do, and what step 3 costs. Cutting entrance
edges and counting components gives 24,430 on the train split, which scales to
31,072 over the roughly 5,300 plans of the whole dataset against the paper's
18,900 apartments, a 64 per cent overcount: 7,563 of those components are pure
circulation and a further thousand-odd are single storerooms, single bedrooms and
single balconies. Step 3 brings the train figure to 14,934, which scales to
18,994, half a per cent above the paper. Seven candidate rules were measured;
the one here was both the closest and the only one that states a dwelling in the
terms a code does.

What the rule cannot do. It reads topology, so a floor whose entrance doors were
labelled `door` rather than `entrance` fuses into one unit, and a maisonette
whose two storeys are two plans counts twice. Seventeen train floors yield no
unit at all under it. None of that is detectable from a single floor's graph, and
all of it is recorded as known residual rather than patched.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from planaudit import geometry as geo
from planaudit.plan import AccessEdge, Plan, Room

ENTRANCE_KINDS = frozenset({"entrance"})

# MSD's room vocabulary as the loader spells it. Circulation is the pair that can
# be either a unit's own hall or the building's shared core; which one it is, is
# topological, and step 3 of the rule is what decides it.
CIRCULATION = frozenset({"corridor", "stairs"})
HABITABLE = frozenset({"bedroom", "livingroom", "kitchen", "dining"})
WET = frozenset({"bathroom"})


@dataclass(frozen=True)
class Units:
    """Which rooms of one floor form which apartment, and what is shared.

    Every index is into `Plan.rooms` of the floor this was derived from.
    `entry_rooms[i]` is the room of unit `i` that carries an entrance edge to a
    shared room, or None where the floor labelled no entrance door for it.
    """

    members: tuple[tuple[int, ...], ...]
    unit_of: tuple[Optional[int], ...]
    shared: tuple[int, ...]
    entry_rooms: tuple[Optional[int], ...]

    def __len__(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class UnitView:
    """One apartment as a Plan of its own, with the floor it was cut from.

    Exposure synthesis needs both: the unit supplies the program and the rings to
    grade, and the floor supplies everything the unit's outer walls can face.
    `keeps` maps each room of `plan` back to its index on `floor`.
    """

    plan: Plan
    floor: Plan
    units: Units
    index: int
    keeps: tuple[int, ...]


def is_unit(categories: Iterable[str]) -> bool:
    """Step 3: does this set of rooms hold a dwelling rather than context.

    A dwelling is a bathroom plus at least one habitable room. Requiring the
    bathroom is what separates an apartment from a communal kitchen, a cellar
    compartment or a single let room off the landing, all of which the component
    construction otherwise hands back as units.
    """
    kinds = set(categories)
    return bool(kinds & WET) and bool(kinds & HABITABLE)


def components(plan: Plan) -> list[list[int]]:
    """Rooms grouped by reachability without passing an entrance door."""
    parent = list(range(len(plan.rooms)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for edge in plan.access:
        if edge.kind in ENTRANCE_KINDS:
            continue
        if not (0 <= edge.a < len(parent) and 0 <= edge.b < len(parent)):
            continue
        ra, rb = find(edge.a), find(edge.b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for index in range(len(parent)):
        groups.setdefault(find(index), []).append(index)
    return [sorted(g) for _, g in sorted(groups.items())]


def assign(plan: Plan) -> Units:
    """Apply steps 1 to 3 of the rule to one floor, and seed step 5."""
    members: list[tuple[int, ...]] = []
    shared: list[int] = []
    unit_of: list[Optional[int]] = [None] * len(plan.rooms)
    for group in components(plan):
        if is_unit(plan.rooms[i].category for i in group):
            for i in group:
                unit_of[i] = len(members)
            members.append(tuple(group))
        else:
            shared.extend(group)
    return Units(
        members=tuple(members),
        unit_of=tuple(unit_of),
        shared=tuple(sorted(shared)),
        entry_rooms=_entry_rooms(plan, unit_of, len(members)),
    )


def _entry_rooms(plan: Plan, unit_of: list[Optional[int]], count: int) -> tuple[Optional[int], ...]:
    """The room of each unit that carries an entrance door onto shared context.

    A unit with several entrance doors keeps the first in edge order, because the
    solver's request describes one entry. A unit with none keeps None, which only
    matters where its outline also touches no shared room: the adapter then has
    nothing to anchor on and puts a synthetic door on the longest outline run.
    """
    out: list[Optional[int]] = [None] * count
    for edge in plan.access:
        if edge.kind not in ENTRANCE_KINDS:
            continue
        for near, far in ((edge.a, edge.b), (edge.b, edge.a)):
            if not (0 <= near < len(unit_of) and 0 <= far < len(unit_of)):
                continue
            unit = unit_of[near]
            if unit is None or unit_of[far] == unit or out[unit] is not None:
                continue
            out[unit] = near
    return tuple(out)


def unit_view(plan: Plan, units: Units, index: int) -> UnitView:
    """Cut one apartment out of its floor as a Plan the adapter can take.

    The rooms of one unit tile a single region whenever the unit is spatially
    contiguous, which is the ordinary case. Where the partition leaves it in
    pieces, the largest piece is the unit and the rooms outside it are dropped,
    because the solver requires the leaves to tile the root exactly. That outline
    is only used to decide membership here; the adapter derives the root it sends
    from the rings it actually emits, under the same rule.

    The apartment carries the extent of the part of the floor it lies in. A
    floor whose rooms close into several parts is refused as a floor and its
    apartments are not, each one lying wholly within one part, so the part and
    not the floor is the region that says where this apartment's building ends.
    An apartment whose rooms somehow span two parts is given none, which is the
    answer this path had for every apartment before the parts were carried.
    """
    from shapely.geometry import Polygon

    group = list(units.members[index])
    tiling = geo.tile([plan.rooms[i].ring for i in group])
    boundary = tiling.outline
    holes, parts = tiling.holes, tiling.outlying_parts + 1
    outline = Polygon(boundary)
    keeps = [i for i in group if outline.contains(Polygon(plan.rooms[i].ring).representative_point())]
    if not keeps:
        raise ValueError(f"{plan.source_id} unit {index} kept no room inside its own outline")
    local = {floor_index: position for position, floor_index in enumerate(keeps)}
    # The extent of the part of the floor this apartment lies in, and not the
    # floor's, which is the largest part alone. A floor drawn in wings closes
    # into one part per wing, every apartment lies wholly within one of them,
    # and an apartment of a wing that is not the largest had no extent to be
    # given before the partition carried every part. Without one the adapter's
    # seam repair has nothing to clip a channel against and reads a slot in the
    # building as a wall.
    floor_rings = list(plan.notes.get("partition_part_rings", ()))
    room_parts = list(plan.notes.get("partition_room_parts", ()))
    lies_in = {room_parts[i] for i in keeps if i < len(room_parts)}
    part_index = lies_in.pop() if len(lies_in) == 1 else None
    extent = floor_rings[part_index] if part_index is not None and part_index < len(floor_rings) else None
    notes = dict(plan.notes)
    # The floor's parts are a fact about the floor, and carrying every part's
    # ring into every apartment of it would repeat the whole building in each.
    notes.pop("partition_part_rings", None)
    notes.pop("partition_room_parts", None)
    notes.update(
        {
            "floor_id": plan.source_id,
            "unit_index": index,
            "unit_of_floor": len(units),
            "unit_outline_holes": holes,
            "unit_outline_parts": parts,
            "unit_rooms_dropped": len(group) - len(keeps),
            "unit_floor_part": part_index,
        }
    )
    return UnitView(
        plan=Plan(
            source=plan.source,
            source_id=f"{plan.source_id}#u{index}",
            rooms=[Room(plan.rooms[i].ring, plan.rooms[i].category, plan.rooms[i].source_index) for i in keeps],
            units_per_meter=plan.units_per_meter,
            source_pixel_m=plan.source_pixel_m,
            floor_extent=list(extent) if extent else None,
            access=_local_access(plan.access, local),
            entry=_entry_point(plan, units, index),
            off_axis_fraction=plan.off_axis_fraction,
            notes=notes,
        ),
        floor=plan,
        units=units,
        index=index,
        keeps=tuple(keeps),
    )


def _local_access(access: list[AccessEdge], local: dict[int, int]) -> list[AccessEdge]:
    return [AccessEdge(local[e.a], local[e.b], e.kind) for e in access if e.a in local and e.b in local]


def _entry_point(plan: Plan, units: Units, index: int) -> Optional[geo.Point]:
    """Centroid of the unit's entry room, which seeds the entry door run."""
    room = units.entry_rooms[index]
    if room is None or not (0 <= room < len(plan.rooms)):
        return None
    return geo.centroid(plan.rooms[room].ring)


def iter_units(plan: Plan) -> Iterator[UnitView]:
    """Yield every apartment of one floor, skipping none silently.

    A unit whose outline cannot be built raises, and the caller counts it as an
    ingest failure, which is the same discipline the floor path already follows.
    """
    units = assign(plan)
    for index in range(len(units)):
        yield unit_view(plan, units, index)
