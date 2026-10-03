"""MSD (Modified Swiss Dwellings) loader: graph_out geometry plus graph_in zoning.

Geometry source. Room rings come from `graph_out[node]["geometry"]`, a plain
Python list of (x, y) tuples in meters, closed (first point repeated). The
dataset format report records that the README's claim of a shapely Polygon is
wrong for the stored pickle, and that `torch` must be importable because the
`centroid` node attribute is a torch tensor even though the geometry is not.

Boundary, and why it cannot just be the union. MSD stores no floor outline and
its room polygons are INTERIORS, so no two rooms touch: the plain union of a
floor's rooms is one disjoint part per room, and its largest part is one room.
The rooms are therefore rebuilt as a partition by planaudit.partition, which
recovers the floor outline by closing the wall gaps, assigns every cell of it to
its nearest room, and takes the boundary from the union of the result. That
makes the flat snapshot's leaves tile its root by construction, which is what
the solver's wall model requires: without it every floor fails to build on its
first interior edge with "carries a single adjacent room after construction".

Rotation. Each floor is rotated by minus its length-weighted dominant wall
direction modulo 90 degrees, so that a building merely sited off the world axes
becomes axis-aligned. The residual off-axis wall fraction after that rotation is
recorded on the Plan. The format report measured this over all 4,167 train
plans: the segment-level rate falls from 90.6 percent to 12.6 percent at the two
degree threshold, so the rotation removes rotation and leaves genuine
irregularity behind, which is what the residual is for.

Category vocabulary. `room_type` is an integer code. MSD ships no legend in the
download, so the names below are inferred from the published MSD class order and
are marked as such. An unrecognised code becomes `msd_unknown_<code>`, which the
adapter counts rather than quietly mapping.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterator, Optional

from planaudit import geometry as geo
from planaudit import partition
from planaudit.plan import AccessEdge, Plan, Room

# Inferred, not authoritative. The MSD download carries no code-to-name legend:
# a search of the README, both zip archives and every text file under data/msd
# found no integer-to-name table, and the README describes the attributes with
# prose examples only. These names are the published MSD class order, treated as
# a hypothesis. An unrecognised code becomes msd_unknown_<code>, which the
# adapter counts rather than absorbing.
#
# The supporting evidence is the zoning coarsening. Over 200 train floors the
# room_type counts are 0:1714 1:582 2:734 3:23 4:1018 5:416 6:281 7:1171 8:781
# and the zoning_type counts are 0:1714 1:2357 2:1868 3:781, which partitions
# exactly as {0} | {1,2,3,4} | {5,6,7} | {8}: one sleeping zone, one living zone,
# one service zone, one outdoor zone. That is consistent with the order below
# and inconsistent with an order that put balcony or bathroom elsewhere.
ROOM_TYPE_NAMES: dict[int, str] = {
    0: "bedroom",
    1: "livingroom",
    2: "kitchen",
    3: "dining",
    4: "corridor",
    5: "stairs",
    6: "storeroom",
    7: "bathroom",
    8: "balcony",
}

ENTRANCE_KINDS = {"entrance"}

# The wall calibration itself, its grid step and its gap-close distance all live
# in planaudit.partition, which is where the sweep that chose them is recorded.
# Nothing here restates either number: the loader asks for the partition and
# writes back whatever step and close it actually used.


def default_root() -> Path:
    here = Path(__file__).resolve().parents[2]
    return here / "data" / "msd" / "train_extracted"


def list_ids(root: Optional[Path] = None) -> list[str]:
    root = Path(root) if root is not None else default_root()
    out = sorted(p.stem for p in (root / "graph_out").glob("*.pickle"))
    return out


def _category(code: object) -> str:
    try:
        key = int(code)
    except (TypeError, ValueError):
        return "msd_unknown_none"
    return ROOM_TYPE_NAMES.get(key, f"msd_unknown_{key}")


def _read_graph(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _entry_point(rings: list[geo.Ring], access: list[AccessEdge]) -> Optional[geo.Point]:
    """Midpoint of the shared wall of the first entrance edge, if there is one."""
    from shapely.geometry import Polygon

    for edge in access:
        if edge.kind not in ENTRANCE_KINDS:
            continue
        if edge.a >= len(rings) or edge.b >= len(rings):
            continue
        try:
            shared = Polygon(rings[edge.a]).exterior.intersection(
                Polygon(rings[edge.b]).exterior
            )
        except Exception:
            continue
        if shared.is_empty:
            continue
        # The shared geometry is a LineString when the two rooms meet along one
        # run, but calibration can leave a MultiLineString or a bare Point, and
        # only the approximate location matters here, so take the centroid.
        point = shared.centroid
        if point.is_empty:
            continue
        return (float(point.x), float(point.y))
    return None


def _access_edges(graph, index_of: dict[object, int]) -> list[AccessEdge]:
    out: list[AccessEdge] = []
    for a, b, attrs in graph.edges(data=True):
        if a not in index_of or b not in index_of:
            continue
        kind = str(attrs.get("connectivity", "unknown"))
        out.append(AccessEdge(index_of[a], index_of[b], kind))
    return out


def _partition_notes(report: partition.PartitionReport) -> dict[str, object]:
    """What the partition had to do to this floor, for the D1 run to count.

    Every field of the report travels, including the ones only one path fills,
    because which path ran is itself one of them and a reader of the notes should
    not have to know which parameters belong to which path.
    """
    return {
        "partition_method": report.method,
        "partition_step_m": report.step,
        "partition_close_m": report.close,
        "partition_wall_gap_m": report.wall_gap,
        "partition_wall_gap_spread_m": report.wall_gap_spread,
        "partition_shared_exact": report.shared_exact,
        "partition_holes_filled": report.holes_filled,
        "partition_holes_over_limit": report.holes_over_limit,
        "partition_resolution_m": report.resolution,
        "partition_precision_m": report.precision,
        "partition_square_m": report.square,
        "partition_feature_m": report.feature,
        "partition_edges_over_two": report.edges_over_two,
        "partition_coverage_valid": report.coverage_valid,
        "partition_faces_merged": report.faces_merged,
        "partition_faces_unplaced": report.faces_unplaced,
        "partition_wrapping_rooms": report.wrapping_rooms,
        "partition_wrapped_m2": report.wrapped_m2,
        "partition_split_rooms": report.split_rooms,
        "partition_split_m2": report.split_m2,
        "partition_residue_refused": report.residue_refused,
        "partition_residue_refused_m2": report.residue_refused_m2,
        # How many parts the floor closed into, which part each returned room
        # lies in, and the outer ring of each part. The rings travel because the
        # extent is a decision the partition made that the rooms cannot express,
        # and `Plan.floor_extent` can carry only the largest of them; a reader
        # cutting one part out of the floor needs that part's own.
        "partition_floor_parts": report.floor_parts,
        "partition_part_rings": [list(ring) for ring in report.part_rings],
        "partition_room_parts": list(report.room_parts),
        "partition_rooms_outside_extent": report.rooms_outside_extent,
        # Doors the source drew across a threshold the closing could not reach,
        # which this floor's partition bridged rather than leaving the rooms
        # they join in separate parts.
        "partition_bridges_used": report.bridges_used,
        "rooms_dropped_by_partition": report.rooms_in - report.rooms_out,
        "partition_area_drift": report.area_drift,
        "partition_room_area_error": report.mean_room_area_error,
    }


def _aligned(source_id: str, root: Optional[Path] = None):
    """Read one floor and rotate it onto its own dominant wall direction.

    Everything that happens before the partition, in one place because two
    callers want exactly it: `load_one`, which goes on to partition, and
    `load_interiors`, which stops here. Returns the resolved root, the raw
    graph, the rotated rings, their categories, the graph node behind each one,
    the rotation applied, and the off-axis fraction that survived it.
    """
    root = Path(root) if root is not None else default_root()
    out_graph = _read_graph(root / "graph_out" / f"{source_id}.pickle")

    node_ids = list(out_graph.nodes())
    raw_rings: list[geo.Ring] = []
    categories: list[str] = []
    kept: list[object] = []
    for node in node_ids:
        attrs = out_graph.nodes[node]
        ring = geo.close_ring([(float(x), float(y)) for x, y in attrs.get("geometry", [])])
        if len(ring) < 3:
            continue
        raw_rings.append(geo.orient_ccw(ring))
        categories.append(_category(attrs.get("room_type")))
        kept.append(node)

    if not raw_rings:
        raise ValueError(f"MSD plan {source_id} has no usable room geometry")

    angle = geo.dominant_direction(raw_rings)
    origin = geo.centroid(raw_rings[0])
    rotated = [geo.rotate_ring(r, -angle, origin) for r in raw_rings]
    residual = geo.off_axis_fraction(rotated)
    return root, out_graph, rotated, categories, kept, angle, residual


def load_interiors(source_id: str, root: Optional[Path] = None) -> Plan:
    """One MSD floor as the dataset stores it: aligned, but never partitioned.

    The rooms of this plan do not tile the floor. The two faces of a shared wall
    sit about 0.16 m apart, so the union of the rooms is one disjoint part per
    room, and nothing the solver ingests may be built from it: the wall model
    needs a partition and `load_one` is the path that supplies one.

    What this is for is measuring the dataset rather than the ingest. The grid
    partition resolves every room onto its step and moves a room's area by up to
    one cell ring, so a geometric statistic taken after it describes the
    partition as much as it describes the floor. A statistic about what MSD's
    layouts ARE reads this; a statistic about what the audit GRADES reads
    `load_one`.

    `entry` comes back None rather than guessed, because the entrance point is
    the midpoint of a shared wall and no two rooms here share one.
    """
    root, out_graph, rotated, categories, kept, angle, residual = _aligned(source_id, root)
    index_of = {node: i for i, node in enumerate(kept)}
    return Plan(
        source="msd",
        source_id=source_id,
        rooms=[Room(rotated[i], categories[i], i) for i in range(len(rotated))],
        units_per_meter=1.0,
        # Nothing on this path resolved anything onto a step. These are the
        # dataset's own corners and MSD has never stated what it resolved them
        # at, while the partition, which is where that value is decided, is the
        # one thing this path does not run. So the reader states the zero it can
        # defend rather than borrowing a step from a path that did not run. The
        # field is not read as a claim of exactness here because nothing reads
        # it: a plan whose rooms do not tile is refused at ingest by
        # construction, and what this path is for is measuring the dataset.
        source_pixel_m=0.0,
        access=_access_edges(out_graph, index_of),
        entry=None,
        off_axis_fraction=residual,
        notes={
            "rotation_radians": -angle,
            "partition_method": "none",
            "zoning_types": _zoning_types(root, source_id, kept),
        },
    )


def load_one(source_id: str, root: Optional[Path] = None, method: Optional[str] = None) -> Plan:
    """Load one MSD floor, rotated to its own dominant wall direction.

    `method` picks the partition path and resolves from `partition.DEFAULT_METHOD`
    when omitted, which since the 2026-09-18 ruling is the rectified path. It was
    the grid between the 2026-09-14 ruling and that one. The argument
    exists so the vector path stays reachable by name for the partition
    comparison, and so a caller that cares can pin a
    path rather than inherit whichever one the default happens to be.
    """
    root, out_graph, rotated, categories, kept, angle, residual = _aligned(source_id, root)
    # The access graph before the partition, indexed alongside `rotated`, which
    # is the only point at which the source's own statement of what connects to
    # what and the rooms it connects are both in hand. The partition is then
    # told which pairs are connected and works out for itself which of them its
    # closing cannot reach; the loader does not own that threshold and does not
    # repeat it. Every remaining index in this function is post-partition.
    drawn = {node: index for index, node in enumerate(kept)}
    bridges = partition.bridge_spans(
        rotated, [(edge.a, edge.b) for edge in _access_edges(out_graph, drawn)]
    )
    rings, survivors, _, report = partition.partition_rings(rotated, method=method, bridges=bridges)

    # The partition can drop a room smaller than one cell, so every index that
    # leaves this function is an index into the surviving list, not the node list.
    position = {old: new for new, old in enumerate(survivors)}
    index_of = {kept[old]: new for old, new in position.items()}
    access = _access_edges(out_graph, index_of)
    notes: dict[str, object] = {
        "rotation_radians": -angle,
        "off_axis_before_partition": residual,
        "zoning_types": _zoning_types(root, source_id, [kept[old] for old in survivors]),
    }
    notes.update(_partition_notes(report))
    return Plan(
        source="msd",
        source_id=source_id,
        rooms=[Room(rings[new], categories[old], old) for old, new in position.items()],
        units_per_meter=1.0,
        # MSD's coordinates are already metres, but what the plan can tell apart
        # is decided by the partition and not by the coordinate unit, so the
        # partition states it. It used to be reconstructed here from the grid's
        # step with the grid's default as a fallback, which read as a fact about
        # the dataset when it is a fact about the path that ran, and which had no
        # honest answer at all for a path that resolves nothing onto a grid.
        source_pixel_m=report.resolution,
        # The floor the partition assigned cells within. MSD stores no outline,
        # so this is the one the partition established at its own close length
        # and the only record of where this floor plan ends; without it the
        # adapter's seam repair reads a slot in the building as a wall.
        floor_extent=report.extent,
        access=access,
        entry=_entry_point(rings, access),
        off_axis_fraction=geo.off_axis_fraction(rings),
        notes=notes,
    )


class FloorInParts(ValueError):
    """A floor whose rooms close into more than one part, refused as a floor.

    The parts are the wings of one building, reached from each other only
    through a core the dataset did not draw, and the whole of the floor is what
    a floor plan is. Grading the largest part under the floor's identity states
    a compactness, a proportion and a room count for a building that does not
    exist, and it does so silently, because the rooms of every other wing were
    dropped by the partition before anything counted them. Refusing is the
    honest outcome and this refusal is where it is recorded.

    It is the floor reading that is refused and not the load. An apartment lies
    wholly within one part, so the parts that stop these rooms being one plan
    leave every apartment whole, and `planaudit.units` grades them from the same
    load. What the parts break is the claim that these rooms are one plan, and
    only the floor path makes that claim.

    `origin` is the prefix a run files the outcome under, so this refusal reports
    as a row of its own rather than among the loader failures, which are defects
    where this is a ruling. The measurements ride on the exception rather than in
    the message, for the reason the adapter's ingest refusal gives: a message naming
    the floor makes every floor read as its own distinct cause.
    """

    origin = "floor"

    def __init__(self, source_id: str, parts: int, rooms_outside: int, m2_outside: float) -> None:
        self.source_id = source_id
        self.parts = parts
        self.rooms_outside = rooms_outside
        self.m2_outside = m2_outside
        super().__init__(
            f"rooms close into {parts} parts,"
            f" {rooms_outside} room(s) and {m2_outside:.3f} m2 outside the largest"
        )


def load_floor(source_id: str, root: Optional[Path] = None, method: Optional[str] = None) -> Plan:
    """One MSD floor as a floor, or a refusal saying it is not one region.

    `load_one` is the load and this is the floor reading of it. The two are
    separate entry points rather than one with a flag, because the units path
    takes the same load and must not be refused by it.
    """
    plan = load_one(source_id, root, method=method)
    parts = int(plan.notes.get("partition_floor_parts", 1) or 1)
    if parts <= 1:
        return plan
    where = list(plan.notes.get("partition_room_parts", ()))
    outside = [index for index in range(len(plan.rooms)) if index < len(where) and where[index] != 0]
    raise FloorInParts(
        source_id,
        parts,
        len(outside),
        sum(abs(geo.signed_area(plan.rooms[index].ring)) for index in outside),
    )


def _zoning_types(root: Path, source_id: str, kept: list[object]) -> list[Optional[int]]:
    """Per-room zoning_type from graph_in, aligned to the kept graph_out nodes.

    graph_in and graph_out share node ids for the same floor, so this is a plain
    lookup. A missing node yields None rather than a guess.
    """
    path = root / "graph_in" / f"{source_id}.pickle"
    if not path.exists():
        return [None] * len(kept)
    try:
        in_graph = _read_graph(path)
    except Exception:
        return [None] * len(kept)
    out: list[Optional[int]] = []
    for node in kept:
        if node in in_graph.nodes:
            value = in_graph.nodes[node].get("zoning_type")
            out.append(None if value is None else int(value))
        else:
            out.append(None)
    return out


def iter_plans(
    limit: Optional[int] = None, root: Optional[Path] = None, method: Optional[str] = None
) -> Iterator[Plan]:
    """Yield MSD plans in sorted id order, skipping ones that will not load.

    A load failure here is an ingest problem and the caller counts it, so the
    exception is re-raised rather than swallowed; `run_d1` catches it per plan.
    """
    ids = list_ids(root)
    if limit is not None:
        ids = ids[:limit]
    for source_id in ids:
        yield load_one(source_id, root, method)
