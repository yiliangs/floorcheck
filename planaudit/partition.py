"""Turn a set of room interiors into an exact partition of the floor.

Why this exists. The solver's wall model requires every interior edge to carry
two adjacent rooms. A dataset that stores room INTERIORS rather than a partition
violates that on every edge: MSD's two faces of a shared wall sit about 0.16 m
apart, so the union of a floor's rooms comes back as one disjoint part per room
and the build fails on its first interior edge with "carries a single adjacent
room after construction".

Three ways out live here and any of them is selectable per call.

The grid path is the default, ruled so on 2026-09-14. It resolves the floor onto
a grid, assigns every cell inside the floor to its nearest room, and rebuilds
each room from its cells. That is a partition by construction, which is why it
builds where the vector path does not, but every ring comes back grid aligned, so
a room's area moves by up to one cell ring and every width statistic quantises to
the step.

The vector path is selectable by name and is the better partition on its own
terms. It measures the wall gap on the floor itself, grows every room outward by
half that gap with mitre joins so the two faces of a wall land on its centreline,
hands any region two grown rooms both claim to the room whose original interior
is nearer, fills the leftover residue where three walls meet at slightly
different offsets, and reduces every coordinate onto one precision grid so shared
edges are bit identical. Room geometry survives: a ring keeps its own corners and
its own angles, and a room's area moves only by the half wall it gains on each
side. The solver still refuses nearly all of it, because three rooms can each
claim the same stretch of wall where three walls meet, and until that is fixed
leading with it spends a doomed round trip on almost every floor. The ruling and
the measurement behind it are recorded with the partition rulings.

Collinear points are dropped only where the vertex is collinear in EVERY ring
that carries it, and only where it is collinear exactly. A vertex where three
rooms meet is a real corner for at least one of them, and dropping it from the
one ring that sees it as flat reopens the gap the snap just closed. Exact rather
than tolerant, because a tolerant drop displaces the edge by the tolerance and
the neighbour's segment stops lying on it.

The rectified path is the third, and it exists because the grid does three jobs
at once and only the first of them needs a grid. Per cell it partitions, it
filters noise, and it squares the plan up, and a reader of its output cannot tell
which of the three moved a wall. Two of its per-cell rules bend a straight source
wall into a one-cell step: a face tilted twelve to sixteen centimetres over its
run crosses a column of cell centres, so containment flips between rows, and a
gap cell equidistant from labelled cells on both sides is assigned by the
distance transform's tie break, which differs row to row at a room corner. Four
incremental repairs to the grid path were tried on 2026-09-18 and every one of
them moved the dents rather than removing them, because every one of them is
still a rule about a cell.

So this path does the same three jobs per EDGE. It takes the vector partition,
which is already exact and faithful and which the solver refuses only over its
wall matching, and nodes it: the merged boundary is snapped once, polygonised,
and every face is given to the room whose interior contains it, with unclaimed
faces and slivers going to the neighbour they share the most boundary with. That
leaves every interior edge carried by exactly two rooms, which is the contract
the vector path breaks where three walls meet. Then it squares up, moving each
near-axis edge onto one line with both of its rooms moving together, and filters
features below the widest wall the corpus draws, 0.40 m, which is a measurement of
this path's own residue and is stated at FEATURE_M. Both of those act through a
map from vertex to vertex applied to every ring at once, which is what keeps a
shared boundary shared. Last, once that pair has settled, it steps the chamfers
the weld leaves: a staircase collapses onto the centres of its classes, and those
centres sit off both of the walls the staircase joined, so what is left is one
edge on neither axis between two edges that are on one. That edge is closed with
two segments that are on axes, through a map from edge to corner decided once for
the whole floor for the same reason the other two maps are. The per-room form of
the same repair does not: applying precision reduction to each room polygon on
its own left gaps up to 2.7 m on every floor it was measured on, on 2026-09-18.

Squaring up and filtering run as a pair, filter first, and this departs from the
order the work item set out. Collapsing a short feature moves the vertices at its
ends, so a run that was vertical before the collapse is left tilted by half the
feature afterwards, and filtering after squaring up reintroduces exactly what
squaring up removed. The pair runs twice and ends on the squaring, which is the
step that leaves the axes true.

Near-axis is decided on a displacement and not on an angle, because the defect is
a displacement: the tilts this path exists to remove run twelve to sixteen
centimetres over two to four metres, which is under half a degree of the two
degree threshold an earlier attempt used. A bare displacement bound would catch a
genuinely slanted wall drawn in short segments, each of which is near-axis on its
own while the chain of them is not, so the bound is also capped as a slope. What
survives both is left alone and reaches the ladder's orthogonality precondition
as it does today.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from planaudit import geometry as geo

# Which path `partition_rings` takes when the caller does not say. The rectified
# path leads: it is the exact vector partition noded, squared per shared edge and
# denoised, so it meets the solver's wall contract without bending a straight
# source wall onto a cell. The grid and the vector paths both stay selectable by
# name, the grid because every MSD number this repository measured before run 9
# was measured on it. Ruled on 2026-09-18, superseding the grid ruling of
# 2026-09-14, see the partition ruling of that date.
DEFAULT_METHOD = "rectified"

# The grid the floor is resolved onto. Chosen by sweeping the first 20 MSD train
# floors through the solver: every step from 0.32 to 0.50 m builds 19 or 20 of
# them, 0.60 builds 18, and below 0.32 the build rate collapses (0.30 builds 5,
# 0.25 builds 1, 0.10 builds 1). What a finer grid buys is not fidelity: the
# mean per-room area error is about 13.5 percent across the whole plateau,
# because it is dominated by the wall area the partition has to hand to some
# room, not by the grid. What it costs is ring complexity, and the solver's
# layout conditions fault on rings with too many short jogs. 0.38 sits inside
# the plateau rather than at either edge.
DEFAULT_STEP_M = 0.38

# How wide a gap between two room interiors still counts as one wall rather than
# as the edge of the floor. MSD's two faces of a shared wall are about 0.16 m
# apart, so the floor outline has to be recovered by closing gaps at least that
# wide before anything can be said about which cell belongs to which room.
DEFAULT_CLOSE_M = 0.20

# The wall gap the vector path falls back on when a floor has no adjacent pair to
# measure, which happens only for a single room floor. Every floor with more than
# one room measures its own.
DEFAULT_GAP_M = 0.16

# Two rooms whose interiors come within this distance face each other across one
# wall. Wider than any MSD interior wall and narrower than any room.
ADJACENT_M = 0.5

# What it takes for two wall faces to count as facing each other: this close to
# parallel, and overlapping by at least this far along their shared direction. A
# pair of rooms that only comes close at a corner is not a wall, and letting it
# into the sample is what drags the median under the wall it should describe.
FACE_ANGLE_DEG = 5.0
FACE_RUN_M = 0.3

# The precision grid every room's coordinates are reduced onto, which is what
# makes a shared edge bit identical on both sides. One micron: five orders of
# magnitude finer than the fallback's cell and well inside the solver's join
# tolerance, so it costs no measurable area or width.
SNAP_M = 1e-6

# A hole in the union of the grown rooms is floor with no room, which under the
# solver's model is an edge with one adjacent room. Anything smaller than this is
# a wall junction and the residue fill routes it through geo.hand_over, which
# gives it to the first candidate in bounding order it leaves simply connected.
# Anything larger is a disagreement between the measured gap and the floor, and
# is reported rather than papered over.
HOLE_LIMIT_M2 = 0.5

# How long a stub carried by three rooms can be and still be a place where three
# rooms meet at a point rather than a real wall. Shorter than any door.
WELD_M = 0.05

# The precision the rectified path snaps the MERGED boundary onto before it
# polygonises. It is not SNAP_M: this snap has to close the six to thirty-nine
# centimetre mismatches where three wall centrelines cross in three places, which
# is why the exact-tolerance helpers above cannot do it, and one micron closes
# nothing. It is applied once, to the boundary of the whole floor, because the
# per-polygon form of the same call repairs each room against itself and opens
# gaps between rooms that were touching.
PRECISION_M = 0.02

# When an edge counts as meant to be vertical or horizontal. Two bounds, and an
# edge has to pass both. The absolute one is one cell, because one cell is
# exactly the displacement the grid path erases: a wall the grid would have
# rendered straight is one this path may straighten, and a wall that moves
# further than a cell over its run is one the grid could not have hidden either.
# It is a cell rather than half a cell because a tilt of 0.21 m over 2.9 m was
# measured on floor 2397 on 2026-09-18 and half a cell misses it.
#
# The slope is the second bound, and what it is for is narrower than it looks. It
# is not the protection for a real slanted wall: the absolute bound is, because a
# wall that moves more than a cell over its run is never touched however shallow
# its slope. What the slope stops is a chain of short segments describing a curve
# or a diagonal, each link of which is within a cell of an axis while the chain
# is not, being straightened one link at a time.
#
# It is 0.5 and not tighter because of what has to pass it. Collapsing a chamfer
# leaves the two walls it joined displaced by half its length, up to 0.19 m, and
# those walls then have to qualify or the chamfer's removal has bought nothing.
# The shortest wall that can carry a chamfer is one cell, so the slope has to
# admit 0.19 over 0.38. Tighter values were measured on 2026-09-18: at 0.10 the
# 46 reviewer units kept 17 rooms whose boundary is not orthogonal, and every one
# of those was residue this module made rather than geometry MSD stated.
#
# What that costs is worth stating rather than discovering. A wall up to about
# seven degrees off an axis over a three metre run is straightened here, and the
# MSD census counts anything past two degrees as non-orthogonal. On the 46
# reviewer units the cost is one segment of 0.84 m on one unit: after the
# loader's rotation, 45 of the 46 carry no off-axis wall length at all.
SQUARE_M = DEFAULT_STEP_M
SQUARE_SLOPE = 0.5

# The smallest feature the rectified path keeps on an axis. It is a measurement
# of this partition's own residue and no longer a fraction of the cell, and the
# measurement is what raised it from half a cell, 0.19 m, to 0.40 m on
# 2026-09-19.
#
# On the 84 pool MSD units the instrument classifies every boundary edge under
# 0.40 m as either matching an edge of the source rings or introduced here. At
# half a cell, 133 of the 148 such edges are introduced rather than drawn, and 98
# of those introduced edges lie between 0.20 and 0.40 m, which is the band this
# constant did not reach. They are not the edges of a pocket. They sit on walls,
# where a wall's faced stretches meet its unfaced ones: a room is cut back to half
# the gap where something faces it and grows by the full reach where nothing does,
# so every point at which facing starts or stops leaves a step of up to 0.25 m,
# and the residue fill ends each strip it hands over with a cut perpendicular to
# the wall. Three attempts to withhold them at a pocket stage reached only the
# compact alcoves and left the population where it was.
#
# The value is the widest wall the corpus draws. Over 16,228 facing pairs from 200
# floors, measured on 2026-09-18, the run-weighted distribution of the gap between
# facing rooms decays to 0.40 m and 95.3 per cent of the run lies below it, so an
# axis feature shorter than 0.40 m is a wall-scale artefact more often than
# something the floor states. The cost is the 14 drawn features per 84 units that
# lie in the same band and are welded away with the residue. That cost is accepted
# and stated rather than avoided, because no criterion tried so far separates the
# two populations by shape.
FEATURE_M = 0.40

# The same, for a segment that is on neither axis. One cell, which is wider, and
# the reason is that the grid path states every wall on an axis: a segment
# shorter than one cell that is on neither is below the length at which the
# source said anything about direction at all. In practice these are the chamfers
# the vector path's mitre joins leave at a corner, 0.12 to 0.33 m long at 16 to
# 35 degrees, and floor 3596 carries seven of them while its raw rings carry
# none. Collapsing one leaves its two neighbours slightly off their axes, which
# is why the squaring runs after the filter and not before it.
CHAMFER_M = DEFAULT_STEP_M

# How many times the filter and the squaring may run as a pair before the loop
# gives up. It is a ceiling and not a count: the pair runs until a round changes
# nothing, because each round can only expose work for the next one. Collapsing a
# chamfer leaves its two walls off their axes for the squaring to find, and
# squaring a wall can shorten a return below the filter's limit. A fixed number
# of rounds stops in the middle of that and leaves exactly the residue the path
# exists to remove.
RECTIFY_ROUNDS = 8

# A face of the noded partition that no room claims, or that is too thin to be a
# room's own floor, goes to the neighbour it shares the most boundary with. These
# are what "too thin" means: an area, and an area over half its perimeter, which
# is the width of the rectangle it would be.
SLIVER_M2 = 0.05
SLIVER_WIDTH_M = 0.05

# How many rounds the sliver merge runs. A sliver can only move to a face that
# already belongs to a room, so a chain of slivers resolves one link per round.
MERGE_ROUNDS = 6


@dataclass
class WallGap:
    """The distribution of nearest-edge distances between facing rooms."""

    median: float
    p25: float
    p75: float
    pairs: int

    @property
    def spread(self) -> float:
        return self.p75 - self.p25


@dataclass
class Uncarried:
    """Floor a one-ring-per-room write-back could not put in a room's ring.

    Two faults, kept apart because they surface as opposite findings and because
    an area without its count, or a count without its area, cannot be read. A
    wrapped room is an interior the outer ring goes on to claim as well, so its
    floor is claimed twice and the solver refuses the plan naming one of its
    edges. A split room is a second piece of the same room that no ring covers,
    so its floor is covered by nobody and the seam repair finds it later as a
    gap, far from the step that made it.
    """

    wrapping_rooms: int = 0
    wrapped_m2: float = 0.0
    split_rooms: int = 0
    split_m2: float = 0.0

    def __add__(self, other: "Uncarried") -> "Uncarried":
        return Uncarried(
            self.wrapping_rooms + other.wrapping_rooms,
            self.wrapped_m2 + other.wrapped_m2,
            self.split_rooms + other.split_rooms,
            self.split_m2 + other.split_m2,
        )


@dataclass
class PartitionReport:
    method: str
    rooms_in: int
    rooms_out: int
    area_before: float
    area_after: float
    mean_room_area_error: float
    step: Optional[float] = None
    close: Optional[float] = None
    wall_gap: Optional[float] = None
    wall_gap_spread: Optional[float] = None
    gap_pairs: int = 0
    shared_exact: Optional[float] = None
    holes_filled: int = 0
    holes_over_limit: int = 0
    # What a one-ring-per-room model could not carry, counted and measured. A
    # count alone cannot be read: one wrapping room costs a closet or a wing.
    wrapping_rooms: int = 0
    wrapped_m2: float = 0.0
    split_rooms: int = 0
    split_m2: float = 0.0
    # Parts of the floor no room could be given without that room enclosing
    # another. Zero on a floor the partition resolved cleanly.
    residue_refused: int = 0
    residue_refused_m2: float = 0.0
    # The smallest length the rings this partition returns can tell apart, which
    # is what a plan's `source_pixel_m` is. Every path fills it, because every
    # path decides it: the grid can tell apart one cell, and the rectified path
    # one snap precision. Before this field the loader reconstructed it from
    # `step` with the grid's default as a fallback, which read as a property of
    # the dataset when it is a property of the method that ran.
    resolution: Optional[float] = None
    # What the rectified path snapped, squared and filtered at. None on the other
    # two paths, which do none of the three.
    precision: Optional[float] = None
    square: Optional[float] = None
    feature: Optional[float] = None
    # The noding's own evidence, which is the whole reason that path exists. An
    # interior edge carried by more than two rooms is the fault the solver refuses
    # the vector path over, so a partition claiming to have fixed it has to say
    # how many are left, and a coverage that is not valid is a gap or an overlap
    # between two rooms that the room areas alone would not show.
    edges_over_two: int = 0
    coverage_valid: Optional[bool] = None
    faces_merged: int = 0
    faces_unplaced: int = 0
    # The floor outline this partition assigned cells within, as one ring per
    # closed part of it, largest first. It is not a statistic like the rest of
    # this record, and it is here because it is the one thing the partition
    # decided that the room set it returns cannot express: the rooms fill the
    # extent, so recovering it from them afterwards gives something larger.
    # Downstream it is what stops a later repair inventing floor the partition
    # ruled was not floor.
    #
    # A floor whose rooms close into more than one part is a real object and not
    # a fault of the closing: the wings of one building, reached from each other
    # only through a core the dataset did not draw. Every part is carried,
    # because a partition that named one of them the floor would be grading one
    # wing under the whole building's identity. `room_parts` says which part
    # each returned room lies in, indexed alongside the returned rings, so a
    # reader can group the rooms by part without repeating the containment test.
    floor_parts: int = 1
    part_rings: tuple[geo.Ring, ...] = ()
    room_parts: tuple[int, ...] = ()
    # Asserted connections the closing could not reach and this partition
    # bridged, from `bridge_spans`. Zero where the caller asserted nothing or
    # where the geometry already closed every connection it states, which is
    # every floor but the ones carrying a threshold wider than the reach. It is
    # reported because a part count read without it cannot be compared with one
    # taken before the assertions were carried.
    bridges_used: int = 0

    @property
    def extent(self) -> Optional[geo.Ring]:
        """The largest part's ring, which is what a single-extent reader wants.

        Derived rather than stored, because a stored copy would be a second
        answer to a question `part_rings` already answers and the two could
        disagree.
        """
        return self.part_rings[0] if self.part_rings else None

    @property
    def rooms_outside_extent(self) -> int:
        """Returned rooms lying in a part other than the largest."""
        return sum(1 for index in self.room_parts if index != 0)

    @property
    def area_drift(self) -> float:
        return 0.0 if self.area_before == 0.0 else abs(self.area_after - self.area_before) / self.area_before


def partition_rings(
    rings: list[geo.Ring],
    method: Optional[str] = None,
    step: Optional[float] = None,
    close: Optional[float] = None,
    gap: Optional[float] = None,
    bridges=(),
) -> tuple[list[geo.Ring], list[int], geo.Ring, PartitionReport]:
    """Rebuild `rings` as a partition. Returns rings, kept indices, boundary, report.

    `method` resolves from `DEFAULT_METHOD`, and the per-path parameters from
    their module constants, at CALL time rather than at definition time, so those
    constants stay the single place each value lives. Binding them as argument
    defaults made them look tunable while every caller that omitted the argument
    silently kept the value frozen at import.

    `bridges` reach the vector path and the rectified path that is built on it,
    and not the grid path. The grid is the comparison path of section 5.4, it
    closes at a different distance from the one `bridge_spans` selects against,
    and its corpus numbers are cited as a measurement of what closing alone
    finds. Stated here rather than dropped in silence.
    """
    method = DEFAULT_METHOD if method is None else method
    if method == "vector":
        return partition_vector(rings, gap=gap, bridges=bridges)
    if method == "grid":
        return partition_grid(rings, step=step, close=close)
    if method == "rectified":
        return partition_rectified(rings, gap=gap, bridges=bridges)
    raise ValueError(f"unknown partition method {method!r}")


def measure_wall_gap(rings: list[geo.Ring], adjacent: Optional[float] = None) -> WallGap:
    """Median and quartiles of the nearest-edge distance between facing rooms."""
    from shapely.geometry import Polygon

    adjacent = ADJACENT_M if adjacent is None else adjacent
    return _gap_of([geo.repaired(Polygon(r)) for r in rings], adjacent)


def bridge_spans(
    rings: list[geo.Ring], pairs, reach: Optional[float] = None
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """The shortest segment across each asserted connection the closing cannot reach.

    `pairs` are index pairs a caller asserts are connected, which for a dataset
    carrying an access graph is a door the source drew. The closing joins two
    rooms when each can grow `reach` toward the other and meet, so a gap wider
    than twice the reach leaves them in separate parts however plainly the
    source states the door. Over MSD those thresholds are 0.50 to 0.68 m wide,
    against 0.14 to 0.30 m for an ordinary interior gap on the same floors, and
    they are nearly all a balcony's sill: the balcony closes into a part of its
    own, the apartment's rooms then span two parts, and the unit view keeps the
    larger piece and drops the balcony.

    Widening the reach is not the repair. Three rounds of width criteria failed
    to tell a threshold from the gap between two wings of a building, because
    the two populations overlap in width. What separates them is not width at
    all: the source states the doorway and says nothing about the wing. So the
    assertion is carried rather than inferred, and only where the geometry
    cannot already reach, which leaves every floor whose access graph says
    nothing new exactly as it was.

    Where the rule lives is a division of labour. Which rooms are connected is
    the dataset's to state; how far the closing reaches and what segment bridges
    it is this module's, because `reach` is this module's constant and the
    segment has to be the one the closing will union in.

    The segment crosses the middle of the threshold and not the shortest way
    across it. Two rooms facing over a wall are nearest at every point of the
    run, so the shortest pair is whichever vertex the distance routine reaches
    first, which is an end of the run; a strip centred there hangs half its
    width off the end of the wall, where it is floor outside the building rather
    than a doorway. The middle is found as the centroid of the region within one
    gap of both rooms, which is the run itself widened evenly at both ends, so
    it is the midpoint of the facing stretch however the two rooms are oriented
    and whether they face over all of their walls or part of them.
    """
    from shapely.geometry import Polygon
    from shapely.ops import nearest_points

    reach = ADJACENT_M / 2.0 if reach is None else reach
    polys = [geo.repaired(Polygon(r)) for r in rings]
    spans: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for first, second in pairs:
        if first == second or not (0 <= first < len(polys) and 0 <= second < len(polys)):
            continue
        one, other = polys[first], polys[second]
        if one.is_empty or other.is_empty:
            continue
        gap = one.distance(other)
        if gap <= 2.0 * reach:
            continue
        between = geo.largest_part(one.buffer(gap).intersection(other.buffer(gap)))
        near_one, near_other = nearest_points(one, other)
        if not between.is_empty and between.area > 0.0:
            middle = between.centroid
            centred_one = nearest_points(one, middle)[0]
            centred_other = nearest_points(other, middle)[0]
            if centred_one.distance(centred_other) > 0.0:
                near_one, near_other = centred_one, centred_other
        spans.append(((float(near_one.x), float(near_one.y)), (float(near_other.x), float(near_other.y))))
    return spans


def partition_vector(
    rings: list[geo.Ring],
    gap: Optional[float] = None,
    snap: Optional[float] = None,
    hole_limit: Optional[float] = None,
    bridges=(),
) -> tuple[list[geo.Ring], list[int], geo.Ring, PartitionReport]:
    """Rebuild `rings` as a partition without leaving the rooms' own geometry.

    The grown rooms are clipped back to the closed floor outline before anything
    else. Without the clip a room on the building edge also grows outward by half
    a wall, which is area no room has and which moves every exterior room's area
    by more than the interior wall it was supposed to gain. With it, a room's
    exterior faces stay exactly where the dataset put them and the only area a
    room gains is the half wall it shares with a neighbour.

    Each part of the closed floor is partitioned on its own, over the rooms whose
    interior point lies in it, and the results are concatenated. Two rooms in
    different parts share no boundary, so nothing is lost by resolving the
    overlaps, the residue and the noding part by part, and a room is clipped
    against the part it is in rather than against a part it is nowhere near. On a
    floor that closes into one part this is the same computation in the same
    order over the same rooms, so nothing about such a floor moves.

    `bridges` from `bridge_spans` join rooms the source states are connected
    across a gap this reach cannot close. Nothing below them changes: the
    bridged rooms simply land in one part, and the threshold strip the bridge
    opens is floor no room claims, which the residue fill hands whole to one of
    them under the same rule as any other residue. The two rooms then share an
    edge, which is what lets a reader tile the apartment as one piece.
    """
    import shapely
    from shapely.geometry import Polygon

    snap = SNAP_M if snap is None else snap
    hole_limit = HOLE_LIMIT_M2 if hole_limit is None else hole_limit
    polys = [_exact(Polygon(r), snap) for r in rings]
    wall = _gap_of(polys, ADJACENT_M)
    reach = ADJACENT_M / 2.0
    parts = [_exact(part, snap) for part in _filled_parts(polys, reach, bridges)]
    home = _home_parts(polys, parts)

    out_rings: list[geo.Ring] = []
    kept: list[int] = []
    room_parts: list[int] = []
    part_rings: list[geo.Ring] = []
    filled = over = refused = 0
    refused_m2 = 0.0
    uncarried = Uncarried()
    for position, part in enumerate(parts):
        members = [index for index, where in enumerate(home) if where == position]
        if not members:
            continue
        # The grown room goes through `_exact` for the reason `_exact` states,
        # and for the same reason the half-wall operand in `_resolve_overlaps`
        # does: a mitre buffer is off the precision grid and can come back self
        # touching at a reflex corner, and GEOS is robust at a fixed grid size
        # only when both operands are already on it. Clipped against one whole
        # floor the raw buffer survived; clipped against a part it threw
        # "Ring edge missing" on floor 3998.
        offsets = [
            geo.largest_part(
                shapely.intersection(
                    _exact(polys[index].buffer(reach, join_style="mitre"), snap), part, grid_size=snap
                )
            )
            for index in members
        ]
        pieces = _resolve_overlaps(offsets, [polys[index] for index in members], snap, gap)
        pieces, part_filled, part_over, part_refused, part_refused_m2 = _fill_residue(
            pieces, part, hole_limit, snap
        )
        part_out, part_kept, part_uncarried = _rings_of(pieces, snap)
        filled += part_filled
        over += part_over
        refused += part_refused
        refused_m2 += part_refused_m2
        uncarried = uncarried + part_uncarried
        ring = _part_ring(part)
        if ring is not None:
            part_rings.append(ring)
        out_rings.extend(part_out)
        kept.extend(members[index] for index in part_kept)
        room_parts.extend([len(part_rings) - 1] * len(part_out))
    boundary = geo.tile(out_rings).outline
    report = PartitionReport(
        method="vector",
        rooms_in=len(polys),
        rooms_out=len(out_rings),
        area_before=sum(p.area for p in polys),
        area_after=sum(abs(geo.signed_area(r)) for r in out_rings),
        mean_room_area_error=_mean_area_error(out_rings, polys, kept),
        wall_gap=wall.median if gap is None else gap,
        wall_gap_spread=wall.spread,
        gap_pairs=wall.pairs,
        shared_exact=_shared_exact(out_rings, boundary),
        holes_filled=filled,
        holes_over_limit=over,
        wrapping_rooms=uncarried.wrapping_rooms,
        wrapped_m2=uncarried.wrapped_m2,
        split_rooms=uncarried.split_rooms,
        split_m2=uncarried.split_m2,
        residue_refused=refused,
        residue_refused_m2=refused_m2,
        # The vector path reduces coordinates onto one micron, but a micron is
        # not what its rings can tell apart: they are the dataset's own corners,
        # and what separates two of them is whatever the dataset resolved. It has
        # never stated that, so this path carries the grid's cell, which is the
        # answer the loader substituted before this field existed.
        resolution=DEFAULT_STEP_M,
        floor_parts=len(part_rings),
        part_rings=tuple(part_rings),
        room_parts=tuple(room_parts),
        bridges_used=len(bridges),
    )
    return out_rings, kept, boundary, report


def partition_grid(
    rings: list[geo.Ring], step: Optional[float] = None, close: Optional[float] = None
) -> tuple[list[geo.Ring], list[int], geo.Ring, PartitionReport]:
    """Rebuild `rings` as a partition by resolving the floor onto a grid.

    Each part of the closed floor is resolved onto its own raster, over the rooms
    whose interior point lies in it. A floor that closes into one part is
    rasterised exactly as it was before, over the same cells in the same order.
    The grid closes at a shorter distance than the vector path, so it finds more
    floors in parts: 987 of MSD's 4,167 train floors at 0.20 m against 294 at
    0.25 m, which is why the two paths could disagree about which part was the
    floor while each kept only one.
    """
    import numpy as np
    import shapely
    from shapely.geometry import Polygon

    step = DEFAULT_STEP_M if step is None else step
    close = DEFAULT_CLOSE_M if close is None else close
    polys = [geo.repaired(Polygon(r)) for r in rings]
    parts = _filled_parts(polys, close)
    home = _home_parts(polys, parts)

    out_rings: list[geo.Ring] = []
    kept: list[int] = []
    room_parts: list[int] = []
    part_rings: list[geo.Ring] = []
    uncarried = Uncarried()
    for position, part in enumerate(parts):
        members = [index for index, where in enumerate(home) if where == position]
        if not members:
            continue
        # One raster per part, over that part's own rooms. A shared lattice over
        # the whole floor would let the nearest-room fill hand a cell in one part
        # the label of a room in another, across a gap no floor spans, and the
        # room would come back in two pieces on opposite sides of the building.
        x0, y0, x1, y1 = part.bounds
        nx = max(int((x1 - x0) / step) + 1, 1)
        ny = max(int((y1 - y0) / step) + 1, 1)
        xs = x0 + (np.arange(nx) + 0.5) * step
        ys = y0 + (np.arange(ny) + 0.5) * step
        grid_x, grid_y = np.meshgrid(xs, ys)

        inside = shapely.contains_xy(part, grid_x, grid_y)
        label = np.full(grid_x.shape, -1, dtype=np.int32)
        for local, index in enumerate(members):
            hit = shapely.contains_xy(polys[index], grid_x, grid_y) & (label < 0)
            label[hit] = local
        label = _fill_nearest(label, inside)

        ring = _part_ring(part)
        if ring is not None:
            part_rings.append(ring)
        for local, index in enumerate(members):
            room_ring, lost = _mask_to_ring(label == local, x0, y0, step)
            uncarried = uncarried + lost
            if room_ring is not None and len(room_ring) >= 3:
                out_rings.append(geo.orient_ccw(room_ring))
                kept.append(index)
                room_parts.append(len(part_rings) - 1)
    boundary = geo.tile(out_rings).outline
    return out_rings, kept, boundary, PartitionReport(
        method="grid",
        rooms_in=len(polys),
        rooms_out=len(out_rings),
        area_before=sum(p.area for p in polys),
        area_after=sum(abs(geo.signed_area(r)) for r in out_rings),
        mean_room_area_error=_mean_area_error(out_rings, polys, kept),
        step=step,
        close=close,
        wrapping_rooms=uncarried.wrapping_rooms,
        wrapped_m2=uncarried.wrapped_m2,
        split_rooms=uncarried.split_rooms,
        split_m2=uncarried.split_m2,
        resolution=step,
        floor_parts=len(part_rings),
        part_rings=tuple(part_rings),
        room_parts=tuple(room_parts),
    )


def partition_rectified(
    rings: list[geo.Ring],
    gap: Optional[float] = None,
    precision: Optional[float] = None,
    square: Optional[float] = None,
    feature: Optional[float] = None,
    bridges=(),
) -> tuple[list[geo.Ring], list[int], geo.Ring, PartitionReport]:
    """Rebuild `rings` as a noded, squared partition without resolving them onto a grid.

    Five steps, and each one of them is the whole of one of the three jobs the
    grid does per cell. The vector partition supplies the exact partition. Noding
    the merged boundary supplies the contract the solver reads, that every
    interior edge carries exactly two rooms. Squaring up supplies the
    orthogonality, and the feature filter the noise rejection. The module
    docstring says why the last two run as a pair and in which order. The fifth
    runs once after that pair has settled, and it is the only one of the five the
    grid never had to do: the weld leaves a chamfer where it collapses a
    staircase, and stepping that chamfer is the orthogonality the squaring cannot
    reach, because an edge that far off an axis is one the squaring is right to
    refuse.
    """
    import shapely
    from shapely.geometry import Polygon

    precision = PRECISION_M if precision is None else precision
    square = SQUARE_M if square is None else square
    feature = FEATURE_M if feature is None else feature

    base_rings, base_kept, _, base = partition_vector(rings, gap=gap, bridges=bridges)
    polys = [geo.repaired(Polygon(r)) for r in rings]
    pieces, merged, unplaced = _faces_by_room(base_rings, precision)

    work: list[list] = []
    carried: list[int] = []
    carried_parts: list[int] = []
    uncarried = Uncarried()
    for index, piece in enumerate(pieces):
        if piece is None or piece.is_empty:
            continue
        whole = geo.repaired(piece)
        shell = geo.largest_part(whole)
        if shell.is_empty or shell.area <= 0.0:
            continue
        uncarried = uncarried + _uncarried_by(shell, geo.parts(whole))
        work.append(_dedupe([(float(x), float(y)) for x, y in shell.exterior.coords]))
        carried.append(base_kept[index])
        carried_parts.append(base.room_parts[index])

    unwelded = work
    for _ in range(RECTIFY_ROUNDS):
        before = work
        work = _square_up(_weld_features(work, feature), square)
        if work == before:
            break
    work = _prune_collinear(_step_chamfers(work, unwelded, _weld_limit(feature)))

    out_rings: list[geo.Ring] = []
    kept: list[int] = []
    room_parts: list[int] = []
    for index, part_index, points in zip(carried, carried_parts, work):
        points = _drop_loops(_dedupe(points))
        if len(points) < 3:
            continue
        if not _simple(points):
            # A ring that touches itself after the welding, which happens where a
            # collapsed feature brings two stretches of one room's boundary onto
            # the same point. It is repaired and not dropped, and not replaced by
            # the ring before the welding either. Dropping it loses a room the
            # partition did carry, and on a four-room apartment that is enough to
            # stop the floor reading as a dwelling at all, which is how two of the
            # reviewer's units disappeared on 2026-09-18 while their area was
            # correct to three decimal places. Falling back is worse than either:
            # the neighbours have already moved, so the old ring no longer lies on
            # them and the coverage opens where it used to close.
            shell = geo.largest_part(geo.repaired(Polygon(points)))
            if shell.is_empty or shell.area <= 0.0:
                continue
            points = _drop_loops(_dedupe([(float(x), float(y)) for x, y in shell.exterior.coords]))
            if len(points) < 3:
                continue
        out_rings.append(geo.orient_ccw(geo.close_ring(points)))
        kept.append(index)
        room_parts.append(part_index)
    boundary = geo.tile(out_rings).outline
    final = [geo.repaired(Polygon(r)) for r in out_rings]
    report = PartitionReport(
        method="rectified",
        rooms_in=len(polys),
        rooms_out=len(out_rings),
        area_before=sum(p.area for p in polys),
        area_after=sum(abs(geo.signed_area(r)) for r in out_rings),
        mean_room_area_error=_mean_area_error(out_rings, polys, kept),
        wall_gap=base.wall_gap,
        wall_gap_spread=base.wall_gap_spread,
        gap_pairs=base.gap_pairs,
        shared_exact=_shared_exact(out_rings, boundary),
        holes_filled=base.holes_filled,
        holes_over_limit=base.holes_over_limit,
        wrapping_rooms=uncarried.wrapping_rooms,
        wrapped_m2=uncarried.wrapped_m2,
        split_rooms=uncarried.split_rooms,
        split_m2=uncarried.split_m2,
        residue_refused=base.residue_refused,
        residue_refused_m2=base.residue_refused_m2,
        # What this path can tell apart is the grid it snapped the merged
        # boundary onto, and nothing coarser: no later step resolves a
        # coordinate, they only move vertices that are already on it. Stating the
        # grid's cell here instead would claim a quantisation this path does not
        # perform and would hand the adapter a seam tolerance nineteen times
        # wider than anything this partition can leave behind.
        resolution=precision,
        precision=precision,
        square=square,
        feature=feature,
        edges_over_two=_edges_over_two(out_rings),
        coverage_valid=bool(shapely.coverage_is_valid(final)) if final else None,
        faces_merged=merged,
        faces_unplaced=unplaced,
        floor_parts=base.floor_parts,
        part_rings=base.part_rings,
        bridges_used=base.bridges_used,
        room_parts=tuple(room_parts),
    )
    return out_rings, kept, boundary, report


def _faces_by_room(rings: list[geo.Ring], precision: float) -> tuple[list, int, int]:
    """Polygonise the merged boundary once and give every face to one room.

    The snap happens here, to the union of every ring's boundary, and not to each
    room in turn. Reducing a room polygon on its own repairs it against itself:
    two rooms that shared an edge come back with that edge in two places and the
    coverage opens a gap, measured up to 2.7 m on 2026-09-18. Reducing the merged
    boundary moves one line, and both rooms are rebuilt from it afterwards, so
    they cannot disagree about where it went.

    A face nobody claims is the residue where three wall centrelines cross in
    three places, and a face too thin to be floor is what the snap leaves when it
    pulls two nearly coincident lines together. Both go to the neighbour they
    share the most boundary with, which is the only rule that does not depend on
    an arbitrary order, and a face touching nothing goes to the nearest room so
    that no floor is dropped in silence.
    """
    import shapely
    from shapely.geometry import LineString, Polygon
    from shapely.ops import polygonize, unary_union

    rooms = [geo.repaired(Polygon(r)) for r in rings]
    # Closed explicitly, first point repeated at the end. A ring in this module
    # is a vertex list with no repeated endpoint, so handing it to LineString as
    # it stands gives an open chain missing the segment that closes it, and
    # polygonise then reads the whole floor as a few large faces rather than one
    # per room. It fails quietly: the faces are valid, every one of them lands
    # inside some room, and the rooms that own none are dropped at the end.
    boundary = unary_union([LineString(list(r) + [r[0]]) for r in rings])
    if precision > 0.0:
        boundary = shapely.set_precision(boundary, grid_size=precision)
    faces = [f for f in polygonize(shapely.node(boundary)) if not f.is_empty and f.area > 0.0]

    owned: dict[int, list] = {index: [] for index in range(len(rooms))}
    loose: list = []
    for face in faces:
        point = face.representative_point()
        owner = next((index for index, room in enumerate(rooms) if room.contains(point)), None)
        if owner is None or _is_sliver(face):
            loose.append(face)
        else:
            owned[owner].append(face)

    merged = len(loose)
    for _ in range(MERGE_ROUNDS):
        if not loose:
            break
        left = []
        for face in loose:
            best, longest = None, 0.0
            for index, taken in owned.items():
                for other in taken:
                    shared = face.boundary.intersection(other.boundary).length
                    if shared > longest:
                        best, longest = index, shared
            if best is None:
                left.append(face)
            else:
                owned[best].append(face)
        if len(left) == len(loose):
            break
        loose = left

    unplaced = len(loose)
    for face in loose:
        point = face.centroid
        owned[min(range(len(rooms)), key=lambda index: rooms[index].distance(point))].append(face)

    return [unary_union(owned[index]) if owned[index] else None for index in range(len(rooms))], merged, unplaced


def _is_sliver(face) -> bool:
    """Whether a face is too small or too thin to be a room's own floor."""
    if face.area < SLIVER_M2:
        return True
    return face.length > 0.0 and (2.0 * face.area / face.length) < SLIVER_WIDTH_M


class _Weld:
    """Vertices that have to end up sharing a coordinate, and what they share.

    A union-find over points rather than a dictionary of replacements, because
    the relation is transitive: three vertices on one wall line arrive as two
    edges and the wall is one line, not two. Every class carries the span it
    already covers so that a join which would stretch it past `limit` is refused,
    which is what stops a genuinely curved wall drawn in short segments from
    collapsing link by link into a single point.
    """

    def __init__(self, limit: Optional[float] = None) -> None:
        self.parent: dict = {}
        self.span: dict = {}
        self.limit = limit

    def find(self, point):
        self.parent.setdefault(point, point)
        root = point
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[point] != root:
            self.parent[point], point = root, self.parent[point]
        return root

    def join(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        box_a = self.span.get(ra, (a[0], a[1], a[0], a[1]))
        box_b = self.span.get(rb, (b[0], b[1], b[0], b[1]))
        box = (
            min(box_a[0], box_b[0]),
            min(box_a[1], box_b[1]),
            max(box_a[2], box_b[2]),
            max(box_a[3], box_b[3]),
        )
        if self.limit is not None and math.hypot(box[2] - box[0], box[3] - box[1]) > self.limit:
            return
        self.parent[rb] = ra
        self.span[ra] = box

    def classes(self) -> dict:
        out: dict = {}
        for point in list(self.parent):
            out.setdefault(self.find(point), []).append(point)
        return out


def _square_bound(a, b, square: float) -> float:
    """How far off an axis this edge may be and still count as meant to be on it.

    Two bounds at once, the tighter of them binding. Half a cell is what makes
    the tilts this path removes near-axis without admitting a real slant of any
    length; the slope is what keeps a chain of short segments off an axis it is
    not on. The snap precision is the floor, because an edge below it is noise
    the snap already declined to resolve and welding it costs nothing.
    """
    length = math.dist(a, b)
    return max(PRECISION_M, min(square, SQUARE_SLOPE * length))


def _square_up(rings: list[list], square: float) -> list[list]:
    """Put every near-axis edge on one line, moving both of its rooms together.

    Decided once for the whole floor and applied to every ring, which is the only
    form of this repair that keeps a shared edge shared: the edge belongs to two
    rooms and moving it in one of them opens a gap in the other. A vertex takes
    its x from the vertical edges it is on and its y from the horizontal ones,
    and a vertex on neither keeps both, so an edge that is genuinely off axis is
    carried through untouched except at the ends it shares with an edge that is
    not.

    The line a class lands on is its length-weighted mean rather than its plain
    mean. A wall faced by one long edge on one side and three short ones on the
    other has four vertices on the fragmented side and two on the other, and the
    plain mean would let the fragmentation decide where the wall is.
    """
    verticals, horizontals = _Weld(), _Weld()
    weight_x: dict = {}
    weight_y: dict = {}
    for ring in rings:
        for a, b in zip(ring, ring[1:] + ring[:1]):
            bound = _square_bound(a, b, square)
            length = math.dist(a, b)
            if abs(b[0] - a[0]) <= bound:
                verticals.join(a, b)
                weight_x[a] = weight_x.get(a, 0.0) + length
                weight_x[b] = weight_x.get(b, 0.0) + length
            if abs(b[1] - a[1]) <= bound:
                horizontals.join(a, b)
                weight_y[a] = weight_y.get(a, 0.0) + length
                weight_y[b] = weight_y.get(b, 0.0) + length
    move_x = _mean_coordinate(verticals, weight_x, 0)
    move_y = _mean_coordinate(horizontals, weight_y, 1)
    return [
        _dedupe([(move_x.get(point, point[0]), move_y.get(point, point[1])) for point in ring])
        for ring in rings
    ]


def _mean_coordinate(weld: _Weld, weight: dict, axis: int) -> dict:
    """One coordinate per vertex, the length-weighted mean of its class."""
    out: dict = {}
    for members in weld.classes().values():
        total = sum(weight.get(point, 0.0) for point in members)
        if total <= 0.0:
            value = sum(point[axis] for point in members) / len(members)
        else:
            value = sum(point[axis] * weight.get(point, 0.0) for point in members) / total
        for point in members:
            out[point] = value
    return out


def _weld_limit(feature: float, chamfer: Optional[float] = None) -> float:
    """How far the weld may move a vertex, which is how far off an axis it can leave one.

    A class of vertices collapses onto its centre and the class's own span is
    capped at this, so an edge the weld left off an axis is off it by at most
    this much: the deviation is what the weld moved the edge's two ends by and
    nothing more. Read by the weld to cap its classes and by the chamfer step to
    tell an edge the weld made from one the source drew.
    """
    return max(feature, CHAMFER_M if chamfer is None else chamfer)


def _on_axis(a, b) -> bool:
    """Whether this edge lies on one of the two axes, at the snap's own precision.

    The one test the weld and the chamfer step both read, because they are two
    halves of one judgement. An edge on an axis is a feature, and short enough it
    is this path's own residue and may be welded away. An edge on neither is a
    direction, and the weld is what puts one there.
    """
    return min(abs(b[0] - a[0]), abs(b[1] - a[1])) <= PRECISION_M


def _weld_features(rings: list[list], feature: float, chamfer: Optional[float] = None) -> list[list]:
    """Collapse every edge too short to be a feature, in every ring that carries it.

    The same map is applied to all the rings, for the same reason the squaring is:
    a feature below the limit on a shared wall is below it for both of that wall's
    rooms, and removing it from one of them is how a coverage comes apart.

    Two limits, because a short segment on an axis and a short segment on neither
    are not the same object. On an axis it is a return or a reveal, and the limit
    is the widest wall the corpus draws, because a return shorter than a wall is
    this partition's own seam more often than a feature of the floor; FEATURE_M
    carries the measurement that fixed it. On neither axis it is a chamfer left
    where two walls were mitred together, and the limit is a whole cell: the
    source states walls on axes, so below one cell an off-axis segment carries no
    direction worth keeping.
    """
    chamfer = CHAMFER_M if chamfer is None else chamfer
    weld = _Weld(limit=_weld_limit(feature, chamfer))
    for ring in rings:
        for a, b in zip(ring, ring[1:] + ring[:1]):
            length = math.dist(a, b)
            if length < (feature if _on_axis(a, b) else chamfer):
                weld.join(a, b)
    move: dict = {}
    for members in weld.classes().values():
        if len(members) < 2:
            continue
        centre = (
            sum(point[0] for point in members) / len(members),
            sum(point[1] for point in members) / len(members),
        )
        for point in members:
            move[point] = centre
    if not move:
        return rings
    return [_drop_loops(_dedupe([move.get(point, point) for point in ring])) for ring in rings]


def _step_chamfers(rings: list[list], unwelded: list[list], limit: float) -> list[list]:
    """Close an off-axis edge the weld left with two segments that are on axes.

    The weld collapses a class of vertices onto the centre of the class. Where a
    staircase of short returns runs between two walls that share neither an x nor
    a y, the centres it collapses onto sit off both of those walls, and what is
    left between them is a single edge on neither axis joining two edges that are
    on one. That edge is a chamfer the floor never drew. The squaring will not
    take it, and is right not to, because it is too far off an axis to be a wall
    that was meant to be on one; and the ladder's R4 precondition reads a segment
    as orthogonal only where its ends share an x or a y to the millimetre, so the
    chamfer is not a rougher wall but a refusal. Raising the feature floor to
    0.40 m on 2026-09-19 made five more of them across the 84 pool units and took
    the units faulting R4 from 18 to 23, which is the measurement this stage
    answers.

    So the chamfer is replaced by the two segments that reach the same two points
    along the axes, through one of the two corners of its bounding box. Both legs
    are then exactly axis aligned by construction rather than to a tolerance,
    because each takes one coordinate unchanged from each end.

    Which corner is decided by area, against the ring as it stood before the
    weld. The two corners contribute exactly opposite triangles, half the product
    of the edge's own run and rise, so this is a choice of sign and not a search:
    the weld either cut floor off this room or annexed it, and the corner goes on
    the side that gives it back. Where the area is already exact the corner is the
    one that continues the edge arriving at the chamfer, so that the step merges
    into that wall rather than adding two corners to it.

    Decided once per edge and applied to every ring that carries it, for the same
    reason the weld and the squaring are decided once for the whole floor: the
    chamfer is a shared boundary, and a corner inserted in one room and not in the
    other is how a coverage comes apart. The two rooms never disagree, because the
    triangle one of them gains is the triangle the other loses and the staircase
    took it from the same side for both, so the sign that gives one room its floor
    back gives the other room its own.

    An edge whose neighbours are themselves off an axis is left alone. A wall the
    source drew at an angle arrives as a chain of such edges, and a chain is what
    tells the two apart: this path's chamfer is a single off-axis edge between two
    on-axis ones, and floor 3453's five metre wall at six degrees is not touched
    because nothing on either side of it is on an axis either.

    A chain is not the only diagonal the source draws, so the edge has also to be
    one the weld could have made. Its smaller extent is the rise it hides, and the
    weld can only have hidden a rise inside its own span cap, which `_weld_limit`
    states. An edge off an axis by more than that was already off it before the
    weld ran. Floor 1561 is what the neighbour test costs on its own: two of its
    rooms are bounded by a single drawn diagonal several metres long with an
    orthogonal wall at each end, and the corner of that diagonal's bounding box
    took thirty seven square metres from one room and gave it to the next, with
    nine of the floor's rings ceasing to be simple on the way.

    This runs once, after the weld and the squaring have settled, and not inside
    their loop. The shorter leg of a step is the rise the staircase hid, which is
    under the feature floor by construction, so a weld run after this one would
    collapse the step straight back into the chamfer it replaced. That leg is the
    cost, and it is the one place the rectified path knowingly returns an axis
    feature shorter than FEATURE_M.
    """
    corner: dict = {}
    for ring, source in zip(rings, unwelded):
        count = len(ring)
        if count < 3:
            continue
        signed = geo.signed_area(ring)
        sign = 1.0 if signed >= 0.0 else -1.0
        error = abs(signed) - abs(geo.signed_area(source))
        for index in range(count):
            a = ring[index]
            b = ring[(index + 1) % count]
            if _on_axis(a, b):
                continue
            if min(abs(b[0] - a[0]), abs(b[1] - a[1])) > limit:
                continue
            if not _on_axis(ring[index - 1], a) or not _on_axis(b, ring[(index + 2) % count]):
                continue
            key = (a, b) if a <= b else (b, a)
            if key in corner:
                continue
            # The two corners of the edge's bounding box. Through the first the
            # ring leaves `a` along y and arrives at `b` along x, through the
            # second the other way round, and the areas they add are exact
            # opposites.
            horizontal_first = (b[0], a[1])
            vertical_first = (a[0], b[1])
            gain = sign * 0.5 * (b[0] - a[0]) * (b[1] - a[1])
            if error > 0.0:
                corner[key] = horizontal_first if gain < 0.0 else vertical_first
            elif error < 0.0:
                corner[key] = horizontal_first if gain > 0.0 else vertical_first
            else:
                arriving_flat = abs(a[1] - ring[index - 1][1]) <= PRECISION_M
                corner[key] = horizontal_first if arriving_flat else vertical_first
    if not corner:
        return rings
    out = []
    for ring in rings:
        count = len(ring)
        points: list = []
        for index in range(count):
            a = ring[index]
            b = ring[(index + 1) % count]
            points.append(a)
            key = (a, b) if a <= b else (b, a)
            if key in corner:
                points.append(corner[key])
        out.append(_drop_loops(_dedupe(points)))
    return out


def _edges_over_two(rings: list[geo.Ring]) -> int:
    """Interior edges carried by more than two rooms, which is the solver's refusal.

    Counted on the finished rings rather than trusted from the noding, because
    every step after the noding moves vertices and one of them could weld two
    rooms onto a third's edge. Zero is the contract; anything else is the vector
    path's fault surviving the repair that exists to remove it.
    """
    owners: dict = {}
    for index, ring in enumerate(rings):
        for a, b in zip(ring, ring[1:] + ring[:1]):
            owners.setdefault(tuple(sorted((a, b))), set()).add(index)
    return sum(1 for rooms in owners.values() if len(rooms) > 2)


def _gap_of(polys, adjacent: float) -> WallGap:
    """Length-weighted distance between facing wall faces, over the whole floor.

    Measured from segment pairs rather than from polygon pairs. `Polygon.distance`
    reports the nearest approach anywhere on two rings, so a pair of rooms that
    merely meets at a corner reports a gap near zero, and an MSD floor has enough
    of those to drag the median well under the wall it is supposed to describe.
    Weighting each pair by the run it faces over gives the wall the floor is
    actually built from.
    """
    import numpy as np

    samples = _facing_pairs(polys, adjacent)
    if not samples:
        return WallGap(DEFAULT_GAP_M, DEFAULT_GAP_M, DEFAULT_GAP_M, 0)
    values = np.array([d for d, _ in samples])
    weights = np.array([w for _, w in samples])
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()

    def pick(quantile: float) -> float:
        return float(values[min(int(np.searchsorted(cumulative, quantile)), len(values) - 1)])

    return WallGap(pick(0.5), pick(0.25), pick(0.75), len(samples))


def _facing_pairs(polys, adjacent: float) -> list[tuple[float, float]]:
    """Every near-parallel pair of wall faces from two rooms, as distance and run."""
    from shapely import STRtree
    from shapely.geometry import LineString, box

    edges = []
    for index, poly in enumerate(polys):
        coords = list(poly.exterior.coords)
        edges.extend((index, a, b) for a, b in zip(coords, coords[1:]))
    if not edges:
        return []
    tree = STRtree([LineString([a, b]) for _, a, b in edges])
    limit = math.cos(math.radians(FACE_ANGLE_DEG))
    out: list[tuple[float, float]] = []
    for position, (room, a, b) in enumerate(edges):
        probe = box(
            min(a[0], b[0]) - adjacent, min(a[1], b[1]) - adjacent,
            max(a[0], b[0]) + adjacent, max(a[1], b[1]) + adjacent,
        )
        for index in tree.query(probe):
            other = int(index)
            if other <= position or edges[other][0] == room:
                continue
            pair = _face(a, b, edges[other][1], edges[other][2], limit, adjacent)
            if pair is not None:
                out.append(pair)
    return out


def _face(a, b, c, d, limit: float, adjacent: float) -> Optional[tuple[float, float]]:
    """Distance and shared run of two wall faces, or None when they do not face."""
    ux, uy = b[0] - a[0], b[1] - a[1]
    vx, vy = d[0] - c[0], d[1] - c[1]
    span, other = math.hypot(ux, uy), math.hypot(vx, vy)
    if span < FACE_RUN_M or other < FACE_RUN_M:
        return None
    if abs(ux * vx + uy * vy) / (span * other) < limit:
        return None
    start, end = _along(a, b, c), _along(a, b, d)
    run = (min(max(start, end), 1.0) - max(min(start, end), 0.0)) * span
    if run < FACE_RUN_M:
        return None
    distance = (_offset(a, c, b) + _offset(a, d, b)) / 2.0
    return None if distance > adjacent else (distance, run)


def _resolve_overlaps(offsets, polys, snap: float, gap: Optional[float]):
    """Split every region two grown rooms both claim along that wall's centreline.

    Every room is grown by the same generous reach, so each wall is contested
    over its whole run by exactly the two rooms that face it, and the seam is
    placed per wall rather than per floor. The floor's median gap is a statistic
    to report, not a length to build on: the MSD gap distribution runs from under
    0.06 m to over 0.40 m, so one global half offset lands the two faces of most
    walls somewhere other than the centreline, leaving a strip that then has to
    be handed whole to one of the rooms.

    A room never loses its own interior. Every point of room j is at least the
    pair's gap away from room i, and the seam sits at half that, so `keep` can
    never reach past room j's own face.
    """
    import shapely
    from shapely import STRtree

    tree = STRtree(offsets)
    pieces = list(offsets)
    for index in range(len(pieces)):
        for position in sorted(int(o) for o in tree.query(offsets[index])):
            if position >= index:
                continue
            overlap = _areal(shapely.intersection(pieces[index], pieces[position], grid_size=snap))
            if overlap.is_empty:
                continue
            wall = polys[index].distance(polys[position]) if gap is None else gap
            # Through `_exact` for the reason `_exact` states. A mitre buffer of
            # a room that turns a sharp reflex corner comes back self touching,
            # and off the precision grid besides, and GEOS is robust at a fixed
            # grid size only when both operands are already on it. Intersecting
            # the raw buffer threw "side location conflict" on the first floor
            # whose second part reached this loop. The offset it moves the seam
            # by is one snap, a micron.
            half = _exact(polys[index].buffer(wall / 2.0 + snap, join_style="mitre"), snap)
            keep = _areal(shapely.intersection(overlap, half, grid_size=snap))
            yours = _areal(shapely.difference(overlap, keep, grid_size=snap))
            pieces[position] = _areal(shapely.difference(pieces[position], keep, grid_size=snap))
            pieces[index] = _areal(shapely.difference(pieces[index], yours, grid_size=snap))
    return [geo.largest_part(p) for p in pieces]


def _fill_residue(pieces, floor, limit: float, snap: float):
    """Give every part of the floor no room claimed to the room `geo.hand_over` picks for it.

    Residue, not holes. Half the measured gap is the right offset for the wall
    the median describes and too little for a thicker one, and the strip that
    survives over a thick wall usually runs out to the building edge rather than
    closing into an interior ring, so a hole-only fill leaves it open. Under the
    solver's model that strip is an interior edge carrying a single adjacent room,
    which is the fault this module exists to avoid, and the grid path's
    nearest-room fill already hands the same space to a room.

    `limit` is the check, not a cap. A part over it is a place where the measured
    gap and the floor disagree by more than a wall junction, and the count is
    reported so a floor leaning on the fill stays visible.

    Which room takes a part is `geo.hand_over`'s rule, which is the rule the
    seam repair uses and the only one in the codebase: the first room in
    bounding order that the part leaves as one simply connected ring. Taking the
    bounding room unconditionally, which is what this did, is the defect the
    hand-over fix removed from the seam repair, and it was the same defect here.
    A residue strip that runs around a third room leaves whoever took it with
    that third room punched out as an interior; `_rings_of` writes a room back
    as its exterior alone, so the third room's floor goes silently to the first
    and the plan claims it twice.

    A part nobody can take is left uncovered and counted with its area. That is
    floor the partition could not assign without inventing an overlap, and
    saying so is better than assigning it: it reaches the seam repair
    downstream, which asks the same question again with the same rule.
    """
    import shapely

    filled = 0
    over = 0
    refused = 0
    refused_m2 = 0.0
    claimed = _areal(shapely.union_all(pieces, grid_size=snap))
    for part in geo.parts(_areal(shapely.difference(floor, claimed, grid_size=snap))):
        owner, grown = geo.hand_over(pieces, part, grid_size=snap)
        if owner is None:
            refused += 1
            refused_m2 += part.area
            continue
        pieces[owner] = grown
        filled += 1
        over += 1 if part.area > limit else 0
    return pieces, filled, over, refused, refused_m2


def _rings_of(pieces, snap: float) -> tuple[list[geo.Ring], list[int], Uncarried]:
    """Outer rings on one common precision grid, noded and stripped of flat points.

    A piece is written back as one outer ring, so two things can be lost here and
    neither may be lost in silence. An interior is a room this piece wraps, and
    the exterior ring alone then claims that room's floor a second time. A second
    part is floor of this piece that no ring covers at all. Both are counted and
    their area is returned with the rings, because each one surfaces far
    downstream as the fault it causes rather than as the drop it was.

    The snap is GEOS precision reduction rather than a hand-rolled merge of
    vertices within a tolerance. Both put every coordinate of every room onto one
    grid, which is what makes two rooms' shared edges bit identical, but a merge
    only moves points: where it collapses a sliver it leaves a ring that touches
    itself, and a self-touching ring is not a polygon. Over the first twelve MSD
    floors the merge left 12 per cent of rooms self-intersecting and 86 per cent
    of interior wall length coincident, against none self-intersecting and 99 per
    cent coincident here, because the reduction repairs the topology it breaks
    instead of only rounding the numbers.
    """
    import shapely
    from shapely.geometry import Polygon

    raw: list[tuple[int, list]] = []
    uncarried = Uncarried()
    for index, piece in enumerate(pieces):
        whole = geo.repaired(shapely.set_precision(piece, snap))
        shell = geo.largest_part(whole)
        if shell.is_empty or shell.area <= 0.0:
            continue
        uncarried = uncarried + _uncarried_by(shell, geo.parts(whole))
        raw.append((index, _dedupe([(float(x), float(y)) for x, y in shell.exterior.coords])))
    noded = _insert_junctions([points for _, points in raw], snap)
    pruned = _weld_stubs(_prune_collinear([_drop_loops(_dedupe(points)) for points in noded]))
    out_rings: list[geo.Ring] = []
    kept: list[int] = []
    for (index, plain), points in zip(raw, pruned):
        ring = points if len(points) >= 3 and _simple(points) else plain
        if len(ring) >= 3:
            out_rings.append(geo.orient_ccw(geo.close_ring(ring)))
            kept.append(index)
    return out_rings, kept, uncarried


def _uncarried_by(shell, lobes: list) -> Uncarried:
    """What writing `shell` back as its outer ring alone leaves behind.

    One place answers this for both paths, because the write-back is the same
    step whether the piece came from a cell mask or from a polygon boolean, and
    the two faults it can commit are the same two. `lobes` is every part the
    piece was in, `shell` the one that is kept.
    """
    from shapely.geometry import Polygon

    lost = Uncarried()
    split_m2 = sum(lobe.area for lobe in lobes) - shell.area
    if split_m2 > 0.0:
        lost.split_rooms = 1
        lost.split_m2 = split_m2
    if len(shell.interiors) > 0:
        lost.wrapping_rooms = 1
        lost.wrapped_m2 = sum(Polygon(hole).area for hole in shell.interiors)
    return lost


def _simple(points: list) -> bool:
    """Whether the finished ring is still a polygon.

    Noding and loop cutting are both edits to a ring that was simple when the
    precision reduction handed it over, and on a few rooms in a thousand they
    leave it crossing itself. That is worth repairing here rather than sending
    on, because an invalid ring is an ingest fault of exactly the kind this
    module exists to remove, and the ring before those two steps is a valid
    answer that only lacks the neighbours' vertices.
    """
    from shapely.geometry import Polygon

    return Polygon(points).is_valid


def _weld_stubs(rings: list[list], limit: Optional[float] = None) -> list[list]:
    """Collapse a short edge that three or more rooms carry into a single point.

    Three rooms meet at a point. This partition can leave them a stub a couple of
    centimetres long instead, because the three wall centrelines bounding them do
    not cross in one place, and the solver reads that stub as an interior edge
    with three adjacent rooms and refuses the floor. Welding the stub's endpoints
    puts the meeting back at a point and moves nothing else.
    """
    limit = WELD_M if limit is None else limit
    for _ in range(3):
        owners: dict = {}
        for index, ring in enumerate(rings):
            for a, b in zip(ring, ring[1:] + ring[:1]):
                owners.setdefault(tuple(sorted((a, b))), set()).add(index)
        merge: dict = {}
        for (a, b), rooms in owners.items():
            if len(rooms) > 2 and math.dist(a, b) <= limit:
                merge[a] = merge[b] = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        if not merge:
            break
        rings = [_dedupe([merge.get(point, point) for point in ring]) for ring in rings]
    return rings


def _insert_junctions(rings: list[list], snap: float) -> list[list]:
    """Put every neighbour's vertex onto the edge it lies on, so the two rings match.

    Two rooms can face each other across a wall with their corners in different
    places: one long edge on one side, three shorter ones on the other. Snapping
    vertices does nothing about that, because on the long edge there is no vertex
    to snap to, and the two boundaries then agree to floating point noise rather
    than exactly. Inserting the neighbour's vertices into the long edge gives
    both sides the same chain of segments, which is what makes a shared boundary
    coincide instead of merely come close.
    """
    from shapely import STRtree
    from shapely.geometry import Point

    points = sorted({point for ring in rings for point in ring})
    if not points:
        return rings
    tree = STRtree([Point(p) for p in points])
    out = []
    for ring in rings:
        built: list = []
        mine = set(ring)
        for index, start in enumerate(ring):
            built.append(start)
            built.extend(_on_edge(tree, points, start, ring[(index + 1) % len(ring)], snap, mine))
        out.append(built)
    return out


def _on_edge(tree, points, start, end, snap: float, mine: set) -> list:
    """The neighbours' vertices lying strictly inside the segment, in order along it.

    A vertex this ring already carries is never inserted, however close to the
    edge it falls. A room with a narrow notch has two faces a hair apart, and
    putting one face's corner onto the other face makes the ring visit that point
    twice, which is a ring self-intersection rather than a shared edge.
    """
    from shapely.geometry import box

    probe = box(
        min(start[0], end[0]) - snap, min(start[1], end[1]) - snap,
        max(start[0], end[0]) + snap, max(start[1], end[1]) + snap,
    )
    hits = []
    for index in tree.query(probe):
        point = points[int(index)]
        if point in mine or _offset(start, point, end) > snap:
            continue
        along = _along(start, end, point)
        if 0.0 < along < 1.0:
            hits.append((along, point))
    return [point for _, point in sorted(hits)]


def _along(start, end, point) -> float:
    """Where `point` projects onto the segment, as a fraction of its length."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    span = dx * dx + dy * dy
    if span == 0.0:
        return 0.0
    return ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / span


def _dedupe(points: list) -> list:
    """Drop repeated neighbours and the closing point, which `close_ring` restores."""
    out: list = []
    for point in points:
        if not out or point != out[-1]:
            out.append(point)
    while len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _drop_loops(points: list) -> list:
    """Cut out any excursion the ring makes back to a vertex it already visited.

    Two vertices a hair apart become one vertex when they snap onto the same
    representative, and the ring then visits that point twice. The shape is
    unchanged at any scale coarser than the snap, but the ring is no longer
    simple, and a self-touching ring is not a polygon the solver can take.
    """
    stack: list = []
    seen: dict = {}
    for point in points:
        if point in seen:
            for dropped in stack[seen[point] + 1:]:
                seen.pop(dropped, None)
            del stack[seen[point] + 1:]
            continue
        seen[point] = len(stack)
        stack.append(point)
    return stack


def _prune_collinear(rings: list[list]) -> list[list]:
    """Drop a vertex only where every ring that carries it sees it as flat.

    A vertex kept by one ring and dropped by its neighbour is exactly how a
    snapped shared edge comes apart again, so the decision is taken once per
    vertex across the whole floor rather than once per ring. Flat means exactly
    collinear: a tolerance here would move the surviving edge by the tolerance
    and lift it off the neighbour's segment.
    """
    corner: dict[tuple[float, float], bool] = {}
    for ring in rings:
        for index, point in enumerate(ring):
            before = ring[index - 1]
            after = ring[(index + 1) % len(ring)]
            corner[point] = corner.get(point, False) or _offset(before, point, after) > 0.0
    out = []
    for ring in rings:
        trimmed = [point for point in ring if corner[point]]
        out.append(trimmed if len(trimmed) >= 3 else ring)
    return out


def _offset(before, point, after) -> float:
    """Perpendicular distance of `point` from the chord through its neighbours."""
    dx, dy = after[0] - before[0], after[1] - before[1]
    span = math.hypot(dx, dy)
    if span == 0.0:
        return 0.0
    return abs(dx * (before[1] - point[1]) - dy * (before[0] - point[0])) / span


def _shared_exact(rings: list[geo.Ring], boundary: geo.Ring) -> float:
    """Share of interior wall length where two rooms carry one coincident edge."""
    from shapely import STRtree
    from shapely.geometry import Polygon

    polys = [geo.repaired(Polygon(r)) for r in rings]
    if len(polys) < 2:
        return 1.0
    tree = STRtree(polys)
    shared = 0.0
    for index, poly in enumerate(polys):
        for other in tree.query(poly):
            if int(other) <= index:
                continue
            shared += poly.boundary.intersection(polys[int(other)].boundary).length
    outline = Polygon(boundary).exterior.length
    interior = (sum(p.exterior.length for p in polys) - outline) / 2.0
    return 1.0 if interior <= 0.0 else min(shared / interior, 1.0)


def _mean_area_error(out_rings: list[geo.Ring], polys, kept: list[int]) -> float:
    """Mean absolute per-room area change, relative to the original interior."""
    errors = [
        abs(abs(geo.signed_area(out_rings[new])) - polys[old].area) / polys[old].area
        for new, old in enumerate(kept)
        if polys[old].area > 0.0
    ]
    return sum(errors) / len(errors) if errors else 0.0


def _areal(geom):
    """Only the parts of an overlay result that carry area.

    An overlay at a fixed grid size can collapse a sliver to a line, and the
    result then carries mixed dimensions, which the next overlay refuses outright
    with "Overlay input is mixed-dimension". Dropping the collapsed parts at each
    step keeps every operand a polygon.
    """
    from shapely.geometry import MultiPolygon, Polygon

    parts = geo.parts(geom)
    if not parts:
        return Polygon()
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def _exact(geom, snap: float):
    """One valid polygon with every coordinate on the precision grid.

    Every overlay in the vector path runs at `grid_size=snap`, and GEOS is only
    robust there when its operands are already on that grid. Reducing the inputs
    once at the top is what keeps the overlays from tripping the
    "found two shells in EdgeRing list" assertion on a floor with a near
    degenerate room.
    """
    import shapely

    return geo.largest_part(geo.repaired(shapely.set_precision(geo.repaired(geom), snap)))


def _part_ring(part) -> Optional[geo.Ring]:
    """One part of `_filled_parts`'s answer as a ring, which is how a Plan carries it."""
    if part is None or part.is_empty or part.area <= 0.0:
        return None
    return geo.orient_ccw(geo.close_ring([(float(x), float(y)) for x, y in part.exterior.coords]))


def _filled_parts(polys, close: float, bridges=()) -> list:
    """The floor outline, one polygon per part the rooms close into, largest first.

    The plain union is the wrong thing to take a part of. When rooms are stored
    as interiors they touch nothing, so the union is one part per room. Growing
    every room by half a wall, unioning, and shrinking back recovers the outline
    the rooms sit in. Interior holes are then dropped, because a hole is a place
    with no room, which under the solver's model is an edge with one adjacent
    room; the nearest-room fill hands that space to a room instead.

    Every part of the closed union is returned and not only the largest. A floor
    whose rooms close into several parts is the ordinary shape of a building
    drawn in wings, and over MSD's 4,167 train floors 294 of them close into
    more than one part at the rectified path's reach by closing alone, carrying
    3,865 rooms and 47,128.5 square metres outside the largest; 138 still do
    once the doorways the source states are bridged. Keeping the largest alone
    silently graded one wing under the whole floor's identity, and the rooms of
    every other wing were clipped against a part they are nowhere near, came
    back empty, and were dropped without a count.

    Within one part the write-back is unchanged: the shrink can itself come back
    in pieces where the growth joined two stretches at a pinch, and the largest
    piece is the part, as it has always been. That keeps the count of parts the
    count of components the rooms actually close into, which is the number the
    refusal downstream is measured against.

    `bridges` are segments the caller asserts connect two rooms, from
    `bridge_spans`. Each is unioned in as a strip one reach wide either side of
    it, with square caps so the strip runs the whole way across the gap and a
    reach into each room, and the strip is then restored after the shrink. It is
    restored because it was never grown: the shrink exists to undo the growth
    applied to the rooms, and a strip a reach wide either side of a segment
    erodes by a reach to nothing, which would pinch the two rooms apart again
    and leave the threshold with no area for the residue fill to hand to
    anybody. What survives is a doorway's worth of floor joining the two rooms,
    which is what the caller asserted was there.
    """
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union

    spans = [
        LineString(bridge).buffer(close, cap_style="square", join_style="mitre") for bridge in bridges
    ]
    grown = unary_union([p.buffer(close, join_style="mitre") for p in polys] + spans)
    out = []
    for component in geo.parts(grown):
        shell = Polygon(component.exterior)
        closed = shell.buffer(-close, join_style="mitre")
        mine = [span for span in spans if component.contains(span.representative_point())]
        if mine:
            closed = unary_union([closed, *mine])
        shrunk = geo.largest_part(closed)
        out.append(shell if shrunk.is_empty or shrunk.area <= 0.0 else Polygon(shrunk.exterior))
    out.sort(key=lambda part: part.area, reverse=True)
    return out


def _home_parts(polys, parts) -> list[int]:
    """Which part of the floor each room lies in, by its own interior point.

    An interior point rather than a centroid, because a room shaped like an L
    has a centroid outside itself. A room that no part contains is given the
    part nearest to it rather than dropped here: it is a room the shrink left
    outside every part, its clip against that part then comes back empty, and
    `_rings_of` accounts for it exactly as it did before this function existed.
    """
    from shapely import STRtree

    if not parts:
        return [0] * len(polys)
    tree = STRtree(parts)
    home: list[int] = []
    for poly in polys:
        if poly.is_empty or poly.area <= 0.0:
            home.append(0)
            continue
        point = poly.representative_point()
        found = [int(i) for i in tree.query(point) if parts[int(i)].contains(point)]
        home.append(found[0] if found else min(range(len(parts)), key=lambda i: parts[i].distance(point)))
    return home


def _fill_nearest(label, inside):
    """Give every inside cell with no room the label of the nearest cell that has one."""
    import numpy as np
    from scipy import ndimage

    missing = label < 0
    if not missing.any() or (~missing).sum() == 0:
        return np.where(inside, label, -1)
    _, indices = ndimage.distance_transform_edt(missing, return_indices=True)
    filled = label[tuple(indices)]
    return np.where(inside, filled, -1)


def _mask_to_ring(mask, x0: float, y0: float, step: float) -> tuple[Optional[geo.Ring], Uncarried]:
    """Union the cells of one label into a polygon and return its outer ring.

    A room is one ring here, in `geo.Ring`, in `plan.Room` and in the solver's
    snapshot node, so a cell set that is not simply connected cannot be emitted
    whole. Two things are therefore discarded, and both are measured as well as
    counted rather than dropped in silence, because each one reaches the solver
    as a construction fault far from here and because a count cannot be read on
    its own: over MSD's 4,167 floors the median discarded part is one cell and
    the largest is 15.88 square metres of a living room.

    An interior ring is a room this room wraps completely. Returning the outer
    ring alone makes this room's ring cover the wrapped one, so the two overlap,
    the wrapped room's every edge groups with one adjacent room, and the solver
    refuses the whole floor naming one of those edges.

    An extra part is a second piece of the same room, reached from the first
    only through another room. Returning the largest part alone leaves that
    piece uncovered, which is a hole in the union when it is interior and a bite
    out of the outline when it is not.

    Nothing here tries to reunite the pieces, and that is a ruling rather than an
    omission. The cause is upstream of the grid: the room's own containment mask
    is already in several parts before any other room contests a cell, because
    the passage between its wings carries no cell centre at this step.
    Reconnecting them would mean claiming cells from a neighbour along a passage
    the grid cannot see, which invents floor where the source may have none, and
    on the MSD rooms that reach here the passage is measured at 0.14 to 0.24 m
    against a 0.38 m step, so the source mostly does have none. Where the passage
    is in fact wider than a cell and the grid was only rendering it as a diagonal
    staircase, the rotation onto the floor's dominant wall direction already
    repairs it: 161 of the 175 floors that split without that rotation stop
    splitting with it, leaving 25 floors and 28 rooms over the whole corpus.
    Specification section 5.2 states the rule and 9.5 carries the residue.
    """
    import numpy as np
    from shapely.geometry import box
    from shapely.ops import unary_union

    if not mask.any():
        return None, Uncarried()
    boxes = []
    for row in range(mask.shape[0]):
        flags = np.concatenate(([0], mask[row].astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(flags))
        for start, stop in zip(edges[0::2], edges[1::2]):
            boxes.append(box(x0 + start * step, y0 + row * step, x0 + stop * step, y0 + (row + 1) * step))
    merged = unary_union(boxes)
    lobes = geo.parts(merged)
    if not lobes:
        return None, Uncarried()
    shell = geo.largest_part(merged)
    ring = geo.close_ring([(float(x), float(y)) for x, y in shell.simplify(0.0).exterior.coords])
    return ring, _uncarried_by(shell, lobes)
