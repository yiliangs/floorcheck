"""Ring geometry helpers shared by the loaders and the adapter.

Nothing here knows about a dataset or about the solver. The two jobs are ring
hygiene (close, deduplicate, orient) and the dominant-direction alignment that
MSD needs, kept here because the off-axis measure it reports is a property of a
ring set rather than of MSD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

Point = tuple[float, float]
Ring = list[Point]


def close_ring(ring: Ring) -> Ring:
    """Drop a repeated closing vertex and any zero-length edge.

    The solver's snapshot rings are implicitly closed, so a duplicated first and
    last point becomes a zero-length segment and a spurious collinear corner.
    """
    out: Ring = []
    for point in ring:
        if out and _same(out[-1], point):
            continue
        out.append((float(point[0]), float(point[1])))
    while len(out) > 1 and _same(out[0], out[-1]):
        out.pop()
    return out


def _same(a: Point, b: Point, tol: float = 1e-9) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def signed_area(ring: Ring) -> float:
    total = 0.0
    for i in range(len(ring)):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % len(ring)]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def orient_ccw(ring: Ring) -> Ring:
    return ring if signed_area(ring) >= 0.0 else list(reversed(ring))


def segments(ring: Ring) -> list[tuple[Point, Point]]:
    return [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]


@dataclass(frozen=True)
class Tiling:
    """What the union of one room set says about whether the rooms tile a floor.

    `rings` are the rooms after any sub-pixel seam between them was closed, in
    the order they were given, so a caller holding a parallel list of tags can
    keep reading the two together. `outline` is the outer ring of the largest
    part of their union, which is the only thing a root node can ever be.

    `closed_m2` is floor that no room covered and that was handed to a
    neighbouring room because the gap was no wider than the tolerance. Which
    neighbour is `hand_over`'s rule: the candidates are the rooms bounding the
    region, tried in `bounding_order`, most-bounding first, and the region
    goes to the first candidate it leaves simply connected, meaning the union
    of that room with the region is one polygon carrying no interior ring. A
    union coming back as more than one polygon disqualifies the candidate on
    the same grounds. If no candidate qualifies, nobody takes the region and
    the floor stays uncovered and is counted there instead.
    `bridged_m2` is the part of that total which lay in an open channel between
    two disjoint rooms rather than in an enclosed void. The two are one total and
    one of its halves rather than two totals, because the floor moves for one
    reason and a caller that wants to cap how much of it moves has to cap the
    whole.
    `uncovered_m2` is what the tolerance did not reach: the surviving holes plus
    the parts of the union the outline cannot contain. It is reported apart from
    `closed_m2` because they are different findings. The first says the source
    frame cannot express a shared wall more precisely than it did, and the
    second says the rooms genuinely do not tile.
    """

    rings: list[Ring]
    outline: Ring
    floor_m2: float
    closed_m2: float
    bridged_m2: float
    uncovered_m2: float
    holes: int
    outlying_parts: int


def bounding_order(polygons: list, region) -> list[int]:
    """The polygons that could take `region`, the one bounding most of it first.

    Sharing a boundary is what makes a room the right one to be handed floor,
    so the order is by shared boundary length, longest first, and a tie goes to
    the lower index. Polygons sharing no boundary at all follow, nearest first,
    because a region still has to go somewhere when a polygon boolean left no
    length behind for anyone.

    The order rather than the winner alone is what the callers need. Handing a
    region over has a second condition beyond bounding most of it, and a
    candidate that fails it passes the region to the next.
    """
    touching: list[tuple[float, int]] = []
    near: list[tuple[float, int]] = []
    for index, poly in enumerate(polygons):
        if poly.is_empty:
            continue
        length = poly.boundary.intersection(region.boundary).length
        if length > 0.0:
            touching.append((-length, index))
        near.append((poly.distance(region), index))
    touching.sort()
    near.sort()
    ordered = [index for _, index in touching]
    seen = set(ordered)
    ordered.extend(index for _, index in near if index not in seen)
    return ordered


def bounding_index(polygons: list, region) -> Optional[int]:
    """The polygon sharing the longest boundary with `region`, or the nearest one."""
    order = bounding_order(polygons, region)
    return order[0] if order else None


def tile(
    rings: list[Ring],
    tolerance_m: float = 0.0,
    bridge_seams: bool = True,
    extent: Optional[Ring] = None,
) -> Tiling:
    """Read the union of `rings`, closing any gap no wider than `tolerance_m`.

    A void is a hole in the union: floor that no room covers. Under the solver's
    two-slot wall model its rim is an interior edge carrying one adjacent room,
    which is geometry the model cannot hold, so a void has to be either closed
    here or refused by the caller.

    A void is closed by giving it to the first candidate in `bounding_order`
    that `hand_over` leaves simply connected, which leaves the two rooms either
    side sharing the wall that the void stood in for. The alternative reading
    of "snap", moving vertices onto one another within the tolerance, cannot
    be used at this tolerance: a plan traced from a
    pixel mask has every coordinate an exact multiple of one pixel, so a snap
    that closed a one-pixel void would equally collapse every genuine
    one-pixel-thick feature in the plan, and a whole wall of rooms with it.

    Width rather than area is the test, because the two say different things. A
    seam the source frame could not express is thin and may be long; a courtyard
    is compact. A void is no wider than `tolerance_m` exactly when eroding it by
    half that leaves nothing, which is what the erosion below asks.

    A gap between two parts of the union is open rather than bounded, so its
    extent cannot be measured by eroding it the way a void's can, and deciding it
    is a mechanism rather than a tolerance. `_bridge_seams` is that mechanism:
    it fills every channel a morphological closing at `tolerance_m` fills and
    that touches two distinct parts, and hands each one to whichever room
    `hand_over` gives it: the candidates are tried in `bounding_order`,
    most-bounding first, and the region goes to the first one it leaves
    simply connected, which is the same rule and the same tie-break an
    enclosed void already gets.

    It is on by default and `bridge_seams=False` turns it off, which is the
    2026-09-16 reversal of that day's earlier ruling on question 13. The two
    shapes are one phenomenon read at one width, so treating them differently
    was an asymmetry between sources rather than a rule: MSD states room
    interiors whose wall faces sit about 0.16 m apart, under one MSD pixel, and
    the grid partition closes them by construction before any of this runs,
    while HouseDiffusion states rooms one pixel apart and was refused whole for
    it. What a source pixel means is the only thing that varies per source, and
    it is stated by the reader in `Plan.source_pixel_m`.

    Which shape a corpus leaves is a fact about the corpus and not about the
    rule. Measured over 600 samples each, House-GAN++ leaves 163 enclosed voids
    against 11 outlying parts, and only 4 of those 11 sit within a pixel of the
    mass. HouseDiffusion is the opposite: 1 enclosed void against 2,124 outlying
    parts, 1,784 of them within one pixel. An outlying part further than a pixel
    from the mass is still a room in the wrong place and is still refused, under
    this rule as under the one before it.

    `extent` is where floor may be invented, and it is the answer to a question
    the width rule does not ask. The width says how wide a closed seam may be
    and says nothing about where it may lie, which costs nothing while the rooms
    are all anyone knows, because then a gap between two of them is inside the
    plan by construction. It stops being free when the step that built the room
    set already decided where the floor was. MSD's partition decides exactly
    that, at 0.20 m, before it can assign a cell to anything, and a channel left
    open afterwards may lie outside that decision: of the seven MSD floors that
    leave one, the share of the channel inside the extent runs from 1.7 to 68.9
    per cent, and on the largest the rule without this clause invents 11.5 square
    metres of floor, four fifths of it in a slot the building does not have.
    So a channel is clipped to `extent` before it is filled, and a caller with
    no such fact passes None and nothing is clipped. An enclosed void is not
    clipped, for the reason `_close_voids` gives: the rooms enclose it, so they
    have already settled the question the extent answers.

    Collinear vertices are dropped from the outline exactly, never within a
    tolerance, because the ring this returns has to keep lying on the rings it
    was built from.

    Shapely is imported here rather than at module scope: this is the one
    routine in the module that needs a polygon boolean, and the ring hygiene
    above stays free of it.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    usable = [ring for ring in rings if len(ring) >= 3]
    polygons = [repaired(Polygon(ring)) for ring in usable]
    limit = repaired(Polygon(extent)) if extent is not None and len(extent) >= 3 else None
    bridged_m2 = 0.0
    closed_m2 = 0.0
    changed: dict[int, object] = {}
    if tolerance_m > 0.0:
        # Bridging first, because it is the step that decides how many parts
        # there are, and closing a void asks which part is the largest.
        if bridge_seams:
            polygons, bridged, bridged_m2 = _bridge_seams(polygons, tolerance_m, limit)
            changed.update(bridged)
        polygons, closed, voids_m2 = _close_voids(polygons, tolerance_m)
        changed.update(closed)
        closed_m2 = bridged_m2 + voids_m2
    merged = unary_union(polygons)
    parts = sorted(getattr(merged, "geoms", [merged]), key=lambda p: p.area, reverse=True)
    if not parts or parts[0].is_empty:
        raise ValueError("the ring union is empty")
    main = parts[0]
    flattened = main.simplify(0.0)
    exterior = main.exterior if flattened.is_empty else flattened.exterior
    outline = orient_ccw(close_ring([(float(x), float(y)) for x, y in exterior.coords]))
    uncovered = sum(Polygon(r).area for r in main.interiors)
    uncovered += sum(part.area for part in parts[1:])
    # Only a room that was actually given a void comes back rewritten. Passing
    # every room through the polygon layer and out again would re-derive rings
    # that nothing asked to change, which is both a way to perturb a plan that
    # had no seam and a way to fail on a self-intersecting ring that the union
    # itself handles perfectly well.
    out_rings = list(usable)
    for index, poly in changed.items():
        rewritten = _ring_of(poly)
        if len(rewritten) >= 3:
            out_rings[index] = rewritten
    return Tiling(
        rings=out_rings,
        outline=outline,
        floor_m2=abs(signed_area(outline)),
        closed_m2=closed_m2,
        bridged_m2=bridged_m2,
        uncovered_m2=uncovered,
        holes=len(main.interiors),
        outlying_parts=len(parts) - 1,
    )


# How far apart two boundaries may be and still count as the same line, meters.
# A mitre closing returns the original coordinates, so the distance this guards
# is zero in exact arithmetic and the constant only absorbs the last bits of a
# polygon boolean. It is not a tolerance anyone rules on: a real seam is a whole
# source pixel wide, seven orders of magnitude above this.
_COINCIDENT_M = 1e-9


def _within(region, limit) -> list:
    """The parts of `region` that lie inside `limit`, or all of it when there is none.

    One clip, used by both repairs, so that "where floor may be invented" is
    asked once and answered the same way for a void and for a channel. A region
    that straddles the extent comes back in pieces rather than whole or not at
    all, because the part of it inside the extent is floor by every test the
    rule applies and the part outside is not, and there is no reading on which
    the second decides the first.
    """
    if limit is None:
        return [region]
    clipped = region.intersection(limit)
    return [
        part
        for part in getattr(clipped, "geoms", [clipped])
        if part.geom_type == "Polygon" and not part.is_empty and part.area > 1e-12
    ]


def hand_over(polygons: list, region, grid_size: Optional[float] = None) -> tuple[Optional[int], object]:
    """The room that takes `region`, and that room grown by it.

    A room is one simple ring everywhere downstream of this module, so a room
    may only be handed floor that leaves it one. That is the second condition
    on an owner, and it is not a formality. Where a region runs around a third
    room, the room bounding most of it comes back from the union as a ring with
    that third room punched out of it, and this module can only write a room
    back as its exterior: the third room's floor would go silently to the
    second and the plan would claim it twice. So the candidates are tried in
    `bounding_order` and the first one the region leaves simply connected takes
    it. When the region would enclose something whoever took it, nobody takes
    it and the floor stays uncovered, which is the honest reading of a plan
    whose rooms cannot be made to tile by moving a seam.

    A union that comes back as more than one polygon is refused on the same
    grounds. It means the region met the room at a point rather than along a
    wall, and keeping the larger half would lose the rest of the region while
    the caller counted all of it as moved.

    This is the rule wherever floor no room covers is given to a room, in this
    module and in the partition, which is why it is public. `grid_size` is the
    precision the union runs at, for a caller whose operands are already on a
    grid and whose next overlay needs the answer on the same one; the default
    of no reduction is what a caller in floating point wants.
    """
    import shapely

    for owner in bounding_order(polygons, region):
        grown = shapely.union_all([polygons[owner], region], grid_size=grid_size)
        if grown.geom_type != "Polygon" or grown.is_empty or len(grown.interiors) > 0:
            continue
        return owner, grown
    return None, None


def _bridge_seams(
    polygons: list, tolerance_m: float, limit=None
) -> tuple[list, dict[int, object], float]:
    """Hand every open channel no wider than `tolerance_m` on under `hand_over`'s rule.

    Answers in the shape `_close_voids` answers in, because the caller does the
    same thing with both: the polygons, which rooms changed, and how much floor
    moved. The two are the same repair on the two shapes uncovered floor takes.

    The channel is read off a morphological closing of the union rather than off
    a pair of rooms, because a channel is not a property of two rooms. A gap
    where three rooms meet is one region with three sides, and pairing the rooms
    would either hand it over twice or leave the junction open. The closing is a
    dilation by half the tolerance followed by an erosion by the same, so the
    floor it adds is exactly the floor a disk of the tolerance's diameter cannot
    reach, which is the same question `_wider_than` asks of an enclosed void and
    the reason the two rules can be stated as one.

    Only a channel touching two distinct parts is filled. The closing also fills
    a narrow notch in the outer boundary of a single part, and that floor is
    outside the plan rather than between two rooms of it: filling it would
    invent floor the generator never emitted. Requiring two parts is what keeps
    this a bridge between rooms rather than a smoothing of the outline.

    The mitre join is not a detail. Every corpus this runs on states its rings
    on an integer lattice, so a mitre dilation and erosion return the original
    coordinates exactly and the channel comes back sharing its boundary with the
    rooms it separates. A round join would approximate both buffers by arcs and
    leave slivers along boundaries that ought to coincide.
    """
    from shapely.ops import unary_union

    merged = unary_union(polygons)
    parts = list(getattr(merged, "geoms", [merged]))
    if len(parts) < 2:
        return polygons, {}, 0.0
    half = tolerance_m / 2.0
    closing = merged.buffer(half, join_style=2).buffer(-half, join_style=2)
    channels = closing.difference(merged)
    out = list(polygons)
    changed: dict[int, object] = {}
    bridged = 0.0
    for whole in getattr(channels, "geoms", [channels]):
        if whole.is_empty or whole.area <= 1e-12:
            continue
        for channel in _within(whole, limit):
            # Asked of the clipped piece and not of the channel it came from: a
            # channel that only reaches a second room outside the extent reaches
            # it through floor the plan does not have.
            if sum(1 for part in parts if part.distance(channel) <= _COINCIDENT_M) < 2:
                continue
            owner, grown = hand_over(out, channel)
            if owner is None:
                continue
            out[owner] = grown
            changed[owner] = grown
            bridged += channel.area
    return out, changed, bridged


def _close_voids(polygons: list, tolerance_m: float) -> tuple[list, dict[int, object], float]:
    """Hand every void no wider than `tolerance_m` on under `hand_over`'s rule.

    Answers with the polygons, which rooms changed, and how much floor moved.
    Naming the changed rooms is what lets the caller leave every other ring
    exactly as it was given.

    No extent is taken here, and the asymmetry with `_bridge_seams` is the point
    rather than an oversight. The extent answers "is this floor inside the
    plan", and for a void the rooms have already answered it: a void is a hole
    in their union, enclosed by rooms on every side, so it is inside whatever
    the rooms are inside. An open channel reaches the outside and the rooms
    settle nothing about it, which is the case the extent exists for. Asking the
    extent about a void is not merely redundant, it is wrong where the two were
    established at different lengths: MSD's extent comes from the raw interiors
    closed at 0.20 m while its rooms are grid masks on a 0.38 m lattice that
    stand up to half a cell past it, so clipping voids there cuts wall junctions
    the width rule accepts. Measured, it refuses 19 MSD floors that tile and
    drops the largest closure from 3.321 to 1.446 square metres.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    merged = unary_union(polygons)
    parts = sorted(getattr(merged, "geoms", [merged]), key=lambda p: p.area, reverse=True)
    if not parts or parts[0].is_empty:
        return polygons, {}, 0.0
    narrow = [
        void
        for void in (Polygon(ring) for ring in parts[0].interiors)
        if void.area > 0.0 and not _wider_than(void, tolerance_m)
    ]
    if not narrow:
        return polygons, {}, 0.0
    out = list(polygons)
    changed: dict[int, object] = {}
    closed = 0.0
    for void in narrow:
        owner, grown = hand_over(out, void)
        if owner is None:
            continue
        out[owner] = grown
        changed[owner] = grown
        closed += void.area
    return out, changed, closed


def _wider_than(void, tolerance_m: float) -> bool:
    """Whether `void` survives being eroded by half `tolerance_m`.

    The residue of an exact fit is zero area computed in floating point, so the
    comparison carries a numerical floor rather than asking for emptiness.
    """
    return void.buffer(-tolerance_m / 2.0).area > 1e-12


def parts(geom) -> list:
    """The polygon components of `geom` that carry area, interiors kept."""
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom] if geom.area > 0.0 else []
    return [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon" and g.area > 0.0]


def largest_part(geom):
    """The component of `geom` carrying the most area, or an empty polygon."""
    from shapely.geometry import Polygon

    found = parts(geom)
    return max(found, key=lambda p: p.area) if found else Polygon()


def repaired(poly):
    """The one valid polygon standing for one room ring: its largest lobe.

    `buffer(0)` is the standard repair, but on a ring that crosses itself it
    answers with one polygon per lobe, and there is then a choice to make about
    what the room is. Every reader of a room downstream of this module takes it
    as one simply connected ring: `_ring_of` writes a room back as its exterior,
    `hand_over` refuses an owner the region would leave in two pieces, and the
    solver's snapshot node carries one ring per room. A bowtie cannot be handed
    on as two lobes without one of those readers picking a lobe anyway, unnamed
    and somewhere else. So the repair collapses to the largest lobe here, at the
    point of repair, and the lobes it drops are discarded. `tile` takes its
    union over the repaired rooms, so a dropped lobe is measured only where
    other rooms enclose it as a hole; on the edge of the plan it lies outside
    the outline and nothing measures it.

    Every reader of a room ring in this module goes through here, because the
    fault the choice creates is a disagreement between readers rather than a bad
    answer from any one of them. `tile` kept the largest lobe and `overlap_area`
    kept them all, so a lobe one dropped and the other kept was counted as
    claimed twice as soon as a neighbour grew over it, on a plan whose rooms in
    fact tile.
    """
    if poly.is_valid:
        return poly
    return largest_part(poly.buffer(0))


def _ring_of(poly) -> Ring:
    return orient_ccw(close_ring([(float(x), float(y)) for x, y in poly.exterior.coords]))


def overlap_area(rings: list[Ring]) -> float:
    """Floor the rings cover more than once: the other half of failing to tile.

    `tile` measures floor no ring covers. This measures floor two rings
    both claim, and the solver reads the two the same way. Where ring A contains
    ring B, A carries no wall face along B's boundary, so every atomic edge of B
    groups with one adjacent room and construction fails naming one of them. The
    union is blind to it: A alone already covers the region, so the outline has
    no hole and the uncovered area is zero.

    Measured as the summed ring areas less the area of their union, which is
    zero for a partition and counts each extra covering once.

    The rings are read through `repaired`, the same repair `tile` reads them
    through, because the two measures are halves of one question and a ring
    either claims a lobe for both of them or for neither.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    usable = [Polygon(ring) for ring in rings if len(ring) >= 3]
    polys = [repaired(poly) for poly in usable]
    if not polys:
        return 0.0
    merged = unary_union(polys)
    return max(sum(poly.area for poly in polys) - merged.area, 0.0)


def contained_rings(rings: list[Ring], share: float = 0.99) -> list[tuple[int, int]]:
    """Pairs (inner, outer) where `share` of ring `inner` lies inside ring `outer`.

    Names which rooms the overlap is made of, so a refusal can say how many
    rooms another room swallowed rather than only how much area was claimed
    twice. Quadratic in the number of rings and only worth running once the
    overlap is known to be non-zero, which is how the adapter calls it.
    """
    from shapely.geometry import Polygon

    polys = []
    for index, ring in enumerate(rings):
        if len(ring) < 3:
            continue
        polys.append((index, repaired(Polygon(ring))))
    pairs: list[tuple[int, int]] = []
    for inner, small in polys:
        if small.area <= 0.0:
            continue
        for outer, large in polys:
            if inner == outer or large.area <= small.area:
                continue
            if small.intersection(large).area / small.area > share:
                pairs.append((inner, outer))
                break
    return pairs


def dominant_direction(rings: list[Ring]) -> float:
    """Length-weighted circular mean of segment angles modulo 90 degrees.

    Angles are multiplied by four so that the 90 degree period maps onto a full
    circle before averaging, then divided back. This is the same estimator the
    dataset format report used to separate whole-building rotation from genuine
    non-orthogonality, so the residual numbers here are comparable to it.
    """
    sin_sum = 0.0
    cos_sum = 0.0
    for ring in rings:
        for (x0, y0), (x1, y1) in segments(ring):
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length == 0.0:
                continue
            angle = math.atan2(dy, dx)
            sin_sum += length * math.sin(4.0 * angle)
            cos_sum += length * math.cos(4.0 * angle)
    if sin_sum == 0.0 and cos_sum == 0.0:
        return 0.0
    return math.atan2(sin_sum, cos_sum) / 4.0


def rotate_ring(ring: Ring, angle: float, origin: Point = (0.0, 0.0)) -> Ring:
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    ox, oy = origin
    out: Ring = []
    for x, y in ring:
        dx, dy = x - ox, y - oy
        out.append((ox + dx * cos_a - dy * sin_a, oy + dx * sin_a + dy * cos_a))
    return out


def off_axis_fraction(rings: list[Ring], threshold_deg: float = 2.0) -> float:
    """Fraction of total wall length more than `threshold_deg` off the axes."""
    total = 0.0
    off = 0.0
    limit = math.radians(threshold_deg)
    for ring in rings:
        for (x0, y0), (x1, y1) in segments(ring):
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length == 0.0:
                continue
            residual = math.atan2(dy, dx) % (math.pi / 2.0)
            residual = min(residual, math.pi / 2.0 - residual)
            total += length
            if residual > limit:
                off += length
    return 0.0 if total == 0.0 else off / total


def quantize_ring(ring: Ring, step: float) -> Ring:
    """Snap every vertex to a multiple of `step`, then re-close the ring.

    Quantizing can collapse a short edge to zero length, so the result goes back
    through close_ring rather than being returned raw.
    """
    if step <= 0.0:
        return close_ring(ring)
    snapped = [(round(x / step) * step, round(y / step) * step) for x, y in ring]
    return close_ring(snapped)


def bounds(ring: Ring) -> tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def centroid(ring: Ring) -> Point:
    area = signed_area(ring)
    if abs(area) < 1e-12:
        n = max(len(ring), 1)
        return (sum(p[0] for p in ring) / n, sum(p[1] for p in ring) / n)
    cx = cy = 0.0
    for i in range(len(ring)):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % len(ring)]
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    return (cx / (6.0 * area), cy / (6.0 * area))
