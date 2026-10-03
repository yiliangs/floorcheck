"""The tiling rule of specification section 6, and the ingest outcomes it produces.

Section 6.2: "A plan whose rooms do not tile the floor they bound is refused at
ingest, and the refusal is its own recorded outcome, counted apart from a solver
construction error and apart from a gate rejection. Tiling has two halves and the
solver reads both the same way."

An ingest refusal is therefore NOT a gate rung. Section 7.3: "A build refusal, an
untiled-rooms refusal and an oracle error all happen before or instead of a
verdict, so they are not rungs and they keep their own rows." This module returns
an :class:`Ingest` whose ``refusal`` is a named outcome, and the ladder never
runs when one is present.

The two halves, in the order section 6 states them:

* the uncovered half, section 6.3, governed by a WIDTH at one source pixel, with
  a backstop on the share of floor handed to rooms as closed seam;
* the twice-claimed half, section 6.4, governed by a SHARE at one part per
  million.

Section 6.3: "The two halves are measured differently, and the difference is the
substance of the 2026-09-15 ruling. The uncovered half is governed by a width and
the twice-claimed half by a share."
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from .constants import CLOSED_SEAM_SHARE, JOIN_TOL_M, OVERLAP_SHARE
from .hygiene import hygiene
from .model import Plan, Ring

# Section 6.2: "by finding every pair where more than 99 per cent of one ring's
# area lies inside another's". This turns a refusal from "this much area was
# claimed twice" into "this many rooms were swallowed by another room".
CONTAINMENT_SHARE = 0.99


@dataclass
class Ingest:
    """What section 6 decides about one plan, before any rung runs."""

    rings: tuple[Ring, ...]
    outline: Polygon | None = None
    refusal: str | None = None
    detail: str = ""
    # Section 6.2: "The refusal carries its measurements on the exception rather
    # than in its message text, so that a tally never has to parse prose back
    # into numbers, and so that it can be counted as one cause."
    measurements: dict[str, float] = field(default_factory=dict)
    swallowed: tuple[tuple[int, int], ...] = ()
    seams_closed: int = 0

    @property
    def refused(self) -> bool:
        return self.refusal is not None


def _polygon(ring: Ring) -> Polygon:
    if len(ring) < 3:
        # Section 2.2 can collapse a schema-valid ring to fewer than three
        # vertices, which bounds no floor. It is carried as a room with no
        # polygon, so section 6 measures the floor without it and section 8.1's
        # rung 1 rejects the plan at this room ("room i is not a ring"), rather
        # than the checker raising, which section 2.6 forbids.
        return Polygon()
    polygon = Polygon(ring)
    if polygon.is_valid:
        return polygon
    # A self-intersecting ring is repaired to the nearest valid geometry so that
    # the area arithmetic below means something. Section 6 never says what to do
    # with one; log finding 9.
    #
    # A bowtie repairs to two lobes, and section 2.1's "One room is one ring"
    # leaves no way to carry both, so the larger lobe is the room and the smaller
    # is discarded. The union below is taken over the repaired rooms, as
    # `planaudit.geometry.tile` takes it, so a dropped lobe is measured only
    # where other rooms enclose it as a hole; on the edge of the plan it lies
    # outside the outline and nothing measures it. Returning the
    # pair would make a room two rings, which is the shape the specification
    # exists to rule out.
    repaired = polygon.buffer(0)
    if isinstance(repaired, Polygon):
        return repaired
    largest = _largest(repaired)
    return largest if largest is not None else Polygon()


def _parts(geometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return [piece for piece in getattr(geometry, "geoms", []) if isinstance(piece, Polygon)]


def _largest(geometry) -> Polygon | None:
    parts = _parts(geometry)
    return max(parts, key=lambda part: part.area) if parts else None


def _survives_erosion(region: Polygon, tolerance: float) -> bool:
    """Is this region wider than the tolerance?

    Section 6.3: "The test is whether the hole survives being eroded by half the
    tolerance, so a hole exactly one pixel wide is closed rather than refused."

    "At most one pixel, not narrower than one pixel. The distinction decides
    whether the rule does anything at all."
    """
    if tolerance <= 0.0:
        # Section 6.3's table gives the synthetic fixture a pixel of 0.0 and
        # holds it "to tiling exactly", so nothing is ever closed.
        return True
    return not region.buffer(-tolerance / 2.0).is_empty


def _bounding_rooms(region: Polygon, polygons: list[Polygon], tolerance: float) -> list[int]:
    """The rooms that bound a region, longest shared boundary first.

    Section 6.3, "Which room takes it": "The candidates are the rooms whose
    boundary the region touches, ordered by how much of the region each of them
    bounds, most first. How much a room bounds is the length of the region's
    boundary it shares, not the area it shares with the region grown by any
    tolerance." A room within half the tolerance of the region is a
    candidate; one that shares no length with it ranks after every room that
    does, nearest first.
    """
    grown = region.buffer(tolerance / 2.0 if tolerance > 0 else JOIN_TOL_M)
    edge = region.boundary.buffer(JOIN_TOL_M / 2.0)
    touching: list[tuple[float, int]] = []
    near: list[tuple[float, int]] = []
    for index, polygon in enumerate(polygons):
        if not polygon.intersects(grown) or polygon.intersection(grown).area <= 0.0:
            continue
        length = polygon.boundary.intersection(edge).length
        if length > 0.0:
            touching.append((-length, index))
        else:
            near.append((polygon.distance(region), index))
    touching.sort()
    near.sort()
    return [index for _, index in touching] + [index for _, index in near]


def _hand_over(region: Polygon, polygons: list[Polygon], order: list[int]) -> int | None:
    """Give a region to the first bounding room it leaves simply connected.

    Section 6.3, "Which room takes it": "The region goes to the first candidate
    in that order that it leaves simply connected, and to nobody if there is no
    such candidate, in which case the region is not closed at all, the floor
    under it stays uncovered, and section 6.2 measures it as uncovered floor like
    any other. A candidate is left simply connected when the union of that room
    with the region comes back as one polygon carrying no interior ring." The
    same section records why the plain rule fails: "Where a region runs around a
    third room, the room bounding most of the region comes back from the union as
    a ring with that third room punched out of it as an interior; the write-back
    drops the interior, and the first room silently acquires the third room's
    floor, which is then claimed twice."

    This implementation reached the rule before the specification did. Log
    finding 8 is the reason the specification now states it this way: the plain
    rule, stated with no condition until 2026-09-16, is known from `STATUS.md`
    W33 to manufacture the 73 spurious overlaps that section 6.4 would then
    refuse.
    """
    for index in order:
        merged = unary_union([polygons[index], region])
        if isinstance(merged, Polygon) and not merged.interiors:
            return index
    return None


def _uncovered_regions(polygons: list[Polygon], union) -> list[Polygon]:
    """Floor no room covers: the union's holes, plus the channels between its parts.

    Section 6.1: "The union is taken as the outer ring of the largest part, with
    two counts reported alongside: how many holes that part carries, and how many
    parts the union came back in. Both say the rings failed to tile one region,
    and structurally they are the same defect seen twice".
    """
    regions: list[Polygon] = []
    for part in _parts(union):
        for interior in part.interiors:
            regions.append(Polygon(interior))
    return regions


def _channel_regions(union, tolerance: float) -> list[Polygon]:
    """The gaps between parts of the union, read off a morphological closing.

    Section 6.1: "An outlying part is separated from the mass by a gap that
    reaches the outside and has no bounded width to erode, so the gap is read off
    a morphological closing of the union at the tolerance instead, and only a
    channel the closing fills that touches two distinct parts of the union is
    taken."
    """
    parts = _parts(union)
    if len(parts) < 2 or tolerance <= 0.0:
        return []
    # Mitre joins rather than round ones, so that the closing of a rectilinear
    # plan has square corners. Section 6.3 records the residue a round band leaves
    # and its size: "the most a bridged channel leaves under the same erosion is
    # 0.0017 square metres, at the corners, where a right-angled band is a pixel
    # and a half across its diagonal". Mitre joins keep that residue at the
    # corners instead of spreading it along the whole seam.
    closed = union.buffer(tolerance / 2.0, join_style="mitre").buffer(
        -tolerance / 2.0, join_style="mitre"
    )
    filled = closed.difference(union)
    channels: list[Polygon] = []
    for region in _parts(filled):
        touching = sum(
            1 for part in parts if part.buffer(tolerance / 10.0).intersects(region)
        )
        if touching >= 2:
            channels.append(region)
    return channels


def ingest(plan: Plan) -> Ingest:
    """Run section 6 over one plan.

    The order is the one section 6.3 states: "Floor that no room covers and that
    is at most one pixel of the plan's own source frame wide is closed into a
    room that bounds it, before the rings are read for tiling and before
    anything else is measured", by the "Which room takes it" rule of the same
    section, and section 6.4: "Closing a seam under section
    6.3 cannot create an overlap, a hole being floor no room covered, so this
    half measures the same quantity whether it runs before or after the closing.
    It is measured after, on the rings the solver will actually see."
    """
    # Section 2.2, run before anything geometric is asserted.
    rings = hygiene([room.ring for room in plan.rooms])
    polygons = [_polygon(ring) for ring in rings]

    source_pixel = plan.source_pixel_m
    if source_pixel is None:
        return Ingest(
            rings=tuple(rings),
            refusal="unknown-source-pixel",
            detail=(
                f"source {plan.source!r} states no smallest length; section 6.3 "
                "requires one and says the pixel is stated by the reader, not inferred"
            ),
        )

    # Section 6.3: "The adapter widens that length by its own quantisation step
    # before applying it, because a length snapped to the grid of section 2.2 is
    # only known to within that grid, and because it gives a plan with no source
    # frame of its own the floating point slack the union arithmetic needs and
    # nothing more."
    tolerance = plan.seam_tolerance_m

    extent = Polygon(plan.floor_extent) if plan.floor_extent else None
    if extent is not None and not extent.is_valid:
        extent = extent.buffer(0)

    union = unary_union(polygons)
    closed_area = 0.0
    seams_closed = 0

    # A hole is judged by erosion and a channel by the closing alone. Section
    # 6.1: a channel "has no bounded width to erode, so the gap is read off a
    # morphological closing of the union at the tolerance instead". Retesting a
    # channel by erosion refuses a one-pixel channel with a bend, whose corner is
    # wider than a pixel across its diagonal; section 6.3 counts that corner
    # residue as already accepted, and `planaudit.geometry._bridge_seams` fills
    # every channel the closing admits.
    holes = [
        region
        for region in _uncovered_regions(polygons, union)
        if not _survives_erosion(region, tolerance)
    ]
    for region in holes + _channel_regions(union, tolerance):
        # Section 6.3, "Where floor may be invented": "Every region either repair
        # would fill is therefore clipped to the plan's floor extent where the
        # plan carries one ... and a plan that carries none is not clipped at all
        # and behaves as before. No clip is the absence of a restriction and not
        # a restriction to nothing." Log finding 10 is why the section now reads
        # "not clipped at all" rather than "clipped to nothing", which read as
        # its own opposite.
        clipped = region.intersection(extent) if extent is not None else region
        for piece in _parts(clipped):
            if piece.area <= 0.0:
                continue
            order = _bounding_rooms(piece, polygons, tolerance)
            target = _hand_over(piece, polygons, order)
            if target is None:
                continue
            merged = unary_union([polygons[target], piece])
            polygons[target] = merged
            closed_area += piece.area
            seams_closed += 1

    union = unary_union(polygons)
    largest = _largest(union)
    if largest is None:
        return Ingest(rings=tuple(rings), refusal="empty-plan", detail="no room has area")

    # Section 6.1: "The union is taken as the outer ring of the largest part".
    outline = Polygon(largest.exterior)
    floor_area = outline.area

    result = Ingest(
        rings=tuple(tuple(polygon.exterior.coords)[:-1] for polygon in polygons),
        outline=outline,
        seams_closed=seams_closed,
    )

    # --- the uncovered half, section 6.3 -----------------------------------
    # Section 6.2: "Measured as the area of the union's holes plus the area of its
    # outlying parts, after the narrow holes have been closed under section 6.3
    # and on the rings the solver will actually see."
    #
    # Section 6.3: "Section 6.2's two terms are the whole measurement, and a
    # channel is not a third term beside them ... Adding a channel as a term of
    # its own would double count an unbridged one." A channel narrow enough to
    # fill has already been filled above, which joins its two parts into one, so
    # a channel that remains shows up here as an outlying part rather than as a
    # term of its own.
    hole_area = sum(region.area for region in _uncovered_regions(polygons, union))
    parts = _parts(union)
    largest_index = max(range(len(parts)), key=lambda i: parts[i].area)
    outlying_area = sum(
        part.area for index, part in enumerate(parts) if index != largest_index
    )
    surviving = hole_area + outlying_area

    result.measurements["uncoveredArea"] = surviving
    result.measurements["holeArea"] = hole_area
    result.measurements["outlyingArea"] = outlying_area
    # Section 6.1: "two counts reported alongside: how many holes that part
    # carries, and how many parts the union came back in".
    result.measurements["holes"] = float(len(parts[largest_index].interiors))
    result.measurements["parts"] = float(len(parts))
    result.measurements["closedSeamArea"] = closed_area
    result.measurements["closedSeamShare"] = closed_area / floor_area if floor_area else 0.0
    result.measurements["floorArea"] = floor_area
    result.measurements["sourcePixelM"] = source_pixel

    if surviving > 0.0:
        # Section 6.3: "The plan is then refused if any uncovered floor survives
        # ... There is no share allowance on the surviving area: it refuses above
        # zero."
        result.refusal = "untiled-rooms"
        result.detail = (
            f"{surviving:.6f} m2 of floor no room covers survives a seam at one "
            f"source pixel ({source_pixel:.6f} m)"
        )
        return result

    if result.measurements["closedSeamShare"] > CLOSED_SEAM_SHARE:
        # Section 6.3: "or if the floor handed to rooms as closed seam exceeds
        # four per cent of the floor". The backstop, not a second tiling test:
        # "The cap is therefore a tripwire against a future widening of the width
        # rule, placed above everything the present rule produces, and it is not a
        # second test of whether a plan tiles."
        result.refusal = "closed-seam-share"
        result.detail = (
            f"{result.measurements['closedSeamShare']:.4f} of the floor was handed "
            f"to rooms as closed seam, above the backstop of {CLOSED_SEAM_SHARE}"
        )
        return result

    # --- the twice-claimed half, section 6.4 -------------------------------
    # "Measured as the summed ring areas less the area of their union, which is
    # zero for a partition and counts each extra covering once."
    twice_claimed = sum(polygon.area for polygon in polygons) - union.area
    result.measurements["twiceClaimedArea"] = max(twice_claimed, 0.0)
    share = (twice_claimed / floor_area) if floor_area else 0.0
    result.measurements["twiceClaimedShare"] = share

    if share > OVERLAP_SHARE:
        result.swallowed = _swallowed_pairs(polygons)
        result.refusal = "rooms-claim-the-same-floor"
        result.detail = (
            f"{twice_claimed:.6f} m2 claimed twice, a share of {share:.3e} of the "
            f"floor, above {OVERLAP_SHARE:.0e}; {len(result.swallowed)} room(s) "
            "lie almost wholly inside another"
        )
        return result

    return result


def _swallowed_pairs(polygons: list[Polygon]) -> tuple[tuple[int, int], ...]:
    """Name the rooms an overlap is made of.

    Section 6.4: "the rooms it is made of are named, by finding every pair where
    more than 99 per cent of one ring's area lies inside another's. That turns a
    refusal from 'this much area was claimed twice' into 'this many rooms were
    swallowed by another room', which is what the tally needs."
    """
    pairs: list[tuple[int, int]] = []
    for inner, small in enumerate(polygons):
        if small.area <= 0.0:
            continue
        for outer, large in enumerate(polygons):
            if inner == outer:
                continue
            if small.intersection(large).area / small.area > CONTAINMENT_SHARE:
                pairs.append((inner, outer))
    return tuple(pairs)
