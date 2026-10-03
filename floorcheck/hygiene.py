"""Ring hygiene and quantisation, specification section 2.2.

    "Before anything geometric is asserted, every ring is closed, deduplicated
    and oriented counter-clockwise, and then quantised onto a common step. The
    step is the `joinTol` constant, whose public derivation in Table 1 of the
    provenance table is half of one RPLAN raster cell, 0.035 m. Collinear
    vertices are dropped exactly, never within a tolerance, and only where the
    vertex is collinear in every ring that carries it. A vertex where three rooms
    meet is a real corner for at least one of them, and dropping it from the one
    ring that sees it as flat reopens a gap that the quantisation has just
    closed; a tolerant drop is worse still, because it displaces the edge by the
    tolerance and the neighbour's segment stops lying on it."

Two things that paragraph leaves implied are decided here and recorded as log
finding 3: the collinear drop runs after the quantisation, not before, and
"every ring that carries it" means exact equality of the quantised coordinate.
"""

from __future__ import annotations

from .constants import JOIN_TOL_M
from .model import Point, Ring


def quantise_point(point: Point, step: float = JOIN_TOL_M) -> Point:
    """Snap one point onto the common step.

    Rounding, not truncation: truncation biases every coordinate toward the
    origin by up to a full step, which would open on one side of a shared wall
    exactly the gap section 2.2 says the quantisation exists to close.
    """
    return (round(point[0] / step) * step, round(point[1] / step) * step)


def dedupe(ring: Ring) -> Ring:
    """Drop repeated vertices, including a closing repeat of the first.

    Section 2.2's "deduplicated". Consecutive duplicates only: a ring that
    revisits a coordinate non-consecutively is a pinched or self-touching ring,
    which is a geometry question section 6 handles, not a hygiene one.
    """
    out: list[Point] = []
    for point in ring:
        if not out or out[-1] != point:
            out.append(point)
    while len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return tuple(out)


def signed_area(ring: Ring) -> float:
    """Twice the signed area by the shoelace sum; positive is counter-clockwise."""
    total = 0.0
    count = len(ring)
    for index in range(count):
        x0, y0 = ring[index]
        x1, y1 = ring[(index + 1) % count]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def orient_ccw(ring: Ring) -> Ring:
    """Section 2.2's "oriented counter-clockwise"."""
    return ring if signed_area(ring) >= 0.0 else tuple(reversed(ring))


def is_collinear(previous: Point, vertex: Point, following: Point) -> bool:
    """Exact collinearity, section 2.2's "dropped exactly, never within a tolerance".

    The cross product is compared to zero with no tolerance at all. After
    quantisation every coordinate is an integer multiple of the step, so the
    cross product of the two differences is an exact multiple of the step squared
    and the comparison is meaningful in floating point only if the multiples are
    recovered first. That is what the integer lattice below does: dividing by the
    step and rounding returns each coordinate to the integer it stands for, so
    the cross product is computed in integers and the word "exactly" means what
    it says.
    """
    (ax, ay), (bx, by), (cx, cy) = previous, vertex, following
    return (bx - ax) * (cy - by) - (by - ay) * (cx - bx) == 0


def _lattice(ring: Ring, step: float) -> tuple[tuple[int, int], ...]:
    return tuple((round(x / step), round(y / step)) for x, y in ring)


def _lattice_collinear(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> bool:
    return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) == 0


def quantise_rings(rings: list[Ring], step: float = JOIN_TOL_M) -> list[Ring]:
    """Close, deduplicate, orient and quantise every ring of a plan.

    The collinear drop is deliberately not done here: it needs every ring at
    once, so it is :func:`drop_collinear_across_rings`, which this function's
    caller runs next.
    """
    out: list[Ring] = []
    for ring in rings:
        snapped = tuple(quantise_point(point, step) for point in ring)
        out.append(orient_ccw(dedupe(snapped)))
    return out


def drop_collinear_across_rings(rings: list[Ring], step: float = JOIN_TOL_M) -> list[Ring]:
    """Drop a vertex only where every ring carrying it sees it as flat.

    Section 2.2: "only where the vertex is collinear in every ring that carries
    it. A vertex where three rooms meet is a real corner for at least one of
    them, and dropping it from the one ring that sees it as flat reopens a gap
    that the quantisation has just closed".

    So the decision is taken once per *coordinate*, over the whole plan, and then
    applied to every ring. A vertex that any ring treats as a corner survives in
    all of them, which is what keeps the neighbour's segment lying on this one's
    edge.

    Log finding 3: "carries it" is read as exact equality of the quantised
    coordinate, which is the only reading that needs no tolerance.
    """
    lattices = [_lattice(ring, step) for ring in rings]

    flat_everywhere: dict[tuple[int, int], bool] = {}
    for lattice in lattices:
        count = len(lattice)
        if count < 3:
            for vertex in lattice:
                flat_everywhere[vertex] = False
            continue
        for index, vertex in enumerate(lattice):
            previous = lattice[index - 1]
            following = lattice[(index + 1) % count]
            flat = _lattice_collinear(previous, vertex, following)
            # A ring that sees the vertex as a corner vetoes the drop for every
            # ring, so the accumulator is a conjunction seeded True.
            flat_everywhere[vertex] = flat_everywhere.get(vertex, True) and flat

    out: list[Ring] = []
    for ring, lattice in zip(rings, lattices):
        kept = [
            point
            for point, vertex in zip(ring, lattice)
            if not flat_everywhere.get(vertex, False)
        ]
        # A ring every one of whose vertices is flat everywhere is degenerate and
        # is left as it was rather than emptied. It bounds no floor, so section 6
        # measures the plan without it and rung 1 of section 8.1 rejects it.
        out.append(tuple(kept) if len(kept) >= 3 else ring)
    return out


def hygiene(rings: list[Ring], step: float = JOIN_TOL_M) -> list[Ring]:
    """Section 2.2 end to end, in the order the section states."""
    return drop_collinear_across_rings(quantise_rings(rings, step), step)
