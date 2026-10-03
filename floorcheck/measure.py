"""The geometric quantities the rungs read.

Each is defined by the sentence of the specification that names it, quoted at the
function. Nothing here is a port of anything: the specification states what each
quantity is in closed form, and closed form is what these compute.
"""

from __future__ import annotations

import math

from shapely.geometry import Polygon

from .model import Ring


def area(ring: Ring) -> float:
    return Polygon(ring).area


def perimeter(ring: Ring) -> float:
    return Polygon(ring).length


def polsby_popper(ring: Ring) -> float:
    """Section 8.2: "A room's score is the Polsby-Popper ratio, four pi times area
    over perimeter squared, reported on the wire as `ppScore`."

    Section 8.2 also gives three exact anchors, which
    ``tests/floorcheck/test_rungs.py`` checks this function against: "a square
    scores pi over four, 0.7854; a two-to-one rectangle scores 0.6981; a
    three-to-one rectangle scores 0.5890."
    """
    length = perimeter(ring)
    if length <= 0.0:
        return 0.0
    return 4.0 * math.pi * area(ring) / (length * length)


def dominant_direction(rings: list[Ring]) -> float:
    """The plan's dominant wall direction, in radians, from section 2.5.

    "the dominant wall direction is estimated as the length-weighted circular mean
    of segment angles modulo 90 degrees, with angles multiplied by four to map the
    90 degree period onto a full circle, and that angle is subtracted before any
    deviation from the nearest axis is measured"

    Applied per plan, as section 2.5 says ("The correction, applied per plan"), not
    per room, so that a rotated building does not have each of its rooms corrected
    to a different angle.
    """
    sin_sum = 0.0
    cos_sum = 0.0
    for ring in rings:
        count = len(ring)
        for index in range(count):
            (x0, y0), (x1, y1) = ring[index], ring[(index + 1) % count]
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length <= 0.0:
                continue
            angle = math.atan2(dy, dx)
            sin_sum += length * math.sin(4.0 * angle)
            cos_sum += length * math.cos(4.0 * angle)
    if sin_sum == 0.0 and cos_sum == 0.0:
        return 0.0
    return math.atan2(sin_sum, cos_sum) / 4.0


def rotate(ring: Ring, angle: float) -> Ring:
    if angle == 0.0:
        return ring
    cosine, sine = math.cos(-angle), math.sin(-angle)
    return tuple((x * cosine - y * sine, x * sine + y * cosine) for x, y in ring)


def bounding_sides(ring: Ring, angle: float = 0.0) -> tuple[float, float]:
    """The room's bounding-box sides, short first, in the plan's dominant frame.

    `floorcheck/category-statistics.md`: "`aspect` is the room's bounding
    box long side over short side, so 1.0 is square. `min width` is the shorter
    bounding-box dimension."
    """
    if not ring:
        # A room whose ring section 2.2 collapsed reaches here empty.
        # It spans nothing, and rung 1 has already rejected the plan at
        # it; an observation beside the ladder must not raise on it.
        return (0.0, 0.0)
    turned = rotate(ring, angle)
    xs = [point[0] for point in turned]
    ys = [point[1] for point in turned]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return (min(width, height), max(width, height))


def squareness(ring: Ring, angle: float = 0.0) -> float:
    """The quantity rung 3 compares against `propReq`.

    Section 8.3: "the check rejects when a room's squareness, the short side over
    the long side of its unrotated bounding box, is below its floor. A description
    of it as long over short is backwards and inverts any value taken from it."

    So this is the reciprocal of the `aspect` column of the category statistics,
    which is the same direction `dataset-p05` emits its requirement in.
    """
    short, long = bounding_sides(ring, angle)
    if long <= 0.0:
        return 0.0
    return short / long


def min_width(ring: Ring, angle: float = 0.0) -> float:
    """The shorter bounding-box dimension, which is what the corpus tables call
    "min width" and what the `dataset-p05` width requirement is a percentile of.

    Section 8.7 puts `minWidth` off the ladder: "The solver applies the width
    requirement while it constructs a layout, not as a check on a finished plan",
    rejecting a span shorter than `minWidth` less `abstol` (section 1.4), which is a
    construction-time test on a cut's cross length rather than on a finished
    room's bounding box. Log finding 18 records that D3 has no access to a cross
    length and uses the bounding-box width the corpus tables use, and that this is
    why the result is reported as an observation rather than as a rung.
    """
    return bounding_sides(ring, angle)[0]
