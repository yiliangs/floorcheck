"""Where a plan's entry lands on its outline.

Specification 2.6, "The entry a converter writes": where the source states an
entry, the converter's entry is the point on the plan's outline nearest to it,
and the distance moved is provenance; where the source states none, the entry is
the midpoint of the outline's longest edge and is marked synthetic. Generated
sources draw the door as a box outside the wall and state its centre, so an
entry off the outline is the door's position and is projected, never refused and
never capped.

The adapter places the solver's entry door from this, and `tools.export_plans`
writes the checker's entry from it, so the two sides of the oracle-agreement
measurement read one rule. It takes the entry already in metres: the scale
conversion is the reader's work (section 2.6); handing this projection an entry
in source units against a metre outline misplaces it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from planaudit import geometry as geo


@dataclass(frozen=True)
class EntryProjection:
    edge: int
    """The index of the outline segment the entry lands on."""
    station: float
    """Where along that segment, from 0 at its start to 1 at its end."""
    point: geo.Point
    """The landed entry, in the outline's frame."""
    offset_m: float
    """How far the stated entry moved to reach `point`; 0.0 when synthetic."""
    synthetic: bool
    """True when the plan stated no entry and the longest edge took it."""


def project_entry(outline: geo.Ring, entry: Optional[geo.Point]) -> EntryProjection:
    """The nearest point of `outline` to `entry`, or its longest edge's midpoint.

    `entry` must be in the same frame as `outline`, metres for every caller.
    """
    best: Optional[tuple[int, float, float]] = None
    for index, (p, q) in enumerate(geo.segments(outline)):
        length = math.hypot(q[0] - p[0], q[1] - p[1])
        if length == 0.0:
            continue
        if entry is None:
            if best is None or -length < best[2]:
                best = (index, 0.5, -length)
            continue
        t = ((entry[0] - p[0]) * (q[0] - p[0]) + (entry[1] - p[1]) * (q[1] - p[1])) / length**2
        t = max(0.0, min(1.0, t))
        dist = math.hypot(p[0] + t * (q[0] - p[0]) - entry[0], p[1] + t * (q[1] - p[1]) - entry[1])
        if best is None or dist < best[2]:
            best = (index, t, dist)
    if best is None:
        raise ValueError("an outline with no segment of positive length carries no entry")
    index, t, dist = best
    p, q = list(geo.segments(outline))[index]
    point = (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
    return EntryProjection(
        edge=index,
        station=t,
        point=point,
        offset_m=0.0 if entry is None else dist,
        synthetic=entry is None,
    )
