"""The dataset-neutral plan record.

Every loader in planaudit.datasets yields `Plan` objects and nothing else. The
grading adapter consumes `Plan` and knows nothing about RPLAN or MSD. That is
the whole point of this module: one representation per concept, so a third
dataset costs a loader and no adapter change.

Units. Coordinates are carried in the source dataset's own units, and the loader
records which through `units_per_meter`. RPLAN lives on a 256 pixel grid whose
metric scale is not published, so RPLAN plans declare an assumed scale and the
adapter applies it. MSD geometry is already in meters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

Point = tuple[float, float]
Ring = list[Point]


@dataclass(frozen=True)
class Room:
    """One enclosed space, as a closed ring plus the loader's category string.

    `category` is the dataset's own vocabulary spelled out, never a solver tag.
    Translation into the solver's RoomTag vocabulary is the adapter's job and is
    counted there, so that an unmapped category is visible as a number rather
    than silently folded into a default.
    """

    ring: Ring
    category: str
    source_index: int


@dataclass(frozen=True)
class AccessEdge:
    """A door or passage between two rooms, by index into `Plan.rooms`.

    `kind` is the dataset's own word for the opening. Negative indices are not
    used; a connection to the outside is expressed by `Plan.entry` instead.
    """

    a: int
    b: int
    kind: str


@dataclass(frozen=True)
class Plan:
    """One floor plan, dataset neutral.

    rooms: the enclosed spaces, in loader order.
    access: doors or passages where the dataset records them, otherwise empty.
    entry: the point where circulation enters the plan, where it is derivable.
    units_per_meter: how many coordinate units make one meter.
    off_axis_fraction: fraction of wall length still more than two degrees off
        the nearest axis after whatever alignment the loader applied. Zero means
        the loader measured it and found none; None means the loader did not
        measure it.
    """

    source: str
    source_id: str
    rooms: list[Room]
    units_per_meter: float
    # The smallest length the plan's source can tell apart, in meters. It is not
    # the same fact as `units_per_meter`, which converts this plan's own
    # coordinates, and the two part company wherever a source states metric
    # geometry that was resolved at a coarser step: MSD's coordinates are meters
    # while its partition resolves the floor onto a 0.38 m grid. For a plan
    # traced from a mask the two agree, because there the native unit is the
    # pixel, and the reader still states it rather than leaving the adapter to
    # infer a source frame it cannot see.
    #
    # The adapter reads it as the width below which a gap between two rooms is
    # the source failing to express a shared wall rather than a void in the
    # floor. A plan whose geometry is exact, the synthetic fixture among them,
    # states 0.0, and the adapter still widens that by one quantisation step,
    # so the fixture is held to a gap of two steps rather than to exact tiling.
    source_pixel_m: float
    # The extent within which floor may be invented, where the step that built
    # this room set established one, and None where it did not.
    #
    # Closing a seam hands a room floor no room covered, and the width rule says
    # how wide that floor may be but not where it may lie. For most sources
    # there is nothing better to say: the rooms are all the reader has, so the
    # floor they bound is the plan and a seam between two of them is inside it
    # by construction. MSD is the exception, because its rooms arrive from a
    # partition that had to decide where the floor was before it could assign
    # anything, and that decision is a fact about the plan which the room set
    # alone cannot carry. Reading it back off the rooms is not the same fact:
    # the rooms fill the extent, so growing them recovers something larger.
    #
    # Stated as a ring rather than a polygon, because that is what this module's
    # geometry is and because the extent every producer of it has is one simple
    # outline with no holes.
    floor_extent: Optional[Ring] = None
    access: list[AccessEdge] = field(default_factory=list)
    entry: Optional[Point] = None
    off_axis_fraction: Optional[float] = None
    notes: dict[str, object] = field(default_factory=dict)

    def categories(self) -> list[str]:
        return [room.category for room in self.rooms]
