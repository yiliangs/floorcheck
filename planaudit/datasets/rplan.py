"""RPLAN loader: room polygons and categories out of Network/data.mat.

Why the .mat and not the rasters. The dataset format report established that the
Zenodo mirror's PNG assets carry only the building footprint outline, three
colours and two white components, with no room segmentation at all. Room
polygons and categories exist only in the MATLAB struct array, so that is what
this loader reads: 80,788 structs, each with `rType` (one category code per
room), `rBoundary` (one polygon per room, same 0..255 frame as the PNGs),
`rEdge` (room-to-room adjacency with an undocumented type code), and `boundary`
(the building outline, with the front door marked).

Ring closure. Both `boundary` and `rBoundary` are stored as OPEN polylines: the
format probe found 0 of 13,596 sampled room polygons with a repeated first
vertex. They are treated as implicitly closed rings, which is also what the
solver's snapshot rings are.

Entry. `boundary` rows are (x, y, dir, isNew). The fourth column is binary and
marks the front door: the rows carrying 1 are the door's two endpoints, so the
entry point is their midpoint. A plan without two such rows gets entry None.

Scale. RPLAN publishes no metric scale for its 256 pixel frame. This loader
declares the frame, RPLAN_FRAME_METERS, and records it on the Plan. The scale
matters to every rung the solver measures in meters, so D2's provenance
table has to rule on it rather than inherit it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Optional

from planaudit import geometry as geo
from planaudit.plan import AccessEdge, Plan, Room

# Inferred, not shipped with this mirror. The archive carries no legend, so this
# is the published RPLAN / Graph2Plan category order. The evidence that it is the
# right order is distributional: over a 2,000 plan sample code 0 occurs exactly
# once per plan (every RPLAN unit has one living room), codes 3 and 7 occur more
# than once per plan on average (bathrooms and second bedrooms), and codes 4, 8,
# 10, 11 and 12 are rare. Codes 13 and above never appear in the room arrays
# because they name walls and openings rather than rooms.
R_TYPE_NAMES: dict[int, str] = {
    0: "livingroom",
    1: "masterroom",
    2: "kitchen",
    3: "bathroom",
    4: "diningroom",
    5: "childroom",
    6: "studyroom",
    7: "secondroom",
    8: "guestroom",
    9: "balcony",
    10: "entrance",
    11: "storage",
    12: "walkin_closet",
}

# The full 256 pixel frame spans this many meters. SETTLED in D2, not assumed:
# Wu et al. section 4 states that each plan sits in a squared region of 18m by
# 18m before being converted to a 256 by 256 image, and the corpus confirms it
# three ways. Converted at 18/256 = 0.0703125 m per pixel, the measured footprint
# areas of all 80,788 records run 62.36 to 119.98 m2, which sits inside the
# paper's own 60 to 120 m2 filter and touches its ceiling to within 0.015 per
# cent; the modal front door is 0.84 m against a code clear width of 0.813 m; and
# bathroom widths run 1.27 to 1.76 m.
RPLAN_FRAME_METERS = 18.0
RPLAN_FRAME_PIXELS = 256.0

_CACHE: dict[str, Any] = {}


def default_path() -> Path:
    here = Path(__file__).resolve().parents[2]
    return here / "data" / "rplan" / "extracted" / "Network" / "data.mat"


def _records(path: Optional[Path] = None):
    """The 80,788 element struct array, loaded once per process.

    data.mat is 93.6 MB and scipy takes seconds over it, so repeated loads would
    dominate the D1 run. The cache is keyed on the resolved path.
    """
    path = Path(path) if path is not None else default_path()
    key = str(path.resolve())
    if key not in _CACHE:
        from scipy.io import loadmat

        _CACHE[key] = loadmat(str(path), struct_as_record=False, squeeze_me=True)["data"]
    return _CACHE[key]


def count(path: Optional[Path] = None) -> int:
    return int(len(_records(path)))


def _category(code: object) -> str:
    try:
        key = int(code)
    except (TypeError, ValueError):
        return "rplan_unknown_none"
    return R_TYPE_NAMES.get(key, f"rplan_unknown_{key}")


def _ring(raw: Any) -> geo.Ring:
    if raw is None:
        return []
    points = [(float(p[0]), float(p[1])) for p in raw] if len(raw) else []
    return geo.close_ring(points)


def _entry_from_boundary(boundary: Any) -> Optional[geo.Point]:
    marked = [row for row in boundary if int(row[3]) == 1]
    if len(marked) < 2:
        return None
    ax, ay = float(marked[0][0]), float(marked[0][1])
    bx, by = float(marked[1][0]), float(marked[1][1])
    return ((ax + bx) / 2.0, (ay + by) / 2.0)


def _access_edges(r_edge: Any, room_count: int) -> list[AccessEdge]:
    """rEdge rows are (room_i, room_j, code). The code is undocumented.

    The archive documents nothing about column 2; the observed range is 0 to 9.
    It plausibly distinguishes wall from door adjacency, but nothing in the
    mirror confirms that, so every edge is carried through as an unknown kind
    and the adapter does not read the code. D2 task 1 owns resolving it.
    """
    out: list[AccessEdge] = []
    if r_edge is None or len(r_edge) == 0:
        return out
    rows = r_edge if hasattr(r_edge[0], "__len__") else [r_edge]
    for row in rows:
        a, b, code = int(row[0]), int(row[1]), int(row[2])
        if 0 <= a < room_count and 0 <= b < room_count and a != b:
            out.append(AccessEdge(a, b, f"rplan_edge_unknown_{code}"))
    return out


def load_index(index: int, path: Optional[Path] = None) -> Plan:
    """Load RPLAN plan `index` in the file's own order."""
    record = _records(path)[index]
    r_type = record.rType
    r_type = r_type if hasattr(r_type, "__len__") else [r_type]
    raw_rooms = record.rBoundary
    raw_rooms = raw_rooms if isinstance(raw_rooms, (list, tuple)) or hasattr(raw_rooms, "dtype") else [raw_rooms]

    rooms: list[Room] = []
    empty = 0
    for i in range(len(r_type)):
        ring = _ring(raw_rooms[i] if i < len(raw_rooms) else None)
        if len(ring) < 3:
            empty += 1
            continue
        rooms.append(Room(geo.orient_ccw(ring), _category(r_type[i]), i))
    if not rooms:
        raise ValueError(f"RPLAN plan {index} has no usable room geometry")

    return Plan(
        source="rplan",
        source_id=str(record.name),
        rooms=rooms,
        units_per_meter=RPLAN_FRAME_PIXELS / RPLAN_FRAME_METERS,
        source_pixel_m=RPLAN_FRAME_METERS / RPLAN_FRAME_PIXELS,
        access=_access_edges(getattr(record, "rEdge", None), len(rooms)),
        entry=_entry_from_boundary(record.boundary),
        off_axis_fraction=geo.off_axis_fraction([room.ring for room in rooms]),
        notes={
            "record_index": index,
            "empty_room_polygons": empty,
            "frame_meters": RPLAN_FRAME_METERS,
        },
    )


def load_one(source_id: str, path: Optional[Path] = None) -> Plan:
    """Load by the struct's own `name` field, scanning in file order."""
    records = _records(path)
    for index in range(len(records)):
        if str(records[index].name) == str(source_id):
            return load_index(index, path)
    raise KeyError(f"RPLAN plan {source_id} not found")


def iter_plans(
    limit: Optional[int] = None,
    path: Optional[Path] = None,
    stride: int = 1,
    offset: int = 0,
) -> Iterator[Plan]:
    """Yield plans in file order. `stride` spreads a small sample over the file.

    `offset` starts the stride walk away from index 0, so an offset sample and
    the offset-0 sample at the same stride visit disjoint indices: offset 10 with
    stride 20 visits 10, 30, 50, ..., which never coincides with 0, 20, 40, ...
    """
    total = count(path)
    taken = 0
    for index in range(offset, total, max(stride, 1)):
        if limit is not None and taken >= limit:
            return
        yield load_index(index, path)
        taken += 1
