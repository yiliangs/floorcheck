"""Every plan the D1 run grades, restated in floorcheck's plan schema.

The oracle-agreement measurement needs one population graded twice, once by the
adapter and the solver and once by the public checker. The two read different
objects: `planaudit.plan.Plan`, which the dataset loaders produce, and the JSON
document `floorcheck.model.plan_from_dict` defines. This module is the only
place that translates between them, and it translates and nothing else.

What that means in practice is that the export carries geometry across a change
of units and a change of container, and performs no cleaning of its own. The
adapter quantises a ring on `joinTol` before the solver sees it; floorcheck
quantises on its own step inside `hygiene`. If the export quantised as well, a
disagreement traceable to the two steps would have been hidden by a third, and
the measurement would be of the export rather than of the two implementations.
So the rings leave here in metres, at the precision the loader stated them, and
each side applies its own hygiene.

Entry is the one exception, and only for the single point `planaudit.entry`
already owns: specification 2.6 makes the converter's entry the point on the
plan's outline nearest the stated entry (or the longest edge's midpoint where
none was stated), and the grading adapter and this module both place a door
from it, so the oracle-agreement measurement needs one rule for where that
point lands, not two implementations of the same projection. This module builds
its own outline from the exported metre rings, independent of the adapter's
tiling, quantisation and door machinery, unary-unions the room polygons and
takes the exterior of the largest part, and calls `planaudit.entry.project_entry`
on it. What travels is only that shared function, never the adapter's outline
or its report.

Units. A `planaudit` plan states its coordinates in its own frame and carries
`units_per_meter` to convert them. floorcheck's schema carries no such field:
section 2 of the specification fixes an input model in metres. So every
coordinate is divided by `units_per_meter` on the way out, which is the same
scaling the adapter's metre conversion applies, less its quantisation.

The source pixel is a different fact and travels separately. Section 6.3 says it
is "stated by the reader, not inferred", and `floorcheck.model.Plan.source_pixel_m`
reads `meta.sourcePixelM` before it falls back to its own table of sources. The
export always states it, which is what lets a source floorcheck's table does not
name be graded at all.

What the schema cannot carry is recorded rather than worked around, because each
omission is a finding for the oracle-agreement measurement:

* `AccessEdge.kind`. floorcheck's `adjacency` is a pair of room indices and
  nothing else, so the kind of opening an edge represents is dropped. The kinds
  ride along in `meta.accessKinds` so a reader can see what was lost, and
  floorcheck never looks at them.
* `Plan.off_axis_fraction` and `Plan.notes`. Provenance the adapter reads and
  the schema has no field for. `notes` is not carried; the one entry the run
  reads, the MSD partition area error, goes to `meta.partitionAreaError`.
* `Room.source_index`. Room order is preserved instead, so an adjacency index
  means the same room on both sides.

Usage mirrors the grading run's own source selection, so the same flags
select the same population:

    python -m tools.export_plans --msd 4167 --msd-units 4000 --rplan 0 --out DIR
    python -m tools.export_plans --baseline houseganpp=4000 --out DIR
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from planaudit import geometry as geo
from planaudit import sources
from planaudit.datasets import generated, msd, rplan
from planaudit.entry import project_entry
from planaudit.plan import Plan

# The floorcheck source token for each group the grading run names. The two
# vocabularies were settled independently and do not coincide: the run names a
# group after the directory its samples live in, and section 6.3's table names a
# source after the system that produced it. Only `rplan` spells the same on both
# sides.
#
# `housediffusion` has no entry in `floorcheck.constants.SOURCE_PIXEL_M` at all,
# which is finding 16 of the implementation log repeating itself one source
# further along: section 2.4 omits a source that section 6.3 then grades. It
# reaches floorcheck as `housediffusion` and is graded on the pixel this export
# states, which is the mechanism section 6.3 provides for exactly this case.
SOURCE_TOKEN: dict[str, str] = {
    "msd": "msd",
    sources.UNITS_SOURCE: "msd",
    "rplan": "rplan",
    "houseganpp": "housegan++",
    "housegan": "housegan",
    "housediffusion": "housediffusion",
    "gsdiff": "gsdiff",
    "synthetic": "synthetic",
}

_UNSAFE = re.compile(r"[^A-Za-z0-9._#-]")


def source_token(group: str) -> str:
    """The name floorcheck knows a run group by."""
    return SOURCE_TOKEN.get(group, group)


def safe_name(source_id: str) -> str:
    """One plan id as one filename, reversibly enough to read back.

    MSD unit ids carry a `#`, which is a filename character on both platforms
    this runs on; a path separator is not, and an id that carries one would
    otherwise write outside the group directory.
    """
    return _UNSAFE.sub("_", source_id)


def _scaled(point: Iterable[float], scale: float) -> list[float]:
    x, y = point
    return [x * scale, y * scale]


def _ring(ring: Iterable[Iterable[float]], scale: float) -> list[list[float]]:
    return [_scaled(point, scale) for point in ring]


def _outline_m(rings: Iterable[list[list[float]]]) -> Optional[geo.Ring]:
    """The outline `project_entry` needs, from already-metre room rings.

    The union of the rooms and nothing else: no tiling tolerance, no seam
    bridging, no quantisation. The adapter's outline goes through all three
    before its own door is placed, and building this one the same way would
    grade the export against itself rather than against the adapter. `None`
    where no ring carries three finite vertices or no union can be built.

    A generated ring may cross itself. Each room is read as `buffer(0)` repairs
    it, so a crossing ring costs its plan at most the projection and never the
    export: every plan written before this function existed is still written,
    and a plan whose rooms admit no union keeps its stated entry, unprojected.

    Collinear vertices are dropped exactly, the way `planaudit.geometry.tile`
    drops them from its own outline: two rooms sharing a straight run of wall
    otherwise leave the union's ring carrying the room boundary's own split
    point as a vertex, which turns one long edge into two shorter, equal-length
    ones and can hand the longest-edge fallback below a tie that the true
    outline does not have.
    """
    import math

    from shapely.errors import GEOSException
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    polygons = []
    for ring in rings:
        if len(ring) < 3 or not all(math.isfinite(c) for point in ring for c in point):
            continue
        polygon = Polygon(ring)
        polygons.append(polygon if polygon.is_valid else polygon.buffer(0))
    polygons = [polygon for polygon in polygons if not polygon.is_empty]
    if not polygons:
        return None
    try:
        main = geo.largest_part(unary_union(polygons))
    except GEOSException:
        return None
    if main.is_empty:
        return None
    flattened = main.simplify(0.0)
    exterior = main.exterior if flattened.is_empty else flattened.exterior
    return geo.orient_ccw(geo.close_ring([(float(x), float(y)) for x, y in exterior.coords]))


def as_document(plan: Plan, group: str, corpus: str) -> dict[str, Any]:
    """One `planaudit` plan as the JSON document floorcheck's schema defines.

    `corpus` is not a field of the schema. It decides which fit the per-token
    requirements are read from, and `floorcheck.report.check` infers it from the
    source when it is not told. The inference is right for the two ground-truth
    corpora and wrong for every generated baseline, which answers to RPLAN's fit
    because a generated sample is an attempt at an RPLAN plan
    (`planaudit.sources.own_corpus`). Carrying it in `meta` keeps the exported
    document self-describing, and the agreement tool passes it to `check`
    explicitly rather than letting the inference run.

    Every MSD room that belongs to an apartment carries that apartment's id as
    `unit` (`planaudit.sources.unit_labels`): a whole floor's rooms the
    apartments `planaudit.units.assign` derives, an apartment's rooms its own
    id. The checker grades a plan stating more than one apartment against the
    whole-floor floors, so the oracle, which selects its floors through the same
    labels, and the checker see the same kind.
    """
    scale = 1.0 / plan.units_per_meter if plan.units_per_meter else 1.0
    rooms: list[dict[str, Any]] = []
    for room, unit in zip(plan.rooms, sources.unit_labels(group, plan)):
        entry: dict[str, Any] = {"category": room.category, "ring": _ring(room.ring, scale)}
        if unit is not None:
            entry["unit"] = unit
        rooms.append(entry)
    document: dict[str, Any] = {
        "name": f"{group}/{plan.source_id}",
        "source": source_token(group),
        "rooms": rooms,
    }
    if plan.floor_extent is not None:
        document["floorExtent"] = _ring(plan.floor_extent, scale)
    outline = _outline_m(room["ring"] for room in document["rooms"])
    metre_entry = tuple(_scaled(plan.entry, scale)) if plan.entry is not None else None
    if outline is None:
        if metre_entry is not None:
            document["entry"] = list(metre_entry)
    else:
        landed = project_entry(outline, metre_entry)
        document["entry"] = list(landed.point)
        document["entrySynthetic"] = landed.synthetic
        document["entryOffsetM"] = landed.offset_m
    if plan.access:
        document["adjacency"] = [[edge.a, edge.b] for edge in plan.access]

    meta: dict[str, Any] = {
        "sourcePixelM": plan.source_pixel_m,
        "unitsPerMeter": plan.units_per_meter,
        "corpus": corpus,
        "group": group,
        "sourceId": plan.source_id,
        "planauditSource": plan.source,
    }
    if plan.access:
        meta["accessKinds"] = [edge.kind for edge in plan.access]
    if plan.off_axis_fraction is not None:
        meta["offAxisFraction"] = plan.off_axis_fraction
    drift = plan.notes.get("partition_room_area_error")
    if drift:
        meta["partitionAreaError"] = float(drift)
    document["meta"] = meta
    return document


def plans(
    group: str, stream: Iterable[object]
) -> Iterator[tuple[Plan, Optional[str]]]:
    """Every plan of one run group, with the loader failure that replaced it.

    The stream is the grading run's own, wrapped in the same `sources.safe` it
    uses, so the population is the run's by construction and not by a second
    reading of the same directories. A stream item that arrives as an `Outcome`
    is a loader failure the run counted and no plan exists for; it is yielded as
    a bare reason so the export's manifest can account for every id the run saw.
    """
    for item in sources.safe(group, stream):
        if isinstance(item, sources.Outcome):
            yield None, f"{item.source_id}: {item.detail}"
            continue
        yield item.plan, None


def write_group(group: str, stream: Iterable[object], out: Path) -> dict[str, Any]:
    """One group's plans, one JSON document each, under `out/<group>/`."""
    corpus = sources.own_corpus(group)
    directory = out / safe_name(group)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    unloadable: list[str] = []
    for plan, failure in plans(group, stream):
        if plan is None:
            unloadable.append(failure)
            continue
        document = as_document(plan, group, corpus)
        path = directory / f"{safe_name(plan.source_id)}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        written.append(plan.source_id)
    return {
        "group": group,
        "corpus": corpus,
        "source": source_token(group),
        "directory": str(directory),
        "written": len(written),
        "unloadable": unloadable,
    }


def _streams(args: argparse.Namespace) -> list[tuple[str, Iterable[object], Path, str]]:
    """The groups a run of these flags would grade, in the run's own order.

    Each group comes with the input it reads and the flag that sets it, so an
    export that finds nothing there can say where it looked.
    """
    out: list[tuple[str, Iterable[object], Path, str]] = []
    msd_root = args.msd_root if args.msd_root is not None else msd.default_root()
    if args.msd:
        out.append(("msd", sources.floor_stream(args.msd, root=args.msd_root), msd_root, "--msd-root"))
    if args.msd_units:
        out.append(
            (sources.UNITS_SOURCE, sources.unit_stream(args.msd_units, root=args.msd_root), msd_root, "--msd-root")
        )
    if args.rplan:
        out.append(
            (
                "rplan",
                rplan.iter_plans(
                    limit=args.rplan,
                    path=args.rplan_path,
                    stride=args.rplan_stride,
                    offset=args.rplan_offset,
                ),
                args.rplan_path if args.rplan_path is not None else rplan.default_path(),
                "--rplan-path",
            )
        )
    for name, count in sources.parse_baselines(args.baseline):
        root = args.samples_root / name if args.samples_root else generated.samples_root(name)
        out.append((name, generated.iter_plans(name, root, limit=count), root, "--samples-root"))
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the plans a D1 run grades into floorcheck's plan schema."
    )
    parser.add_argument("--msd", type=int, default=0)
    parser.add_argument("--msd-units", type=int, default=0)
    parser.add_argument("--rplan", type=int, default=0)
    parser.add_argument("--rplan-stride", type=int, default=997)
    parser.add_argument(
        "--rplan-offset",
        type=int,
        default=0,
        help="Start the stride walk at this file-order index, so offset 10 with stride 20"
        " visits 10, 30, ... and is disjoint from the offset-0 sample.",
    )
    parser.add_argument("--baseline", action="append", default=[], metavar="NAME=N")
    parser.add_argument(
        "--msd-root",
        type=Path,
        default=None,
        help="the extracted MSD training split (the directory holding graph_out/)."
        " Default: data/msd/train_extracted beside the package.",
    )
    parser.add_argument(
        "--rplan-path",
        type=Path,
        default=None,
        help="RPLAN's data.mat. Default: data/rplan/extracted/Network/data.mat beside the package.",
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=None,
        help="the directory holding one subdirectory of samples per --baseline NAME."
        " Default: data/samples beside the package.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def merge_manifest(path: Path, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The manifest at `path` with each of `entries` replacing its group's entry.

    A directory is often exported one group at a time, so the manifest is merged
    per group rather than rewritten: a group this run wrote gets this run's entry,
    and every other group keeps the entry of the run that wrote it, in place.
    Each entry carries its own `exportedAt`, so the manifest says which run
    wrote which directory.
    """
    kept = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    fresh = {entry["group"]: entry for entry in entries}
    merged = [fresh.pop(entry.get("group"), entry) for entry in kept]
    return merged + list(fresh.values())


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entries = []
    for name, stream, where, flag in _streams(args):
        entry = {**write_group(name, stream, args.out), "exportedAt": stamp}
        entries.append(entry)
        print(f"{entry['group']}: {entry['written']} plans to {entry['directory']}")
        for reason in entry["unloadable"]:
            print(f"  no plan: {reason}")
        if not entry["written"]:
            print(f"  looked for {name} plans in {where}, set with {flag}")
    path = args.out / "manifest.json"
    path.write_text(json.dumps(merge_manifest(path, entries), indent=2), encoding="utf-8")
    loaded = sum(entry["written"] for entry in entries)
    failed = sum(len(entry["unloadable"]) for entry in entries)
    print(f"exported {loaded} plans, {failed} failed to load")
    # An export that wrote no plan has produced nothing to grade, which a caller
    # checking only the exit status must not read as success.
    return 0 if loaded else 1


if __name__ == "__main__":
    raise SystemExit(main())
