"""Shared reader for generated baseline samples on disk.

Every baseline sampler under tools/baselines writes the same record, one JSON
object per line, under data/samples/<baseline>/bucket-<n>.jsonl beside a
manifest.json. This module turns one such line into a `Plan`; the per baseline
modules (planaudit.datasets.houseganpp, planaudit.datasets.gsdiff) carry only
the source name, the sample directory and the prose about what that baseline
does and does not emit.

Why one reader and not one per baseline. The thing that differs between two
generative baselines is how a sample is produced, not how it is read. Giving
each a private copy of the same JSON walk would put two representations behind
one concept, and the second one would drift. The samplers carry the difference
and the record schema absorbs it: a baseline that emits no doors writes an empty
`doors` list and an empty `access` list, and a baseline that cannot say where
circulation enters writes `entry: null`. Those are then visible in the loaded
Plan as an empty `access` and `entry is None` rather than being invented here.

Units. Each record states `units_per_meter` for its own native frame, and also
carries the metre conversion of every ring under the `_m` keys so that the
frame arithmetic is auditable from the file alone without re-running the model.
The Plan is built on the native coordinates, as RPLAN's own loader is, and the
adapter applies `units_per_meter`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

from planaudit import geometry as geo
from planaudit.plan import AccessEdge, Plan, Room


def samples_root(baseline: str) -> Path:
    here = Path(__file__).resolve().parents[2]
    return here / "data" / "samples" / baseline


def shard_paths(root: Path) -> list[Path]:
    """Every bucket shard under `root`, in bucket order."""
    return sorted(root.glob("*.jsonl"))


def manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sample_total(book: dict) -> Optional[int]:
    """How many samples the manifest says were written, whichever way it says it.

    Two sampling runs, two spellings. A baseline conditioned per room-count
    bucket records `samples_per_bucket` and no total; GSDiff conditions on a plan
    id split, so it has no buckets to record and states `written` instead. A
    reader that knows only the first renders the second as nought, which is what
    the GSDiff cells of 2026-09-16 say: "2911 of 0 generated".
    """
    if "written" in book:
        return int(book["written"])
    per_bucket = book.get("samples_per_bucket")
    if per_bucket:
        return sum(int(value) for value in per_bucket.values())
    return None


def checkpoints(book: dict) -> list[tuple[str, str]]:
    """Every checkpoint that produced these samples, as (role, sha256) pairs.

    A model is not always one checkpoint. GSDiff samples through three, a graph
    encoder, an edge model and a diffusion model, and naming any one of them
    would name the wrong thing, so the manifest records them under `checkpoints`
    keyed by role. The mask-tracing baselines carry a single `checkpoint` and its
    `checkpoint_sha256`. Both come back as a list, so a caller states what the
    manifest actually holds rather than assuming there is one.

    Returning empty is a real answer and means the manifest records no checkpoint
    at all, which a run document must say rather than paper over.
    """
    several = book.get("checkpoints")
    if isinstance(several, dict) and several:
        return [
            (role, str(entry.get("sha256", "unknown")))
            for role, entry in sorted(several.items())
        ]
    single = book.get("checkpoint")
    if single:
        return [(str(single), str(book.get("checkpoint_sha256", "unknown")))]
    return []


def _ring(raw: Any) -> geo.Ring:
    if not raw:
        return []
    return geo.close_ring([(float(p[0]), float(p[1])) for p in raw])


def _rooms(record: dict) -> list[Room]:
    rooms: list[Room] = []
    for entry in record.get("rooms", []):
        ring = _ring(entry.get("ring"))
        if len(ring) < 3:
            continue
        rooms.append(
            Room(geo.orient_ccw(ring), str(entry["category"]), int(entry.get("node_index", len(rooms))))
        )
    return rooms


def _access(record: dict, room_count: int) -> list[AccessEdge]:
    out: list[AccessEdge] = []
    for edge in record.get("access", []):
        a, b = int(edge["a"]), int(edge["b"])
        if a != b and 0 <= a < room_count and 0 <= b < room_count:
            out.append(AccessEdge(a, b, str(edge.get("kind", "unknown"))))
    return out


def plan_from_record(record: dict, source: str) -> Plan:
    """One sample line as a Plan, on the baseline's own native frame."""
    rooms = _rooms(record)
    if not rooms:
        raise ValueError(f"{source} sample {record.get('sample_id')} has no usable room geometry")
    entry = record.get("entry")
    notes = dict(record.get("notes", {}))
    notes.update(
        {
            "bucket": record.get("bucket"),
            "rplan_id": record.get("rplan_id"),
            "seed": record.get("seed"),
            "frame": record.get("frame"),
            "frame_meters": record.get("frame_meters"),
            "emits_doors": bool(record.get("doors")),
        }
    )
    upm = float(record["units_per_meter"])
    return Plan(
        source=source,
        source_id=str(record["sample_id"]),
        rooms=rooms,
        units_per_meter=upm,
        # A plan traced from a mask has no native unit but the source pixel, so
        # the pixel size is the reciprocal of the scale the sampler recorded.
        source_pixel_m=1.0 / upm if upm else 0.0,
        access=_access(record, len(rooms)),
        entry=(float(entry[0]), float(entry[1])) if entry else None,
        off_axis_fraction=geo.off_axis_fraction([room.ring for room in rooms]),
        notes=notes,
    )


def iter_records(root: Path, limit: Optional[int] = None) -> Iterator[dict]:
    taken = 0
    for shard in shard_paths(root):
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if limit is not None and taken >= limit:
                    return
                taken += 1
                yield json.loads(line)


def iter_plans(source: str, root: Path, limit: Optional[int] = None) -> Iterator[Plan]:
    for record in iter_records(root, limit):
        yield plan_from_record(record, source)


def load_one(source: str, root: Path, source_id: str) -> Plan:
    for record in iter_records(root):
        if str(record.get("sample_id")) == str(source_id):
            return plan_from_record(record, source)
    raise KeyError(f"{source} sample {source_id} not found under {root}")


def list_ids(root: Path, limit: Optional[int] = None) -> list[str]:
    return [str(record["sample_id"]) for record in iter_records(root, limit)]
