"""Calibrate the rung 2 and rung 3 floors at the plan level, as a false-rejection rate.

The compactness floor (rung 2) and the proportions floors (rung 3) are room-level
corpus fifth percentiles, but each rung judges a plan by its worst room, so the
share of real plans a rung refuses grows with the number of rooms a plan holds.
This tool fits both floors instead so that each rung alone refuses a stated share
X of real built plans, and measures what that does to the graded margin of the
run of record.

- Rung 2: a plan's score is its worst Polsby-Popper over its non-hallway rooms,
  as `floorcheck.ladder.rung_room_compactness` reduces it. The fitted floor is the
  plan-level X quantile of that score.
- Rung 3: the per-category floor shape is kept and driven by one knob p: every
  category that carries a proportions floor today gets the p quantile of its own
  rooms' unrotated squareness (the quantity `rung_room_proportions` reads), the
  bedroom cap of 0.60 still applies, and p is chosen by bisection so that the
  plan-level refusal equals X. Today's floors are the same shape at p = 0.05 of
  the full corpus, read through the tabled aspect 95th percentile.

Each real kind is fitted separately (RPLAN apartments, MSD apartments, MSD
whole floors), since the worst of n rooms depends on n. A plan's kind is the one
the checker grades it as, `floorcheck.requirements.floor_kind` over its corpus
and the apartments its rooms state (`Room.unit`), so an MSD whole floor stating
one apartment or none counts toward the MSD apartments. Generated plans are
graded against the RPLAN floors, as `grade` does for a plan whose corpus is
RPLAN. The run groups stay the unit of every rate and margin; each plan in a
group is held to its own kind's floors.

Every plan is read once through `floorcheck` (ingest, then rungs 1, 2, 3 and 4
evaluated independently, with no short-circuit) into a feature file, and every
floor after that is applied to those features. The walk's pass is the conjunction
of the independent verdicts, which is checked against `grade` on every plan at
today's floors before any other number is written.

`--emit` writes the committed floors table both sides grade against,
`floorcheck/plan-level-floors.json`: each kind's two floors fitted at
`--level` (default 0.05) on all of that kind's real plans, `propReq` stored
before the bedroom cap, which the rung applies. Feature extraction grades every
plan through `grade`, which reads the committed table, so the reproduction line
of a run that emits reports against the table that was committed when the
features were read; rerun with `--rebuild` after emitting to check the composed
walk against `grade` at the emitted floors.

Run from the repository root (the data paths are the main checkout's gitignored
`data/`). The table of record was emitted from the run-13 exports, which carry
`Room.unit`:

    python -m tools.plan_level_floor data/derived/panel/run13-plans --workers 8 \\
        --buildings data/derived/msd-buildings.json \\
        --features data/derived/plan-level-features-run13.jsonl \\
        --emit floorcheck/plan-level-floors.json \\
        --out plan-level-floor.gen.md --json plan-level-floor.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from floorcheck.constants import ABS_TOL_M  # noqa: E402
from floorcheck.ladder import (  # noqa: E402
    BEDROOM_CLASS,
    BEDROOM_PROP_CAP,
    NOT_RUN,
    REJECTED,
    grade,
    rung_room_appendices,
    rung_room_outlines,
)
from floorcheck.measure import polsby_popper, squareness  # noqa: E402
from floorcheck.requirements import (  # noqa: E402
    FLOOR_KINDS,
    KIND_CORPUS,
    compactness_floor,
    floor_kind,
    requirements,
)
from floorcheck.scope import unit_count  # noqa: E402
from floorcheck.tags import is_hallway, tag_for  # noqa: E402

from tools.intervals import BASELINES, DENOMINATORS, GROUND_TRUTH, Margin  # noqa: E402

# group -> corpus of the real run groups; the MSD grid-path group is the whole floors.
REAL = {"rplan": "rplan", "msd units": "msd", "msd": "msd"}
KIND_LABEL = {"rplan": "RPLAN apartments", "msd-unit": "MSD apartments", "msd-floor": "MSD whole floors"}
GROUPS = ("rplan", "msd units", "msd") + BASELINES
LEVELS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
SPLITS = 20
SEED = 20260929
RECTANGULAR = 0.95
ROOM_BINS = ((1, 4), (5, 6), (7, 8), (9, 10), (11, 14), (15, 19), (20, 10_000))
OK = "ok"
MARGIN_HEADER = {
    "graded": "margin over graded plans",
    "admitted": "margin over admitted plans",
    "verdict": "margin over plans that reached a verdict",
}


# --- features, one pass through floorcheck --------------------------------


def rectangularity(ring) -> float:
    """Room area over the area of its minimum-area oriented bounding rectangle."""
    from shapely.geometry import Polygon

    polygon = Polygon(ring)
    box = polygon.minimum_rotated_rectangle.area
    return polygon.area / box if box > 0.0 else 0.0


def _verdict(result) -> str:
    """REJECTED and NOT_RUN kept, every other verdict (passed, skipped) read as ok."""
    return result.verdict if result.verdict in (REJECTED, NOT_RUN) else OK


def extract(document: dict[str, Any], corpus: str) -> dict[str, Any]:
    """One plan's floor-independent features and its `grade` pass at today's floors."""
    from floorcheck.ingest import ingest
    from floorcheck.model import plan_from_dict

    out: dict[str, Any] = {"admitted": False, "fault": None, "kind": _document_kind(document, corpus)}
    try:
        plan = plan_from_dict(document, name=str(document.get("name", "plan")))
        out["kind"] = floor_kind(corpus, unit_count(plan))
        result = ingest(plan)
    except Exception as error:  # a plan the model or ingest cannot read
        out["fault"] = type(error).__name__
        out["gradePassed"] = False
        return out
    if result.refused:
        out["gradePassed"] = False
        return out
    rings = list(result.rings)
    out["admitted"] = True
    out["gradePassed"] = bool(grade(plan, rings, source="dataset-p05", corpus=corpus).passed)
    out["r1"] = _verdict(rung_room_outlines(rings))
    out["r4"] = _verdict(rung_room_appendices(plan, rings))
    rooms = []
    for room, ring in zip(plan.rooms, rings):
        hallway = bool(is_hallway(room.category))
        rooms.append(
            [
                tag_for(room.category),
                int(hallway),
                None if hallway else round(float(polsby_popper(ring)), 6),
                round(float(squareness(ring)), 6),
                round(float(rectangularity(ring)), 6),
            ]
        )
    out["rooms"] = rooms
    return out


def _document_kind(document: dict[str, Any], corpus: str) -> str:
    """The kind of a document the model cannot read, from its rooms' `unit` labels.
    Such a plan is never admitted, so no floor is fitted or applied on it."""
    rooms = document.get("rooms") if isinstance(document.get("rooms"), list) else []
    units = {room.get("unit") for room in rooms if isinstance(room, dict)} - {None}
    return floor_kind(corpus, len(units))


def _extract_path(args: tuple[str, str, str]) -> dict[str, Any]:
    path, group, corpus = args
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = document.get("meta") if isinstance(document.get("meta"), dict) else {}
    record = extract(document, corpus)
    record.update(group=group, corpus=corpus, id=str(meta.get("sourceId") or Path(path).stem))
    return record


def plan_jobs(plans: Path) -> list[tuple[str, str, str]]:
    """Every plan document of the six groups, with its group and corpus."""
    jobs = []
    for directory in sorted(p for p in plans.iterdir() if p.is_dir()):
        for path in sorted(directory.glob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            meta = document.get("meta") if isinstance(document.get("meta"), dict) else {}
            group = meta.get("group") or directory.name
            corpus = meta.get("corpus") or "rplan"
            if group in GROUPS:
                jobs.append((str(path), group, corpus))
    return jobs


def build_features(plans: Path, features: Path, workers: Optional[int] = None) -> None:
    jobs = plan_jobs(plans)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    features.parent.mkdir(parents=True, exist_ok=True)
    partial = features.with_name(features.name + ".partial")  # a crashed run leaves no cache
    with partial.open("w", encoding="utf-8") as sink:
        if workers == 1:
            for record in map(_extract_path, jobs):
                sink.write(json.dumps(record, separators=(",", ":")) + "\n")
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for record in pool.map(_extract_path, jobs, chunksize=64):
                    sink.write(json.dumps(record, separators=(",", ":")) + "\n")
    partial.replace(features)


# --- the analysis, on features alone --------------------------------------


@dataclass
class Source:
    """One group's plans as arrays; rooms flattened with their plan index."""

    group: str
    corpus: str
    ids: list[str]
    kind: np.ndarray  # object array: the floor kind each plan is graded as
    admitted: np.ndarray
    rest_ok: np.ndarray  # rungs 1 and 4 not rejected and run
    r1_ok: np.ndarray  # rung 1 run and not rejected
    r4_fault: np.ndarray
    grade_passed: np.ndarray
    n_rooms: np.ndarray
    worst_pp: np.ndarray  # nan where every room is a hallway or not admitted
    worst_rect: np.ndarray
    room_plan: np.ndarray
    room_tag: np.ndarray  # object array of tag names
    room_sq: np.ndarray
    buildings: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.ids)


def source_from_records(group: str, corpus: str, records: list[dict[str, Any]]) -> Source:
    n = len(records)
    admitted = np.zeros(n, bool)
    rest_ok = np.zeros(n, bool)
    r1_ok = np.zeros(n, bool)
    r4_fault = np.zeros(n, bool)
    grade_passed = np.zeros(n, bool)
    n_rooms = np.zeros(n, int)
    worst_pp = np.full(n, np.nan)
    worst_rect = np.full(n, np.nan)
    room_plan, room_tag, room_sq = [], [], []
    kind = np.asarray([record.get("kind") or floor_kind(corpus, 0) for record in records], object)
    for i, record in enumerate(records):
        grade_passed[i] = bool(record.get("gradePassed"))
        if not record.get("admitted"):
            continue
        admitted[i] = True
        rest_ok[i] = record["r1"] == OK and record["r4"] == OK
        r1_ok[i] = record["r1"] == OK
        r4_fault[i] = record["r4"] == NOT_RUN
        rooms = record["rooms"]
        n_rooms[i] = len(rooms)
        for tag, hallway, pp, sq, rect in rooms:
            room_plan.append(i)
            room_tag.append(tag)
            room_sq.append(sq)
            # strict < keeps the first worst room, as the rung's own reduction does
            if not hallway and (math.isnan(worst_pp[i]) or pp < worst_pp[i]):
                worst_pp[i], worst_rect[i] = pp, rect
    return Source(
        group,
        corpus,
        [r["id"] for r in records],
        kind,
        admitted,
        rest_ok,
        r1_ok,
        r4_fault,
        grade_passed,
        n_rooms,
        worst_pp,
        worst_rect,
        np.asarray(room_plan, int),
        np.asarray(room_tag, object),
        np.asarray(room_sq, float),
    )


def load_sources(features: Path) -> dict[str, Source]:
    grouped: dict[str, list[dict[str, Any]]] = {g: [] for g in GROUPS}
    corpora: dict[str, str] = {}
    with features.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            grouped[record["group"]].append(record)
            corpora[record["group"]] = record["corpus"]
    return {g: source_from_records(g, corpora[g], r) for g, r in grouped.items() if r}


def gather(parts: list[tuple[Source, np.ndarray]], name: str, corpus: str) -> Source:
    """The masked plans of several sources as one source, rooms renumbered."""
    ids: list[str] = []
    plan_arrays: dict[str, list[np.ndarray]] = {
        key: [] for key in ("kind", "admitted", "rest_ok", "r1_ok", "r4_fault", "grade_passed", "n_rooms", "worst_pp", "worst_rect")
    }
    room_plan, room_tag, room_sq, buildings = [], [], [], []
    offset = 0
    for source, mask in parts:
        chosen = np.flatnonzero(mask)
        renumber = np.full(source.size, -1, int)
        renumber[chosen] = np.arange(len(chosen)) + offset
        ids += [source.ids[i] for i in chosen]
        if source.buildings:
            buildings += [source.buildings[i] for i in chosen]
        for key, values in plan_arrays.items():
            values.append(getattr(source, key)[chosen])
        rooms = mask[source.room_plan]
        room_plan.append(renumber[source.room_plan[rooms]])
        room_tag.append(source.room_tag[rooms])
        room_sq.append(source.room_sq[rooms])
        offset += len(chosen)

    def joined(values: list[np.ndarray], dtype) -> np.ndarray:
        return np.concatenate(values) if values else np.asarray([], dtype)

    return Source(
        name,
        corpus,
        ids,
        joined(plan_arrays["kind"], object),
        joined(plan_arrays["admitted"], bool),
        joined(plan_arrays["rest_ok"], bool),
        joined(plan_arrays["r1_ok"], bool),
        joined(plan_arrays["r4_fault"], bool),
        joined(plan_arrays["grade_passed"], bool),
        joined(plan_arrays["n_rooms"], int),
        joined(plan_arrays["worst_pp"], float),
        joined(plan_arrays["worst_rect"], float),
        joined(room_plan, int),
        joined(room_tag, object),
        joined(room_sq, float),
        buildings,
    )


def kind_view(sources: dict[str, Source], kind: str, masks: dict[str, np.ndarray]) -> Source:
    """Every masked real plan graded as `kind`, whichever run group it came from."""
    parts = [(sources[g], masks[g] & (sources[g].kind == kind)) for g in REAL if g in sources]
    return gather(parts, kind, KIND_CORPUS[kind])


@dataclass(frozen=True)
class Floors:
    """One kind's floors: the rung 2 number and the rung 3 per-tag vector."""

    compactness: float
    proportions: dict[str, float]
    knob: Optional[float] = None  # the rung 3 quantile p, when fitted


def current_floors(kind: str) -> Floors:
    """The committed floors of one kind, as `grade` reads them."""
    reqs = requirements("dataset-p05", kind, None)
    return Floors(
        compactness_floor("dataset-p05", kind, None),
        {tag: r.prop_req for tag, r in reqs.items() if r.prop_req > 0.0},
    )


def fitted_tags(kind: str) -> list[str]:
    """The categories a kind's rung 3 floors are fitted for: every RoomTag the
    category statistics of the kind's corpus carry, the categories `grade` reads
    a requirement for."""
    return sorted(requirements("dataset-p05", kind, None))


def capped(proportions: dict[str, float]) -> dict[str, float]:
    return {t: min(f, BEDROOM_PROP_CAP) if t in BEDROOM_CLASS else f for t, f in proportions.items()}


def rung2_fails(source: Source, floor: float) -> np.ndarray:
    """Plan refused by rung 2 alone; a plan with no scored room passes."""
    with np.errstate(invalid="ignore"):
        return source.admitted & (source.worst_pp < floor)


def rung3_fails(source: Source, proportions: dict[str, float]) -> np.ndarray:
    """Plan refused by rung 3 alone, with the bedroom cap and the ABS_TOL admission."""
    floors = capped(proportions)
    room_floor = np.array([floors.get(t, 0.0) for t in source.room_tag], float)
    failing = (room_floor > 0.0) & (source.room_sq < room_floor - ABS_TOL_M)
    count = np.bincount(source.room_plan, weights=failing.astype(float), minlength=source.size)
    return source.admitted & (count > 0)


def ladder_passes(source: Source, floors: Floors) -> np.ndarray:
    """The walk's pass: every independent verdict clear, so order does not matter."""
    return (
        source.admitted
        & source.rest_ok
        & ~rung2_fails(source, floors.compactness)
        & ~rung3_fails(source, floors.proportions)
    )


def by_kind(source: Source, floors: dict[str, Floors], verdict) -> np.ndarray:
    """`verdict(source, kind floors)` on every plan, each under its own kind's floors."""
    out = np.zeros(source.size, bool)
    for kind in set(source.kind):
        out |= verdict(source, floors[kind]) & (source.kind == kind)
    return out


def rung2_fails_by_kind(source: Source, floors: dict[str, Floors]) -> np.ndarray:
    return by_kind(source, floors, lambda s, f: rung2_fails(s, f.compactness))


def rung3_fails_by_kind(source: Source, floors: dict[str, Floors]) -> np.ndarray:
    return by_kind(source, floors, lambda s, f: rung3_fails(s, f.proportions))


def ladder_passes_by_kind(source: Source, floors: dict[str, Floors]) -> np.ndarray:
    return by_kind(source, floors, ladder_passes)


def fit_rung2(scores: np.ndarray, level: float) -> float:
    """The floor that refuses `level` of these plans (nan scores always pass)."""
    n = len(scores)
    finite = np.sort(scores[~np.isnan(scores)])
    k = int(round(level * n))
    if k <= 0:
        return float(finite[0])
    if k >= len(finite):
        return float(np.nextafter(finite[-1], np.inf))
    return float((finite[k - 1] + finite[k]) / 2.0)


def tag_quantiles(source: Source, mask: np.ndarray, tags: Iterable[str], p: float) -> dict[str, float]:
    """Each tag's p quantile of squareness over the rooms of the masked plans."""
    keep = mask[source.room_plan]
    out = {}
    for tag in tags:
        values = source.room_sq[keep & (source.room_tag == tag)]
        if len(values):
            out[tag] = float(np.quantile(values, p))
    return out


def fit_rung3(source: Source, mask: np.ndarray, tags: Iterable[str], level: float, steps: int = 40):
    """The knob p, and its floors, whose plan-level refusal over `mask` is nearest `level`."""
    tags = list(tags)
    keep = mask & source.admitted
    total = int(keep.sum())
    # squareness per tag over the fit rooms, sorted once
    room_keep = keep[source.room_plan]
    pools = {t: np.sort(source.room_sq[room_keep & (source.room_tag == t)]) for t in tags}
    pools = {t: v for t, v in pools.items() if len(v)}

    def floors_at(p: float) -> dict[str, float]:
        return {t: float(np.quantile(v, p)) for t, v in pools.items()}

    def refusal(p: float) -> float:
        return float((rung3_fails(source, floors_at(p)) & keep).sum()) / total

    lo, hi = 0.0, 1.0
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        if refusal(mid) < level:
            lo = mid
        else:
            hi = mid
    best = min((lo, hi), key=lambda p: abs(refusal(p) - level))
    return best, floors_at(best)


def fit(source: Source, mask: np.ndarray, level: float, tags: Iterable[str]) -> Floors:
    """Both floors fitted on the masked plans of one real kind."""
    keep = mask & source.admitted
    p, proportions = fit_rung3(source, keep, tags, level)
    return Floors(fit_rung2(source.worst_pp[keep], level), proportions, p)


def fit_kinds(sources: dict[str, Source], masks: dict[str, np.ndarray], level: float) -> dict[str, Floors]:
    """Every kind's floors, fitted on its masked real plans across the run groups."""
    out = {}
    for kind in FLOOR_KINDS:
        view = kind_view(sources, kind, masks)
        if view.admitted.any():
            out[kind] = fit(view, np.ones(view.size, bool), level, fitted_tags(kind))
    return out


def rate(flags: np.ndarray, mask: np.ndarray) -> float:
    total = int(mask.sum())
    return float((flags & mask).sum()) / total if total else float("nan")


def point_margin(rates: dict[str, float]) -> tuple[float, str, str]:
    """The headline margin: lowest ground-truth rate less highest baseline rate."""
    low = min((g for g in GROUND_TRUTH if g in rates), key=lambda g: rates[g])
    high = max((g for g in BASELINES if g in rates), key=lambda g: rates[g])
    return rates[low] - rates[high], low, high


# --- the splits ------------------------------------------------------------


def floor_of(source_id: str) -> str:
    return source_id.split("#u", 1)[0]


def attach_buildings(sources: dict[str, Source], key: dict[str, str]) -> None:
    """The building of every MSD plan; a floor the key does not hold is its own."""
    for group in ("msd units", "msd"):
        if group in sources:
            s = sources[group]
            s.buildings = [key.get(floor_of(i), f"floor:{floor_of(i)}") for i in s.ids]


def split_masks(sources: dict[str, Source], seed: int) -> dict[str, np.ndarray]:
    """The fit half of every real kind: by plan for RPLAN, by building for MSD."""
    rng = np.random.default_rng(seed)
    masks = {}
    if "rplan" in sources:
        n = sources["rplan"].size
        order = rng.permutation(n)
        mask = np.zeros(n, bool)
        mask[order[: n // 2]] = True
        masks["rplan"] = mask
    buildings = sorted({b for g in ("msd units", "msd") if g in sources for b in sources[g].buildings})
    chosen = set(np.asarray(buildings, object)[rng.permutation(len(buildings))[: len(buildings) // 2]])
    for group in ("msd units", "msd"):
        if group in sources:
            masks[group] = np.array([b in chosen for b in sources[group].buildings], bool)
    return masks


def graded_rates(sources: dict[str, Source], floors: dict[str, Floors], masks: dict[str, np.ndarray]) -> dict[str, float]:
    """Pass rate over graded plans for every group, each plan under its kind's floors."""
    return {g: rate(ladder_passes_by_kind(s, floors), masks[g]) for g, s in sources.items()}


def precondition_stops(source: Source, floors: Floors) -> np.ndarray:
    """Admitted plans the walk carries to rung 4 and its precondition then stops.

    The walk reaches rung 4 only on a plan rungs 1, 2 and 3 all clear, so a plan a
    shape check rejects first has a verdict, a rejection, even when its walls would
    also fail the precondition (specification 9.4). Rung 4 is the only precondition
    fault the features carry."""
    return (
        source.admitted
        & source.r4_fault
        & source.r1_ok
        & ~rung2_fails(source, floors.compactness)
        & ~rung3_fails(source, floors.proportions)
    )


def precondition_stops_by_kind(source: Source, floors: dict[str, Floors]) -> np.ndarray:
    return by_kind(source, floors, precondition_stops)


def denominator_mask(source: Source, denominator: str, stopped: np.ndarray) -> np.ndarray:
    """The plans one of the run of record's denominators counts (`tools.cells.Cell.total`):
    graded is every plan, admitted drops the plans ingest refused or could not read,
    and reached a verdict further drops `stopped`, the admitted plans the walk carried
    to the rung 4 precondition and that precondition stopped (`precondition_stops`,
    specification 9.4). Which plans those are depends on the floors."""
    if denominator == "graded":
        return np.ones(source.size, bool)
    if denominator == "admitted":
        return source.admitted
    if denominator == "verdict":
        return source.admitted & ~stopped
    raise ValueError(f"unknown denominator {denominator!r}")


def denominator_rates(
    sources: dict[str, Source], floors: dict[str, Floors], masks: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Pass rate of every group on each of the three denominators, same numerator.
    A plan stopped on a precondition never passes, so it leaves only the denominator."""
    passes = {g: ladder_passes_by_kind(s, floors) for g, s in sources.items()}
    stopped = {g: precondition_stops_by_kind(s, floors) for g, s in sources.items()}
    return {
        d: {g: rate(passes[g], masks[g] & denominator_mask(s, d, stopped[g])) for g, s in sources.items()}
        for d, _ in DENOMINATORS
    }


def sweep(sources: dict[str, Source], levels=LEVELS, splits=SPLITS, seed=SEED) -> dict[str, Any]:
    """Measurement 1: fit on one half, read the other, over `splits` random halves."""
    real = [g for g in REAL if g in sources]
    everything = {g: np.ones(s.size, bool) for g, s in sources.items()}
    current = {k: current_floors(k) for k in FLOOR_KINDS}
    rows: dict[float, dict[str, list]] = {}
    for level in levels:
        acc: dict[str, list] = {}
        for split in range(splits):
            masks = split_masks(sources, seed + split)
            fitted = fit_kinds(sources, masks, level)
            held = {g: ~masks[g] for g in real}
            view = {**everything, **held}
            for k in fitted:
                s = kind_view(sources, k, held)
                h = s.admitted
                acc.setdefault(f"{k}|floor2", []).append(fitted[k].compactness)
                acc.setdefault(f"{k}|knob3", []).append(fitted[k].knob)
                acc.setdefault(f"{k}|r2", []).append(rate(rung2_fails(s, fitted[k].compactness), h))
                acc.setdefault(f"{k}|r3", []).append(rate(rung3_fails(s, fitted[k].proportions), h))
                everyone = np.ones(s.size, bool)
                acc.setdefault(f"{k}|ladder", []).append(1.0 - rate(ladder_passes(s, fitted[k]), everyone))
                acc.setdefault(f"{k}|ladder_now", []).append(1.0 - rate(ladder_passes(s, current[k]), everyone))
            by_denominator = denominator_rates(sources, fitted, view)
            new = by_denominator["graded"]
            now = graded_rates(sources, current, view)
            m, low, high = point_margin(new)
            acc.setdefault("margin", []).append(m)
            acc.setdefault("margin_now", []).append(point_margin(now)[0])
            acc.setdefault("pair", []).append(f"{low} over {high}")
            for g in BASELINES:
                if g in new:
                    acc.setdefault(f"{g}|pass", []).append(new[g])
            for g in GROUND_TRUTH:
                if g in new:
                    acc.setdefault(f"{g}|pass", []).append(new[g])
            # the same margin on each denominator, its pair re-chosen there, as `Margin` does
            for d, rates in by_denominator.items():
                m, low, high = point_margin(rates)
                if d != "graded":
                    acc.setdefault(f"margin|{d}", []).append(m)
                acc.setdefault(f"pair|{d}", []).append(f"{low} over {high}")
                for g in GROUND_TRUTH + BASELINES:
                    if g in rates:
                        acc.setdefault(f"{g}|pass|{d}", []).append(rates[g])
        rows[level] = acc
    return rows


def full_fit(sources: dict[str, Source], level: float) -> dict[str, Floors]:
    return fit_kinds(sources, {g: np.ones(s.size, bool) for g, s in sources.items()}, level)


def by_room_count(sources: dict[str, Source], floors: dict[str, Floors]) -> dict[str, list[dict[str, Any]]]:
    """Measurement 2: each rung's refusal by room count, per source."""
    out = {}
    for g, s in sources.items():
        r2 = rung2_fails_by_kind(s, floors)
        r3 = rung3_fails_by_kind(s, floors)
        rows = []
        for lo, hi in ROOM_BINS:
            mask = s.admitted & (s.n_rooms >= lo) & (s.n_rooms <= hi)
            rows.append({"bin": (lo, hi), "plans": int(mask.sum()), "r2": rate(r2, mask), "r3": rate(r3, mask)})
        out[g] = rows
    return out


def overlap(sources: dict[str, Source], floors: dict[str, Floors]) -> dict[str, dict[str, Any]]:
    """Measurement 3: rungs 2, 3 and the rung 4 precondition, each on every admitted plan."""
    out = {}
    for g, s in sources.items():
        a = s.admitted
        r2 = rung2_fails_by_kind(s, floors)
        r3 = rung3_fails_by_kind(s, floors)
        r4 = s.r4_fault & a
        patterns = {}
        for b2 in (0, 1):
            for b3 in (0, 1):
                for b4 in (0, 1):
                    patterns[f"{b2}{b3}{b4}"] = int((a & (r2 == b2) & (r3 == b3) & (r4 == b4)).sum())
        rect = r2 & (s.worst_rect >= RECTANGULAR)
        out[g] = {
            "admitted": int(a.sum()),
            "patterns": patterns,
            "r2": int(r2.sum()),
            "r3": int(r3.sum()),
            "r4": int(r4.sum()),
            "r2_rect": int(rect.sum()),
            "r2_rect_r3": int((rect & r3).sum()),
            "r2_nonrect": int((r2 & ~rect).sum()),
            "r2_nonrect_r3": int((r2 & ~rect & r3).sum()),
        }
    return out


def reproduce(sources: dict[str, Source]) -> dict[str, Any]:
    """Today's floors on every plan: the composed walk against `grade`, and the headline."""
    current = {k: current_floors(k) for k in FLOOR_KINDS}
    disagreements = {}
    cells = {}
    for g, s in sources.items():
        composed = ladder_passes_by_kind(s, current)
        disagreements[g] = int((composed != s.grade_passed).sum())
        cells[g] = _Cell(int(composed.sum()), s.size)
    margin = Margin({g: cells[g] for g in GROUND_TRUTH + BASELINES if g in cells}, "graded")
    return {
        "disagreements": disagreements,
        "rates": {g: (c.passed, c.graded) for g, c in cells.items()},
        "margin": margin.fixed[0],
        "low": margin.low_name,
        "high": margin.high_name,
        "reselected": [margin.reselected[1], margin.reselected[2]],
    }


@dataclass
class _Cell:
    passed: int
    graded: int

    def total(self, denominator: str) -> int:
        return self.graded


# --- the record ------------------------------------------------------------


def pct(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f}"


def spread(values: list[float], scale: float = 100.0, digits: int = 1) -> str:
    v = np.asarray(values, float)
    return f"{scale * v.mean():.{digits}f} ({scale * v.min():.{digits}f} to {scale * v.max():.{digits}f})"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render(result: dict[str, Any]) -> str:
    rep = result["reproduction"]
    out = ["# Plan-level floors: generated tables", ""]
    out.append(
        # `Margin` reports in percentage points already
        f"Reproduction at today's floors: margin {rep['margin']:+.1f} "
        f"({rep['reselected'][0]:+.1f} to {rep['reselected'][1]:+.1f}), "
        f"{rep['low']} over {rep['high']}; composed walk disagrees with `grade` on "
        f"{sum(rep['disagreements'].values())} plans."
    )
    out.append("")
    sweep_rows = result["sweep"]
    real = [k for k in FLOOR_KINDS if f"{k}|r2" in next(iter(sweep_rows.values()))]
    out += ["## Sweep, held-out half, mean (range) over splits", ""]
    rows = []
    for level, acc in sweep_rows.items():
        m = np.asarray(acc["margin"]) * 100
        now = np.asarray(acc["margin_now"]) * 100
        rows.append(
            [pct(float(level), 0), f"{m.mean():+.1f} ({m.min():+.1f} to {m.max():+.1f})",
             f"{now.mean():+.1f}", max(set(acc["pair"]), key=acc["pair"].count)]
            + [spread(acc[f"{g}|ladder"]) for g in real]
        )
    out.append(table(["X", "margin, fitted floors", "margin, today's floors", "pair"]
                     + [f"ladder refusal, {KIND_LABEL[g]}" for g in real], rows))
    out.append("")
    rows = []
    for level, acc in sweep_rows.items():
        for g in real:
            rows.append([pct(float(level), 0), KIND_LABEL[g], spread(acc[f"{g}|floor2"], 1.0, 4),
                         spread(acc[f"{g}|knob3"], 1.0, 3), spread(acc[f"{g}|r2"]), spread(acc[f"{g}|r3"]),
                         spread(acc[f"{g}|ladder_now"])])
    out.append(table(["X", "kind", "rung 2 floor", "rung 3 knob p", "rung 2 refusal", "rung 3 refusal",
                      "ladder refusal, today's floors"], rows))
    out.append("")
    rows = []
    for level, acc in sweep_rows.items():
        rows.append([pct(float(level), 0)] + [spread(acc[f"{g}|pass"]) for g in GROUND_TRUTH + BASELINES
                                              if f"{g}|pass" in acc])
    out.append(table(["X"] + [f"{g} pass" for g in GROUND_TRUTH + BASELINES
                              if f"{g}|pass" in next(iter(sweep_rows.values()))], rows))
    out.append("")
    out += ["## Margin and pass rates on the three denominators, fitted floors, held-out half", ""]
    rows = []
    for level, acc in sweep_rows.items():
        row = [pct(float(level), 0)]
        for d, _ in DENOMINATORS:
            m = np.asarray(acc["margin" if d == "graded" else f"margin|{d}"]) * 100
            pairs = acc[f"pair|{d}"]
            row += [f"{m.mean():+.1f} ({m.min():+.1f} to {m.max():+.1f})", max(set(pairs), key=pairs.count)]
        rows.append(row)
    out.append(table(["X"] + [h for d, _ in DENOMINATORS for h in (MARGIN_HEADER[d], "pair")], rows))
    out.append("")
    first = next(iter(sweep_rows.values()))
    for d, label in DENOMINATORS:
        shown = [g for g in GROUND_TRUTH + BASELINES if f"{g}|pass|{d}" in first]
        rows = [[pct(float(level), 0)] + [spread(acc[f"{g}|pass|{d}"]) for g in shown]
                for level, acc in sweep_rows.items()]
        out.append(table(["X"] + [f"{g} pass, {label}" for g in shown], rows))
        out.append("")
    for name, floors in result["fullFloors"].items():
        out += [f"## Floors fitted on every real plan, {name}", ""]
        rows = [[KIND_LABEL[g], f"{f['compactness']:.4f}", "n/a" if f["knob"] is None else f"{f['knob']:.3f}",
                 ", ".join(f"{t} {v:.3f}" for t, v in sorted(capped(f["proportions"]).items()))]
                for g, f in floors.items()]
        out.append(table(["kind", "rung 2 floor", "rung 3 knob p", "rung 3 floors (bedroom cap applied)"], rows))
        out.append("")
    out += ["## Refusal by room count at X = 5, floors fitted on every real plan", ""]
    counts = result["roomCount"]
    rows = []
    for g, bins in counts.items():
        for b in bins:
            if b["plans"]:
                lo, hi = b["bin"]
                label = f"{lo}+" if hi >= 10_000 else f"{lo}-{hi}"
                rows.append([g, label, str(b["plans"]), pct(b["r2"]), pct(b["r3"])])
    out.append(table(["source", "rooms", "plans", "rung 2 refusal", "rung 3 refusal"], rows))
    out.append("")
    for name, table_ in result["overlap"].items():
        out += [f"## Overlap of rungs 2, 3 and the rung 4 precondition, {name}", ""]
        rows = []
        for g, o in table_.items():
            p = o["patterns"]
            share = o["r2_rect_r3"] / o["r2_rect"] if o["r2_rect"] else float("nan")
            rows.append([g, str(o["admitted"]), str(o["r2"]), str(o["r3"]), str(o["r4"]),
                         str(p["110"] + p["111"]), str(p["101"] + p["111"]), str(p["011"] + p["111"]), str(p["111"]),
                         str(p["100"]), str(p["010"]), str(p["001"]),
                         f"{o['r2_rect']} ({o['r2_rect_r3']}, {pct(share)})"])
        out.append(table(["source", "admitted", "r2", "r3", "r4 fault", "r2&r3", "r2&r4", "r3&r4", "all three",
                          "r2 only", "r3 only", "r4 only", "r2 rect. (also r3, share)"], rows))
        out.append("")
    return "\n".join(out)


def analyse(sources: dict[str, Source], splits: int = SPLITS, seed: int = SEED) -> dict[str, Any]:
    result: dict[str, Any] = {"reproduction": reproduce(sources)}
    result["sweep"] = sweep(sources, splits=splits, seed=seed)
    at5 = full_fit(sources, 0.05)
    now = {k: current_floors(k) for k in FLOOR_KINDS}
    result["fullFloors"] = {
        "today": {g: vars(f) for g, f in now.items()},
        **{f"X = {pct(level, 0)}": {g: vars(f) for g, f in full_fit(sources, level).items()} for level in LEVELS},
    }
    result["roomCount"] = by_room_count(sources, at5)
    result["overlap"] = {"today's floors": overlap(sources, now), "X = 5": overlap(sources, at5)}
    return result


def emitted_table(sources: dict[str, Source], level: float, provenance: dict[str, Any]) -> dict[str, Any]:
    """The floors table `grade` and the oracle read, fitted at `level` on every real plan.

    Floors are rounded to four decimals, and `propReq` is stored before the
    bedroom cap. Beside the table this returns, under `fit`, what the rounded
    floors do on the plans they were fitted on: each kind's plan count, the
    rung 3 knob, and the share refused by rung 2 alone, rung 3 alone and the two
    together. The `fit` block is reported, not written.
    """
    floors = full_fit(sources, level)
    kinds: dict[str, Any] = {}
    fit_report: dict[str, Any] = {}
    everything = {g: np.ones(s.size, bool) for g, s in sources.items()}
    for kind, fitted in floors.items():
        rounded = Floors(
            round(fitted.compactness, 4),
            {tag: round(value, 4) for tag, value in sorted(fitted.proportions.items())},
            fitted.knob,
        )
        kinds[kind] = {"compactness": rounded.compactness, "propReq": rounded.proportions}
        view = kind_view(sources, kind, everything)
        admitted = view.admitted
        r2 = rung2_fails(view, rounded.compactness)
        r3 = rung3_fails(view, rounded.proportions)
        fit_report[kind] = {
            "plans": int(view.size),
            "admitted": int(admitted.sum()),
            "knob": fitted.knob,
            "rung2Refusal": rate(r2, admitted),
            "rung3Refusal": rate(r3, admitted),
            "rungs2and3Refusal": rate(r2 | r3, admitted),
        }
    return {"level": level, "provenance": provenance, "kinds": kinds, "fit": fit_report}


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="floorcheck-floors", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("plans", type=Path, help="the plan exports of the run (data/derived/panel/run13-plans)")
    parser.add_argument("--buildings", type=Path, default=REPO / "data" / "derived" / "msd-buildings.json")
    parser.add_argument("--features", type=Path, default=REPO / "data" / "derived" / "plan-level-features.jsonl")
    parser.add_argument("--rebuild", action="store_true", help="re-read every plan even if the features exist")
    parser.add_argument("--splits", type=int, default=SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--emit", type=Path, default=None, help="write the floors table (floorcheck/plan-level-floors.json)")
    parser.add_argument("--level", type=float, default=0.05, help="the per-rung refusal the emitted floors are fitted at")
    args = parser.parse_args(argv)

    if args.rebuild or not args.features.exists():
        build_features(args.plans, args.features, args.workers)
    sources = load_sources(args.features)
    from tools.msd_buildings import load

    attach_buildings(sources, load(cache=args.buildings))
    if args.emit:
        counts = {g: s.size for g, s in sources.items() if g in REAL}
        provenance = {
            # The arguments the table depends on; output and cache paths do not change it.
            "command": (
                f"python -m tools.plan_level_floor {args.plans.as_posix()}"
                f" --emit {args.emit.as_posix()} --level {args.level:g}"
            ),
            "commit": _commit(),
            "input": (
                f"every real plan of {args.plans.as_posix()}, by run group {counts}, each plan"
                " fitted under the kind floorcheck grades it as (floor_kind over its corpus and"
                " the apartments its rooms state); floors rounded to four decimals, propReq"
                " before the bedroom cap"
            ),
        }
        emitted = emitted_table(sources, args.level, provenance)
        fit_report = emitted.pop("fit")
        args.emit.write_text(json.dumps(emitted, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.emit}")
        for kind, row in fit_report.items():
            print(
                f"  {kind}: {row['plans']} plans, {row['admitted']} admitted, knob {row['knob']:.4f}, "
                f"rung 2 alone {pct(row['rung2Refusal'], 2)}, rung 3 alone {pct(row['rung3Refusal'], 2)}, "
                f"rungs 2 and 3 {pct(row['rungs2and3Refusal'], 2)}"
            )
    if not any(g in sources for g in BASELINES):
        # Fitting floors needs only built plans; every table below sets them against a generator.
        print("no margin was formed because no generator group is present; the margin tables are skipped")
        return 0
    result = analyse(sources, args.splits, args.seed)
    text = render(result)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
