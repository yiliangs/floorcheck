"""The plan sources a run reads.

Each source yields plans to grade, or an outcome already decided because a plan
never became one: the MSD floors and apartments with their fallback partition,
the loader guard that turns an exception into a build failure, the parsing of
the generated baselines a run names, and the corpus each source answers to.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional

from planaudit import exposure, partition, units
from planaudit.datasets import msd
from planaudit.plan import Plan

BUILD_FAILURE = "BUILD FAILURE"
UNITS_SOURCE = "msd units"

# A stream item is either an Attempt to grade or an Outcome the stream already
# decided because the plan never became one.


@dataclass
class Attempt:
    """One request to grade, and how to rebuild it on the fallback partition.

    Which partition a plan was built on is a property of the plan, so only the
    stream that built it can rebuild it, and the fallback has to travel with the
    plan rather than be inferred at the verdict. A stream with no fallback yields
    `rebuild=None` and one build failure stands.
    """

    plan: Plan
    context: Optional[exposure.ExposureContext] = None
    rebuild: Optional[Callable[[], "Attempt"]] = None


@dataclass
class Outcome:
    source: str
    source_id: str
    build_failure: bool
    accepted: bool
    stage: str
    detail: str = ""
    unmapped: Optional[Counter] = None
    area_error: float = 0.0
    tokens: int = 0
    wall: Optional[dict[str, float]] = None
    path: str = ""
    # Floor area no room covered, set only when the adapter refused the plan
    # because its rooms do not tile. The 2026-09-14 ruling makes that refusal an
    # outcome in its own right, so the tally counts it by type and reports the
    # area. Grouping it by message text does not survive corpus scale: the
    # message names the plan, so every plan reads as its own distinct cause.
    untiled_m2: Optional[float] = None
    # The other half of the same refusal. A plan can fail either or both, so the
    # two are carried apart rather than summed into one number that would say
    # neither how much floor had no room nor how much had two.
    overlap_m2: Optional[float] = None
    # Floor the adapter's seam tolerance handed to a neighbouring room, set
    # whenever the adapter got far enough to measure it. A run document reports
    # it beside the refusals so that the tolerance is accountable for what it
    # moved and not only for what it stopped refusing.
    closed_m2: float = 0.0
    # The part of `closed_m2` that lay in an open channel between two rooms that
    # do not touch rather than in an enclosed void. The two shapes are what the
    # sources differ in, a corpus stating a partition leaving voids where its
    # walls do not meet and a corpus stating disjoint rooms leaving channels, so
    # a cell reporting only the total cannot say which corpus it read.
    bridged_m2: float = 0.0
    # Rooms the grid partition rebuilt in more than one piece, and the area of
    # every piece but the largest, which is what emitting one ring per room
    # costs. Read from the plan's own notes rather than measured here, because
    # the loss happens where the partition assembles a room from its cells, well
    # before the adapter sees the plan.
    split_rooms: int = 0
    split_m2: float = 0.0
    # The tally row an adapter refusal belongs in, taken from the refusal class
    # itself at the moment it was caught. Set only on that path: a build failure
    # from anywhere else is still placed by the origin prefix of `detail`, which
    # is what says which project owns the bug.
    refusal_row: Optional[str] = None
    # Whether the rung named in `stage` rejected on its own precondition rather
    # than on the rule it exists to apply. The solver reports the two the same way
    # in `stage`, and they are different findings about the checker: a rung that
    # refuses a room because its wall loop carries a non-orthogonal segment has
    # not measured that room against anything, so counting it beside the rooms
    # the rung did measure reports a geometry fault as a policy verdict.
    precondition_fault: bool = False


def safe(source: str, stream: Iterable[object]) -> Iterator[Attempt | Outcome]:
    """Yield attempts, turning a loader exception into a build failure outcome."""
    iterator = iter(stream)
    index = 0
    while True:
        try:
            yield _as_attempt(next(iterator))
        except StopIteration:
            return
        except Exception as exc:
            yield Outcome(source, f"<load {index}>", True, False, BUILD_FAILURE, f"loader: {exc}")
        index += 1


def _as_attempt(item: object) -> Attempt | Outcome:
    """Accept a bare Plan or a (Plan, context) pair from a stream with no fallback."""
    if isinstance(item, (Attempt, Outcome)):
        return item
    if isinstance(item, tuple):
        plan, context = item
        return Attempt(plan, context)
    return Attempt(item)  # type: ignore[arg-type]


def floor_stream(limit: int, root=None) -> Iterator[object]:
    """Every MSD floor on the default partition, with the grid retry where one exists.

    `load_floor` and not `load_one`, because this is the stream that reads the
    rooms as one plan: a floor whose rooms close into more than one part is
    refused here and graded apartment by apartment in `unit_stream` instead.
    """
    for source_id in msd.list_ids(root)[:limit]:
        yield _or_fallback("msd", source_id, root, lambda sid: msd.load_floor(sid, root))


def _floor_rebuild(source_id: str, root) -> Optional[Callable[[], Attempt]]:
    """The grid retry for a floor the default path could not build, or None.

    The retry was built for one case, the vector path, whose junction refusals the
    grid could rescue, and it is offered only when that path leads. Under the grid
    it was pointless, re-partitioning on the grid asking the solver the identical
    question. Under the rectified path, which leads since the 2026-09-18 ruling,
    it would not be pointless and that is exactly the objection: a build rate that
    counts some floors on one partition and the rest on another describes no
    partition, and the corpus numbers that ruling reads were measured on the
    rectified path alone. A refusal is recorded, as a grid refusal was.
    """
    if partition.DEFAULT_METHOD != "vector":
        return None
    return lambda: Attempt(msd.load_floor(source_id, root, method="grid"))


def _or_fallback(source: str, source_id: str, root, load) -> Attempt | Outcome:
    """An Attempt on the default path, or on the fallback when the default cannot load.

    A partition that raises is a stronger form of the failure the solver reports
    and has the same remedy, so it takes the same fallback rather than becoming a
    loader failure the run can do nothing about. Only the vector path has a
    fallback, so on any other default the loader failure is what the run records.

    The prefix is the failure's own where it states one. `loader` says this
    repository could not read the source, which is a defect; a refusal that is a
    ruling about what the source holds names its own origin and files under its
    own row, so the two are never summed into one count.
    """
    rebuild = _floor_rebuild(source_id, root)
    try:
        return Attempt(load(source_id), None, rebuild)
    except Exception as first:
        if rebuild is None:
            return Outcome(
                source, source_id, True, False, BUILD_FAILURE, f"{_origin(first)}: {first}"[:200]
            )
        try:
            return rebuild()
        except Exception as second:
            return Outcome(
                source, source_id, True, False, BUILD_FAILURE,
                f"{_origin(first)}: {first}; fallback: {second}"[:200],
            )


def _origin(exc: BaseException) -> str:
    """The tally prefix a load failure carries, `loader` unless it named one."""
    return str(getattr(exc, "origin", "loader"))


def unit_stream(limit: int, root=None) -> Iterator[object]:
    """Every apartment of the first `limit` MSD floors, with its own context.

    A floor whose membership cannot be derived, or a unit whose outline cannot be
    built, becomes one build failure and the rest of the floor still runs. The
    plain `safe` wrapper cannot do this: a generator that raises is closed, so
    one bad unit would truncate the whole stream.
    """
    for source_id in msd.list_ids(root)[:limit]:
        item = _or_fallback(UNITS_SOURCE, source_id, root, lambda sid: msd.load_one(sid, root))
        if isinstance(item, Outcome):
            yield item
            continue
        plan = item.plan
        try:
            assignment = units.assign(plan)
        except Exception as exc:
            yield Outcome(UNITS_SOURCE, plan.source_id, True, False, BUILD_FAILURE, f"units: {exc}")
            continue
        for index in range(len(assignment)):
            try:
                view = units.unit_view(plan, assignment, index)
                yield Attempt(view.plan, exposure.context_for(view), _unit_rebuild(plan.source_id, index, root))
            except Exception as exc:
                yield Outcome(
                    UNITS_SOURCE, f"{plan.source_id}#u{index}", True, False, BUILD_FAILURE, f"unit: {exc}"
                )


def _unit_rebuild(floor_id: str, index: int, root) -> Optional[Callable[[], Attempt]]:
    """Rebuild one apartment from its floor on the grid partition, or None.

    Membership is derived from the rooms, so the grid floor has to be reassigned
    rather than reusing the vector assignment. A grid floor that drops a room can
    end up with fewer units, and the missing index raises, which the caller
    counts as the fallback failing.

    Offered on the same condition as `_floor_rebuild`, and for the same reason.
    This one used to be offered unconditionally, which cost nothing while the grid
    itself led, because the retry rebuilt the identical request. Under the
    rectified path it would silently rescue a unit on a different partition from
    the one the run is measuring, and a verdict mix assembled from two partitions
    describes neither.
    """
    if partition.DEFAULT_METHOD != "vector":
        return None

    def rebuild() -> Attempt:
        floor = msd.load_one(floor_id, root, method="grid")
        view = units.unit_view(floor, units.assign(floor), index)
        return Attempt(view.plan, exposure.context_for(view))

    return rebuild


def parse_baselines(pairs: list[str]) -> list[tuple[str, int]]:
    """Parse every --baseline NAME=N into the ordered source list a run grades.

    A baseline is a directory under data/samples, not a code path: the reader in
    `planaudit.datasets.generated` takes the name, so House-GAN and
    HouseDiffusion cost a flag rather than a module. The name is checked as a
    bare directory name here so a typo fails on the command line rather than as
    an empty group halfway through a run.
    """
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    for pair in pairs:
        name, sep, value = pair.partition("=")
        name = name.strip()
        if not sep or not name:
            raise SystemExit(f"--baseline wants NAME=N, got {pair!r}")
        if "/" in name or "\\" in name or name.startswith("."):
            raise SystemExit(f"--baseline NAME is a directory under data/samples, got {name!r}")
        if name in seen:
            raise SystemExit(f"--baseline {name} named twice")
        try:
            count = int(value)
        except ValueError:
            raise SystemExit(f"--baseline {name} wants a whole number, got {value!r}") from None
        if count <= 0:
            raise SystemExit(f"--baseline {name} wants a positive count, got {count}")
        seen.add(name)
        out.append((name, count))
    return out


def own_corpus(source: str) -> str:
    """The corpus a source answers to, before any calibration override.

    A source derived from MSD answers to MSD. Everything else answers to RPLAN:
    the RPLAN ground truth because it is RPLAN, the synthetic probe because it
    belongs to no corpus and the run document says so rather than leaving it
    implicit, and every generated baseline because a House-GAN++, House-GAN,
    HouseDiffusion or GSDiff sample is an attempt at an RPLAN plan, conditioned
    on an RPLAN graph on the RPLAN frame, so the corpus it has to answer to is
    RPLAN's.

    The answer is a rule rather than a table because the baseline vocabulary is
    open: a run names a baseline by its directory under `data/samples`, so a
    directory added tomorrow has to be answered without an edit here. It is also
    why the rule reads a prefix. The repository spells these sources two ways
    that committed artifacts fix, the run group names (`msd`, `msd units`,
    `rplan`, `synthetic`, a baseline directory) and the corpus-walk labels
    (`msd-floors`, `msd-units`, `houseganpp`), and both spellings carry the MSD
    prefix on exactly the MSD sources. One rule therefore answers both, which is
    what keeps the run and the scans from grading one plan against two floors.

    A calibration override replaces this answer rather than setting a second one
    beside it. The name returned is also the name of the fit over that corpus.
    """
    return "msd" if source.startswith("msd") else "rplan"


def unit_labels(source: str, plan: Plan) -> tuple[Optional[str], ...]:
    """The apartment each room of `plan` belongs to, one entry per room.

    This is what the exporter writes as floorcheck's `Room.unit`, and what the
    oracle counts to pick a plan's floor kind
    (`floorcheck.requirements.floor_kind`), so both sides grade one plan against
    one set of floors. An MSD apartment, which `units.unit_view` marks with its
    `unit_index`, labels every room with its own id; an MSD whole floor labels
    each room with the apartment `units.assign` derives for it, `u<index>` as in
    the `#u<index>` source ids of the apartment stream, and a room outside every
    apartment with None. A floor the derivation cannot read states no
    apartment. RPLAN and generated plans state none.
    """
    empty: tuple[Optional[str], ...] = (None,) * len(plan.rooms)
    if own_corpus(source) != "msd":
        return empty
    index = plan.notes.get("unit_index")
    if index is not None:
        return (f"u{index}",) * len(plan.rooms)
    try:
        assignment = units.assign(plan)
    except Exception:
        return empty
    return tuple(None if unit is None else f"u{unit}" for unit in assignment.unit_of)


def apartment_count(source: str, plan: Plan) -> int:
    """How many distinct apartments `unit_labels` states for `plan`."""
    return len({label for label in unit_labels(source, plan) if label is not None})
