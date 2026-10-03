"""The checker's own pass rates and margin, on a directory of exported plans.

Grades every plan in a directory with floorcheck and prints, per source, the
pass rate with a Wilson 95 per cent interval under three denominators, and the
margin between the lowest ground truth and the highest baseline with its
intervals (see `tools.intervals`).

The directory holds one subdirectory per source group with one JSON plan
document each, plus a `manifest.json` listing each group's `group`, `corpus`
and `directory`. A document's own `meta.group` and `meta.corpus` take
precedence over the manifest.

    python -m tools.checker_outcomes plans/ --out outcomes.md

`--profile` adds where each plan stopped, per check, and the share of plans
with no verdict. `--buildings PATH` adds the margin with an interval that
resamples whole MSD buildings, read from the cache `tools.msd_buildings` writes.

`--suppress CHECK` regrades with one check's rejection suppressed: the check is
evaluated, and a plan it would reject continues to the checks below. With
`roomProportions` this is the counterfactual the paper reports:

    python -m tools.checker_outcomes plans/ --suppress roomProportions
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from floorcheck.ladder import _WIRE_NAMES, NOT_RUN, REJECTED
from floorcheck.model import PlanRejected, plan_from_dict
from floorcheck.report import check
from tools import cluster_bootstrap
from tools.intervals import (
    BASELINES,
    DENOMINATORS,
    GROUND_TRUTH,
    GROUP_LABELS,
    ORDER,
    Margin,
    margin_rows,
    rate_rows,
    table,
    wilson,
)

ACCEPTED = "accepted"

# Not one of floorcheck's own outcomes: the checker raised where it should have
# answered, recorded as a refusal rather than a crash or an acceptance.
CHECKER_FAULT = "the checker raised"


def checker_verdict(
    document: dict[str, Any],
    corpus: str,
    statistics: str | os.PathLike | None = None,
    suppress: Iterable[str] = (),
) -> dict[str, Any]:
    """floorcheck's report on one plan document, reduced to its verdict.

    The corpus is passed explicitly rather than left to `check`'s inference from
    the source: a generated baseline answers to RPLAN's fit. ``statistics`` is
    forwarded to `floorcheck.report.check` unchanged; `None` keeps floorcheck's
    own committed default. ``suppress`` names checks whose rejection is
    suppressed (`floorcheck.ladder.grade`): each is still evaluated, and a plan
    it would reject continues to the checks below.

    Returns a dict with `ingest`, `stoppedAt`, `passed`, `reason` and `notRun`.
    """
    try:
        plan = plan_from_dict(document, name=str(document.get("name", "plan")))
    except PlanRejected as rejected:
        return {
            "ingest": rejected.reason,
            "stoppedAt": None,
            "passed": None,
            "reason": rejected.detail,
            "notRun": None,
        }
    try:
        report = check(
            plan, source="dataset-p05", corpus=corpus, statistics=statistics, suppress=suppress
        ).to_dict()
    except Exception as error:
        # The checker must grade invalid input with a reject reason rather than
        # an exception. A defect that breaks that is counted here as a class of
        # its own instead of stopping the whole measurement.
        return {
            "ingest": CHECKER_FAULT,
            "stoppedAt": None,
            "passed": None,
            "reason": f"{type(error).__name__}: {error}",
            "notRun": None,
        }
    ingest = report["ingest"]["outcome"]
    ladder = report.get("ladder")
    if ladder is None:
        return {
            "ingest": ingest,
            "stoppedAt": None,
            "passed": None,
            "reason": report["ingest"].get("detail", ""),
            "notRun": None,
        }
    stopped = next((rung for rung in ladder if rung["verdict"] == REJECTED), None)
    not_run = next((rung for rung in ladder if rung["verdict"] == NOT_RUN), None)
    if stopped is None and not_run is not None:
        # The not-run rule of section 9.4: the plan is admitted and no rung
        # rejected it, but a rung declined to measure it (a precondition
        # fault), so it reached no verdict at all, neither a pass
        # nor a rejection.
        return {
            "ingest": ingest,
            "stoppedAt": None,
            "passed": None,
            "reason": not_run.get("reason", ""),
            "notRun": not_run["name"],
        }
    return {
        "ingest": ingest,
        "stoppedAt": stopped["name"] if stopped else None,
        "passed": stopped is None,
        "reason": stopped.get("reason", "") if stopped else "",
        "notRun": None,
    }


@dataclass
class Verdict:
    """The checker's verdict on one plan."""

    group: str
    source_id: str
    corpus: Optional[str]
    checker_ingest: str
    checker_passed: Optional[bool]
    checker_not_run: Optional[str]
    reason: str = ""
    checker_stopped_at: Optional[str] = None

    @property
    def checker_admitted(self) -> bool:
        return self.checker_ingest == ACCEPTED

    @classmethod
    def from_report(
        cls, group: str, source_id: str, corpus: Optional[str], verdict: dict[str, Any]
    ) -> "Verdict":
        """Built from the dict `checker_verdict` returns."""
        return cls(
            group=group,
            source_id=source_id,
            corpus=corpus,
            checker_ingest=verdict["ingest"],
            checker_passed=verdict["passed"],
            checker_not_run=verdict["notRun"],
            reason=verdict.get("reason") or "",
            checker_stopped_at=verdict.get("stoppedAt"),
        )


def _last_component(path: str) -> str:
    """The last component of a path written on either platform."""
    parts = [part for part in re.split(r"[\\/]", path) if part]
    return parts[-1] if parts else ""


def _manifest(plans: Path) -> dict[str, dict[str, Any]]:
    """The manifest's entries, keyed by the last component of their directory."""
    path = plans / "manifest.json"
    if not path.is_file():
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for entry in json.loads(path.read_text(encoding="utf-8")):
        directory = entry.get("directory")
        key = _last_component(str(directory)) if directory else str(entry.get("group", ""))
        entries[key] = entry
    return entries


def grade_directory(
    plans: Path, statistics: str | os.PathLike | None = None, suppress: Iterable[str] = ()
) -> dict[str, list[Verdict]]:
    """Grade every plan document under `plans`, grouped by source group.

    Group and corpus come from each document's `meta.group` and `meta.corpus`,
    else from the `manifest.json` entry whose `directory` ends in the plan's
    subdirectory name, else the group is the subdirectory name and the corpus is
    left for floorcheck to infer. Groups and documents are read in sorted order.
    """
    plans = Path(plans)
    manifest = _manifest(plans)
    groups: dict[str, list[Verdict]] = {}
    for directory in sorted(p for p in plans.iterdir() if p.is_dir()):
        entry = manifest.get(directory.name, {})
        for path in sorted(directory.glob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            meta = document.get("meta") if isinstance(document.get("meta"), dict) else {}
            group = meta.get("group") or entry.get("group") or directory.name
            corpus = meta.get("corpus") or entry.get("corpus")
            source_id = meta.get("sourceId") or path.stem
            verdict = checker_verdict(document, corpus, statistics, suppress)
            groups.setdefault(group, []).append(
                Verdict.from_report(group, str(source_id), corpus, verdict)
            )
    return groups


@dataclass
class Outcomes:
    """One group's counts under the checker, in the shape `tools.intervals` reads."""

    graded: int
    passed: int
    admitted: int
    verdict: int

    def total(self, denominator: str) -> int:
        if denominator == "graded":
            return self.graded
        if denominator == "admitted":
            return self.admitted
        if denominator == "verdict":
            return self.verdict
        raise ValueError(f"unknown denominator {denominator!r}")

    @classmethod
    def of(cls, items: Iterable[Any]) -> "Outcomes":
        """Counts from items carrying `checker_passed`, `checker_admitted` and
        `checker_not_run`, such as `Verdict`."""
        items = list(items)
        admitted = sum(1 for p in items if p.checker_admitted)
        not_run: Counter = Counter(p.checker_not_run for p in items if p.checker_not_run)
        return cls(
            graded=len(items),
            passed=sum(1 for p in items if p.checker_passed is True),
            admitted=admitted,
            verdict=admitted - sum(not_run.values()),
        )


def suppression_note(suppress: Iterable[str]) -> str:
    """The line a counterfactual table carries under its heading."""
    names = ", ".join(f"`{name}`" for name in sorted(set(suppress)))
    return (
        f"Counterfactual: the rejection of {names} is suppressed. The check is "
        "evaluated, and a plan it would reject continues to the checks below."
    )


def checker_own_table(groups: dict[str, list[Any]]) -> list[str]:
    """The checker's own pass rates and margin, read off its own verdicts alone.

    Graded is every plan; admitted is the plans the checker ingested; reached a
    verdict is admitted less the plans a rung declined to measure: a precondition
    fault at `roomAppendices` (section 9.4's not-run rule). Only
    groups that are a ground truth or a baseline take part, and a population
    with no pair of the two prints its rate rows and no margin.
    """
    cells = {
        group: Outcomes.of(items)
        for group, items in groups.items()
        if group in GROUND_TRUTH or group in BASELINES
    }

    lines = ["## The checker's own pass rates and margin", ""]
    lines.append(
        "A plan admitted but left with no verdict, at roomAppendices on its"
        " orthogonality precondition (section 9.4), is counted here as a"
        " precondition stop, so"
        " \"reached a verdict\" below subtracts it from \"admitted\" rather"
        " than treating the two as equal (see this function's docstring)."
    )
    lines.append("")

    if not cells:
        lines.append("This population has no source that is a ground truth or a baseline.")
        return lines

    for denominator, label in DENOMINATORS:
        rows = rate_rows(cells, denominator)
        if not rows:
            continue
        lines.append(f"### Denominator: {label}")
        lines.append("")
        lines.append(table(rows, ["source", "passed", label, "pass rate", "95 per cent interval"]))
        lines.append("")

    margins = []
    for denominator, _ in DENOMINATORS:
        try:
            margins.append(Margin(cells, denominator))
        except SystemExit:
            continue
    if margins:
        lines.append(
            table(
                margin_rows(margins),
                [
                    "denominator",
                    "pair",
                    "margin",
                    "bootstrap, fixed pair",
                    "Newcombe, fixed pair",
                    "bootstrap, re-selected",
                ],
            )
        )
    else:
        lines.append(
            "No margin can be formed: this population carries no ground-truth"
            " source, no baseline source, or only one of the two."
        )
    return lines


# ---------------------------------------------------------------------------
# Where each plan stopped, and the share with no verdict (`--profile`).

# The ladder's rungs in the order floorcheck runs them.
RUNG_ORDER = tuple(_WIRE_NAMES[number] for number in sorted(_WIRE_NAMES))

REFUSED_AT_INGEST = "refused at ingest"
CHECKER_RAISED = "checker raised"
PRECONDITION_STOP = "precondition stop"
PASSED = "passed"
# An admitted plan that failed with no rung recorded; `checker_verdict` never
# writes one, but a hand-built verdict can.
UNNAMED_RUNG = "rejected, rung not recorded"


def stop_class(item: Any) -> str:
    """Where one plan stopped: refused, raised, a rung's name, a precondition stop or passed."""
    if not item.checker_admitted:
        return CHECKER_RAISED if item.checker_ingest == CHECKER_FAULT else REFUSED_AT_INGEST
    if item.checker_not_run:
        return PRECONDITION_STOP
    if item.checker_passed is True:
        return PASSED
    return item.checker_stopped_at or UNNAMED_RUNG


def ordered_groups(groups: dict[str, list[Any]]) -> list[str]:
    """The groups in the order the rate tables use, then any other group by name."""
    return [g for g in ORDER if g in groups] + sorted(g for g in groups if g not in ORDER)


def _label(group: str) -> str:
    return GROUP_LABELS.get(group, group)


def profile_columns(groups: dict[str, list[Any]]) -> list[str]:
    """The stop classes as columns: ingest, raised, every rung met in ladder order,
    any rung the ladder does not name, precondition stop, passed."""
    seen = {stop_class(item) for items in groups.values() for item in items}
    fixed = {REFUSED_AT_INGEST, CHECKER_RAISED, PRECONDITION_STOP, PASSED}
    rungs = [rung for rung in RUNG_ORDER if rung in seen]
    other = sorted(seen - fixed - set(RUNG_ORDER))
    return [REFUSED_AT_INGEST, CHECKER_RAISED, *rungs, *other, PRECONDITION_STOP, PASSED]


def refusal_profile(groups: dict[str, list[Any]]) -> dict[str, Counter]:
    """Per group, how many plans stopped in each class; each group's counts sum to its plans."""
    return {group: Counter(stop_class(item) for item in groups[group]) for group in ordered_groups(groups)}


def _share(count: int, total: int) -> str:
    return f"{count:,} ({100.0 * count / total:.1f} per cent)" if total else f"{count:,}"


def profile_rows(groups: dict[str, list[Any]]) -> tuple[list[str], list[list[str]]]:
    """The header and rows of the refusals-by-check table."""
    columns = profile_columns(groups)
    rows = []
    for group, counts in refusal_profile(groups).items():
        graded = len(groups[group])
        rows.append([_label(group), f"{graded:,}"] + [_share(counts.get(c, 0), graded) for c in columns])
    return ["source", "graded", *columns], rows


def no_verdict_rows(groups: dict[str, list[Any]]) -> list[list[str]]:
    """Per group: graded, admitted, reached a verdict, no verdict and its share of graded."""
    rows = []
    for group in ordered_groups(groups):
        cell = Outcomes.of(groups[group])
        missing = cell.graded - cell.verdict
        interval = wilson(missing, cell.graded)
        share = f"{interval[0]:.1f}" if interval else "n/a"
        bounds = f"{interval[1]:.1f} to {interval[2]:.1f}" if interval else "n/a"
        rows.append(
            [_label(group), f"{cell.graded:,}", f"{cell.admitted:,}", f"{cell.verdict:,}",
             f"{missing:,}", share, bounds]
        )
    return rows


def profile_lines(groups: dict[str, list[Any]]) -> list[str]:
    """The `--profile` sections: refusals by check, and the share with no verdict."""
    header, rows = profile_rows(groups)
    lines = ["## Refusals by check", ""]
    lines.append(
        "Each plan in exactly one class: refused at ingest, the checker raised, the"
        " first rung that rejected it, a precondition stop (admitted, no rung"
        " rejected, a rung declined to measure), or passed. Each cell is a count"
        " and its share of graded plans."
    )
    lines += ["", table(rows, header), ""]
    lines += ["## Share with no verdict", ""]
    lines.append(
        "No verdict is graded less reached a verdict: refused at ingest, the checker"
        " raised, or a precondition stop. The interval is Wilson, 95 per cent."
    )
    lines += [
        "",
        table(
            no_verdict_rows(groups),
            ["source", "graded", "admitted", "reached a verdict", "no verdict",
             "per cent of graded", "95 per cent interval"],
        ),
    ]
    return lines


# ---------------------------------------------------------------------------
# The margin with MSD buildings resampled whole (`--buildings`).


def floor_of(source_id: str) -> str:
    """The MSD floor id a unit's id names (`<floor id>#u<index>`); a floor's own id is itself."""
    return source_id.split("#u", 1)[0]


def margin_groups(groups: dict[str, list[Any]]) -> list[str]:
    return [g for g in ordered_groups(groups) if g in GROUND_TRUTH or g in BASELINES]


def plan_flags(item: Any) -> list[int]:
    """Graded, admitted, reached a verdict and passed for one plan."""
    admitted = int(item.checker_admitted)
    verdict = int(item.checker_admitted and not item.checker_not_run)
    return [1, admitted, verdict, int(item.checker_passed is True)]


def cluster_labels(items: list[Any], buildings: dict[str, str]) -> list[str]:
    """One label per plan: an MSD plan's building, every other plan its own."""
    return [
        f"building {buildings[floor_of(item.source_id)]}" if item.corpus == "msd" else f"plan {index}"
        for index, item in enumerate(items)
    ]


def building_sources(
    groups: dict[str, list[Any]], buildings: dict[str, str]
) -> dict[str, cluster_bootstrap.Source]:
    """Every ground-truth and baseline group as a cluster-bootstrap source.

    Raises SystemExit naming how many MSD floors the buildings key lacks.
    """
    names = margin_groups(groups)
    missing = {
        floor_of(item.source_id)
        for g in names
        for item in groups[g]
        if item.corpus == "msd" and floor_of(item.source_id) not in buildings
    }
    if missing:
        raise SystemExit(
            f"{len(missing):,} MSD floors are missing from the buildings cache;"
            " build it over every floor of the export with floorcheck-buildings"
        )
    return {
        g: cluster_bootstrap.Source(
            g,
            np.array([plan_flags(item) for item in groups[g]], dtype=np.int64).reshape(-1, 4),
            np.array(cluster_labels(groups[g], buildings), dtype=object),
        )
        for g in names
    }


def building_margin_lines(
    groups: dict[str, list[Any]],
    buildings: dict[str, str],
    replicates: int = cluster_bootstrap.REPLICATES,
    seed: int = cluster_bootstrap.SEED,
) -> list[str]:
    """The `--buildings` section: the re-selected margin with buildings resampled whole."""
    sources = building_sources(groups, buildings)
    lines = ["## Margin with buildings resampled whole", ""]
    counts = ", ".join(
        f"{_label(g)} {len(np.unique(s.clusters)):,}" for g, s in sources.items()
    )
    lines.append(f"Clusters per group: {counts or 'none'}.")
    lines.append("")
    if not any(g in sources for g in GROUND_TRUTH) or not any(b in sources for b in BASELINES):
        lines.append(
            "No margin can be formed: this population carries no ground-truth"
            " source, no baseline source, or only one of the two."
        )
        return lines
    intervals = cluster_bootstrap.bootstrap_many(
        sources.values(),
        [cluster_bootstrap.margin(0, d) for d, _ in DENOMINATORS],
        replicates=replicates,
        seed=seed,
    )
    rows = [
        [label, f"{interval.point:+.1f}", interval.text()]
        for (_, label), interval in zip(DENOMINATORS, intervals)
    ]
    lines.append(
        table(rows, ["denominator", "margin, re-selected",
                     f"95 per cent interval, {replicates:,} cluster resamples, seed {seed}"])
    )
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="floorcheck-outcomes",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("plans", type=Path, help="directory of exported plan documents")
    parser.add_argument(
        "--category-statistics",
        default=None,
        metavar="PATH",
        help="category-statistics table to grade against (default: floorcheck's own)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, metavar="PATH", help="write the Markdown here"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="also print where each plan stopped, per check, and the share with no verdict",
    )
    parser.add_argument(
        "--buildings",
        type=Path,
        default=None,
        metavar="PATH",
        help="buildings cache from floorcheck-buildings: also print the margin with MSD buildings resampled whole",
    )
    parser.add_argument(
        "--suppress",
        action="append",
        default=[],
        choices=RUNG_ORDER,
        metavar="CHECK",
        help="counterfactual: evaluate CHECK but suppress its rejection, so a plan it "
        "would reject continues to the checks below (repeatable; e.g. roomProportions)",
    )
    parser.add_argument("--replicates", type=int, default=cluster_bootstrap.REPLICATES)
    parser.add_argument("--seed", type=int, default=cluster_bootstrap.SEED)
    args = parser.parse_args(argv)

    buildings = None
    if args.buildings is not None:
        from tools import msd_buildings

        buildings = msd_buildings.load(msd_buildings.THRESHOLD, cache=args.buildings)
    groups = grade_directory(args.plans, args.category_statistics, args.suppress)
    lines = checker_own_table(groups)
    if args.suppress:
        # The heading is the shipped table's; say at once that this is not it.
        lines[1:1] = ["", suppression_note(args.suppress)]
    if args.profile:
        lines += [""] + profile_lines(groups)
    if buildings is not None:
        lines += [""] + building_margin_lines(groups, buildings, args.replicates, args.seed)
    text = "\n".join(lines) + "\n"
    if args.out is None:
        print(text, end="")
    else:
        args.out.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
