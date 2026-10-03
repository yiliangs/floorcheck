"""Pass rates and margins with 95 per cent intervals, on per-source counts.

Every function here reads a source's counts through a small duck-typed shape: an
object with an integer `passed` and a method `total(denominator)` that returns
the count under one of the three denominators in `DENOMINATORS`. The rate rows
put a Wilson score interval on each source's pass rate. The margin is the lowest
ground-truth pass rate minus the highest baseline pass rate, reported with three
intervals: a parametric bootstrap on the fixed pair, Newcombe's hybrid score
interval on the same pair, and a parametric bootstrap that re-selects the two
extremes on every draw. The re-selected interval is the one to quote, because
which source is lowest or highest is itself estimated from the same counts.

Standard library only.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Protocol

# The group names, and how the tables label them.
GROUP_LABELS = {
    "msd": "MSD ground truth, grid path",
    "msd units": "MSD ground truth, units path",
    "rplan": "RPLAN ground truth",
    "houseganpp": "House-GAN++",
    "housegan": "House-GAN",
    "housediffusion": "HouseDiffusion",
}

# Ground truth and baselines, for the margin. `msd` is the grid-path row, held
# out of the margin as a build-rate measure rather than a gate row.
GROUND_TRUTH = ("rplan", "msd units")
BASELINES = ("houseganpp", "housegan", "housediffusion")

# The three denominators, keyed as `total` takes them and labelled as the tables
# name them. Graded plans is every plan looked at; admitted plans subtracts the
# plans refused before the ladder; reached a verdict further subtracts admitted
# plans a rung declined to measure.
DENOMINATORS = (
    ("graded", "graded plans"),
    ("admitted", "admitted plans"),
    ("verdict", "reached a verdict"),
)

# 95 per cent, two-sided.
Z = 1.959963984540054

# Draws for the parametric bootstrap, and the seed that makes it reproduce.
DRAWS = 20000
SEED = 20260916

ORDER = ("rplan", "msd units", "houseganpp", "housegan", "housediffusion")


class Counts(Protocol):
    """The shape every function here reads a source's counts through."""

    passed: int

    def total(self, denominator: str) -> int: ...


def table(rows: list[list[str]], header: list[str]) -> str:
    """A Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join(lines)


def wilson(passed: int, total: int) -> Optional[tuple[float, float, float]]:
    """The Wilson score interval, as (point, low, high) in per cent.

    Wilson rather than the normal approximation because a pass rate near zero or
    one would otherwise get a bound outside the unit interval.
    """
    if total <= 0:
        return None
    point = passed / total
    centre = (passed + Z * Z / 2) / (total + Z * Z)
    half = (Z / (total + Z * Z)) * math.sqrt(passed * (total - passed) / total + Z * Z / 4)
    return (100.0 * point, 100.0 * max(0.0, centre - half), 100.0 * min(1.0, centre + half))


def newcombe(
    passed_a: int, total_a: int, passed_b: int, total_b: int
) -> Optional[tuple[float, float, float]]:
    """Newcombe's hybrid score interval on a difference of two independent rates.

    Built from the two Wilson intervals, with no simulation, so it is the check
    on the bootstrap rather than a second opinion.
    """
    first = wilson(passed_a, total_a)
    second = wilson(passed_b, total_b)
    if first is None or second is None:
        return None
    point_a, low_a, high_a = first
    point_b, low_b, high_b = second
    difference = point_a - point_b
    low = difference - math.hypot(point_a - low_a, high_b - point_b)
    high = difference + math.hypot(high_a - point_a, point_b - low_b)
    return (difference, low, high)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("no draws")
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_fixed(
    passed_a: int, total_a: int, passed_b: int, total_b: int, *, draws: int = DRAWS, seed: int = SEED
) -> Optional[tuple[float, float, float]]:
    """A parametric bootstrap on one named pair, as (point, low, high) in per cent."""
    if total_a <= 0 or total_b <= 0:
        return None
    rng = random.Random(seed)
    rate_a = passed_a / total_a
    rate_b = passed_b / total_b
    differences = sorted(
        100.0 * (rng.binomialvariate(total_a, rate_a) / total_a)
        - 100.0 * (rng.binomialvariate(total_b, rate_b) / total_b)
        for _ in range(draws)
    )
    return (
        100.0 * (rate_a - rate_b),
        _percentile(differences, 0.025),
        _percentile(differences, 0.975),
    )


def bootstrap_reselected(
    ground_truth: list[tuple[int, int]],
    baselines: list[tuple[int, int]],
    *,
    draws: int = DRAWS,
    seed: int = SEED,
) -> Optional[tuple[float, float, float]]:
    """A parametric bootstrap that takes the extremes again on every draw.

    Each argument is a list of (passed, total) for the sources of that group. The
    reported margin is the lowest ground truth minus the highest baseline, and
    which source is lowest or highest is itself estimated from the same counts,
    so an interval that fixes the pair understates the spread.
    """
    if not ground_truth or not baselines:
        return None
    rng = random.Random(seed)
    truth = [(total, passed / total) for passed, total in ground_truth]
    baseline = [(total, passed / total) for passed, total in baselines]
    differences = sorted(
        min(100.0 * rng.binomialvariate(total, rate) / total for total, rate in truth)
        - max(100.0 * rng.binomialvariate(total, rate) / total for total, rate in baseline)
        for _ in range(draws)
    )
    point = 100.0 * min(rate for _, rate in truth) - 100.0 * max(rate for _, rate in baseline)
    return (point, _percentile(differences, 0.025), _percentile(differences, 0.975))


def rate_rows(cells: dict[str, Counts], denominator: str) -> list[list[str]]:
    """One row per source in `ORDER`: passed, total, pass rate and its interval."""
    rows = []
    for group in ORDER:
        cell = cells.get(group)
        if cell is None:
            continue
        interval = wilson(cell.passed, cell.total(denominator))
        if interval is None:
            continue
        point, low, high = interval
        rows.append(
            [
                GROUP_LABELS[group],
                f"{cell.passed:,}",
                f"{cell.total(denominator):,}",
                f"{point:.1f}",
                f"{low:.1f} to {high:.1f}",
            ]
        )
    return rows


def _extreme(cells: dict[str, Counts], names, denominator: str, pick) -> Optional[str]:
    scored = []
    for name in names:
        cell = cells.get(name)
        if cell is None:
            continue
        total = cell.total(denominator)
        if total > 0:
            scored.append((cell.passed / total, name))
    if not scored:
        return None
    return pick(scored)[1]


class Margin:
    """One denominator's margin, with every interval put on it.

    Raises SystemExit when the counts carry no ground truth or no baseline with
    a nonzero total under this denominator, so there is no margin to report.
    """

    def __init__(
        self, cells: dict[str, Counts], denominator: str, *, draws: int = DRAWS, seed: int = SEED
    ) -> None:
        self.denominator = denominator
        self.low_name = _extreme(cells, GROUND_TRUTH, denominator, min)
        self.high_name = _extreme(cells, BASELINES, denominator, max)
        if self.low_name is None or self.high_name is None:
            raise SystemExit(f"the {denominator} denominator has no margin to report")
        low_cell = cells[self.low_name]
        high_cell = cells[self.high_name]
        pair = (
            low_cell.passed,
            low_cell.total(denominator),
            high_cell.passed,
            high_cell.total(denominator),
        )
        self.fixed = bootstrap_fixed(*pair, draws=draws, seed=seed)
        self.hybrid = newcombe(*pair)
        self.reselected = bootstrap_reselected(
            [(cells[n].passed, cells[n].total(denominator)) for n in GROUND_TRUTH if n in cells],
            [(cells[n].passed, cells[n].total(denominator)) for n in BASELINES if n in cells],
            draws=draws,
            seed=seed,
        )
        self.low_rate = wilson(low_cell.passed, low_cell.total(denominator))
        self.high_rate = wilson(high_cell.passed, high_cell.total(denominator))
        assert self.fixed and self.hybrid and self.reselected
        assert self.low_rate and self.high_rate

    @property
    def rates_overlap(self) -> bool:
        """Whether the two extreme sources' own intervals overlap.

        A weaker question than whether the margin clears zero: two intervals can
        overlap while the interval on their difference still excludes zero.
        """
        return self.low_rate[1] <= self.high_rate[2] and self.high_rate[1] <= self.low_rate[2]

    @property
    def excludes_zero(self) -> bool:
        return self.reselected[1] > 0 or self.reselected[2] < 0


def margin_rows(margins: list[Margin]) -> list[list[str]]:
    """One row per margin: denominator, pair, point and the three intervals."""
    return [
        [
            dict(DENOMINATORS)[m.denominator],
            f"{GROUP_LABELS[m.low_name]} over {GROUP_LABELS[m.high_name]}",
            f"{m.fixed[0]:+.1f}",
            f"{m.fixed[1]:+.1f} to {m.fixed[2]:+.1f}",
            f"{m.hybrid[1]:+.1f} to {m.hybrid[2]:+.1f}",
            f"{m.reselected[1]:+.1f} to {m.reselected[2]:+.1f}",
        ]
        for m in margins
    ]
