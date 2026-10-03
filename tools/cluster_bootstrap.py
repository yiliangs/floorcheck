"""Cluster bootstrap intervals for a margin between sources of graded plans.

`tools.intervals` treats every graded plan as independent. A corpus whose plans
share a building is not: several MSD floors often belong to one building. This
module resamples whole clusters instead of plans and reports the same
re-selected margin `tools.intervals.Margin` quotes, the lowest-passing ground
truth less the highest-passing baseline, recomputed in every replicate.

A source is a matrix of per-plan indicator columns and a cluster label per
plan. The columns are, per ruler, graded, admitted, reached a verdict and
passed. A replicate draws, for every source independently, as many clusters as
the source has, uniformly with replacement, and sums their columns. A source
whose plans are independent gives every plan a cluster of its own.

A held-out shift is a paired statistic: the same plans graded under two rulers
side by side in one matrix, so a replicate resamples a cluster once and carries
its outcome under both rulers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from tools.intervals import BASELINES, GROUND_TRUTH

REPLICATES = 10000
SEED = 20260928
CHUNK = 250

# Column offsets inside one ruler's block of four.
GRADED, ADMITTED, VERDICT, PASSED = range(4)
DENOMINATOR_COLUMN = {"graded": GRADED, "admitted": ADMITTED, "verdict": VERDICT}


@dataclass
class Source:
    """One source's plans: indicator columns (plans x 4 per ruler) and cluster labels."""

    name: str
    flags: np.ndarray
    clusters: np.ndarray

    def totals(self) -> np.ndarray:
        """Column sums over every plan."""
        return self.flags.sum(axis=0)

    def cluster_totals(self) -> np.ndarray:
        """Column sums per cluster, clusters x columns."""
        _, labels = np.unique(self.clusters, return_inverse=True)
        out = np.zeros((labels.max() + 1, self.flags.shape[1]), dtype=np.int64)
        np.add.at(out, labels, self.flags)
        return out


def rate(totals: np.ndarray, ruler: int, denominator: str) -> np.ndarray:
    """Per cent passing under `ruler` over `denominator`, along the last axis."""
    base = 4 * ruler
    return 100.0 * totals[..., base + PASSED] / totals[..., base + DENOMINATOR_COLUMN[denominator]]


def reselected(totals: dict[str, np.ndarray], ruler: int, denominator: str) -> np.ndarray:
    """Lowest-passing ground truth less highest-passing baseline."""
    truth = np.min([rate(totals[g], ruler, denominator) for g in GROUND_TRUTH if g in totals], axis=0)
    base = np.max([rate(totals[b], ruler, denominator) for b in BASELINES if b in totals], axis=0)
    return truth - base


def replicate_totals(source: Source, replicates: int, rng: np.random.Generator) -> np.ndarray:
    """Column totals of `replicates` cluster resamples, replicates x columns."""
    per_cluster = source.cluster_totals().astype(np.float64)
    clusters = per_cluster.shape[0]
    out = np.empty((replicates, per_cluster.shape[1]))
    for start in range(0, replicates, CHUNK):
        size = min(CHUNK, replicates - start)
        weights = rng.multinomial(clusters, np.full(clusters, 1.0 / clusters), size=size)
        out[start:start + size] = weights @ per_cluster
    return out


@dataclass
class Interval:
    point: float
    low: float
    high: float

    def text(self) -> str:
        return f"{self.low:+.1f} to {self.high:+.1f}"


def bootstrap_many(
    sources: Iterable[Source],
    statistics: Sequence[Callable[[dict[str, np.ndarray]], np.ndarray]],
    *,
    replicates: int = REPLICATES,
    seed: int = SEED,
) -> list[Interval]:
    """Percentile intervals of each statistic over one set of cluster resamples."""
    sources = list(sources)
    rng = np.random.default_rng(seed)
    observed = {s.name: s.totals().astype(np.float64) for s in sources}
    drawn = {s.name: replicate_totals(s, replicates, rng) for s in sources}
    out = []
    for statistic in statistics:
        low, high = np.percentile(statistic(drawn), [2.5, 97.5])
        out.append(Interval(float(statistic(observed)), float(low), float(high)))
    return out


def bootstrap(sources, statistic, *, replicates: int = REPLICATES, seed: int = SEED) -> Interval:
    """Percentile interval of `statistic` over cluster resamples of every source."""
    return bootstrap_many(sources, [statistic], replicates=replicates, seed=seed)[0]


def margin(ruler: int, denominator: str) -> Callable[[dict[str, np.ndarray]], np.ndarray]:
    return lambda totals: reselected(totals, ruler, denominator)


def shift(denominator: str, before: int = 0, after: int = 1) -> Callable[[dict[str, np.ndarray]], np.ndarray]:
    """The re-selected margin under ruler `after` less that under ruler `before`."""
    return lambda totals: reselected(totals, after, denominator) - reselected(totals, before, denominator)


def reclustered(source: Source, cluster_of: Callable[[str], str], ids: Sequence[str]) -> Source:
    """The same plans with cluster labels from `cluster_of` over their ids."""
    return Source(source.name, source.flags, np.array([cluster_of(i) for i in ids], dtype=object))
