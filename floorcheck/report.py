"""One plan in, one JSON report out.

The report's shape follows the item's own requirement: the ingest outcome first,
then the rungs in ladder order, with the requirement source named in the header.
Everything a reader needs to place a verdict is in the header, because section 9.4
says two conditions "travel with those numbers and neither may be dropped", and a
verdict printed without its requirement source is a number nobody can place.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from . import __version__
from .ingest import Ingest, ingest
from .ladder import Graded, construction_observations, grade
from .measure import dominant_direction
from .model import Plan, PlanRejected, load_plan, plan_from_json
from .requirements import floor_kind, requirements
from .scope import AREA_PROGRAM, exposure_refusals, unit_count


@dataclass
class Report:
    plan: str
    source: str
    requirement_source: str
    corpus: str
    floor_kind: str
    ingest: Ingest
    graded: Graded | None
    observations: list[dict[str, Any]]
    refusals: list[dict[str, str]]
    # Rungs whose rejection was suppressed for a counterfactual (`grade`'s
    # `suppress`); named in the header only when there are any.
    suppressed: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        header = {
            "checker": "floorcheck",
            "version": __version__,
            "specification": "the transfer specification, public edition",
            "plan": self.plan,
            "source": self.source,
            "requirementSource": self.requirement_source,
            "corpus": self.corpus,
            "floorKind": self.floor_kind,
        }
        if self.suppressed:
            header["suppressed"] = sorted(self.suppressed)
        ingest_block: dict[str, Any] = {
            "outcome": self.ingest.refusal or "accepted",
            "seamsClosed": self.ingest.seams_closed,
            "measurements": {
                key: round(value, 9) for key, value in self.ingest.measurements.items()
            },
        }
        if self.ingest.refusal:
            ingest_block["detail"] = self.ingest.detail
        if self.ingest.swallowed:
            ingest_block["swallowed"] = [list(pair) for pair in self.ingest.swallowed]

        body: dict[str, Any] = {
            "header": header,
            "ingest": ingest_block,
            "refusals": self.refusals,
        }

        if self.graded is None:
            # Section 7.3: an ingest refusal "happens before or instead of a
            # verdict, so [it is] not [a rung]". No rung row is invented for a
            # plan that never reached the ladder.
            body["ladder"] = None
            body["verdict"] = "no verdict: refused at ingest"
            return body

        body["ladder"] = [
            {
                "rung": result.number,
                "name": result.name,
                "verdict": result.verdict,
                **({"reason": result.reason} if result.reason else {}),
                **({"worst": result.worst} if result.worst else {}),
            }
            for result in self.graded.rungs
        ]
        body["observations"] = {"minWidth": self.observations}
        if self.graded.stopped_at is not None:
            body["verdict"] = f"stopped at {self.graded.stopped_at}"
        elif self.graded.not_run is not None:
            body["verdict"] = f"no verdict: {self.graded.not_run} not run"
        else:
            body["verdict"] = "passed every transferable rung"
        return body


def check(
    plan: Plan,
    source: str = "dataset-p05",
    corpus: str | None = None,
    statistics: str | os.PathLike | None = None,
    suppress: Iterable[str] = (),
) -> Report:
    """Grade one plan. ``statistics`` names the `dataset-p05` category-statistics
    table to grade against; ``None`` keeps the committed default at
    `floorcheck.requirements.DEFAULT_CATEGORY_STATISTICS_PATH`. ``suppress`` is
    passed to `grade`: rungs whose rejection is suppressed for a counterfactual."""
    suppressed = frozenset(suppress)
    corpus = corpus or ("msd" if plan.source == "msd" else "rplan")
    kind = floor_kind(corpus, unit_count(plan))
    result = ingest(plan)

    refusals = [
        {"name": refusal.name, "kind": refusal.kind, "reason": refusal.reason}
        for refusal in exposure_refusals(plan)
    ]
    refusals.append(
        {"name": AREA_PROGRAM.name, "kind": AREA_PROGRAM.kind, "reason": AREA_PROGRAM.reason}
    )

    if result.refused:
        return Report(
            plan=plan.name,
            source=plan.source,
            requirement_source=source,
            corpus=corpus,
            floor_kind=kind,
            ingest=result,
            graded=None,
            observations=[],
            refusals=refusals,
            suppressed=suppressed,
        )

    rings = list(result.rings)
    graded = grade(
        plan, rings, source=source, corpus=corpus, statistics=statistics, suppress=suppressed
    )
    reqs = requirements(source, kind, statistics)
    observations = construction_observations(plan, rings, reqs, dominant_direction(rings))

    return Report(
        plan=plan.name,
        source=plan.source,
        requirement_source=source,
        corpus=corpus,
        floor_kind=kind,
        ingest=result,
        graded=graded,
        observations=observations,
        refusals=refusals,
        suppressed=suppressed,
    )


def check_json(
    text: str,
    source: str = "dataset-p05",
    name: str = "plan",
    statistics: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Grade one plan given as JSON text, returning a reject reason rather than
    raising.

    The checker's contract: it "must grade any plan in
    RPLAN or MSD format, including invalid generator output (overlaps, rooms
    outside the boundary, missing entries), with a reject reason rather than an
    exception".
    """
    try:
        plan = plan_from_json(text, name=name)
    except PlanRejected as rejected:
        return {
            "header": {"checker": "floorcheck", "version": __version__, "plan": name},
            "ingest": {"outcome": rejected.reason, "detail": rejected.detail},
            "ladder": None,
            "verdict": "no verdict: the input model of section 2 cannot express this plan",
        }
    return check(plan, source=source, statistics=statistics).to_dict()


def check_file(
    path: str, source: str = "dataset-p05", statistics: str | os.PathLike | None = None
) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        return {
            "header": {"checker": "floorcheck", "version": __version__, "plan": path},
            "ingest": {"outcome": "unreadable-file", "detail": str(error)},
            "ladder": None,
            "verdict": "no verdict: the file could not be read",
        }
    return check_json(text, source=source, name=path, statistics=statistics)


__all__ = ["Report", "check", "check_file", "check_json", "load_plan", "unit_count"]
