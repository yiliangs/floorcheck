"""`floorcheck <plan.json>`.

The interface is fixed as "`floorcheck <plan.json>`
reading a documented neutral plan schema ... Output: verdict per rung plus the
reject-reason vocabulary, machine-readable and human-readable."

The exit status is 0 when the checker ran, whatever it decided, and 2 only when it
could not run at all. A refusal is a result, not an error: the same document
requires that invalid generator output be graded "with a reject reason rather than
an exception", and a non-zero status for a refusal would make a training loop
treat a graded rejection as a crash.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .report import check_file
from .requirements import DEFAULT_CATEGORY_STATISTICS_PATH, SOURCE_NAMES


def _human(report: dict[str, Any]) -> str:
    lines: list[str] = []
    header = report.get("header", {})
    lines.append(f"plan            {header.get('plan', '?')}")
    lines.append(f"dataset source  {header.get('source', '?')}")
    lines.append(f"requirements    {header.get('requirementSource', '?')}")
    if header.get("corpus"):
        lines.append(f"corpus          {header['corpus']}")
    if header.get("floorKind"):
        lines.append(f"floor kind      {header['floorKind']}")
    lines.append("")

    ingest = report.get("ingest", {})
    lines.append(f"ingest          {ingest.get('outcome', '?')}")
    if ingest.get("detail"):
        lines.append(f"                {ingest['detail']}")
    if ingest.get("seamsClosed"):
        lines.append(f"                {ingest['seamsClosed']} seam(s) closed")
    lines.append("")

    ladder = report.get("ladder")
    if ladder is None:
        lines.append("ladder          not reached")
    else:
        lines.append("ladder")
        for row in ladder:
            lines.append(f"  {row['rung']}. {row['name']:<20} {row['verdict']}")
            if row.get("reason"):
                lines.append(f"     {row['reason']}")

    refusals = report.get("refusals") or []
    if refusals:
        lines.append("")
        lines.append("refused, with the section that refuses them")
        for row in refusals:
            lines.append(f"  {row['name']:<20} {row['kind']}")

    lines.append("")
    lines.append(f"verdict         {report.get('verdict', '?')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="floorcheck",
        description=(
            "Grade one floor plan against the transferable subset of the gate "
            "ladder, as specified in the public edition of the transfer specification. A "
            "clean-room implementation: no private solver code, no FFI, no production "
            "constant."
        ),
    )
    parser.add_argument("plan", help="path to a plan JSON file")
    parser.add_argument(
        "--requirements",
        choices=list(SOURCE_NAMES),
        default="dataset-p05",
        help=(
            "which public requirement source to grade against. Section 1.3: D3 "
            "may implement standards and dataset-p05 and may not implement "
            "stand-in. Default: dataset-p05."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="machine-readable JSON (default) or the human-readable summary",
    )
    parser.add_argument(
        "--category-statistics",
        default=None,
        metavar="PATH",
        help=(
            "the category-statistics table a dataset-p05 run is graded against. "
            f"Default: the committed {DEFAULT_CATEGORY_STATISTICS_PATH}."
        ),
    )
    args = parser.parse_args(argv)

    try:
        report = check_file(
            args.plan, source=args.requirements, statistics=args.category_statistics
        )
    except Exception as error:  # the checker itself broke, which is status 2
        print(f"floorcheck: internal error: {error}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=False))
    else:
        print(_human(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
