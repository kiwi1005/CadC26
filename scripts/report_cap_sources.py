#!/usr/bin/env python3
"""Report exact official-v10 cap attribution from HCFP JSON evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.cap_margin import build_cap_report, render_markdown  # noqa: E402
from hcfp.score_attribution import CAP_LOG  # noqa: E402
from hcfp.verify import ALPHA, BETA, soft_violation_normalized  # noqa: E402


_PRIMARY_BLOCKERS = ("none", "hard", "soft", "area", "hpwl", "projection", "mixed")
_STAGES = ("raw", "projected", "post", "final")


def _canonicalize_payload(payload: object) -> object:
    """Add harmless aliases used by evaluator reports before attribution."""

    if isinstance(payload, list):
        return [_canonicalize_payload(row) for row in payload]
    if not isinstance(payload, Mapping):
        return payload
    result = {key: _canonicalize_payload(value) for key, value in payload.items()}
    if "max_possible_violations" not in result:
        maximum = result.get("max_soft_violations")
        if maximum is not None:
            result["max_possible_violations"] = maximum
    return result


def _source_rows(payload: object) -> list[dict[str, object]]:
    """Flatten the supported input envelopes for report-side enrichment."""

    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    lanes = payload.get("lanes")
    if isinstance(lanes, Mapping):
        rows: list[dict[str, object]] = []
        for lane, values in lanes.items():
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, Mapping):
                    continue
                row = dict(value)
                row.setdefault("lane", lane)
                row.setdefault("source", lane)
                rows.append(row)
        return rows
    for field in ("test_results", "results", "cases"):
        values = payload.get(field)
        if isinstance(values, list):
            return [dict(row) for row in values if isinstance(row, Mapping)]
    return []


def _case_metadata(payload: object, test_id: object) -> Mapping[str, object] | None:
    if not isinstance(payload, Mapping):
        return None
    metadata = payload.get("case_metadata")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get(str(test_id), metadata.get(test_id))
    return value if isinstance(value, Mapping) else None


def _variants(row: Mapping[str, object] | None) -> list[Mapping[str, object]]:
    if not isinstance(row, Mapping):
        return []
    variants = [row]
    for key in ("selected", "stages", "raw", "projected", "post", "post_bdp", "final"):
        value = row.get(key)
        if isinstance(value, Mapping):
            variants.append(value)
    return variants


def _lookup(
    row: Mapping[str, object] | None,
    names: Sequence[str],
) -> object | None:
    for variant in _variants(row):
        for name in names:
            value = variant.get(name)
            if value is not None:
                return value
    return None


def _match_row(case: Mapping[str, object], rows: list[dict[str, object]]) -> Mapping[str, object] | None:
    test_id = case.get("test_id")
    lane = case.get("lane")
    source = case.get("source")
    matches = [
        row
        for row in rows
        if test_id is None or row.get("test_id") == test_id
    ]
    if lane is not None:
        lane_matches = [row for row in matches if row.get("lane") == lane]
        if lane_matches:
            matches = lane_matches
    if source is not None:
        source_matches = [row for row in matches if row.get("source") == source]
        if source_matches:
            matches = source_matches
    return matches[0] if matches else None


def _stage_margin(row: Mapping[str, object] | None, stage: str) -> object | None:
    if not isinstance(row, Mapping):
        return None
    aliases = {
        "raw": ("raw_cap_margin", "raw_margin"),
        "projected": ("projected_cap_margin", "projected_margin"),
        "post": ("post_cap_margin", "post_repair_cap_margin"),
        "final": ("final_cap_margin", "post_repair_cap_margin", "post_cap_margin", "cap_margin"),
    }
    for value in _variants(row):
        nested = value.get(stage)
        if isinstance(nested, Mapping) and nested.get("cap_margin") is not None:
            return nested["cap_margin"]
        for name in aliases.get(stage, ()):
            if value.get(name) is not None:
                return value[name]
    return None


def _positions(row: Mapping[str, object] | None) -> object | None:
    if not isinstance(row, Mapping):
        return None
    for variant in _variants(row):
        for name in (
            "positions",
            "geometry",
            "candidate_geometry",
            "post_repair_geometry",
            "final_positions",
        ):
            value = variant.get(name)
            if isinstance(value, (list, tuple)):
                return value
    return None


def _derived_utilization(row: Mapping[str, object] | None) -> float | None:
    direct = _lookup(row, ("utilization",))
    if direct is not None:
        try:
            value = float(direct)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None
    boxes = _positions(row)
    if not isinstance(boxes, (list, tuple)) or not boxes:
        return None
    try:
        parsed = [tuple(float(part) for part in box[:4]) for box in boxes]
        if any(len(box) != 4 or not all(math.isfinite(part) for part in box) for box in parsed):
            return None
        if any(width <= 0.0 or height <= 0.0 for _, _, width, height in parsed):
            return None
        occupied = sum(width * height for _, _, width, height in parsed)
        left = min(x for x, _, _, _ in parsed)
        bottom = min(y for _, y, _, _ in parsed)
        right = max(x + width for x, _, width, _ in parsed)
        top = max(y + height for _, y, _, height in parsed)
        envelope = (right - left) * (top - bottom)
    except (TypeError, ValueError):
        return None
    if envelope <= 0.0:
        return None
    return occupied / envelope


def _membership(ids: list[int]) -> list[list[int]]:
    return [
        [int(value == group) for value in ids]
        for group in sorted(set(ids))
        if group > 0
    ]


def _metadata_soft_breakdown(
    row: Mapping[str, object] | None,
    metadata: Mapping[str, object] | None,
) -> tuple[int, int, int, int, float] | None:
    """Reconstruct official soft counts via the repository's exact verifier."""

    boxes = _positions(row)
    constraints = metadata.get("constraints") if isinstance(metadata, Mapping) else None
    if not isinstance(boxes, (list, tuple)) or not isinstance(constraints, (list, tuple)):
        return None
    try:
        n = int(metadata.get("block_count", len(boxes)))
        rows = [tuple(int(float(value)) for value in item[:5]) for item in constraints[:n]]
        if n <= 0 or len(boxes) != n or len(rows) != n or any(len(item) != 5 for item in rows):
            return None
        case = {
            "boundary_bits": [
                [int(bool(row[4] & bit)) for bit in (1, 2, 4, 8)] for row in rows
            ],
            "group_membership": _membership([row[3] for row in rows]),
            "mib_membership": _membership([row[2] for row in rows]),
        }
        soft = soft_violation_normalized(case, boxes)
    except (TypeError, ValueError, IndexError):
        return None
    return (
        soft.raw_boundary,
        soft.raw_grouping,
        soft.raw_mib,
        soft.maximum,
        soft.total,
    )


def _quality_component(gap: object) -> float:
    try:
        value = max(0.0, float(gap))
    except (TypeError, ValueError):
        return 0.0
    return math.log1p(ALPHA * value)


def _primary_blocker(case: Mapping[str, object]) -> str:
    """Classify the first cap-crossing counterfactual deterministically."""

    if not bool(case.get("hard_feasible")):
        return "hard"
    try:
        log_cost = float(case["log_uncapped_cost"])
        soft = float(case["soft_contribution"])
        runtime = float(case["runtime_contribution"])
    except (KeyError, TypeError, ValueError):
        return "mixed"
    if log_cost <= CAP_LOG:
        return "none"
    if runtime + float(case.get("quality_contribution", 0.0)) <= CAP_LOG:
        return "soft"
    hpwl = _quality_component(case.get("hpwl_gap", 0.0))
    area = _quality_component(case.get("area_gap", 0.0))
    if soft + runtime + hpwl <= CAP_LOG:
        return "area"
    if soft + runtime + area <= CAP_LOG:
        return "hpwl"
    if bool(case.get("projection_dominated")):
        return "projection"
    return "mixed"


def _enrich_case(
    case: dict[str, object],
    row: Mapping[str, object] | None,
    metadata: Mapping[str, object] | None,
) -> None:
    """Attach source-specific evidence without changing scorer semantics."""

    for target, aliases in (
        ("boundary_violations", ("boundary_violations",)),
        ("grouping_violations", ("grouping_violations",)),
        ("mib_violations", ("mib_violations",)),
        ("max_possible_violations", ("max_possible_violations", "max_soft_violations")),
        ("violations_relative", ("violations_relative", "relative_violations")),
    ):
        value = _lookup(row, aliases)
        if value is not None:
            case[target] = value

    breakdown = _metadata_soft_breakdown(row, metadata)
    supplied_relative = case.get("violations_relative")
    if breakdown is not None:
        boundary, grouping, mib, maximum, relative = breakdown
        try:
            consistent = supplied_relative is None or math.isclose(
                float(supplied_relative), relative, rel_tol=1.0e-12, abs_tol=1.0e-12
            )
        except (TypeError, ValueError):
            consistent = False
        if consistent:
            denominator = max(maximum, 1)
            case.update(
                boundary_violations=boundary,
                grouping_violations=grouping,
                mib_violations=mib,
                total_soft_violations=boundary + grouping + mib,
                max_possible_violations=maximum,
                violations_relative=relative,
                boundary_contribution=BETA * boundary / denominator,
                grouping_contribution=BETA * grouping / denominator,
                mib_contribution=BETA * mib / denominator,
                soft_contribution=BETA * relative,
                soft_breakdown_available=True,
                soft_breakdown_source="case_metadata",
            )

    for stage in _STAGES:
        key = f"{stage}_cap_margin"
        if stage == "final":
            key = "final_cap_margin"
        value = _stage_margin(row, stage)
        if value is None and stage == "final":
            value = case.get("post_repair_cap_margin", case.get("cap_margin"))
        if value is None:
            value = case.get(key)
        case[key] = value
    if case.get("post_repair_cap_margin") is None:
        case["post_repair_cap_margin"] = case.get("final_cap_margin")
    raw_margin = case.get("raw_cap_margin")
    final_margin = case.get("final_cap_margin")
    if raw_margin is not None and final_margin is not None:
        try:
            if float(raw_margin) > 0.0 and float(final_margin) <= 0.0:
                case["projection_dominated"] = True
        except (TypeError, ValueError):
            pass
    case["capped_cost"] = case.get("official_capped_cost")

    source = _lookup(row, ("candidate_source", "source", "candidate_type", "candidate_source_type"))
    if source is not None:
        case["candidate_source"] = source
    utilization = _derived_utilization(row)
    case["utilization"] = utilization
    displacement = _lookup(
        row,
        (
            "projection_displacement",
            "post_repair_projection_displacement",
            "repair_displacement",
        ),
    )
    case["projection_displacement"] = (
        displacement if displacement is not None else case.get("repair_displacement")
    )
    primary = _primary_blocker(case)
    case["primary_blocker"] = primary
    case["primary_blocker_classification"] = primary
    case["primary_blocker_active"] = (
        not bool(case.get("hard_feasible"))
        or float(case.get("cap_margin", 0.0)) <= 0.0
    )


def _enrich_report(report: dict[str, object], payload: object) -> dict[str, object]:
    rows = _source_rows(payload)
    cases = report.get("cases")
    if not isinstance(cases, list):
        return report
    for case in cases:
        if isinstance(case, dict):
            _enrich_case(
                case,
                _match_row(case, rows),
                _case_metadata(payload, case.get("test_id")),
            )
    summary = report.get("summary")
    if isinstance(summary, dict):
        counts = Counter(
            str(case.get("primary_blocker", "mixed"))
            for case in cases
            if isinstance(case, Mapping)
        )
        summary["primary_blocker_counts"] = {
            name: counts.get(name, 0) for name in _PRIMARY_BLOCKERS
        }
        summary["soft_breakdown_available_cases"] = sum(
            bool(case.get("soft_breakdown_available"))
            for case in cases
            if isinstance(case, Mapping)
        )
    limitations = report.get("schema_limitations")
    if isinstance(limitations, list):
        report["schema_limitations"] = [
            item
            for item in limitations
            if not str(item).startswith("Some inputs expose only violations_relative")
            or not all(
                bool(case.get("soft_breakdown_available"))
                for case in cases
                if isinstance(case, Mapping)
            )
        ]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="oracle, benchmark, or minimal case JSON"
    )
    parser.add_argument("--output", required=True, help="destination JSON report")
    parser.add_argument(
        "--markdown",
        help="destination Markdown summary (default: output path with .md suffix)",
    )
    parser.add_argument(
        "--runtime-factor",
        type=float,
        default=1.0,
        help="local attribution default when a row has no runtime_factor",
    )
    parser.add_argument(
        "--lane",
        help="select one lane from a benchmark report instead of reporting every lane",
    )
    args = parser.parse_args(argv)

    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if args.lane:
        lanes = payload.get("lanes") if isinstance(payload, dict) else None
        if not isinstance(lanes, dict) or args.lane not in lanes:
            raise ValueError(f"benchmark input has no lane {args.lane!r}")
        payload = dict(payload)
        payload["lanes"] = {args.lane: lanes[args.lane]}
    report_payload = _canonicalize_payload(payload)
    report = build_cap_report(
        report_payload, default_runtime_factor=args.runtime_factor
    )
    _enrich_report(report, payload)
    output = Path(args.output)
    markdown = Path(args.markdown) if args.markdown else output.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(output)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
