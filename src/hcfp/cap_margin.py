"""Normalize benchmark/oracle rows into per-case score-cap attribution."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import math
from typing import Any

from hcfp.score_attribution import attribute_score, attribute_score_from_relative


CLASSIFICATIONS = ("none", "hard", "soft", "quality", "mixed", "projection")
SOFT_COUNT_FIELDS = (
    "boundary_violations",
    "grouping_violations",
    "mib_violations",
    "max_possible_violations",
)


def attribute_record(
    record: Mapping[str, Any],
    *,
    default_runtime_factor: float = 1.0,
) -> dict[str, Any]:
    """Attribute one canonical or legacy evaluator row."""

    hard_feasible = _required_bool(record, "hard_feasible", "is_feasible", "feasible")
    hpwl_gap = _required(record, "hpwl_gap")
    area_gap = _required(record, "area_gap")
    runtime_factor = record.get("runtime_factor", default_runtime_factor)
    count_fields = [field for field in SOFT_COUNT_FIELDS if field in record]
    if count_fields and len(count_fields) != len(SOFT_COUNT_FIELDS):
        missing = sorted(set(SOFT_COUNT_FIELDS) - record.keys())
        raise ValueError(f"soft-count record is missing {missing}")
    if count_fields:
        attribution = attribute_score(
            hpwl_gap,
            area_gap,
            boundary_violations=_integer(record["boundary_violations"]),
            grouping_violations=_integer(record["grouping_violations"]),
            mib_violations=_integer(record["mib_violations"]),
            max_possible_violations=_integer(record["max_possible_violations"]),
            hard_feasible=hard_feasible,
            runtime_factor=runtime_factor,
        )
        supplied_relative = record.get("violations_relative")
        if supplied_relative is not None and not math.isclose(
            float(supplied_relative),
            attribution.violations_relative,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "violations_relative is inconsistent with the raw soft counts"
            )
    else:
        attribution = attribute_score_from_relative(
            hpwl_gap,
            area_gap,
            _required(record, "violations_relative"),
            hard_feasible=hard_feasible,
            runtime_factor=runtime_factor,
        )
    result = attribution.as_dict()
    for field in (
        "test_id",
        "block_count",
        "candidate_index",
        "source",
        "lane",
        "stage",
        "repair_displacement",
    ):
        if field in record:
            result[field] = record[field]
    return result


def build_cap_report(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    default_runtime_factor: float = 1.0,
) -> dict[str, Any]:
    """Build a deterministic per-case report from supported JSON schemas."""

    runtime = float(default_runtime_factor)
    if not math.isfinite(runtime) or runtime < 0.0:
        raise ValueError("default_runtime_factor must be finite and non-negative")
    cases, input_schema = _normalize_cases(payload, runtime)
    cases.sort(
        key=lambda row: (
            int(row.get("test_id", -1)),
            str(row.get("lane", "")),
            str(row.get("source", "")),
        )
    )
    limitations = _schema_limitations(cases)
    return {
        "schema_version": 1,
        "input_schema": input_schema,
        "default_runtime_factor": runtime,
        "cases": cases,
        "summary": _summary(cases),
        "schema_limitations": limitations,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact, deterministic per-case Markdown summary."""

    summary = report["summary"]
    lines = [
        "# HCFP exact cap attribution",
        "",
        f"Cases: {summary['cases']}; capped: {summary['capped_cases']}; "
        f"hard feasible: {summary['hard_feasible_cases']}.",
        "",
        "| Case | Blocks | Source | Raw margin | Projected margin | Post margin | "
        "B / G / M | Soft fixes | Quality gap | Blocker |",
        "| ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for case in report["cases"]:
        stages = case["stages"]
        lines.append(
            "| "
            + " | ".join(
                (
                    _text(case.get("test_id")),
                    _text(case.get("block_count")),
                    _text(case.get("source", case.get("lane", "-"))),
                    _number(_stage_value(stages, "raw", "cap_margin")),
                    _number(_stage_value(stages, "projected", "cap_margin")),
                    _number(_stage_value(stages, "post", "cap_margin")),
                    " / ".join(
                        _number(case.get(field))
                        for field in (
                            "boundary_contribution",
                            "grouping_contribution",
                            "mib_contribution",
                        )
                    ),
                    _text(case.get("required_soft_fixes_to_uncap")),
                    _number(case.get("required_quality_gap_to_uncap")),
                    _text(case["blocker_classification"]),
                )
            )
            + " |"
        )
    limitations = report.get("schema_limitations", [])
    if limitations:
        lines.extend(("", "## Schema limitations", ""))
        lines.extend(f"- {item}" for item in limitations)
    return "\n".join(lines) + "\n"


def projection_dominated(stages: Mapping[str, Mapping[str, Any]]) -> bool:
    """Return true when a complete paired path makes a raw-feasible score worse."""

    if not all(stage in stages for stage in ("raw", "projected", "post")):
        return False
    raw, projected, post = (stages[name] for name in ("raw", "projected", "post"))
    if not bool(raw["hard_feasible"]):
        return False
    if not bool(projected["hard_feasible"]) or not bool(post["hard_feasible"]):
        return True
    return float(post["log_uncapped_cost"]) > float(raw["log_uncapped_cost"]) + 1.0e-9


def _normalize_cases(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    runtime: float,
) -> tuple[list[dict[str, Any]], str]:
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return [_flat_case(row, runtime) for row in payload], "row_list"
    if not isinstance(payload, Mapping):
        raise TypeError("input JSON must be an object or a list of case rows")
    lanes = payload.get("lanes")
    if isinstance(lanes, Mapping):
        cases = []
        for lane, rows in lanes.items():
            if not isinstance(rows, list):
                raise ValueError(f"lane {lane!r} must contain a list of rows")
            for row in rows:
                cases.append(
                    _flat_case(_with_defaults(row, lane=lane, source=lane), runtime)
                )
        return cases, "benchmark_report"
    for field in ("test_results", "results"):
        rows = payload.get(field)
        if isinstance(rows, list):
            return [_flat_case(row, runtime) for row in rows], "evaluator_results"
    rows = payload.get("cases")
    if not isinstance(rows, list):
        raise ValueError(
            "input JSON must contain cases, lanes, test_results, or results"
        )
    cases = []
    schemas = set()
    for row in rows:
        if _is_oracle_case(row):
            cases.append(_oracle_case(row, runtime))
            schemas.add("oracle_report")
        elif "stages" in row or all(
            stage in row for stage in ("raw", "projected", "post")
        ):
            cases.append(_staged_case(row, runtime))
            schemas.add("staged_cases")
        else:
            cases.append(_flat_case(row, runtime))
            schemas.add("minimal_cases")
    return cases, schemas.pop() if len(schemas) == 1 else "mixed_cases"


def _flat_case(row: Mapping[str, Any], runtime: float) -> dict[str, Any]:
    attributed = attribute_record(row, default_runtime_factor=runtime)
    return _case_result(row, {"post": attributed})


def _staged_case(row: Mapping[str, Any], runtime: float) -> dict[str, Any]:
    source = row.get("stages", row)
    if not isinstance(source, Mapping):
        raise ValueError("stages must be an object")
    stage_rows = {
        stage: source[stage]
        for stage in ("raw", "projected", "post")
        if isinstance(source.get(stage), Mapping)
    }
    if not stage_rows:
        raise ValueError("staged case must contain raw, projected, or post metrics")
    defaults = {
        key: row[key]
        for key in (
            "test_id",
            "block_count",
            "candidate_index",
            "source",
            "repair_displacement",
        )
        if key in row
    }
    stages = {
        name: attribute_record(
            _with_defaults(metrics, stage=name, **defaults),
            default_runtime_factor=runtime,
        )
        for name, metrics in stage_rows.items()
    }
    return _case_result(row, stages)


def _oracle_case(row: Mapping[str, Any], runtime: float) -> dict[str, Any]:
    incumbent = row["incumbent"]
    if not isinstance(incumbent, Mapping):
        raise ValueError("oracle incumbent must be an object")
    candidate_index = _integer(_required(incumbent, "candidate_index"))
    raw = _candidate_at(row["raw"], candidate_index, "raw")
    projected = _candidate_at(row["post_bdp"], candidate_index, "post_bdp")
    defaults = {
        "test_id": row.get("test_id"),
        "block_count": row.get("block_count"),
        "candidate_index": candidate_index,
        "source": incumbent.get("source"),
    }
    stages = {
        "raw": attribute_record(
            _with_defaults(raw, stage="raw", **defaults),
            default_runtime_factor=runtime,
        ),
        "projected": attribute_record(
            _with_defaults(projected, stage="projected", **defaults),
            default_runtime_factor=runtime,
        ),
        "post": attribute_record(
            _with_defaults(incumbent, stage="post", **defaults),
            default_runtime_factor=runtime,
        ),
    }
    return _case_result(row, stages)


def _case_result(
    row: Mapping[str, Any],
    stages: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    final_name = next(name for name in ("post", "projected", "raw") if name in stages)
    final = stages[final_name]
    result = dict(final)
    for field in (
        "test_id",
        "block_count",
        "candidate_index",
        "source",
        "lane",
        "repair_displacement",
    ):
        if field in row:
            result[field] = row[field]
        elif field in final:
            result[field] = final[field]
    result.setdefault("source", result.get("lane"))
    result["candidate_source"] = result["source"]
    result.setdefault("repair_displacement", None)
    for stage in ("raw", "projected", "post"):
        result[f"{stage}_cap_margin"] = _stage_value(stages, stage, "cap_margin")
    result["post_repair_cap_margin"] = result["post_cap_margin"]
    dominated = projection_dominated(stages)
    result["projection_dominated"] = dominated
    if dominated:
        result["blocker_classification"] = "projection"
    result["stages"] = stages
    return result


def _candidate_at(stage: Any, index: int, name: str) -> Mapping[str, Any]:
    if not isinstance(stage, Mapping) or not isinstance(stage.get("candidates"), list):
        raise ValueError(f"oracle {name} stage must contain candidates")
    for candidate in stage["candidates"]:
        if _integer(_required(candidate, "candidate_index")) == index:
            return candidate
    raise ValueError(f"oracle {name} stage has no candidate_index {index}")


def _is_oracle_case(row: Any) -> bool:
    return (
        isinstance(row, Mapping)
        and isinstance(row.get("raw"), Mapping)
        and isinstance(row.get("post_bdp"), Mapping)
        and "incumbent" in row
    )


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter({name: 0 for name in CLASSIFICATIONS})
    by_source: dict[str, Counter[str]] = defaultdict(
        lambda: Counter({name: 0 for name in CLASSIFICATIONS})
    )
    for case in cases:
        classification = str(case["blocker_classification"])
        counts[classification] += 1
        by_source[str(case.get("source", case.get("lane", "unknown")))][
            classification
        ] += 1
    return {
        "cases": len(cases),
        "hard_feasible_cases": sum(bool(case["hard_feasible"]) for case in cases),
        "capped_cases": sum(bool(case["is_capped"]) for case in cases),
        "soft_breakdown_available_cases": sum(
            bool(case["soft_breakdown_available"]) for case in cases
        ),
        "projection_evaluable_cases": sum(
            all(stage in case["stages"] for stage in ("raw", "projected", "post"))
            for case in cases
        ),
        "projection_dominated_cases": sum(
            bool(case["projection_dominated"]) for case in cases
        ),
        "classification_counts": dict(counts),
        "classification_counts_by_source": {
            source: dict(source_counts)
            for source, source_counts in sorted(by_source.items())
        },
    }


def _schema_limitations(cases: list[dict[str, Any]]) -> list[str]:
    limitations = []
    if any(not case["soft_breakdown_available"] for case in cases):
        limitations.append(
            "Some inputs expose only violations_relative; boundary, grouping, and MIB "
            "contributions and integer soft-fix counts are unavailable for those cases."
        )
    if any(
        not all(stage in case["stages"] for stage in ("raw", "projected", "post"))
        for case in cases
    ):
        limitations.append(
            "Projection dominance is classified only for paired raw, projected, and post stages."
        )
    if any(case.get("repair_displacement") is None for case in cases):
        limitations.append(
            "Repair displacement is absent from at least one input case."
        )
    return limitations


def _with_defaults(row: Mapping[str, Any], **defaults: Any) -> dict[str, Any]:
    result = {key: value for key, value in defaults.items() if value is not None}
    result.update(row)
    return result


def _required(record: Mapping[str, Any], name: str) -> Any:
    if name not in record:
        raise ValueError(f"score record is missing {name}")
    return record[name]


def _required_bool(record: Mapping[str, Any], *names: str) -> bool:
    for name in names:
        if name in record:
            value = record[name]
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
            return value
    raise ValueError(f"score record is missing one of {list(names)}")


def _integer(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("integer field cannot be a bool")
    result = int(value)
    if result != value:
        raise TypeError(f"expected an integer, got {value!r}")
    return result


def _stage_value(stages: Mapping[str, Any], stage: str, field: str) -> Any:
    value = stages.get(stage)
    return value.get(field) if isinstance(value, Mapping) else None


def _number(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def _text(value: Any) -> str:
    return "-" if value is None else str(value).replace("|", "\\|")
