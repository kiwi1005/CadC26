#!/usr/bin/env python3
"""Verify Oracle Step7R close claims against local Step7Q/R artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EPS = 1.0e-9
STRICT_EPS = 1.0e-7


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step7q-rows",
        type=Path,
        default=Path("artifacts/research/step7q_objective_slot_replay_rows.jsonl"),
    )
    parser.add_argument(
        "--step7r-gradient-rows",
        type=Path,
        default=Path("artifacts/research/step7r_c_gradient_replay_rows.jsonl"),
    )
    parser.add_argument(
        "--step7r-close",
        type=Path,
        default=Path("artifacts/research/step7r_close_decision.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7s_oracle_claim_verification.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("artifacts/research/step7s_oracle_claim_verification.md"),
    )
    args = parser.parse_args()
    q_rows = read_jsonl(args.step7q_rows)
    r_rows = read_jsonl(args.step7r_gradient_rows) if args.step7r_gradient_rows.exists() else []
    close = load_json(args.step7r_close) if args.step7r_close.exists() else {}
    summary = {
        "schema": "step7s_oracle_claim_verification_v1",
        "step7q_rows_path": str(args.step7q_rows),
        "step7r_gradient_rows_path": str(args.step7r_gradient_rows),
        "step7r_close_path": str(args.step7r_close),
        "strict_meaningful_eps": STRICT_EPS,
        "hinge_scalarization_check": hinge_scalarization_check(q_rows),
        "step7q_avnr_dedup_check": avnr_dedup_check(q_rows),
        "step7q_hinge_cap_check": hinge_cap_check(q_rows),
        "step7r_gradient_check": gradient_check(r_rows),
        "close_decision_consistency": close_decision_consistency(close),
    }
    summary["decision"] = verification_decision(summary)
    summary["next_recommendation"] = next_recommendation(summary)
    write_json(args.out, summary)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(compact(summary), indent=2, sort_keys=True))


def hinge_scalarization_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[float] = []
    checked = 0
    for row in rows:
        for prefix in ("official_before_quality", "official_after_quality"):
            q = row.get(prefix)
            if not isinstance(q, dict):
                continue
            expected = official_formula(q)
            actual = to_float(q.get("cost"))
            errors.append(abs(expected - actual))
            checked += 1
    max_error = max(errors, default=None)
    return {
        "checked_quality_records": checked,
        "max_abs_error": max_error,
        "passes_1e_12": max_error is not None and max_error <= 1.0e-12,
        "formula": "exp(2*Violationsrelative)*(1+0.5*max(0,HPWLgap)+0.5*max(0,Areagap_bbox))",
    }


def official_formula(q: dict[str, Any]) -> float:
    h = max(0.0, to_float(q.get("HPWLgap")))
    a = max(0.0, to_float(q.get("Areagap_bbox")))
    s = to_float(q.get("Violationsrelative"))
    return math.exp(2.0 * s) * (1.0 + 0.5 * h + 0.5 * a)


def avnr_dedup_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    avnr = [row for row in rows if is_avnr(row)]
    by_key: dict[tuple[str, int, tuple[float, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in avnr:
        by_key[dedup_key(row)].append(row)
    unique_rows = []
    for key, group in sorted(by_key.items(), key=lambda item: (item[0][0], item[0][1])):
        exemplar = group[0]
        unique_rows.append(
            {
                "case_id": key[0],
                "block_id": key[1],
                "target_box": list(key[2]),
                "occurrences": len(group),
                "official_like_cost_delta": exemplar.get("official_like_cost_delta"),
                "hpwl_delta": exemplar.get("hpwl_delta"),
                "bbox_area_delta": exemplar.get("bbox_area_delta"),
                "soft_constraint_delta": exemplar.get("soft_constraint_delta"),
            }
        )
    return {
        "avnr_row_count": len(avnr),
        "unique_candidate_count": len(by_key),
        "dedup_key": "(case_id, block_id, target_box rounded 1e-9)",
        "unique_candidates": unique_rows,
        "passes_oracle_claim_27_to_2": len(avnr) == 27 and len(by_key) == 2,
    }


def hinge_cap_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    avnr = [row for row in rows if is_avnr(row)]
    unique: dict[tuple[str, int, tuple[float, ...]], dict[str, Any]] = {}
    for row in avnr:
        unique.setdefault(dedup_key(row), row)
    checks = []
    for key, row in sorted(unique.items(), key=lambda item: (item[0][0], item[0][1])):
        before = row["official_before_quality"]
        after = row["official_after_quality"]
        predicted_delta = official_formula(after) - official_formula(before)
        hpwl_only_cap = 0.5 * math.exp(2.0 * to_float(before["Violationsrelative"])) * max(
            0.0, to_float(before["HPWLgap"])
        )
        observed_improvement = -to_float(row.get("official_like_cost_delta"))
        checks.append(
            {
                "case_id": key[0],
                "block_id": key[1],
                "target_box": list(key[2]),
                "before_hpwl_gap": before.get("HPWLgap"),
                "before_bbox_gap": before.get("Areagap_bbox"),
                "before_soft": before.get("Violationsrelative"),
                "after_hpwl_gap": after.get("HPWLgap"),
                "observed_official_delta": row.get("official_like_cost_delta"),
                "predicted_official_delta_from_formula": predicted_delta,
                "prediction_abs_error": abs(
                    predicted_delta - to_float(row.get("official_like_cost_delta"))
                ),
                "hpwl_hinge_cleanup_cap": hpwl_only_cap,
                "observed_improvement_over_cap": observed_improvement / hpwl_only_cap
                if hpwl_only_cap > 0
                else None,
                "strict_threshold_factor": STRICT_EPS / observed_improvement
                if observed_improvement > 0
                else None,
            }
        )
    return {
        "unique_avnr_checked": len(checks),
        "checks": checks,
        "all_formula_errors_below_1e_12": all(
            item["prediction_abs_error"] <= 1.0e-12 for item in checks
        ),
        "all_improvements_below_strict_eps": all(
            -to_float(item["observed_official_delta"]) < STRICT_EPS for item in checks
        ),
    }


def gradient_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"available": False}
    avnr = [row for row in rows if is_avnr(row)]
    hpwl_improving = [row for row in rows if to_float(row.get("hpwl_delta")) < -EPS]
    by_key = {dedup_key(row) for row in avnr if row.get("target_box") is not None}
    case_counts = Counter(str(row.get("case_id")) for row in rows)
    return {
        "available": True,
        "variant_count": len(rows),
        "hpwl_strict_improvement_count": len(hpwl_improving),
        "hpwl_strict_improvement_rate": len(hpwl_improving) / max(len(rows), 1),
        "avnr_row_count": len(avnr),
        "avnr_unique_candidate_count": len(by_key),
        "represented_case_count": len(case_counts),
        "largest_case_share": max(case_counts.values(), default=0) / max(len(rows), 1),
        "case_counts": dict(case_counts),
    }


def close_decision_consistency(close: dict[str, Any]) -> dict[str, Any]:
    finding = close.get("step7q_f_avnr_duplication_finding", {}) if close else {}
    return {
        "available": bool(close),
        "decision": close.get("decision"),
        "phase4_gate_open": close.get("phase4_gate_open"),
        "reported_step7q_f_avnr_count": finding.get(
            "step7q_f_actual_all_vector_nonregressing_count_reported"
        ),
        "reported_unique_candidate_count_after_dedup": finding.get(
            "step7q_f_unique_candidate_count_after_dedup"
        ),
    }


def verification_decision(summary: dict[str, Any]) -> str:
    scalar = summary["hinge_scalarization_check"]
    dedup = summary["step7q_avnr_dedup_check"]
    cap = summary["step7q_hinge_cap_check"]
    if (
        scalar["passes_1e_12"]
        and dedup["passes_oracle_claim_27_to_2"]
        and cap["all_formula_errors_below_1e_12"]
        and cap["all_improvements_below_strict_eps"]
    ):
        return "oracle_hinge_and_duplication_claims_verified"
    return "oracle_claims_need_followup"


def next_recommendation(summary: dict[str, Any]) -> str:
    if summary["decision"] == "oracle_hinge_and_duplication_claims_verified":
        return (
            "implement_step7s_critical_cone_or_active_soft_certificate_"
            "before_more_operator_search"
        )
    return "resolve_verification_mismatch_before_planning_next_method"


def markdown(summary: dict[str, Any]) -> str:
    dedup = summary["step7q_avnr_dedup_check"]
    cap = summary["step7q_hinge_cap_check"]
    scalar = summary["hinge_scalarization_check"]
    lines = [
        "# Step7S Oracle Claim Verification",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        f"- hinge_scalarization_max_abs_error: {scalar['max_abs_error']}",
        f"- AVNR rows: {dedup['avnr_row_count']}",
        f"- AVNR unique candidates: {dedup['unique_candidate_count']}",
        f"- all unique AVNR improvements below 1e-7: {cap['all_improvements_below_strict_eps']}",
        "",
        "## Unique AVNR candidates",
        "",
    ]
    for row in dedup["unique_candidates"]:
        lines.append(
            "- case={case_id}, block={block_id}, occurrences={occurrences}, "
            "official_delta={official_like_cost_delta}, hpwl_delta={hpwl_delta}, "
            "bbox_delta={bbox_area_delta}, soft_delta={soft_constraint_delta}".format(**row)
        )
    lines.extend(["", f"Next: `{summary['next_recommendation']}`", ""])
    return "\n".join(lines)


def is_avnr(row: dict[str, Any]) -> bool:
    return bool(row.get("hard_feasible_nonnoop")) and all(
        to_float(row.get(key)) <= EPS
        for key in ("hpwl_delta", "bbox_area_delta", "soft_constraint_delta")
    )


def dedup_key(row: dict[str, Any]) -> tuple[str, int, tuple[float, ...]]:
    return (
        str(row.get("case_id")),
        int(row.get("block_id", -1)),
        tuple(round(to_float(value), 9) for value in row.get("target_box", [])),
    )


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": summary["decision"],
        "hinge_max_abs_error": summary["hinge_scalarization_check"]["max_abs_error"],
        "avnr_row_count": summary["step7q_avnr_dedup_check"]["avnr_row_count"],
        "avnr_unique_candidate_count": summary["step7q_avnr_dedup_check"][
            "unique_candidate_count"
        ],
        "next_recommendation": summary["next_recommendation"],
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
