#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from puzzleplace.alternatives.pareto_slack_fit_edits import (
    metric_report,
    pareto_front_report,
    route_report,
)
from puzzleplace.alternatives.regional_legalizer import (
    build_regional_legalizer_candidates,
    decision_for_step7i_r,
    evaluate_regional_legalizer_candidates,
    failure_attribution_report,
    feasibility_report,
    local_starvation_recovery_report,
)
from puzzleplace.train.dataset_bc import load_validation_cases


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2))


def _load_cases(case_ids: list[int]) -> dict[int, Any]:
    cases = load_validation_cases(case_limit=max(case_ids) + 1)
    return {case_id: cases[case_id] for case_id in case_ids}


def _render_visualizations(
    rows: list[dict[str, Any]], feasibility: dict[str, Any], output_dir: Path
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "step7i_r_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for stale in viz_dir.glob("*.png"):
        stale.unlink()
    paths = [
        _bar_chart(
            plt,
            feasibility["legalizer_attempt_count_by_strategy"],
            viz_dir / "legalizer_attempt_count_by_strategy.png",
        ),
        _bar_chart(
            plt,
            feasibility["route_count_after_legalizer"],
            viz_dir / "route_count_after_legalizer.png",
        ),
        _bar_chart(
            plt,
            {
                "hard_feasible": feasibility["hard_feasible_count_after_legalizer"],
                "infeasible": feasibility["source_step7i_candidate_count"]
                - feasibility["hard_feasible_count_after_legalizer"],
            },
            viz_dir / "hard_feasible_after_legalizer.png",
        ),
        _scatter(
            plt,
            rows,
            "hpwl_delta",
            "official_like_cost_delta",
            viz_dir / "hpwl_vs_official_after_legalizer.png",
        ),
    ]
    _write_index(paths, viz_dir, feasibility)
    return paths


def _bar_chart(plt: Any, counts: dict[str, int], path: Path) -> str:
    names = list(counts)
    values = [int(counts[name]) for name in names]
    fig, ax = plt.subplots(figsize=(max(5.2, len(names) * 0.9), 3.8), dpi=120)
    ax.bar(range(len(names)), values, color="#0f766e")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("count")
    ax.set_title(path.stem)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _scatter(plt: Any, rows: list[dict[str, Any]], x_key: str, y_key: str, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=120)
    for status in sorted({str(row["legalizer_status"]) for row in rows}):
        group = [row for row in rows if row["legalizer_status"] == status]
        ax.scatter(
            [float(row.get(x_key, 0.0)) for row in group],
            [float(row.get(y_key, 0.0)) for row in group],
            s=18,
            alpha=0.7,
            label=status,
        )
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(path.stem)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _write_index(paths: list[str], viz_dir: Path, feasibility: dict[str, Any]) -> None:
    sections = []
    for path in paths:
        name = Path(path).name
        sections.append(
            "<section>"
            f"<h2>{html.escape(name)}</h2>"
            f"<img src='{html.escape(name)}' style='max-width:100%;border:1px solid #ddd'/>"
            "</section>"
        )
    snippet = html.escape(
        json.dumps(
            {
                "hard_feasible_count_after_legalizer": feasibility[
                    "hard_feasible_count_after_legalizer"
                ],
                "official_like_improving_candidate_count": feasibility[
                    "official_like_improving_candidate_count"
                ],
                "no_op_after_legalizer_count": feasibility["no_op_after_legalizer_count"],
            },
            indent=2,
        )
    )
    (viz_dir / "index.html").write_text(
        "\n".join(
            [
                "<!doctype html><meta charset='utf-8'>",
                "<title>Step7I-R regional legalizer probe</title>",
                "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:32px}pre{background:#f8fafc;padding:12px}</style>",
                "<h1>Step7I-R regional legalizer probe</h1>",
                "<h2>Summary</h2>",
                f"<pre>{snippet}</pre>",
                *sections,
            ]
        )
    )


def _decision_md(
    decision: str,
    feasibility: dict[str, Any],
    starvation: dict[str, Any],
    pareto: dict[str, Any],
    failure: dict[str, Any],
) -> str:
    summary = {
        "decision": decision,
        "source_step7i_candidate_count": feasibility["source_step7i_candidate_count"],
        "attempted_regional_candidate_count": feasibility["attempted_regional_candidate_count"],
        "attempted_macro_candidate_count": feasibility["attempted_macro_candidate_count"],
        "legalizer_attempt_count_by_strategy": feasibility["legalizer_attempt_count_by_strategy"],
        "route_count_before_legalizer": feasibility["route_count_before_legalizer"],
        "route_count_after_legalizer": feasibility["route_count_after_legalizer"],
        "hard_feasible_count_after_legalizer": feasibility["hard_feasible_count_after_legalizer"],
        "hard_feasible_rate_after_legalizer": feasibility["hard_feasible_rate_after_legalizer"],
        "non_original_non_noop_hard_feasible_count_after_legalizer": feasibility[
            "non_original_non_noop_hard_feasible_count_after_legalizer"
        ],
        "infeasible_after_regional_edit_count": feasibility["infeasible_after_regional_edit_count"],
        "step7i_infeasible_after_regional_edit_baseline": feasibility[
            "step7i_infeasible_after_regional_edit_baseline"
        ],
        "overlap_resolved_count": feasibility["overlap_resolved_count"],
        "no_feasible_slot_count": feasibility["no_feasible_slot_count"],
        "region_capacity_failure_count": feasibility["region_capacity_failure_count"],
        "fixed_preplaced_conflict_count": feasibility["fixed_preplaced_conflict_count"],
        "MIB_group_closure_conflict_count": feasibility["MIB_group_closure_conflict_count"],
        "macro_closure_requires_supernode_count": feasibility[
            "macro_closure_requires_supernode_count"
        ],
        "no_op_after_legalizer_count": feasibility["no_op_after_legalizer_count"],
        "official_like_improving_candidate_count": feasibility[
            "official_like_improving_candidate_count"
        ],
        "regional_official_like_improving_count": feasibility[
            "regional_official_like_improving_count"
        ],
        "macro_official_like_improving_count": feasibility["macro_official_like_improving_count"],
        "official_like_improvement_density": feasibility["official_like_improvement_density"],
        "original_inclusive_pareto_non_empty_count": pareto[
            "original_inclusive_pareto_non_empty_count"
        ],
        "local_starvation_case_recovery_count": starvation["local_starvation_case_recovery_count"],
        "cases_recovered_from_no_official_like_local_candidate": starvation[
            "cases_recovered_from_no_official_like_local_candidate"
        ],
        "cases_recovered_from_no_preselection_retained": starvation[
            "cases_recovered_from_no_preselection_retained"
        ],
        "failure_counts": failure["counts"],
    }
    return (
        "\n".join(
            [
                "# Step7I-R Route-Specific Regional Legalizer / Repacker Probe",
                "",
                f"Decision: `{decision}`",
                "",
                "## Scope",
                "",
                "- Sidecar-only regional legalizer probe over Step7I candidates.",
                "- No contest runtime integration, finalizer changes, RL, or "
                "full global optimization.",
                "- Step7G route classification remains active; global candidates stay report-only.",
                "- Infeasible/no-op candidates remain in reporting.",
                "",
                "## Summary",
                "",
                "```json",
                json.dumps(summary, indent=2),
                "```",
                "",
                "## Interpretation",
                "",
                _interpretation(decision),
            ]
        )
        + "\n"
    )


def _interpretation(decision: str) -> str:
    if decision == "promote_regional_legalizer_to_step7c_hybrid":
        return "Regional legalization made Step7I candidates feasible and metric-relevant."
    if decision == "refine_regional_legalizer":
        return "Feasibility improved, but regional metric gains remain too sparse for promotion."
    if decision == "refine_region_assignment_costs":
        return (
            "Legalization can produce feasible alternatives, but selected region targets "
            "do not improve metrics."
        )
    if decision == "pivot_to_macro_closure_generator":
        return "MIB/group closure dominates legalizer failures; build macro closure generator next."
    if decision == "pivot_to_analytical_force_map":
        return "Region assignment direction appears weak; try analytical force maps."
    if decision == "pivot_to_training_retrieval_prior":
        return "Heuristic region targets lack enough topology guidance."
    if decision == "revisit_region_flow_assumptions":
        return "Coarse regions/windows appear poorly matched to legalizable moves."
    return "Legalizer attempts are insufficient for a confident regional conclusion."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step7i-route", default="artifacts/research/step7i_route_report.json")
    parser.add_argument(
        "--step7i-starvation", default="artifacts/research/step7i_local_starvation_recovery.json"
    )
    parser.add_argument("--output-dir", default="artifacts/research")
    parser.add_argument("--case-id", type=int, action="append", dest="case_ids")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_data = _load_json(Path(args.step7i_route))
    source_rows = list(source_data["rows"])
    if args.case_ids:
        source_rows = [row for row in source_rows if int(row["case_id"]) in set(args.case_ids)]
    case_ids = sorted({int(row["case_id"]) for row in source_rows})
    candidates, attempts = build_regional_legalizer_candidates(source_rows, _load_cases(case_ids))
    rows = evaluate_regional_legalizer_candidates(candidates)
    feasibility = feasibility_report(rows, source_rows)
    route = route_report(rows)
    metrics = metric_report(rows)
    pareto = pareto_front_report(rows)
    failure = failure_attribution_report(rows)
    starvation = local_starvation_recovery_report(rows, _load_json(Path(args.step7i_starvation)))
    decision = decision_for_step7i_r(feasibility, starvation)
    visualizations = _render_visualizations(rows, feasibility, output_dir)

    _write_json(output_dir / "step7i_r_legalizer_candidates.json", {"rows": rows})
    _write_json(output_dir / "step7i_r_legalizer_attempts.json", {"rows": attempts})
    _write_json(output_dir / "step7i_r_route_report.json", route | {"rows": rows})
    _write_json(output_dir / "step7i_r_feasibility_report.json", feasibility)
    _write_json(output_dir / "step7i_r_metric_report.json", metrics)
    _write_json(output_dir / "step7i_r_pareto_report.json", pareto)
    _write_json(output_dir / "step7i_r_failure_attribution.json", failure)
    _write_json(output_dir / "step7i_r_local_starvation_recovery.json", starvation)
    (output_dir / "step7i_r_decision.md").write_text(
        _decision_md(decision, feasibility, starvation, pareto, failure)
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "source_step7i_candidate_count": feasibility["source_step7i_candidate_count"],
                "attempted_regional_candidate_count": feasibility[
                    "attempted_regional_candidate_count"
                ],
                "attempted_macro_candidate_count": feasibility["attempted_macro_candidate_count"],
                "hard_feasible_count_after_legalizer": feasibility[
                    "hard_feasible_count_after_legalizer"
                ],
                "infeasible_after_regional_edit_count": feasibility[
                    "infeasible_after_regional_edit_count"
                ],
                "official_like_improving_candidate_count": feasibility[
                    "official_like_improving_candidate_count"
                ],
                "local_starvation_case_recovery_count": starvation[
                    "local_starvation_case_recovery_count"
                ],
                "visualizations": len(visualizations),
                "output": str(output_dir / "step7i_r_decision.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
