#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from puzzleplace.alternatives.dominance_slack_preselection import (
    DEFAULT_RANK_BUDGET,
    decision_for_step7c_real_f,
    objective_vector_table,
    preselect_dominance_slack_candidates,
    rank_budget_sensitivity,
    retained_measurement_report,
)
from puzzleplace.alternatives.pareto_slack_fit_edits import (
    dominance_report,
    metric_report,
    pareto_front_report,
    route_report,
    sensitivity_report,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _route_rows(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return data["rows"]
    raise ValueError(f"missing rows in {path}")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2))


def _render_visualizations(
    retained: list[dict[str, Any]], filter_summary: dict[str, Any], output_dir: Path
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "step7c_real_f_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for stale in viz_dir.glob("*.png"):
        stale.unlink()
    paths = [
        _bar_chart(
            plt, filter_summary["filter_reason_counts"], viz_dir / "filter_reason_counts.png"
        ),
        _bar_chart(
            plt,
            filter_summary["retained_strategy_counts"],
            viz_dir / "retained_strategy_counts.png",
        ),
        _scatter(
            plt,
            retained,
            "hpwl_delta",
            "official_like_cost_delta",
            viz_dir / "retained_hpwl_vs_official.png",
        ),
        _scatter(
            plt,
            retained,
            "displacement_magnitude",
            "official_like_cost_delta",
            viz_dir / "retained_displacement_vs_official.png",
        ),
    ]
    _write_index(paths, viz_dir, filter_summary)
    return paths


def _bar_chart(plt: Any, counts: dict[str, int], path: Path) -> str:
    names = list(counts)
    values = [int(counts[name]) for name in names]
    fig, ax = plt.subplots(figsize=(max(5.0, len(names) * 0.8), 3.8), dpi=120)
    ax.bar(range(len(names)), values, color="#2563eb")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("count")
    ax.set_title(path.stem)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _scatter(plt: Any, rows: list[dict[str, Any]], x_key: str, y_key: str, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=120)
    ax.scatter(
        [float(row.get(x_key, 0.0)) for row in rows],
        [float(row.get(y_key, 0.0)) for row in rows],
        s=16,
        alpha=0.7,
        color="#0f766e",
    )
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.axvline(0.0, color="#9ca3af", linewidth=0.8)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(path.stem)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _write_index(paths: list[str], viz_dir: Path, filter_summary: dict[str, Any]) -> None:
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
                "source_candidate_count_from_E": filter_summary["source_candidate_count_from_E"],
                "retained_candidate_count": filter_summary["retained_candidate_count"],
                "E_winner_preservation_rate": filter_summary["E_winner_preservation_rate"],
            },
            indent=2,
        )
    )
    (viz_dir / "index.html").write_text(
        "\n".join(
            [
                "<!doctype html><meta charset='utf-8'>",
                "<title>Step7C-real-F dominance-aware preselection</title>",
                "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:32px}pre{background:#f8fafc;padding:12px}</style>",
                "<h1>Step7C-real-F dominance-aware preselection</h1>",
                "<h2>Summary</h2>",
                f"<pre>{snippet}</pre>",
                *sections,
            ]
        )
    )


def _decision_md(
    decision: str,
    feasibility: dict[str, Any],
    route: dict[str, Any],
    filter_summary: dict[str, Any],
    rank_sensitivity: dict[str, Any],
) -> str:
    summary = {
        "real_case_count": feasibility["real_case_count"],
        "source_candidate_count_from_E": feasibility["source_candidate_count_from_E"],
        "generated_candidate_count": feasibility["generated_candidate_count"],
        "filtered_candidate_count": feasibility["filtered_candidate_count"],
        "retained_fraction": feasibility["retained_fraction"],
        "filter_reason_counts": feasibility["filter_reason_counts"],
        "route_count_by_class": route["route_count_by_class"],
        "non_global_candidate_rate": route["non_global_candidate_rate"],
        "invalid_local_attempt_rate": route["invalid_local_attempt_rate"],
        "official_like_hard_feasible_rate": feasibility["official_like_hard_feasible_rate"],
        "hpwl_improving_count": feasibility["hpwl_improving_count"],
        "official_like_cost_improving_count": feasibility["official_like_cost_improving_count"],
        "hpwl_to_official_like_conversion_rate": feasibility[
            "hpwl_to_official_like_conversion_rate"
        ],
        "official_like_cost_improvement_density": feasibility[
            "official_like_cost_improvement_density"
        ],
        "dominated_by_original_rate": feasibility["dominated_by_original_rate"],
        "E_winner_preservation_count": filter_summary["E_winner_preservation_count"],
        "E_winner_preservation_rate": filter_summary["E_winner_preservation_rate"],
        "original_inclusive_pareto_non_empty_count": feasibility[
            "original_inclusive_pareto_non_empty_count"
        ],
    }
    return (
        "\n".join(
            [
                "# Step7C-real-F Dominance-Aware Slack Slot Preselection",
                "",
                f"Decision: `{decision}`",
                "",
                "## Scope",
                "",
                "- Sidecar-only preselection replay over Step7C-real-E local slack candidates.",
                "- No runtime solver integration, finalizer changes, RL, or full iteration.",
                "- Filtered candidates are summarized; original-inclusive Pareto remains active.",
                "",
                "## Summary",
                "",
                "```json",
                json.dumps(summary, indent=2),
                "```",
                "",
                "## Rank-budget sensitivity",
                "",
                "```json",
                json.dumps(rank_sensitivity["sensitivity_to_rank_budget_or_top_k"], indent=2),
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
    if decision == "promote_to_step7c_local_lane_iteration":
        return (
            "Dominance-aware preselection reduces dominated probes while preserving enough E-stage "
            "wins and improving density/conversion. A bounded local-lane iteration "
            "sidecar is justified."
        )
    if decision == "refine_dominance_preselection":
        return (
            "Preselection improves density, but the retained winner set is still too sparse for "
            "promotion. Refine rank features or per-case budgets."
        )
    if decision == "combine_preselection_with_route_specific_legalizer":
        return "Selected candidates look promising but need route-specific legalizer support."
    if decision == "pivot_to_coarse_region_planner":
        return "Local slack-fit remains too sparse; pivot to coarse region planning."
    if decision == "pivot_to_macro_closure_generator":
        return "Macro/group constraints dominate remaining failures."
    if decision == "revisit_local_slack_lane_ceiling":
        return "Local slack wins exist, but preselection cannot make the lane practical enough."
    return "Preselection evidence is insufficient for a clear next implementation direction."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step7c-real-e-route", default="artifacts/research/step7c_real_e_route_report.json"
    )
    parser.add_argument("--output-dir", default="artifacts/research")
    parser.add_argument("--rank-budget", type=int, default=DEFAULT_RANK_BUDGET)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = _route_rows(Path(args.step7c_real_e_route))
    retained, filter_summary = preselect_dominance_slack_candidates(
        source_rows, rank_budget=args.rank_budget
    )
    pareto = pareto_front_report(retained)
    dominance = dominance_report(retained)
    metrics = metric_report(retained)
    route = route_report(retained)
    epsilon_sensitivity = sensitivity_report(retained)
    rank_sensitivity = rank_budget_sensitivity(source_rows)
    feasibility = retained_measurement_report(retained, filter_summary)
    decision = decision_for_step7c_real_f(feasibility, route)
    visualizations = _render_visualizations(retained, filter_summary, output_dir)

    _write_json(output_dir / "step7c_real_f_preselection_candidates.json", retained)
    _write_json(output_dir / "step7c_real_f_filter_report.json", filter_summary)
    _write_json(
        output_dir / "step7c_real_f_objective_vectors.json", objective_vector_table(retained)
    )
    _write_json(output_dir / "step7c_real_f_pareto_front.json", pareto)
    _write_json(output_dir / "step7c_real_f_dominance_report.json", dominance)
    _write_json(output_dir / "step7c_real_f_metric_report.json", metrics)
    _write_json(output_dir / "step7c_real_f_route_report.json", route | {"rows": retained})
    _write_json(
        output_dir / "step7c_real_f_sensitivity_report.json",
        epsilon_sensitivity | rank_sensitivity,
    )
    (output_dir / "step7c_real_f_decision.md").write_text(
        _decision_md(decision, feasibility, route, filter_summary, rank_sensitivity)
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "source_candidate_count_from_E": feasibility["source_candidate_count_from_E"],
                "generated_candidate_count": feasibility["generated_candidate_count"],
                "filtered_candidate_count": feasibility["filtered_candidate_count"],
                "official_like_cost_improving_count": feasibility[
                    "official_like_cost_improving_count"
                ],
                "official_like_cost_improvement_density": feasibility[
                    "official_like_cost_improvement_density"
                ],
                "hpwl_to_official_like_conversion_rate": feasibility[
                    "hpwl_to_official_like_conversion_rate"
                ],
                "dominated_by_original_rate": feasibility["dominated_by_original_rate"],
                "E_winner_preservation_rate": feasibility["E_winner_preservation_rate"],
                "invalid_local_attempt_rate": route["invalid_local_attempt_rate"],
                "official_like_hard_feasible_rate": feasibility["official_like_hard_feasible_rate"],
                "visualizations": len(visualizations),
                "output": str(output_dir / "step7c_real_f_decision.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
