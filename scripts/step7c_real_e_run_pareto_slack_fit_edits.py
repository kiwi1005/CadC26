#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from puzzleplace.alternatives.pareto_slack_fit_edits import (
    build_pareto_slack_fit_candidates,
    decision_for_step7c_real_e,
    dominance_report,
    evaluate_pareto_slack_fit_edits,
    failure_attribution_report,
    feasibility_and_improvement_report,
    metric_report,
    objective_vector_table,
    pareto_front_report,
    route_report,
    sensitivity_report,
)
from puzzleplace.train.dataset_bc import load_validation_cases


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2))


def _case_ids(descriptors: list[dict[str, Any]], explicit: list[int] | None) -> list[int]:
    if explicit:
        return explicit
    return sorted({int(row["case_id"]) for row in descriptors})


def _load_cases(case_ids: list[int]) -> dict[int, Any]:
    cases = load_validation_cases(case_limit=max(case_ids) + 1)
    return {case_id: cases[case_id] for case_id in case_ids}


def _render_visualizations(
    rows: list[dict[str, Any]], feasibility: dict[str, Any], output_dir: Path
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "step7c_real_e_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for stale in viz_dir.glob("*.png"):
        stale.unlink()
    paths = [
        _bar_chart(
            plt,
            feasibility["candidate_count_by_strategy"],
            viz_dir / "candidate_count_by_strategy.png",
        ),
        _bar_chart(plt, feasibility["route_count_by_class"], viz_dir / "route_count_by_class.png"),
        _scatter(
            plt,
            rows,
            "hpwl_delta",
            "official_like_cost_delta",
            viz_dir / "hpwl_vs_official_delta.png",
        ),
        _scatter(
            plt,
            rows,
            "bbox_area_delta",
            "official_like_cost_delta",
            viz_dir / "bbox_vs_official_delta.png",
        ),
        _scatter(
            plt,
            rows,
            "displacement_magnitude",
            "official_like_cost_delta",
            viz_dir / "displacement_vs_official_delta.png",
        ),
    ]
    _write_index(paths, viz_dir, feasibility)
    return paths


def _bar_chart(plt: Any, counts: dict[str, int], path: Path) -> str:
    names = list(counts)
    values = [int(counts[name]) for name in names]
    fig, ax = plt.subplots(figsize=(max(5.0, len(names) * 0.8), 3.8), dpi=120)
    ax.bar(range(len(names)), values, color="#4f46e5")
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
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_strategy.setdefault(str(row["strategy"]), []).append(row)
    for strategy, group in sorted(by_strategy.items()):
        ax.scatter(
            [float(row[x_key]) for row in group],
            [float(row[y_key]) for row in group],
            label=strategy,
            alpha=0.65,
            s=14,
        )
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.axvline(0.0, color="#9ca3af", linewidth=0.8)
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
                "official_like_cost_improving_count": feasibility[
                    "official_like_cost_improving_count"
                ],
                "official_like_cost_improvement_density": feasibility[
                    "official_like_cost_improvement_density"
                ],
                "hpwl_gain_to_official_like_conversion_rate": feasibility[
                    "hpwl_gain_to_official_like_conversion_rate"
                ],
            },
            indent=2,
        )
    )
    (viz_dir / "index.html").write_text(
        "\n".join(
            [
                "<!doctype html><meta charset='utf-8'>",
                "<title>Step7C-real-E Pareto slack-fit local lane</title>",
                "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:32px}pre{background:#f8fafc;padding:12px}</style>",
                "<h1>Step7C-real-E Pareto slack-fit local lane</h1>",
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
    dominance: dict[str, Any],
    sensitivity: dict[str, Any],
) -> str:
    summary = {
        "real_case_count": feasibility["real_case_count"],
        "candidate_count_by_strategy": feasibility["candidate_count_by_strategy"],
        "route_count_by_class": route["route_count_by_class"],
        "non_global_candidate_rate": route["non_global_candidate_rate"],
        "invalid_local_attempt_rate": route["invalid_local_attempt_rate"],
        "official_like_hard_feasible_rate": feasibility["official_like_hard_feasible_rate"],
        "hpwl_improving_count": feasibility["hpwl_improving_count"],
        "official_like_cost_improving_count": feasibility["official_like_cost_improving_count"],
        "hpwl_gain_to_official_like_conversion_rate": feasibility[
            "hpwl_gain_to_official_like_conversion_rate"
        ],
        "official_like_cost_improvement_density": feasibility[
            "official_like_cost_improvement_density"
        ],
        "original_inclusive_pareto_non_empty_count": feasibility[
            "original_inclusive_pareto_non_empty_count"
        ],
        "candidates_dominated_by_original_count": dominance[
            "candidates_dominated_by_original_count"
        ],
        "hpwl_only_winner_rejected_by_pareto_count": dominance[
            "hpwl_only_winner_rejected_by_pareto_count"
        ],
        "bbox_regression_among_hpwl_winners": dominance["bbox_regression_among_hpwl_winners"],
        "soft_regression_among_hpwl_winners": dominance["soft_regression_among_hpwl_winners"],
    }
    return (
        "\n".join(
            [
                "# Step7C-real-E Pareto-Constrained Slack-Fit Local Lane Expansion",
                "",
                f"Decision: `{decision}`",
                "",
                "## Scope",
                "",
                "- Sidecar-only local slack-fit/HPWL lane expansion.",
                "- Step7G route safety remains active; global candidates are report-only.",
                "- Ranking uses constrained Pareto objective vectors, "
                "not hand-tuned bbox/soft thresholds.",
                "- Original layout remains in every per-case Pareto comparison.",
                "",
                "## Summary",
                "",
                "```json",
                json.dumps(summary, indent=2),
                "```",
                "",
                "## Sensitivity",
                "",
                "```json",
                json.dumps(sensitivity["sensitivity_to_epsilon_or_tolerance"], indent=2),
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
            "Expanded local slack-fit candidates materially improved official-like wins while "
            "constrained Pareto retained route safety and rejected HPWL-only false positives. "
            "A bounded local-lane iteration sidecar is justified next."
        )
    if decision == "refine_pareto_slack_fit_generator":
        return (
            "Constrained Pareto accounting works, but the expanded local "
            "generator still has sparse "
            "official-like wins. Refine local slot generation/coverage before iteration."
        )
    if decision == "build_route_specific_legalizers":
        return (
            "Good local candidates are being blocked by hard feasibility "
            "or route-specific repair limits."
        )
    if decision == "pivot_to_coarse_region_planner":
        return (
            "Expanded local moves cannot create enough metric signal; "
            "consider coarse region planning."
        )
    if decision == "pivot_to_macro_closure_generator":
        return "MIB/group closure dominates the remaining signal; isolate macro closure generation."
    if decision == "revisit_objective_vector_model":
        return (
            "Official-like metrics conflict with the objective vector "
            "or required fields are incomplete."
        )
    return "Metric signal is too weak for a confident direction."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step7h-predictions", default="artifacts/research/step7h_route_predictions.json"
    )
    parser.add_argument("--output-dir", default="artifacts/research")
    parser.add_argument("--case-ids", nargs="*", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptors = _load_json(Path(args.step7h_predictions), [])
    case_ids = _case_ids(descriptors, args.case_ids)
    cases_by_id = _load_cases(case_ids)

    candidates = build_pareto_slack_fit_candidates(case_ids, cases_by_id)
    rows = evaluate_pareto_slack_fit_edits(candidates)
    vectors = objective_vector_table(rows)
    pareto = pareto_front_report(rows)
    dominance = dominance_report(rows)
    metrics = metric_report(rows)
    route = route_report(rows)
    failures = failure_attribution_report(rows)
    sensitivity = sensitivity_report(rows)
    feasibility = feasibility_and_improvement_report(rows, pareto, dominance)
    decision = decision_for_step7c_real_e(feasibility, route, pareto)
    visualizations = _render_visualizations(rows, feasibility, output_dir)

    _write_json(
        output_dir / "step7c_real_e_edit_candidates.json",
        [
            {
                "case_id": c.case_id,
                "candidate_id": c.candidate_id,
                "strategy": c.strategy,
                "variant_index": c.variant_index,
                "real_block_ids_changed": list(c.changed_blocks),
                "construction_status": c.construction_status,
                "construction_notes": list(c.construction_notes),
                "proxy_hpwl_delta": c.proxy_hpwl_delta,
                "proxy_bbox_delta": c.proxy_bbox_delta,
                "proxy_soft_delta": c.proxy_soft_delta,
                "displacement_magnitude": c.displacement_magnitude,
            }
            for c in candidates
        ],
    )
    _write_json(output_dir / "step7c_real_e_objective_vectors.json", vectors)
    _write_json(output_dir / "step7c_real_e_pareto_front.json", pareto)
    _write_json(output_dir / "step7c_real_e_dominance_report.json", dominance)
    _write_json(output_dir / "step7c_real_e_metric_report.json", metrics)
    _write_json(output_dir / "step7c_real_e_route_report.json", route | {"rows": rows})
    _write_json(output_dir / "step7c_real_e_failure_attribution.json", failures)
    _write_json(output_dir / "step7c_real_e_sensitivity_report.json", sensitivity)
    (output_dir / "step7c_real_e_decision.md").write_text(
        _decision_md(decision, feasibility, route, dominance, sensitivity)
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "real_case_count": feasibility["real_case_count"],
                "candidate_count": len(rows),
                "official_like_cost_improving_count": feasibility[
                    "official_like_cost_improving_count"
                ],
                "hpwl_improving_count": feasibility["hpwl_improving_count"],
                "hpwl_gain_to_official_like_conversion_rate": feasibility[
                    "hpwl_gain_to_official_like_conversion_rate"
                ],
                "official_like_cost_improvement_density": feasibility[
                    "official_like_cost_improvement_density"
                ],
                "invalid_local_attempt_rate": route["invalid_local_attempt_rate"],
                "official_like_hard_feasible_rate": feasibility["official_like_hard_feasible_rate"],
                "original_inclusive_pareto_non_empty_count": pareto[
                    "original_inclusive_pareto_non_empty_count"
                ],
                "visualizations": len(visualizations),
                "output": str(output_dir / "step7c_real_e_decision.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
