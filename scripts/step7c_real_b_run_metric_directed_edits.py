#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from puzzleplace.alternatives.metric_directed_edits import (
    build_metric_directed_edit_candidates,
    decision_for_step7c_real_b,
    evaluate_metric_directed_edits,
    failure_attribution_report,
    feasibility_report,
    metric_report,
    pareto_report,
    route_report,
    strategy_ablation_report,
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
    rows: list[dict[str, Any]],
    feasibility: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "step7c_real_b_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for stale in viz_dir.glob("*.png"):
        stale.unlink()
    paths = [
        _bar_chart(
            plt,
            feasibility["candidate_count_by_strategy"],
            viz_dir / "candidate_count_by_strategy.png",
        ),
        _bar_chart(
            plt,
            feasibility["improving_count_by_strategy"],
            viz_dir / "improving_count_by_strategy.png",
        ),
        _bar_chart(plt, feasibility["route_count_by_class"], viz_dir / "route_count_by_class.png"),
        _histogram(
            plt,
            [float(row["official_like_cost_delta"]) for row in rows],
            viz_dir / "official_like_cost_delta_distribution.png",
            xlabel="official-like cost delta",
        ),
        _histogram(
            plt,
            [float(row["hpwl_delta"]) for row in rows],
            viz_dir / "hpwl_delta_distribution.png",
            xlabel="HPWL proxy delta",
        ),
    ]
    _write_index(paths, viz_dir, feasibility)
    return paths


def _bar_chart(plt: Any, counts: dict[str, int], path: Path) -> str:
    names = list(counts)
    values = [int(counts[name]) for name in names]
    fig, ax = plt.subplots(figsize=(max(5.0, len(names) * 0.75), 3.8), dpi=120)
    ax.bar(range(len(names)), values, color="#4f46e5")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("candidate count")
    ax.set_title(path.stem)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.05, str(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _histogram(plt: Any, values: list[float], path: Path, *, xlabel: str) -> str:
    fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=120)
    ax.hist(values, bins=min(14, max(3, len(values) // 3)), color="#0f766e", edgecolor="white")
    ax.axvline(0.0, color="#111827", linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("candidate count")
    ax.set_title(path.stem)
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
                "improvement_density": feasibility["improvement_density"],
                "official_like_cost_improvement_density": feasibility[
                    "official_like_cost_improvement_density"
                ],
            },
            indent=2,
        )
    )
    (viz_dir / "index.html").write_text(
        "\n".join(
            [
                "<!doctype html><meta charset='utf-8'>",
                "<title>Step7C-real-B metric-directed edits</title>",
                "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:32px}pre{background:#f8fafc;padding:12px}</style>",
                "<h1>Step7C-real-B metric-directed edits</h1>",
                "<h2>Summary</h2>",
                f"<pre>{snippet}</pre>",
                *sections,
            ]
        )
    )


def _write_decision_md(
    *,
    decision: str,
    feasibility: dict[str, Any],
    route: dict[str, Any],
    metrics: dict[str, Any],
    pareto: dict[str, Any],
    failures: dict[str, Any],
) -> str:
    summary = {
        "real_case_count": feasibility["real_case_count"],
        "official_like_cost_improving_count": feasibility["official_like_cost_improving_count"],
        "improvement_density": feasibility["improvement_density"],
        "official_like_cost_improvement_density": feasibility[
            "official_like_cost_improvement_density"
        ],
        "route_count_by_class": route["route_count_by_class"],
        "non_global_candidate_rate": route["non_global_candidate_rate"],
        "invalid_local_attempt_rate": route["invalid_local_attempt_rate"],
        "official_like_hard_feasible_rate": feasibility["official_like_hard_feasible_rate"],
        "regional_macro_non_noop_rate": feasibility["regional_macro_non_noop_rate"],
        "regional_macro_feasible_rate": feasibility["regional_macro_feasible_rate"],
        "original_inclusive_pareto_non_empty_count": pareto[
            "original_inclusive_pareto_non_empty_count"
        ],
    }
    return (
        "\n".join(
            [
                "# Step7C-real-B Official-Metric-Directed Real Edit Targeting",
                "",
                f"Decision: `{decision}`",
                "",
                "## Scope",
                "",
                "- Sidecar-only one-step validation on real FloorSet validation cases.",
                "- Step7G routing remains a safety layer, not a hard discard gate.",
                "- Global candidates are report-only and are not sent to bounded-local repair.",
                "- Original layouts are included in every Pareto comparison.",
                "",
                "## Summary",
                "",
                "```json",
                json.dumps(summary, indent=2),
                "```",
                "",
                "## Strategy ablation",
                "",
                "```json",
                json.dumps(feasibility["candidate_count_by_strategy"], indent=2),
                "```",
                "",
                "## Metric distributions",
                "",
                "```json",
                json.dumps(metrics["overall"], indent=2),
                "```",
                "",
                "## Failure attribution",
                "",
                "```json",
                json.dumps(failures["counts"], indent=2),
                "```",
                "",
                "## Step7C-real-A comparison",
                "",
                "```json",
                json.dumps(feasibility["step7c_real_a_comparison"], indent=2),
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
    if decision == "promote_to_step7c_multi_iteration_sidecar":
        return (
            "Metric-directed targeting increased official-like improving candidates while "
            "preserving route safety, feasibility, and original-inclusive Pareto coverage. "
            "A bounded multi-iteration sidecar is now justified."
        )
    if decision == "refine_metric_directed_generators":
        return (
            "Official-metric targeting improves over Step7C-real-A but the signal is still "
            "not dense enough for multi-iteration search. Refine target selection and "
            "route-specific construction before promotion."
        )
    if decision == "build_route_specific_legalizers":
        return (
            "Metric direction exists, but too many useful regional/macro candidates are "
            "blocked by no-op or feasibility limits. Build route-specific legalizers next."
        )
    if decision == "pivot_to_coarse_region_planner":
        return (
            "Local/regional metric-directed edits remain structurally weak; "
            "pivot to a region planner."
        )
    if decision == "pivot_to_macro_level_move_generator":
        return "MIB/group closure dominates the failures; isolate macro-level move generation."
    if decision == "revisit_official_metric_model":
        return "Official-like metric behavior is inconsistent enough to revisit evaluator modeling."
    return "The metric signal is too weak or incomplete for a reliable Step7C promotion decision."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step7h-predictions", default="artifacts/research/step7h_route_predictions.json"
    )
    parser.add_argument(
        "--real-a-feasibility", default="artifacts/research/step7c_real_a_feasibility_report.json"
    )
    parser.add_argument("--output-dir", default="artifacts/research")
    parser.add_argument("--case-ids", nargs="*", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptors = _load_json(Path(args.step7h_predictions), [])
    real_a_feasibility = _load_json(Path(args.real_a_feasibility), {})
    case_ids = _case_ids(descriptors, args.case_ids)
    cases_by_id = _load_cases(case_ids)

    candidates = build_metric_directed_edit_candidates(case_ids, cases_by_id)
    rows = evaluate_metric_directed_edits(candidates)
    route = route_report(rows)
    feasibility = feasibility_report(
        rows,
        real_case_count=len(cases_by_id),
        real_a_feasibility=real_a_feasibility,
    )
    metrics = metric_report(rows)
    pareto = pareto_report(rows)
    failures = failure_attribution_report(rows)
    ablation = strategy_ablation_report(rows)
    decision = decision_for_step7c_real_b(feasibility, route, pareto, metrics)
    visualizations = _render_visualizations(rows, feasibility, output_dir)

    _write_json(
        output_dir / "step7c_real_b_edit_candidates.json",
        [
            {
                "case_id": candidate.case_id,
                "candidate_id": candidate.candidate_id,
                "strategy": candidate.strategy,
                "descriptor_locality_class": candidate.descriptor_locality_class,
                "real_block_ids_changed": list(candidate.changed_blocks),
                "macro_closure_block_ids": list(candidate.macro_closure_blocks),
                "construction_status": candidate.construction_status,
                "construction_notes": list(candidate.construction_notes),
                "internal_trial_count": candidate.internal_trial_count,
                "internal_feasible_trial_count": candidate.internal_feasible_trial_count,
            }
            for candidate in candidates
        ],
    )
    _write_json(output_dir / "step7c_real_b_route_report.json", route | {"rows": rows})
    _write_json(output_dir / "step7c_real_b_feasibility_report.json", feasibility)
    _write_json(output_dir / "step7c_real_b_metric_report.json", metrics)
    _write_json(output_dir / "step7c_real_b_pareto_report.json", pareto)
    _write_json(output_dir / "step7c_real_b_failure_attribution.json", failures)
    _write_json(output_dir / "step7c_real_b_strategy_ablation.json", ablation)
    (output_dir / "step7c_real_b_decision.md").write_text(
        _write_decision_md(
            decision=decision,
            feasibility=feasibility,
            route=route,
            metrics=metrics,
            pareto=pareto,
            failures=failures,
        )
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "real_case_count": feasibility["real_case_count"],
                "official_like_cost_improving_count": feasibility[
                    "official_like_cost_improving_count"
                ],
                "improvement_density": feasibility["improvement_density"],
                "official_like_cost_improvement_density": feasibility[
                    "official_like_cost_improvement_density"
                ],
                "route_count_by_class": route["route_count_by_class"],
                "non_global_candidate_rate": route["non_global_candidate_rate"],
                "invalid_local_attempt_rate": route["invalid_local_attempt_rate"],
                "official_like_hard_feasible_rate": feasibility["official_like_hard_feasible_rate"],
                "original_inclusive_pareto_non_empty_count": pareto[
                    "original_inclusive_pareto_non_empty_count"
                ],
                "visualizations": len(visualizations),
                "output": str(output_dir / "step7c_real_b_decision.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
