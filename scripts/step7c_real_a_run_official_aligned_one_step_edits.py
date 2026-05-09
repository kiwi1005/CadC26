#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from puzzleplace.alternatives.real_placement_edits import (
    build_real_edit_candidates,
    confusion_report,
    decision_for_step7c_real_a,
    evaluate_real_edits,
    failure_attribution_report,
    feasibility_report,
    metric_report,
    pareto_report,
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
    rows: list[dict[str, Any]], metrics: dict[str, Any], output_dir: Path
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "step7c_real_a_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for stale in viz_dir.glob("*.png"):
        stale.unlink()
    paths = [
        _bar_chart(
            plt, _counts(rows, "actual_locality_class"), viz_dir / "real_route_count_by_class.png"
        ),
        _bar_chart(plt, _counts(rows, "family"), viz_dir / "real_edit_count_by_family.png"),
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
    _write_index(paths, viz_dir, metrics)
    return paths


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        out[value] = out.get(value, 0) + 1
    return out


def _bar_chart(plt: Any, counts: dict[str, int], path: Path) -> str:
    names = list(counts)
    values = [int(counts[name]) for name in names]
    fig, ax = plt.subplots(figsize=(max(5.0, len(names) * 0.85), 3.8), dpi=120)
    ax.bar(range(len(names)), values, color="#7c3aed")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("real edit count")
    ax.set_title(path.stem)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.05, str(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _histogram(plt: Any, values: list[float], path: Path, *, xlabel: str) -> str:
    fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=120)
    ax.hist(values, bins=min(12, max(3, len(values) // 2)), color="#0891b2", edgecolor="white")
    ax.axvline(0.0, color="#111827", linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("candidate count")
    ax.set_title(path.stem)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _write_index(paths: list[str], viz_dir: Path, metrics: dict[str, Any]) -> None:
    sections = []
    for path in paths:
        name = Path(path).name
        sections.append(
            "<section>"
            f"<h2>{html.escape(name)}</h2>"
            f"<img src='{html.escape(name)}' style='max-width:100%;border:1px solid #ddd'/>"
            "</section>"
        )
    metric_snippet = html.escape(
        json.dumps(
            {
                "official_like_cost_delta_distribution": metrics[
                    "official_like_cost_delta_distribution"
                ],
                "hpwl_delta_distribution": metrics["hpwl_delta_distribution"],
            },
            indent=2,
        )
    )
    (viz_dir / "index.html").write_text(
        "\n".join(
            [
                "<!doctype html><meta charset='utf-8'>",
                "<title>Step7C-real-A official-aligned one-step edits</title>",
                "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:32px}pre{background:#f8fafc;padding:12px}</style>",
                "<h1>Step7C-real-A official-aligned one-step edits</h1>",
                "<h2>Metric summary</h2>",
                f"<pre>{metric_snippet}</pre>",
                *sections,
            ]
        )
    )


def _write_decision_md(
    *,
    decision: str,
    feasibility: dict[str, Any],
    confusion: dict[str, Any],
    metrics: dict[str, Any],
    pareto: dict[str, Any],
    failures: dict[str, Any],
) -> str:
    summary = {
        "real_case_count": feasibility["real_case_count"],
        "descriptor_candidate_count": feasibility["descriptor_candidate_count"],
        "real_edit_candidate_count": feasibility["real_edit_candidate_count"],
        "real_route_count_by_class": feasibility["real_route_count_by_class"],
        "real_non_global_candidate_rate": feasibility["real_non_global_candidate_rate"],
        "descriptor_to_real_route_stability": confusion["descriptor_to_real_route_stability"],
        "invalid_local_attempt_rate": feasibility["invalid_local_attempt_rate"],
        "official_like_hard_feasible_rate": feasibility["official_like_hard_feasible_rate"],
        "actual_safe_improvement_count": feasibility["actual_safe_improvement_count"],
        "original_inclusive_pareto_non_empty_count": pareto[
            "original_inclusive_pareto_non_empty_count"
        ],
    }
    return (
        "\n".join(
            [
                "# Step7C-real-A Official-Aligned One-Step Real Placement Edit Loop",
                "",
                f"Decision: `{decision}`",
                "",
                "## Scope",
                "",
                "- Sidecar-only one-step validation on real FloorSet validation cases.",
                "- Global candidates are report-only and are not sent to bounded-local repair.",
                "- Original layouts are included in every Pareto comparison.",
                "",
                "## Real edit summary",
                "",
                "```json",
                json.dumps(summary, indent=2),
                "```",
                "",
                "## Official-aligned metric summary",
                "",
                "```json",
                json.dumps(
                    {
                        "hpwl_delta_distribution": metrics["hpwl_delta_distribution"],
                        "bbox_area_delta_distribution": metrics["bbox_area_delta_distribution"],
                        "official_like_cost_delta_distribution": metrics[
                            "official_like_cost_delta_distribution"
                        ],
                        "mib_group_boundary_soft_delta": metrics["mib_group_boundary_soft_delta"],
                    },
                    indent=2,
                ),
                "```",
                "",
                "## Descriptor vs real-route behavior",
                "",
                "```json",
                json.dumps(confusion, indent=2),
                "```",
                "",
                "## Failure attribution",
                "",
                "```json",
                json.dumps(failures["counts"], indent=2),
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
            "Route-aware real placement edits preserved non-global diversity, kept invalid local "
            "attempts near zero, retained original-inclusive Pareto fronts, and produced visible "
            "real metric improvements. The next step may be a bounded multi-iteration sidecar loop."
        )
    if decision == "refine_real_edit_generators":
        return (
            "The route and legality signals survive on real FloorSet placements, "
            "but metric gains are not strong enough to justify multi-iteration "
            "search yet. Improve real edit targeting before promoting Step7C."
        )
    if decision == "build_route_specific_legalizers":
        return (
            "Candidate routing produced promising lanes, but real edits fail hard "
            "legality often enough that route-specific legalizers are next."
        )
    if decision == "revisit_proxy_to_real_assumptions":
        return "Step7H/Step7C-thin descriptor classes did not reliably predict real edit routes."
    if decision == "pivot_to_coarse_region_planner":
        return (
            "Useful real movement is mostly regional/global; a coarse region "
            "planner is the safer next lane."
        )
    if decision == "pivot_to_macro_level_move_generator":
        return (
            "Macro closure dominates one-step behavior; macro-level move "
            "generation/legalization should be isolated."
        )
    return (
        "Official-aligned metric access or coverage is insufficient for a "
        "confident promotion decision."
    )


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
    selected_descriptors = [row for row in descriptors if int(row["case_id"]) in set(case_ids)]
    cases_by_id = _load_cases(case_ids)
    candidates = build_real_edit_candidates(selected_descriptors, cases_by_id)
    rows = evaluate_real_edits(candidates)
    confusion = confusion_report(rows)
    feasibility = feasibility_report(
        rows,
        descriptor_candidate_count=len(selected_descriptors),
        real_case_count=len(cases_by_id),
    )
    metrics = metric_report(rows)
    pareto = pareto_report(rows)
    failures = failure_attribution_report(rows)
    decision = decision_for_step7c_real_a(feasibility, confusion, pareto, metrics)
    visualizations = _render_visualizations(rows, metrics, output_dir)

    _write_json(
        output_dir / "step7c_real_a_real_edit_candidates.json",
        [
            {
                "case_id": candidate.case_id,
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "descriptor_locality_class": candidate.descriptor_locality_class,
                "real_block_ids_changed": list(candidate.changed_blocks),
                "macro_closure_block_ids": list(candidate.macro_closure_blocks),
                "route_lane": candidate.route_lane,
                "construction_status": candidate.construction_status,
                "construction_notes": list(candidate.construction_notes),
            }
            for candidate in candidates
        ],
    )
    _write_json(output_dir / "step7c_real_a_actual_routes.json", rows)
    _write_json(output_dir / "step7c_real_a_feasibility_report.json", feasibility)
    _write_json(output_dir / "step7c_real_a_metric_report.json", metrics)
    _write_json(output_dir / "step7c_real_a_pareto_report.json", pareto)
    _write_json(output_dir / "step7c_real_a_failure_attribution.json", failures)
    (output_dir / "step7c_real_a_decision.md").write_text(
        _write_decision_md(
            decision=decision,
            feasibility=feasibility,
            confusion=confusion,
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
                "real_edit_candidate_count": feasibility["real_edit_candidate_count"],
                "real_route_count_by_class": feasibility["real_route_count_by_class"],
                "real_non_global_candidate_rate": feasibility["real_non_global_candidate_rate"],
                "invalid_local_attempt_rate": feasibility["invalid_local_attempt_rate"],
                "official_like_hard_feasible_rate": feasibility["official_like_hard_feasible_rate"],
                "actual_safe_improvement_count": feasibility["actual_safe_improvement_count"],
                "original_inclusive_pareto_non_empty_count": pareto[
                    "original_inclusive_pareto_non_empty_count"
                ],
                "descriptor_to_real_route_stability": confusion[
                    "descriptor_to_real_route_stability"
                ],
                "visualizations": len(visualizations),
                "output": str(output_dir / "step7c_real_a_decision.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
