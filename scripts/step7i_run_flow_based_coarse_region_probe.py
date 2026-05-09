#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from puzzleplace.alternatives.coarse_region_flow import (
    assignment_cost_correlation_report,
    build_flow_assignment_candidates,
    build_region_maps,
    decision_for_step7i,
    evaluate_flow_assignment_candidates,
    failure_attribution_report,
    feasibility_report,
    local_starvation_recovery_report,
)
from puzzleplace.alternatives.pareto_slack_fit_edits import (
    metric_report,
    pareto_front_report,
    route_report,
)
from puzzleplace.train.dataset_bc import load_validation_cases


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2))


def _case_ids(trace_path: Path, explicit: list[int] | None) -> list[int]:
    if explicit:
        return sorted(explicit)
    trace = _load_json(trace_path)
    return sorted({int(row["case_id"]) for row in trace.get("rows", [])})


def _load_cases(case_ids: list[int]) -> dict[int, Any]:
    cases = load_validation_cases(case_limit=max(case_ids) + 1)
    return {case_id: cases[case_id] for case_id in case_ids}


def _render_visualizations(
    rows: list[dict[str, Any]], feasibility: dict[str, Any], output_dir: Path
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "step7i_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for stale in viz_dir.glob("*.png"):
        stale.unlink()
    paths = [
        _bar_chart(
            plt,
            feasibility["candidate_count_by_assignment_type"],
            viz_dir / "candidate_count_by_assignment_type.png",
        ),
        _bar_chart(plt, feasibility["route_count_by_class"], viz_dir / "route_count_by_class.png"),
        _scatter(
            plt,
            rows,
            "assignment_total_cost",
            "official_like_cost_delta",
            viz_dir / "assignment_cost_vs_official_delta.png",
        ),
        _scatter(
            plt,
            rows,
            "hpwl_delta",
            "official_like_cost_delta",
            viz_dir / "hpwl_vs_official_delta.png",
        ),
    ]
    _write_index(paths, viz_dir, feasibility)
    return paths


def _bar_chart(plt: Any, counts: dict[str, int], path: Path) -> str:
    names = list(counts)
    values = [int(counts[name]) for name in names]
    fig, ax = plt.subplots(figsize=(max(5.2, len(names) * 0.9), 3.8), dpi=120)
    ax.bar(range(len(names)), values, color="#7c3aed")
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
    for cls in sorted({str(row["actual_locality_class"]) for row in rows}):
        group = [row for row in rows if row["actual_locality_class"] == cls]
        ax.scatter(
            [float(row.get(x_key, 0.0)) for row in group],
            [float(row.get(y_key, 0.0)) for row in group],
            s=18,
            alpha=0.7,
            label=cls,
        )
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(path.stem)
    ax.legend(fontsize=7)
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
                "regional_candidate_count": feasibility["regional_candidate_count"],
                "official_like_improving_candidate_count": feasibility[
                    "official_like_improving_candidate_count"
                ],
                "official_like_improvement_density": feasibility[
                    "official_like_improvement_density"
                ],
            },
            indent=2,
        )
    )
    (viz_dir / "index.html").write_text(
        "\n".join(
            [
                "<!doctype html><meta charset='utf-8'>",
                "<title>Step7I flow-based coarse region probe</title>",
                "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:32px}pre{background:#f8fafc;padding:12px}</style>",
                "<h1>Step7I flow-based coarse region probe</h1>",
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
        "real_case_count": feasibility["real_case_count"],
        "assignment_candidate_count": feasibility["assignment_candidate_count"],
        "candidate_count_by_assignment_type": feasibility["candidate_count_by_assignment_type"],
        "route_count_by_class": feasibility["route_count_by_class"],
        "regional_candidate_count": feasibility["regional_candidate_count"],
        "global_candidate_count": feasibility["global_candidate_count"],
        "non_global_candidate_rate": feasibility["non_global_candidate_rate"],
        "non_noop_regional_candidate_count": feasibility["non_noop_regional_candidate_count"],
        "regional_feasible_or_legalizer_promising_count": feasibility[
            "regional_feasible_or_legalizer_promising_count"
        ],
        "official_like_improving_candidate_count": feasibility[
            "official_like_improving_candidate_count"
        ],
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
        "failure_counts": failure["required_failure_counts"],
    }
    return (
        "\n".join(
            [
                "# Step7I Flow-Based Coarse Region Probe",
                "",
                f"Decision: `{decision}`",
                "",
                "## Scope",
                "",
                "- Sidecar-only coarse-region assignment probe.",
                "- No contest runtime integration, finalizer changes, RL, or "
                "full global optimization.",
                "- Step7G route classification remains active; global candidates are report-only.",
                "- Original layout remains in Pareto comparison.",
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
    if decision == "promote_region_flow_planner":
        return (
            "Coarse region assignment recovered local-starved cases with useful "
            "non-global regional candidates."
        )
    if decision == "refine_region_assignment_costs":
        return (
            "The probe creates plausible non-global regional options, but metric signal "
            "or ranking is still weak."
        )
    if decision == "build_regional_legalizer":
        return (
            "Assignment targets look promising, but concrete regional edits need a "
            "route-specific legalizer."
        )
    if decision == "pivot_to_analytical_force_map":
        return (
            "Flow-style costs do not correlate with actual deltas; try continuous "
            "force maps instead."
        )
    if decision == "pivot_to_macro_closure_generator":
        return "Macro/group closure dominates the observed failures."
    if decision == "pivot_to_training_retrieval_prior":
        return "Heuristic region assignment lacks enough topology signal."
    if decision == "revisit_region_probe_assumptions":
        return (
            "The current coarse regions or assignment construction do not produce "
            "useful regional evidence."
        )
    return "Region signal is inconclusive."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-iter0-trace", default="artifacts/research/step7c_local_iter0_iteration_trace.json"
    )
    parser.add_argument("--output-dir", default="artifacts/research")
    parser.add_argument("--case-id", type=int, action="append", dest="case_ids")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = _load_json(Path(args.local_iter0_trace))
    case_ids = _case_ids(Path(args.local_iter0_trace), args.case_ids)
    cases_by_id = _load_cases(case_ids)
    region_map = build_region_maps(case_ids, cases_by_id)
    candidates = build_flow_assignment_candidates(case_ids, cases_by_id)
    rows = evaluate_flow_assignment_candidates(candidates)
    feasibility = feasibility_report(rows)
    route = route_report(rows)
    metrics = metric_report(rows)
    pareto = pareto_front_report(rows)
    failure = failure_attribution_report(rows)
    starvation = local_starvation_recovery_report(rows, trace)
    corr = assignment_cost_correlation_report(rows)
    decision = decision_for_step7i(feasibility, starvation, corr)
    visualizations = _render_visualizations(rows, feasibility, output_dir)

    _write_json(output_dir / "step7i_region_map.json", region_map)
    _write_json(output_dir / "step7i_assignment_candidates.json", {"rows": rows})
    _write_json(output_dir / "step7i_route_report.json", route | {"rows": rows})
    _write_json(output_dir / "step7i_feasibility_report.json", feasibility)
    _write_json(output_dir / "step7i_metric_report.json", metrics | corr)
    _write_json(output_dir / "step7i_pareto_report.json", pareto)
    _write_json(output_dir / "step7i_failure_attribution.json", failure)
    _write_json(output_dir / "step7i_local_starvation_recovery.json", starvation)
    (output_dir / "step7i_decision.md").write_text(
        _decision_md(decision, feasibility, starvation, pareto, failure)
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "real_case_count": feasibility["real_case_count"],
                "assignment_candidate_count": feasibility["assignment_candidate_count"],
                "route_count_by_class": feasibility["route_count_by_class"],
                "regional_candidate_count": feasibility["regional_candidate_count"],
                "official_like_improving_candidate_count": feasibility[
                    "official_like_improving_candidate_count"
                ],
                "local_starvation_case_recovery_count": starvation[
                    "local_starvation_case_recovery_count"
                ],
                "visualizations": len(visualizations),
                "output": str(output_dir / "step7i_decision.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
