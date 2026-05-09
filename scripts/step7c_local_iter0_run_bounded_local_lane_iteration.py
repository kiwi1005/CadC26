#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from puzzleplace.search.local_lane_iteration import (
    DEFAULT_ITERATION_LIMIT,
    run_bounded_local_lane_iteration,
)
from puzzleplace.train.dataset_bc import load_validation_cases


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2))


def _case_ids(preselection_path: Path, explicit: list[int] | None) -> list[int]:
    if explicit:
        return sorted(explicit)
    rows = _load_json(preselection_path)
    if not isinstance(rows, list):
        raise ValueError(f"expected list in {preselection_path}")
    return sorted({int(row["case_id"]) for row in rows})


def _load_cases(case_ids: list[int]) -> dict[int, Any]:
    cases = load_validation_cases(case_limit=max(case_ids) + 1)
    return {case_id: cases[case_id] for case_id in case_ids}


def _render_visualizations(reports: dict[str, Any], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "step7c_local_iter0_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for stale in viz_dir.glob("*.png"):
        stale.unlink()
    paths = [
        _bar_chart(
            plt,
            reports["candidate_report"]["candidates_generated_by_iteration"],
            viz_dir / "candidates_generated_by_iteration.png",
        ),
        _bar_chart(
            plt,
            reports["candidate_report"]["candidates_retained_by_preselection_by_iteration"],
            viz_dir / "candidates_retained_by_iteration.png",
        ),
        _bar_chart(
            plt,
            reports["candidate_report"]["selected_edits_by_iteration"],
            viz_dir / "selected_edits_by_iteration.png",
        ),
        _case_delta_chart(
            plt,
            reports["metric_report"]["cumulative_official_like_cost_delta_by_case"],
            viz_dir / "cumulative_official_delta_by_case.png",
        ),
    ]
    _write_index(paths, viz_dir, reports)
    return paths


def _bar_chart(plt: Any, counts: dict[str, int], path: Path) -> str:
    names = list(counts)
    values = [int(counts[name]) for name in names]
    fig, ax = plt.subplots(figsize=(max(4.8, len(names) * 0.9), 3.6), dpi=120)
    ax.bar(range(len(names)), values, color="#2563eb")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=0)
    ax.set_ylabel("count")
    ax.set_title(path.stem)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _case_delta_chart(plt: Any, deltas: dict[str, float], path: Path) -> str:
    names = list(deltas)
    values = [float(deltas[name]) for name in names]
    colors = ["#16a34a" if value < 0 else "#dc2626" if value > 0 else "#64748b" for value in values]
    fig, ax = plt.subplots(figsize=(max(5.0, len(names) * 0.55), 3.6), dpi=120)
    ax.bar(range(len(names)), values, color=colors)
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=0)
    ax.set_ylabel("cumulative official-like cost delta")
    ax.set_title(path.stem)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _write_index(paths: list[str], viz_dir: Path, reports: dict[str, Any]) -> None:
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
                "decision": reports["decision"],
                "selected_edit_count": reports["candidate_report"]["selected_edit_count"],
                "skipped_cases_by_reason": reports["candidate_report"]["skipped_cases_by_reason"],
                "cumulative_official_like_cost_delta_by_case": reports["metric_report"][
                    "cumulative_official_like_cost_delta_by_case"
                ],
            },
            indent=2,
        )
    )
    (viz_dir / "index.html").write_text(
        "\n".join(
            [
                "<!doctype html><meta charset='utf-8'>",
                "<title>Step7C-local-iter-0 bounded local lane iteration</title>",
                "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:32px}pre{background:#f8fafc;padding:12px}</style>",
                "<h1>Step7C-local-iter-0 bounded local lane iteration</h1>",
                "<h2>Summary</h2>",
                f"<pre>{snippet}</pre>",
                *sections,
            ]
        )
    )


def _decision_md(reports: dict[str, Any]) -> str:
    summary = {
        "decision": reports["decision"],
        "real_case_count": reports["candidate_report"]["real_case_count"],
        "configured_iteration_limit": reports["candidate_report"]["configured_iteration_limit"],
        "completed_iterations_by_case": reports["iteration_trace"]["completed_iterations_by_case"],
        "candidates_generated_by_iteration": reports["candidate_report"][
            "candidates_generated_by_iteration"
        ],
        "candidates_retained_by_preselection_by_iteration": reports["candidate_report"][
            "candidates_retained_by_preselection_by_iteration"
        ],
        "selected_edits_by_iteration": reports["candidate_report"]["selected_edits_by_iteration"],
        "skipped_cases_by_reason": reports["candidate_report"]["skipped_cases_by_reason"],
        "route_count_by_iteration": reports["route_report"]["route_count_by_iteration"],
        "invalid_local_attempt_rate": reports["route_report"]["invalid_local_attempt_rate"],
        "official_like_hard_feasible_rate_by_iteration": reports["feasibility_report"][
            "official_like_hard_feasible_rate_by_iteration"
        ],
        "cumulative_official_like_cost_delta_by_case": reports["metric_report"][
            "cumulative_official_like_cost_delta_by_case"
        ],
        "candidate_starvation_count": reports["failure_attribution"]["candidate_starvation_count"],
        "sequential_feasibility_collapse_count": reports["failure_attribution"][
            "sequential_feasibility_collapse_count"
        ],
        "metric_regression_after_application_count": reports["failure_attribution"][
            "metric_regression_after_application_count"
        ],
    }
    return (
        "\n".join(
            [
                "# Step7C-local-iter-0 Bounded Local-Lane Iteration Sidecar",
                "",
                f"Decision: `{reports['decision']}`",
                "",
                "## Scope",
                "",
                "- Sidecar-only bounded local-lane iteration prototype.",
                "- No contest runtime integration, finalizer changes, RL, or "
                "full large-scale search.",
                "- Candidates are regenerated from the updated placement state each iteration.",
                "- Step7G locality routing remains active; global lanes stay report-only.",
                "- Both current-inclusive and original-inclusive Pareto status are reported.",
                "",
                "## Summary",
                "",
                "```json",
                json.dumps(summary, indent=2),
                "```",
                "",
                "## Interpretation",
                "",
                _interpretation(str(reports["decision"])),
            ]
        )
        + "\n"
    )


def _interpretation(decision: str) -> str:
    if decision == "promote_to_step7c_local_lane_search":
        return (
            "Sequential regeneration preserved useful local slack-fit wins without route or "
            "feasibility collapse. A broader local-lane search sidecar is justified."
        )
    if decision == "refine_sequential_local_preselection":
        return (
            "Sequential application found some cumulative official-like gains, but the signal is "
            "not yet broad enough for promotion. Refine per-state preselection/selection policy."
        )
    if decision == "improve_state_regeneration_after_edit":
        return (
            "Updated-state maps/routes/metrics appear stale or inconsistent after edit application."
        )
    if decision == "add_route_specific_local_legalizer":
        return (
            "Selected local edits need route-specific legalization support before broader search."
        )
    if decision == "pivot_to_coarse_region_planner":
        return (
            "Local sequential candidate regeneration starved too quickly; "
            "coarse region planning is favored."
        )
    if decision == "revisit_local_lane_iteration_assumptions":
        return "Artifact-replay wins did not reproduce under sequential real placement states."
    return "Sequential evidence is unstable or insufficient for a confident promotion."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step7c-real-f-candidates",
        default="artifacts/research/step7c_real_f_preselection_candidates.json",
    )
    parser.add_argument("--output-dir", default="artifacts/research")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATION_LIMIT)
    parser.add_argument("--rank-budget", type=int, default=12)
    parser.add_argument("--case-id", type=int, action="append", dest="case_ids")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_ids = _case_ids(Path(args.step7c_real_f_candidates), args.case_ids)
    reports = run_bounded_local_lane_iteration(
        case_ids,
        _load_cases(case_ids),
        iteration_limit=args.iterations,
        rank_budget=args.rank_budget,
    )
    visualizations = _render_visualizations(reports, output_dir)

    _write_json(output_dir / "step7c_local_iter0_iteration_trace.json", reports["iteration_trace"])
    _write_json(
        output_dir / "step7c_local_iter0_candidate_report.json", reports["candidate_report"]
    )
    _write_json(output_dir / "step7c_local_iter0_route_report.json", reports["route_report"])
    _write_json(
        output_dir / "step7c_local_iter0_feasibility_report.json", reports["feasibility_report"]
    )
    _write_json(output_dir / "step7c_local_iter0_metric_report.json", reports["metric_report"])
    _write_json(output_dir / "step7c_local_iter0_pareto_report.json", reports["pareto_report"])
    _write_json(
        output_dir / "step7c_local_iter0_failure_attribution.json",
        reports["failure_attribution"],
    )
    (output_dir / "step7c_local_iter0_decision.md").write_text(_decision_md(reports))

    print(
        json.dumps(
            {
                "decision": reports["decision"],
                "real_case_count": reports["candidate_report"]["real_case_count"],
                "configured_iteration_limit": reports["candidate_report"][
                    "configured_iteration_limit"
                ],
                "selected_edit_count": reports["candidate_report"]["selected_edit_count"],
                "skipped_cases_by_reason": reports["candidate_report"]["skipped_cases_by_reason"],
                "invalid_local_attempt_rate": reports["route_report"]["invalid_local_attempt_rate"],
                "cumulative_official_like_cost_delta_by_case": reports["metric_report"][
                    "cumulative_official_like_cost_delta_by_case"
                ],
                "visualizations": len(visualizations),
                "output": str(output_dir / "step7c_local_iter0_decision.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
