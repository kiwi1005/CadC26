#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from puzzleplace.diagnostics.edit_signal_forensics import (
    decision_for_step7c_real_d,
    failure_taxonomy,
    feature_delta_correlation,
    missing_field_report,
    slot_scoring_calibration,
    summary_counts,
    tradeoff_report,
    unified_candidate_table,
    winner_loser_examples,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _route_rows(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return data["rows"]
    if isinstance(data, list):
        return data
    raise ValueError(f"cannot find route rows in {path}")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2))


def _render_visualizations(
    table: list[dict[str, Any]],
    summaries: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "step7c_real_d_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for stale in viz_dir.glob("*.png"):
        stale.unlink()
    paths = [
        _bar_chart(
            plt,
            summaries["candidate_count_by_source_step"],
            viz_dir / "candidate_count_by_source_step.png",
        ),
        _bar_chart(
            plt,
            summaries["official_like_improving_count_by_strategy"],
            viz_dir / "official_like_improving_count_by_strategy.png",
        ),
        _scatter(
            plt,
            table,
            "hpwl_delta",
            "official_like_cost_delta",
            viz_dir / "hpwl_vs_official_delta.png",
        ),
        _scatter(
            plt,
            table,
            "bbox_area_delta",
            "official_like_cost_delta",
            viz_dir / "bbox_vs_official_delta.png",
        ),
        _scatter(
            plt,
            [row for row in table if row["window_block_count"] is not None],
            "window_block_count",
            "official_like_cost_delta",
            viz_dir / "window_size_vs_official_delta.png",
        ),
    ]
    _write_index(paths, viz_dir, summaries)
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
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.05, str(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _scatter(
    plt: Any,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    path: Path,
) -> str:
    fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=120)
    colors = {"step7c_real_b": "#0f766e", "step7c_real_c": "#7c3aed"}
    for source in sorted({row["source_step"] for row in rows}):
        group = [row for row in rows if row["source_step"] == source]
        ax.scatter(
            [float(row[x_key]) for row in group],
            [float(row[y_key]) for row in group],
            label=source,
            alpha=0.75,
            s=20,
            c=colors.get(source, "#111827"),
        )
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.axvline(0.0, color="#9ca3af", linewidth=0.8)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(path.stem)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _write_index(paths: list[str], viz_dir: Path, summaries: dict[str, Any]) -> None:
    sections = []
    for path in paths:
        name = Path(path).name
        sections.append(
            "<section>"
            f"<h2>{html.escape(name)}</h2>"
            f"<img src='{html.escape(name)}' "
            "style='max-width:100%;border:1px solid #ddd'/>"
            "</section>"
        )
    snippet = html.escape(
        json.dumps(
            {
                "candidate_count_by_source_step": summaries["candidate_count_by_source_step"],
                "official_like_improving_count_by_strategy": summaries[
                    "official_like_improving_count_by_strategy"
                ],
            },
            indent=2,
        )
    )
    (viz_dir / "index.html").write_text(
        "\n".join(
            [
                "<!doctype html><meta charset='utf-8'>",
                "<title>Step7C-real-D edit signal forensics</title>",
                "<style>body{font-family:sans-serif;margin:24px}"
                "section{margin-bottom:32px}pre{background:#f8fafc;padding:12px}</style>",
                "<h1>Step7C-real-D edit signal forensics</h1>",
                "<h2>Summary</h2>",
                f"<pre>{snippet}</pre>",
                *sections,
            ]
        )
    )


def _decision_md(
    *,
    decision: str,
    table: list[dict[str, Any]],
    summaries: dict[str, Any],
    correlations: dict[str, Any],
    tradeoffs: dict[str, Any],
    slot: dict[str, Any],
    failures: dict[str, Any],
    examples: dict[str, Any],
    missing: dict[str, Any],
) -> str:
    b_rows = [row for row in table if row["source_step"] == "step7c_real_b"]
    c_rows = [row for row in table if row["source_step"] == "step7c_real_c"]
    compact = {
        "decision": decision,
        "candidate_count_by_source_step": summaries["candidate_count_by_source_step"],
        "b_official_like_improving_count": sum(
            int(row["official_like_improving"]) for row in b_rows
        ),
        "c_official_like_improving_count": sum(
            int(row["official_like_improving"]) for row in c_rows
        ),
        "hpwl_delta_vs_official_like_delta": correlations[
            "hpwl_delta_vs_official_like_delta_summary"
        ],
        "tradeoff_overall": tradeoffs["overall"],
        "failure_decision_counts": failures["decision_relevant_counts"],
        "missing_optional_counts": missing["optional_missing_counts"],
    }
    return (
        "\n".join(
            [
                "# Step7C-real-D Edit Signal Forensics and Slot Scoring Calibration",
                "",
                f"Decision: `{decision}`",
                "",
                "## Scope",
                "",
                "- Sidecar-only forensic comparison of Step7C-real-B and Step7C-real-C.",
                "- No runtime solver integration, finalizer changes, RL, or new generator.",
                "- Non-improving, infeasible, no-op, and global report-only rows "
                "remain in the table.",
                "",
                "## Summary",
                "",
                "```json",
                json.dumps(compact, indent=2),
                "```",
                "",
                "## Winner/loser signal",
                "",
                "```json",
                json.dumps(
                    {
                        "top_positive_examples": examples["top_positive_examples"][:5],
                        "top_negative_examples": examples["top_negative_examples"][:5],
                    },
                    indent=2,
                ),
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
    if decision == "expand_slack_fit_local_lane":
        return (
            "The only repeatable official-like winners are local one-block HPWL/slack-fit moves "
            "with zero bbox and soft penalty. Step7C-real-C window repacks preserved safety but "
            "added noisy multi-block/window effects and reduced official-like improvement density. "
            "The next implementation should expand the slack-fit/local HPWL lane with better "
            "candidate coverage and strict bbox/soft neutral filters before "
            "another window generator."
        )
    if decision == "refine_slot_scoring":
        return (
            "Metric pressure is useful, but candidate/slot ranking is misaligned "
            "with official-like "
            "deltas. Improve slot/window scoring before generating more moves."
        )
    if decision == "build_route_specific_legalizers":
        return (
            "Promising candidates are blocked by hard feasibility, "
            "no-op behavior, or repair limits."
        )
    if decision == "pivot_to_coarse_region_planner":
        return (
            "Local/window moves are mostly feasible but too weak "
            "for meaningful official-like gains."
        )
    if decision == "pivot_to_macro_closure_generator":
        return (
            "MIB/group or regional topology dominates the useful signal; focus on macro closures."
        )
    if decision == "build_training_retrieval_prior":
        return (
            "Heuristic pressure features do not explain winners; "
            "topology retrieval/templates are needed."
        )
    if decision == "revisit_official_metric_model":
        return (
            "Official-like metric fields are incomplete or inconsistent enough "
            "to revisit evaluator modeling."
        )
    return "Artifact gaps are too large to distinguish the next implementation path."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step7c-real-b-route", default="artifacts/research/step7c_real_b_route_report.json"
    )
    parser.add_argument(
        "--step7c-real-c-route", default="artifacts/research/step7c_real_c_route_report.json"
    )
    parser.add_argument("--output-dir", default="artifacts/research")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    b_rows = _route_rows(Path(args.step7c_real_b_route))
    c_rows = _route_rows(Path(args.step7c_real_c_route))
    table = unified_candidate_table(b_rows, c_rows)
    summaries = summary_counts(table)
    correlations = feature_delta_correlation(table)
    examples = winner_loser_examples(table)
    tradeoffs = tradeoff_report(table)
    slot = slot_scoring_calibration(table)
    failures = failure_taxonomy(table)
    missing = missing_field_report(table)
    decision = decision_for_step7c_real_d(table, summaries, correlations, failures, missing)
    visualizations = _render_visualizations(table, summaries, output_dir)

    _write_json(output_dir / "step7c_real_d_unified_candidate_table.json", table)
    _write_json(output_dir / "step7c_real_d_feature_delta_correlation.json", correlations)
    _write_json(output_dir / "step7c_real_d_winner_loser_examples.json", examples)
    _write_json(output_dir / "step7c_real_d_tradeoff_report.json", tradeoffs)
    _write_json(output_dir / "step7c_real_d_slot_scoring_calibration.json", slot)
    _write_json(
        output_dir / "step7c_real_d_failure_taxonomy.json", failures | {"summaries": summaries}
    )
    (output_dir / "step7c_real_d_decision.md").write_text(
        _decision_md(
            decision=decision,
            table=table,
            summaries=summaries,
            correlations=correlations,
            tradeoffs=tradeoffs,
            slot=slot,
            failures=failures,
            examples=examples,
            missing=missing,
        )
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "candidate_count_by_source_step": summaries["candidate_count_by_source_step"],
                "official_like_improving_count_by_strategy": summaries[
                    "official_like_improving_count_by_strategy"
                ],
                "hpwl_gain_to_official_improve_rate": tradeoffs["overall"][
                    "hpwl_gain_to_official_improve_rate"
                ],
                "missing_required_counts": missing["required_missing_counts"],
                "visualizations": len(visualizations),
                "output": str(output_dir / "step7c_real_d_decision.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
