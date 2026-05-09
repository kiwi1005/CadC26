#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from puzzleplace.alternatives.route_aware_layout_edits import (
    build_layout_edit_candidates,
    confusion_report,
    decision_for_step7c_thin,
    evaluate_layout_edits,
    feasibility_report,
    pareto_report,
)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2))


def _render_visualizations(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "step7c_thin_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for stale in viz_dir.glob("*.png"):
        stale.unlink()
    paths = [
        _bar_chart(
            plt,
            _counts(rows, "actual_locality_class"),
            viz_dir / "actual_route_count_by_class.png",
        ),
        _bar_chart(plt, _counts(rows, "family"), viz_dir / "actual_edit_count_by_family.png"),
    ]
    _write_index(paths, viz_dir)
    return paths


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        out[value] = out.get(value, 0) + 1
    return out


def _bar_chart(plt: Any, counts: dict[str, int], path: Path) -> str:
    names = list(counts)
    values = [counts[name] for name in names]
    fig, ax = plt.subplots(figsize=(max(5.0, len(names) * 0.85), 3.8), dpi=120)
    ax.bar(range(len(names)), values, color="#16a34a")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("actual edit count")
    ax.set_title(path.stem)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.05, str(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _write_index(paths: list[str], viz_dir: Path) -> None:
    sections = []
    for path in paths:
        name = Path(path).name
        sections.append(
            "<section>"
            f"<h2>{html.escape(name)}</h2>"
            f"<img src='{html.escape(name)}' style='max-width:100%;border:1px solid #ddd'/>"
            "</section>"
        )
    (viz_dir / "index.html").write_text(
        "\n".join(
            [
                "<!doctype html><meta charset='utf-8'>",
                "<title>Step7C-thin route-aware layout edit loop</title>",
                "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:32px}</style>",
                "<h1>Step7C-thin route-aware layout edit loop</h1>",
                *sections,
            ]
        )
    )


def _write_decision_md(
    *,
    decision: str,
    feasibility: dict[str, Any],
    confusion: dict[str, Any],
    pareto: dict[str, Any],
) -> str:
    return (
        "\n".join(
            [
                "# Step7C-thin Route-Aware Layout Edit Loop",
                "",
                f"Decision: `{decision}`",
                "",
                "## Actual edit summary",
                "",
                "```json",
                json.dumps(
                    {
                        "descriptor_candidate_count": feasibility[
                            "descriptor_candidate_count"
                        ],
                        "actual_edit_candidate_count": feasibility[
                            "actual_edit_candidate_count"
                        ],
                        "actual_route_count_by_class": feasibility[
                            "actual_route_count_by_class"
                        ],
                        "actual_non_global_candidate_rate": feasibility[
                            "actual_non_global_candidate_rate"
                        ],
                        "invalid_local_attempt_rate": feasibility[
                            "invalid_local_attempt_rate"
                        ],
                        "actual_hard_feasible_rate": feasibility[
                            "actual_hard_feasible_rate"
                        ],
                        "actual_safe_improvement_count": feasibility[
                            "actual_safe_improvement_count"
                        ],
                        "regional_macro_preservation_count": feasibility[
                            "regional_macro_preservation_count"
                        ],
                        "global_report_only_count": feasibility[
                            "global_report_only_count"
                        ],
                    },
                    indent=2,
                ),
                "```",
                "",
                "## Descriptor vs actual routing",
                "",
                "```json",
                json.dumps(confusion, indent=2),
                "```",
                "",
                "## Pareto report",
                "",
                "```json",
                json.dumps(
                    {
                        "actual_pareto_front_non_empty_count": pareto[
                            "actual_pareto_front_non_empty_count"
                        ],
                        "original_inclusive_case_count": pareto[
                            "original_inclusive_case_count"
                        ],
                    },
                    indent=2,
                ),
                "```",
                "",
                "## Interpretation",
                "",
                "Step7C-thin converts a deterministic subset of Step7H proxy lanes "
                "into concrete rectangle layout edits. The result distinguishes "
                "descriptor quality from actual edit behavior and keeps global "
                "candidates report-only rather than sending them to bounded-local repair.",
            ]
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step7h-predictions",
        default="artifacts/research/step7h_route_predictions.json",
    )
    parser.add_argument("--output-dir", default="artifacts/research")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptors = _load_json(Path(args.step7h_predictions), [])
    candidates = build_layout_edit_candidates(descriptors)
    rows = evaluate_layout_edits(candidates)
    confusion = confusion_report(rows)
    feasibility = feasibility_report(rows, descriptor_candidate_count=len(descriptors))
    pareto = pareto_report(rows)
    decision = decision_for_step7c_thin(feasibility, confusion, pareto)
    visualizations = _render_visualizations(rows, output_dir)

    _write_json(
        output_dir / "step7c_thin_layout_edit_candidates.json",
        [
            {
                "case_id": candidate.case_id,
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "descriptor_locality_class": candidate.descriptor_locality_class,
                "block_count": candidate.block_count,
                "route_lane": candidate.route_lane,
            }
            for candidate in candidates
        ],
    )
    _write_json(output_dir / "step7c_thin_actual_routes.json", rows)
    _write_json(output_dir / "step7c_thin_feasibility_report.json", feasibility)
    _write_json(output_dir / "step7c_thin_pareto_report.json", pareto)
    _write_json(output_dir / "step7c_thin_confusion_report.json", confusion)
    (output_dir / "step7c_thin_decision.md").write_text(
        _write_decision_md(
            decision=decision,
            feasibility=feasibility,
            confusion=confusion,
            pareto=pareto,
        )
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "actual_edit_candidate_count": feasibility["actual_edit_candidate_count"],
                "actual_route_count_by_class": feasibility["actual_route_count_by_class"],
                "actual_non_global_candidate_rate": feasibility[
                    "actual_non_global_candidate_rate"
                ],
                "invalid_local_attempt_rate": feasibility["invalid_local_attempt_rate"],
                "actual_hard_feasible_rate": feasibility["actual_hard_feasible_rate"],
                "actual_pareto_front_non_empty_count": pareto[
                    "actual_pareto_front_non_empty_count"
                ],
                "route_stability_descriptor_to_edit": confusion[
                    "route_stability_descriptor_to_edit"
                ],
                "visualizations": len(visualizations),
                "output": str(output_dir / "step7c_thin_decision.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
