#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from puzzleplace.alternatives.route_aware_candidates import (
    candidate_diversity_report,
    decision_for_step7h,
    generate_route_aware_candidates,
    infer_case_block_counts,
    min_slack_by_case,
    pareto_report,
    predict_candidates,
    synthetic_probe_candidates,
    synthetic_probe_report,
)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2))


def _render_visualizations(diversity: dict[str, Any], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "step7h_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for stale in viz_dir.glob("*.png"):
        stale.unlink()
    paths: list[str] = []
    class_counts = diversity["candidate_count_by_class"]
    family_counts = diversity["candidate_count_by_family"]
    paths.append(_bar_chart(plt, class_counts, viz_dir / "candidate_count_by_class.png"))
    paths.append(_bar_chart(plt, family_counts, viz_dir / "candidate_count_by_family.png"))
    _write_index(paths, viz_dir)
    return paths


def _bar_chart(plt: Any, counts: dict[str, int], path: Path) -> str:
    names = list(counts)
    values = [int(counts[name]) for name in names]
    fig, ax = plt.subplots(figsize=(max(5.0, len(names) * 0.7), 3.6), dpi=120)
    ax.bar(range(len(names)), values, color="#2563eb")
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
                "<title>Step7H route-aware candidate diversification</title>",
                "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:32px}</style>",
                "<h1>Step7H route-aware candidate diversification</h1>",
                *sections,
            ]
        )
    )


def _write_decision_md(
    *,
    decision: str,
    diversity: dict[str, Any],
    pareto: dict[str, Any],
    synthetic: dict[str, Any],
    step7g_decision: str,
) -> str:
    return (
        "\n".join(
            [
                "# Step7H Route-Aware Candidate Diversification",
                "",
                f"Decision: `{decision}`",
                "",
                "## Grounding",
                "",
                f"- Step7G decision: {step7g_decision}",
                "- Step7G router remains a safety layer, not a discard gate.",
                "- This is sidecar-only; no contest runtime/finalizer changes.",
                "",
                "## Candidate diversity",
                "",
                "```json",
                json.dumps(
                    {
                        "candidate_count": diversity["candidate_count"],
                        "candidate_count_by_class": diversity["candidate_count_by_class"],
                        "non_global_candidate_rate": diversity["non_global_candidate_rate"],
                        "local_candidate_rate": diversity["local_candidate_rate"],
                        "regional_candidate_rate": diversity["regional_candidate_rate"],
                        "macro_candidate_rate": diversity["macro_candidate_rate"],
                        "global_candidate_rate": diversity["global_candidate_rate"],
                        "invalid_local_attempt_rate": diversity["invalid_local_attempt_rate"],
                    },
                    indent=2,
                ),
                "```",
                "",
                "## Pareto / safety summary",
                "",
                "```json",
                json.dumps(
                    {
                        "pareto_front_non_empty_count": pareto[
                            "pareto_front_non_empty_count"
                        ],
                        "useful_regional_macro_candidate_count": pareto[
                            "useful_regional_macro_candidate_count"
                        ],
                        "safe_improvement_preservation": pareto[
                            "safe_improvement_preservation"
                        ],
                    },
                    indent=2,
                ),
                "```",
                "",
                "## Synthetic probe report",
                "",
                "```json",
                json.dumps(
                    {
                        "pass_count": synthetic["pass_count"],
                        "total": synthetic["total"],
                        "under_predicted_globality": synthetic[
                            "under_predicted_globality"
                        ],
                        "over_predicted_globality": synthetic[
                            "over_predicted_globality"
                        ],
                        "router_class_confusion": synthetic["router_class_confusion"],
                    },
                    indent=2,
                ),
                "```",
                "",
                "## Interpretation",
                "",
                "Step7H separates intrinsically global candidates from smaller "
                "deterministic local/regional/macro lanes before repair. The "
                "candidate set is still proxy/sidecar, so Step7C should validate "
                "actual layout construction and repair, but the all-global Step7G "
                "failure mode is no longer forced by the candidate surface itself.",
            ]
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step7g-maps", default="artifacts/research/step7g_locality_maps.json")
    parser.add_argument(
        "--step7g-predictions",
        default="artifacts/research/step7g_move_locality_predictions.json",
    )
    parser.add_argument(
        "--step7g-routing",
        default="artifacts/research/step7g_routing_results.json",
    )
    parser.add_argument("--step7g-decision", default="artifacts/research/step7g_decision.md")
    parser.add_argument("--output-dir", default="artifacts/research")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step7g_maps = _load_json(Path(args.step7g_maps), [])
    step7g_predictions = _load_json(Path(args.step7g_predictions), [])
    step7g_routing = _load_json(Path(args.step7g_routing), {})
    step7g_decision = Path(args.step7g_decision).read_text().splitlines()[2]

    block_counts = infer_case_block_counts(step7g_predictions)
    slack = min_slack_by_case(step7g_maps)
    case_ids = [int(row["case_id"]) for row in step7g_predictions]
    candidates = generate_route_aware_candidates(
        case_ids=case_ids,
        block_counts=block_counts,
        min_slack=slack,
    )
    predictions = predict_candidates(candidates)
    synthetic_candidates = synthetic_probe_candidates()
    synthetic_predictions = predict_candidates(synthetic_candidates)
    diversity = candidate_diversity_report(predictions)
    synthetic = synthetic_probe_report(synthetic_predictions)
    preserved_cases = step7g_routing.get("quality", {}).get(
        "safe_improvement_cases_after_routing_preserved", []
    )
    pareto = pareto_report(predictions, preserved_step7g_safe_cases=preserved_cases)
    decision = decision_for_step7h(diversity, pareto, synthetic)
    visualizations = _render_visualizations(diversity, output_dir)

    _write_json(output_dir / "step7h_candidate_diversity.json", diversity)
    _write_json(output_dir / "step7h_route_predictions.json", predictions)
    _write_json(output_dir / "step7h_pareto_report.json", pareto)
    _write_json(output_dir / "step7h_synthetic_probe_report.json", synthetic)
    (output_dir / "step7h_decision.md").write_text(
        _write_decision_md(
            decision=decision,
            diversity=diversity,
            pareto=pareto,
            synthetic=synthetic,
            step7g_decision=step7g_decision,
        )
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "candidate_count": diversity["candidate_count"],
                "candidate_count_by_class": diversity["candidate_count_by_class"],
                "invalid_local_attempt_rate": diversity["invalid_local_attempt_rate"],
                "useful_regional_macro_candidate_count": pareto[
                    "useful_regional_macro_candidate_count"
                ],
                "synthetic_probe_pass": f"{synthetic['pass_count']}/{synthetic['total']}",
                "visualizations": len(visualizations),
                "output": str(output_dir / "step7h_decision.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
