#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def main() -> None:
    research_dir = ROOT / "artifacts" / "research"
    payload = json.loads((research_dir / "generalization_followup_smallcheckpoints.json").read_text())
    rows = payload["rows"]

    untrained_rows = [row for row in rows if row["variant"] == "untrained"]
    trained_rows = [row for row in rows if row["variant"] in {"bc", "awbc"}]
    best_untrained = min(untrained_rows, key=lambda row: float(row["avg_official_cost"]))

    comparisons = []
    for row in trained_rows:
        comparisons.append(
            {
                "variant": row["variant"],
                "seed": row["seed"],
                "delta_official_cost_vs_best_untrained": float(row["avg_official_cost"]) - float(best_untrained["avg_official_cost"]),
                "delta_mean_displacement_vs_best_untrained": float(row["avg_mean_displacement"]) - float(best_untrained["avg_mean_displacement"]),
                "delta_hpwl_gap_vs_best_untrained": float(row["avg_hpwl_gap"]) - float(best_untrained["avg_hpwl_gap"]),
                "delta_area_gap_vs_best_untrained": float(row["avg_area_gap"]) - float(best_untrained["avg_area_gap"]),
                "delta_soft_violations_vs_best_untrained": float(row["avg_total_soft_violations"]) - float(best_untrained["avg_total_soft_violations"]),
                "delta_boundary_violations_vs_best_untrained": float(row["avg_boundary_violations"]) - float(best_untrained["avg_boundary_violations"]),
                "delta_grouping_violations_vs_best_untrained": float(row["avg_grouping_violations"]) - float(best_untrained["avg_grouping_violations"]),
                "delta_preserved_x_order_fraction_vs_best_untrained": float(row["avg_preserved_x_order_fraction"]) - float(best_untrained["avg_preserved_x_order_fraction"]),
            }
        )

    result = {
        "source_artifact": "artifacts/research/generalization_followup_smallcheckpoints.json",
        "best_untrained": {
            "seed": best_untrained["seed"],
            "avg_official_cost": best_untrained["avg_official_cost"],
            "avg_mean_displacement": best_untrained["avg_mean_displacement"],
            "avg_hpwl_gap": best_untrained["avg_hpwl_gap"],
            "avg_area_gap": best_untrained["avg_area_gap"],
            "avg_total_soft_violations": best_untrained["avg_total_soft_violations"],
            "avg_boundary_violations": best_untrained["avg_boundary_violations"],
            "avg_grouping_violations": best_untrained["avg_grouping_violations"],
            "avg_preserved_x_order_fraction": best_untrained["avg_preserved_x_order_fraction"],
        },
        "comparisons": comparisons,
        "interpretation": {
            "observed_pattern": (
                "trained variants can reduce displacement and often preserve x-order better than the best untrained baseline, "
                "but still lose on official cost"
            ),
            "confirmed_driver": (
                "the cost gap is now directly visible in the official quality terms: trained runs have materially worse HPWL gap and area gap than the best untrained baseline"
            ),
            "non_explanation": (
                "soft-violation counts and boundary counts alone do not explain the gap, because trained seed-0 rows improve those while still losing badly on official cost"
            ),
            "conclusion": (
                "lower repair displacement is not translating into better downstream quality; the trained proposals appear to be easier to legalize but worse on HPWL/bbox-related objective terms"
            ),
        },
    }
    (research_dir / "generalization_cost_decomposition.json").write_text(json.dumps(result, indent=2))

    lines = [
        "# Generalization Cost Decomposition",
        "",
        f"- source artifact: `{result['source_artifact']}`",
        f"- best untrained seed: `{best_untrained['seed']}`",
        f"- best untrained mean official cost: `{best_untrained['avg_official_cost']:.3f}`",
        "",
        "| Variant | Seed | Δ official cost | Δ displacement | Δ HPWL gap | Δ area gap | Δ soft violations | Δ boundary | Δ grouping |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['variant']} | {row['seed']} | "
            f"{row['delta_official_cost_vs_best_untrained']:.3f} | "
            f"{row['delta_mean_displacement_vs_best_untrained']:.3f} | "
            f"{row['delta_hpwl_gap_vs_best_untrained']:.3f} | "
            f"{row['delta_area_gap_vs_best_untrained']:.3f} | "
            f"{row['delta_soft_violations_vs_best_untrained']:.3f} | "
            f"{row['delta_boundary_violations_vs_best_untrained']:.3f} | "
            f"{row['delta_grouping_violations_vs_best_untrained']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- Trained checkpoints do **not** currently beat the best untrained baseline on official cost in `validation-0..19`.",
            "- Lower mean displacement is **not sufficient** to produce a better official score.",
            "- The direct cost drivers now appear in the artifact: trained runs, especially seed 0, have much worse `HPWL gap` and `area gap` than the best untrained baseline.",
            "- Boundary/soft-violation counts are not the main explanation, because trained seed-0 runs improve those and still lose badly on cost.",
            "- This branch should be treated as a documented negative result: the current trained checkpoints are easier to legalize, but worse on the official quality objective.",
        ]
    )
    (research_dir / "generalization_cost_decomposition.md").write_text("\n".join(lines))
    print(json.dumps({"comparisons": len(comparisons)}, indent=2))


if __name__ == "__main__":
    main()
