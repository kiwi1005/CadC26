#!/usr/bin/env python3
"""Build Step7O Phase0/1 training-demand prior artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.ml.training_demand_prior import (
    build_input_inventory,
    build_training_demand_atlas,
)


def _case_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["inventory", "atlas"], required=True)
    parser.add_argument(
        "--step7n-phase0-summary",
        type=Path,
        default=Path("artifacts/research/step7n_phase0_lineage_summary.json"),
    )
    parser.add_argument(
        "--step7ml-g-inventory",
        type=Path,
        default=Path("artifacts/research/step7ml_g_data_inventory.json"),
    )
    parser.add_argument(
        "--step7ml-g-schema",
        type=Path,
        default=Path("artifacts/research/step7ml_g_schema_report.json"),
    )
    parser.add_argument(
        "--step7ml-g-layout-prior",
        type=Path,
        default=Path("artifacts/research/step7ml_g_layout_prior_examples.json"),
    )
    parser.add_argument(
        "--step7ml-g-region-heatmap",
        type=Path,
        default=Path("artifacts/research/step7ml_g_region_heatmap_examples.json"),
    )
    parser.add_argument(
        "--step7ml-i-quality",
        type=Path,
        default=Path("artifacts/research/step7ml_i_quality_gate_report.json"),
    )
    parser.add_argument(
        "--step7ml-j-quality",
        type=Path,
        default=Path("artifacts/research/step7ml_j_quality_gate_report.json"),
    )
    parser.add_argument(
        "--step7ml-k-quality",
        type=Path,
        default=Path("artifacts/research/step7ml_k_quality_gate_report.json"),
    )
    parser.add_argument(
        "--inventory-out",
        type=Path,
        default=Path("artifacts/research/step7o_phase0_input_inventory.json"),
    )
    parser.add_argument(
        "--guard-markdown-out",
        type=Path,
        default=Path("artifacts/research/step7o_phase0_stop_guard.md"),
    )
    parser.add_argument(
        "--validation-cases",
        type=_case_list,
        default=[19, 24, 25, 51, 76, 79, 91, 99],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7o_phase1_training_demand_atlas.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("artifacts/research/step7o_phase1_training_demand_summary.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("artifacts/research/step7o_phase1_training_demand_summary.md"),
    )
    args = parser.parse_args()

    if args.mode == "inventory":
        summary = build_input_inventory(
            args.step7n_phase0_summary,
            args.step7ml_g_inventory,
            args.step7ml_i_quality,
            args.step7ml_j_quality,
            args.step7ml_k_quality,
            args.inventory_out,
            args.guard_markdown_out,
            step7ml_g_schema_path=args.step7ml_g_schema,
        )
        print(json.dumps({"decision": summary["decision"]}, indent=2, sort_keys=True))
        return

    summary = build_training_demand_atlas(
        args.step7ml_g_inventory,
        args.step7ml_g_schema,
        args.step7ml_g_layout_prior,
        args.step7ml_g_region_heatmap,
        args.validation_cases,
        args.out,
        args.summary_out,
        args.markdown_out,
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "atlas_row_count": summary["atlas_row_count"],
                "represented_case_count": summary["represented_case_count"],
                "forbidden_validation_label_term_count": summary[
                    "forbidden_validation_label_term_count"
                ],
                "phase2_gate_open": summary["phase2_gate_open"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
