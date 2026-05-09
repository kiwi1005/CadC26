#!/usr/bin/env python3
"""Run Step7O Phase2 prior calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.experiments.step7o_prior_calibration import run_prior_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas",
        type=Path,
        default=Path("artifacts/research/step7o_phase1_training_demand_atlas.jsonl"),
    )
    parser.add_argument(
        "--step7ml-i-candidates",
        type=Path,
        default=Path("artifacts/research/step7ml_i_decoded_candidates.json"),
    )
    parser.add_argument(
        "--step7ml-k-candidates",
        type=Path,
        default=Path("artifacts/research/step7ml_k_invariant_candidates.json"),
    )
    parser.add_argument(
        "--step7n-i-candidates",
        type=Path,
        default=Path("artifacts/research/step7n_i_quality_filtered_candidates.json"),
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
        "--rows-out",
        type=Path,
        default=Path("artifacts/research/step7o_phase2_prior_calibration_rows.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("artifacts/research/step7o_phase2_prior_calibration_summary.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("artifacts/research/step7o_phase2_prior_calibration_summary.md"),
    )
    args = parser.parse_args()
    summary = run_prior_calibration(
        args.atlas,
        args.step7ml_i_candidates,
        args.step7ml_k_candidates,
        args.step7n_i_candidates,
        args.rows_out,
        args.summary_out,
        args.markdown_out,
        step7ml_j_quality_path=args.step7ml_j_quality,
        step7ml_k_quality_path=args.step7ml_k_quality,
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "top_budget_count": summary["top_budget_count"],
                "top_budget_official_like_improving_count": summary[
                    "top_budget_official_like_improving_count"
                ],
                "phase3_gate_open": summary["phase3_gate_open"],
                "gnn_rl_gate_open": summary["gnn_rl_gate_open"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
