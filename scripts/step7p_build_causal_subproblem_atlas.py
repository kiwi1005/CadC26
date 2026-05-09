#!/usr/bin/env python3
"""Run Step7P Phase1 causal subproblem atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.experiments.step7p_causal_subproblem_atlas import build_causal_subproblem_atlas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stagnation-lock", type=Path, required=True)
    parser.add_argument("--step7ml-i-candidates", type=Path, required=True)
    parser.add_argument("--step7ml-k-candidates", type=Path, required=True)
    parser.add_argument("--step7m-phase2-summary", type=Path, required=True)
    parser.add_argument("--step7m-phase4-summary", type=Path, required=True)
    parser.add_argument("--step7n-i-target-quality", type=Path, required=True)
    parser.add_argument("--validation-cases", type=parse_cases, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--failures-out", type=Path, required=True)
    args = parser.parse_args()
    summary = build_causal_subproblem_atlas(
        args.stagnation_lock,
        args.step7ml_i_candidates,
        args.step7ml_k_candidates,
        args.step7m_phase2_summary,
        args.step7m_phase4_summary,
        args.step7n_i_target_quality,
        args.validation_cases,
        args.out,
        args.summary_out,
        args.markdown_out,
        args.failures_out,
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "subproblem_count": summary["subproblem_count"],
                "represented_case_count": summary["represented_case_count"],
                "phase2_gate_open": summary["phase2_gate_open"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_cases(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
