#!/usr/bin/env python3
"""Run Step7P Phase0 stagnation lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.experiments.step7p_stagnation_lock import build_stagnation_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step7l-summary", type=Path, required=True)
    parser.add_argument("--step7m-phase2", type=Path, required=True)
    parser.add_argument("--step7m-phase3", type=Path, required=True)
    parser.add_argument("--step7m-phase4", type=Path, required=True)
    parser.add_argument("--step7n-phase0", type=Path, required=True)
    parser.add_argument("--step7ml-i-quality", type=Path, required=True)
    parser.add_argument("--step7ml-j-quality", type=Path, required=True)
    parser.add_argument("--step7ml-k-quality", type=Path, required=True)
    parser.add_argument("--step7n-i-target-quality", type=Path, required=True)
    parser.add_argument("--step7o-phase2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--candidate-audit-out", type=Path, required=True)
    args = parser.parse_args()
    summary = build_stagnation_lock(
        args.step7l_summary,
        args.step7m_phase2,
        args.step7m_phase3,
        args.step7m_phase4,
        args.step7n_phase0,
        args.step7ml_i_quality,
        args.step7ml_j_quality,
        args.step7ml_k_quality,
        args.step7n_i_target_quality,
        args.step7o_phase2,
        args.out,
        args.markdown_out,
        args.candidate_audit_out,
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "phase1_gate_open": summary["phase1_gate_open"],
                "step7ml_winner_baseline": summary["step7ml_winner_baseline"],
                "step7o_phase3_gate_open": summary["step7o_phase3_gate_open"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
