#!/usr/bin/env python3
"""Build Step7Q-A non-leaky operator-learning examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.ml.step7q_operator_learning import build_operator_learning_data_mart


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--replay-rows", type=Path, required=True)
    parser.add_argument("--blocker", type=Path, required=True)
    parser.add_argument("--branch-summary", type=Path, required=True)
    parser.add_argument("--examples-out", type=Path, required=True)
    parser.add_argument("--label-summary-out", type=Path, required=True)
    parser.add_argument("--feature-summary-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    summary = build_operator_learning_data_mart(
        args.atlas,
        args.requests,
        args.replay_rows,
        args.blocker,
        args.branch_summary,
        args.examples_out,
        args.label_summary_out,
        args.feature_summary_out,
        args.summary_out,
        args.markdown_out,
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "example_count": summary["example_count"],
                "feature_label_leakage_count": summary["feature_label_leakage_count"],
                "strict_supervision_enabled": summary["strict_supervision_enabled"],
                "allowed_next_phase": summary["allowed_next_phase"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
