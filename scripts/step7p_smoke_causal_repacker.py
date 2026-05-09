#!/usr/bin/env python3
"""Run Step7P Phase2 synthetic causal repacker smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.repack.causal_closure_repacker import run_synthetic_repacker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--operator-contract-out", type=Path, required=True)
    args = parser.parse_args()
    summary = run_synthetic_repacker(
        args.fixtures, args.report_out, args.markdown_out, args.operator_contract_out
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "fixture_count": summary["fixture_count"],
                "phase3_gate_open": summary["phase3_gate_open"],
                "gnn_rl_gate_open": summary["gnn_rl_gate_open"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
