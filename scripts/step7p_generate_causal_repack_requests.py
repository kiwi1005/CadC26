#!/usr/bin/env python3
"""Run Step7P Phase3 causal request generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.experiments.step7p_causal_request_replay import generate_causal_requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subproblem-atlas", type=Path, required=True)
    parser.add_argument("--operator-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    summary = generate_causal_requests(
        args.subproblem_atlas,
        args.operator_contract,
        args.out,
        args.summary_out,
        args.markdown_out,
    )
    print(json.dumps(compact(summary), indent=2, sort_keys=True))


def compact(summary: dict[str, object]) -> dict[str, object]:
    keys = ("decision", "request_count", "represented_case_count", "phase3_replay_gate_open")
    return {key: summary.get(key) for key in keys}


if __name__ == "__main__":
    main()
