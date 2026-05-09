#!/usr/bin/env python3
"""Run Step7P Phase4 causal family ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.experiments.step7p_causal_request_replay import run_family_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-rows", type=Path, required=True)
    parser.add_argument("--rows-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    summary = run_family_ablation(
        args.replay_rows, args.rows_out, args.summary_out, args.markdown_out
    )
    print(json.dumps(compact(summary), indent=2, sort_keys=True))


def compact(summary: dict[str, object]) -> dict[str, object]:
    keys = ("decision", "family_count", "passed_family_count", "gnn_rl_gate_open")
    return {key: summary.get(key) for key in keys}


if __name__ == "__main__":
    main()
