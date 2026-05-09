#!/usr/bin/env python3
"""Run Step7P Phase3 bounded causal replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.experiments.step7p_causal_request_replay import replay_causal_requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--replay-rows-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--failures-out", type=Path, required=True)
    args = parser.parse_args()
    _ = args.base_dir
    summary = replay_causal_requests(
        args.requests,
        args.replay_rows_out,
        args.summary_out,
        args.markdown_out,
        args.failures_out,
    )
    print(json.dumps(compact(summary), indent=2, sort_keys=True))


def compact(summary: dict[str, object]) -> dict[str, object]:
    keys = ("decision", "request_count", "strict_meaningful_winner_count", "phase4_gate_open")
    return {key: summary.get(key) for key in keys}


if __name__ == "__main__":
    main()
