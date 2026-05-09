#!/usr/bin/env python3
"""Run Step7R Phase 1.5 fresh-metric replay for the k=2 swap deck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.ml.step7r_swap_replay import replay_swap_deck


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--examples", type=Path, default=Path("artifacts/research/step7q_operator_examples.jsonl")
    )
    parser.add_argument(
        "--swap-deck", type=Path, default=Path("artifacts/research/step7r_swap_source_deck.jsonl")
    )
    parser.add_argument(
        "--replay-rows-out",
        type=Path,
        default=Path("artifacts/research/step7r_swap_replay_rows.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("artifacts/research/step7r_swap_replay_summary.json"),
    )
    parser.add_argument(
        "--failures-out",
        type=Path,
        default=Path("artifacts/research/step7r_swap_replay_failures_by_case.json"),
    )
    parser.add_argument("--floorset-root", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Process workers. Defaults to min(48, n_rows) after max-candidates slicing.",
    )
    parser.add_argument("--auto-download", action="store_true")
    args = parser.parse_args()

    summary = replay_swap_deck(
        args.base_dir,
        args.examples,
        args.swap_deck,
        args.replay_rows_out,
        args.summary_out,
        args.failures_out,
        floorset_root=args.floorset_root,
        max_candidates=args.max_candidates,
        auto_download=args.auto_download,
        n_workers=args.n_workers,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
