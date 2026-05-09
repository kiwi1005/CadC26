#!/usr/bin/env python3
"""Review Step7T active-soft strict winners for Phase4/integration readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.experiments.step7t_phase4_review import (
    load_json,
    review_step7t_phase4,
    write_review_markdown,
)


def optional_json(path: Path) -> dict | None:
    return load_json(path) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/research/step7t_active_soft_summary.json"),
    )
    parser.add_argument(
        "--visual-sanity",
        type=Path,
        default=Path("artifacts/research/step7t_visual_sanity.json"),
    )
    parser.add_argument(
        "--step7s-summary",
        type=Path,
        default=Path("artifacts/research/step7s_critical_cone_summary.json"),
    )
    parser.add_argument(
        "--step7q-summary",
        type=Path,
        default=Path("artifacts/research/step7q_objective_slot_replay_summary.json"),
    )
    parser.add_argument(
        "--step7r-decision",
        type=Path,
        default=Path("artifacts/research/step7r_close_decision.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7t_phase4_review.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("artifacts/research/step7t_phase4_review.md"),
    )
    args = parser.parse_args()

    review = review_step7t_phase4(
        load_json(args.summary),
        load_json(args.visual_sanity),
        step7s_summary=optional_json(args.step7s_summary),
        step7q_summary=optional_json(args.step7q_summary),
        step7r_decision=optional_json(args.step7r_decision),
        source_summary_path=str(args.summary),
        visual_sanity_path=str(args.visual_sanity),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")
    args.markdown_out.write_text(write_review_markdown(review), encoding="utf-8")
    print(
        {
            "out": str(args.out),
            "markdown_out": str(args.markdown_out),
            "decision": review["decision"],
            "sidecar_phase4_review_pass": review["sidecar_phase4_review_pass"],
            "runtime_integration_gate_open": review["runtime_integration_gate_open"],
            "strict_winner_cases": review["strict_winner_cases"],
            "recommended_next_experiment": review["recommended_next_experiment"],
        }
    )


if __name__ == "__main__":
    main()
