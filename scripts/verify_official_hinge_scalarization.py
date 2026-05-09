#!/usr/bin/env python3
"""Step7S Phase E1: verify official_like_cost matches the public hinge formula.

For each replay row with `official_before_quality` and `official_after_quality`,
compute predicted cost via:

    c_pred(h, a, v) = (1 + 0.5 * (max(0,h) + max(0,a))) * exp(2*v)

and compare against the stored `cost`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ALPHA = 0.5
BETA = 2.0


def predicted_cost(h: float, a: float, v: float) -> float:
    return (1.0 + ALPHA * (max(0.0, h) + max(0.0, a))) * math.exp(BETA * v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=Path,
        default=Path("artifacts/research/step7q_objective_slot_replay_rows.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7s_hinge_scalarization_check.json"),
    )
    parser.add_argument("--abs-tolerance", type=float, default=1e-12)
    parser.add_argument("--rel-tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    rows = []
    with args.rows.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    diffs_before: list[dict[str, Any]] = []
    diffs_after: list[dict[str, Any]] = []
    max_abs_err = 0.0
    max_rel_err = 0.0
    pair_count = 0

    for row in rows:
        for label, key, sink in (
            ("before", "official_before_quality", diffs_before),
            ("after", "official_after_quality", diffs_after),
        ):
            quality = row.get(key)
            if not quality or not quality.get("feasible", True):
                continue
            h = float(quality.get("HPWLgap", 0.0))
            a = float(quality.get("Areagap_bbox", 0.0))
            v = float(quality.get("Violationsrelative", 0.0))
            stored = float(quality.get("cost", 0.0))
            predicted = predicted_cost(h, a, v)
            abs_err = abs(predicted - stored)
            rel_err = abs_err / max(abs(stored), 1e-300)
            max_abs_err = max(max_abs_err, abs_err)
            max_rel_err = max(max_rel_err, rel_err)
            pair_count += 1
            if abs_err > args.abs_tolerance and rel_err > args.rel_tolerance:
                sink.append(
                    {
                        "case_id": row.get("case_id"),
                        "block_id": row.get("block_id"),
                        "side": label,
                        "h": h,
                        "a": a,
                        "v": v,
                        "stored": stored,
                        "predicted": predicted,
                        "abs_err": abs_err,
                        "rel_err": rel_err,
                    }
                )

    verdict = (
        "pass"
        if max_abs_err <= args.abs_tolerance or max_rel_err <= args.rel_tolerance
        else "fail"
    )
    summary = {
        "schema": "step7s_hinge_scalarization_check_v1",
        "verdict": verdict,
        "rows_checked": len(rows),
        "pair_count": pair_count,
        "abs_tolerance": args.abs_tolerance,
        "rel_tolerance": args.rel_tolerance,
        "alpha": ALPHA,
        "beta": BETA,
        "formula": "(1 + 0.5*(max(0,h) + max(0,a))) * exp(2*v)",
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "mismatches_before": diffs_before[:10],
        "mismatches_before_total": len(diffs_before),
        "mismatches_after": diffs_after[:10],
        "mismatches_after_total": len(diffs_after),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "rows_checked": len(rows),
                "pair_count": pair_count,
                "max_abs_err": max_abs_err,
                "max_rel_err": max_rel_err,
            }
        )
    )


if __name__ == "__main__":
    main()
