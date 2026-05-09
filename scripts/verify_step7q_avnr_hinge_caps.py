#!/usr/bin/env python3
"""Step7S Phase E3: verify AVNR official improvement equals HPWL hinge cap.

For each AVNR row with bbox_delta=0 and soft_delta=0, the maximum possible
official improvement under the hinge scalarization is:

    U = 0.5 * exp(2 * s_baseline) * max(0, h_baseline)

We compare U against -ΔC. Oracle predicted exact match for the two unique
Step7Q-F AVNR candidates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

EPS = 1e-9


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
        default=Path("artifacts/research/step7s_avnr_hinge_cap_check.json"),
    )
    parser.add_argument("--rel-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    rows = []
    with args.rows.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    records: list[dict[str, Any]] = []
    max_rel_err = 0.0
    pass_count = 0
    skip_count = 0

    for row in rows:
        if not row.get("actual_all_vector_nonregressing"):
            continue
        bbox = float(row.get("bbox_area_delta", 0.0))
        soft = float(row.get("soft_constraint_delta", 0.0))
        if abs(bbox) > EPS or abs(soft) > EPS:
            skip_count += 1
            continue
        before = row.get("official_before_quality") or {}
        h0 = float(before.get("HPWLgap", 0.0))
        s0 = float(before.get("Violationsrelative", 0.0))
        cap = 0.5 * math.exp(2.0 * s0) * max(0.0, h0)
        observed = -float(row.get("official_like_cost_delta", 0.0))
        rel_err = abs(cap - observed) / max(abs(cap), 1e-300)
        max_rel_err = max(max_rel_err, rel_err)
        passed = rel_err <= args.rel_tolerance
        pass_count += int(passed)
        records.append(
            {
                "case_id": row.get("case_id"),
                "block_id": row.get("block_id"),
                "h0": h0,
                "s0": s0,
                "predicted_cap": cap,
                "observed_neg_delta_c": observed,
                "rel_err": rel_err,
                "matches": passed,
            }
        )

    verdict = "pass" if records and pass_count == len(records) else "fail"
    summary = {
        "schema": "step7s_avnr_hinge_cap_check_v1",
        "verdict": verdict,
        "rel_tolerance": args.rel_tolerance,
        "checked_avnr_count": len(records),
        "pass_count": pass_count,
        "skipped_with_nonzero_bbox_or_soft": skip_count,
        "max_rel_err": max_rel_err,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "checked_avnr_count": len(records),
                "pass_count": pass_count,
                "max_rel_err": max_rel_err,
            }
        )
    )


if __name__ == "__main__":
    main()
