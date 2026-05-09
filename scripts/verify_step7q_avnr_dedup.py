#!/usr/bin/env python3
"""Step7S Phase E2: formalize Step7Q-F AVNR row dedup count.

Group AVNR rows by (case_id, block_id, target_box) and confirm the unique
candidate count. Step7R's ad-hoc analysis observed 27 -> 2.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


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
        default=Path("artifacts/research/step7s_avnr_unique_candidate_check.json"),
    )
    parser.add_argument("--expected-avnr-count", type=int, default=27)
    parser.add_argument("--expected-unique-count", type=int, default=2)
    args = parser.parse_args()

    rows = []
    with args.rows.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    avnr = [row for row in rows if bool(row.get("actual_all_vector_nonregressing"))]
    by_unique: dict[tuple[str, int, tuple[float, ...]], list[dict[str, Any]]] = {}
    for row in avnr:
        target = row.get("target_box") or []
        key = (
            str(row.get("case_id")),
            int(row.get("block_id", -1)),
            tuple(round(float(v), 9) for v in target),
        )
        by_unique.setdefault(key, []).append(row)

    unique_records = []
    for (case_id, block_id, target), occurrences in sorted(
        by_unique.items(), key=lambda kv: (kv[0][0], kv[0][1])
    ):
        sample = occurrences[0]
        unique_records.append(
            {
                "case_id": case_id,
                "block_id": block_id,
                "target_box": list(target),
                "occurrences": len(occurrences),
                "official_like_cost_delta": sample.get("official_like_cost_delta"),
                "hpwl_delta": sample.get("hpwl_delta"),
                "bbox_area_delta": sample.get("bbox_area_delta"),
                "soft_constraint_delta": sample.get("soft_constraint_delta"),
            }
        )

    verdict = (
        "pass"
        if len(avnr) == args.expected_avnr_count
        and len(by_unique) == args.expected_unique_count
        else "fail"
    )
    summary = {
        "schema": "step7s_avnr_unique_candidate_check_v1",
        "verdict": verdict,
        "rows_total": len(rows),
        "avnr_row_count": len(avnr),
        "expected_avnr_count": args.expected_avnr_count,
        "unique_candidate_count": len(by_unique),
        "expected_unique_count": args.expected_unique_count,
        "occurrence_distribution": dict(Counter(len(v) for v in by_unique.values())),
        "unique_candidates": unique_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "avnr_row_count": len(avnr),
                "unique_candidate_count": len(by_unique),
            }
        )
    )


if __name__ == "__main__":
    main()
