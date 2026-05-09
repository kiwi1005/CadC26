#!/usr/bin/env python3
"""Step7S threshold/noise audit for the two unique Step7Q-F AVNR candidates.

This is intentionally a sidecar-only diagnostic.  It answers the narrow
threshold question from the Step7S Oracle response:

* are the observed 1e-8-scale AVNR improvements reproducible under exact replay?
* is there any same-process replay jitter large enough to justify treating
  MEANINGFUL_COST_EPS=1e-7 as a numerical-noise floor?

The script does *not* lower the project gate.  It emits evidence that the
current 1e-7 threshold is a project-defined meaningfulness floor rather than a
measured replay-noise floor.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from puzzleplace.experiments.step7l_learning_guided_replay import load_validation_cases
from puzzleplace.geometry.legality import positions_from_case_targets
from puzzleplace.ml.step7q_fresh_metric_replay import actual_delta, official_like_evaluator

STRICT_EPS = 1.0e-7
U_DOUBLE = 2.0**-53
Box = tuple[float, float, float, float]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def target_decimal_places(target_box: list[Any]) -> int:
    places = 0
    for value in target_box:
        exponent = Decimal(str(value)).as_tuple().exponent
        places = max(places, max(0, -int(exponent)))
    return places


def unique_avnr_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, tuple[float, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("actual_all_vector_nonregressing"):
            continue
        target = tuple(round(float(v), 9) for v in row.get("target_box", []))
        grouped[(str(row.get("case_id")), int(row.get("block_id", -1)), target)].append(row)

    unique: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        exemplar = dict(group[0])
        exemplar["_occurrences"] = len(group)
        exemplar["_dedup_key"] = {
            "case_id": key[0],
            "block_id": key[1],
            "target_box": list(key[2]),
        }
        unique.append(exemplar)
    return unique


def replay_candidate(
    base_dir: Path,
    case_id: int,
    block_id: int,
    target_box: Box,
    repeats: int,
) -> dict[str, Any]:
    cases = load_validation_cases(base_dir, [case_id])
    case = cases[case_id]
    baseline: list[Box] = [tuple(map(float, box)) for box in positions_from_case_targets(case)]
    candidate = list(baseline)
    candidate[block_id] = target_box

    deltas: list[dict[str, float]] = []
    before_costs: list[float] = []
    after_costs: list[float] = []
    for _ in range(repeats):
        before = official_like_evaluator(case, baseline)
        after = official_like_evaluator(case, candidate)
        delta = actual_delta(before, after)
        deltas.append({k: float(v) for k, v in delta.items()})
        before_costs.append(float(before["quality"]["cost"]))
        after_costs.append(float(after["quality"]["cost"]))

    cost_deltas = [row["official_like_cost_delta"] for row in deltas]
    hpwl_deltas = [row["hpwl_delta"] for row in deltas]
    bbox_deltas = [row["bbox_area_delta"] for row in deltas]
    soft_deltas = [row["soft_constraint_delta"] for row in deltas]
    return {
        "case_id": case_id,
        "block_id": block_id,
        "target_box": list(target_box),
        "repeat_count": repeats,
        "cost_delta_min": min(cost_deltas),
        "cost_delta_max": max(cost_deltas),
        "cost_delta_spread": max(cost_deltas) - min(cost_deltas),
        "hpwl_delta_spread": max(hpwl_deltas) - min(hpwl_deltas),
        "bbox_delta_spread": max(bbox_deltas) - min(bbox_deltas),
        "soft_delta_spread": max(soft_deltas) - min(soft_deltas),
        "before_cost_spread": max(before_costs) - min(before_costs),
        "after_cost_spread": max(after_costs) - min(after_costs),
        "representative_delta": deltas[0],
    }


def gamma_m(m_ops: int) -> float:
    numerator = float(m_ops) * U_DOUBLE
    if numerator >= 1.0:
        return float("inf")
    return numerator / (1.0 - numerator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--rows",
        type=Path,
        default=Path("artifacts/research/step7q_objective_slot_replay_rows.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7s_threshold_noise_audit.json"),
    )
    parser.add_argument("--repeat-count", type=int, default=50)
    parser.add_argument(
        "--m-ops-assumption",
        type=int,
        default=200_000,
        help="Conservative floating-operation count used for gamma_m roundoff guard.",
    )
    parser.add_argument("--safety-multiplier", type=float, default=64.0)
    args = parser.parse_args()

    rows = read_jsonl(args.rows)
    unique_rows = unique_avnr_rows(rows)
    replay_records: list[dict[str, Any]] = []
    max_replay_delta_abs_error = 0.0
    max_cost_spread = 0.0
    max_abs_cost = 0.0
    max_target_decimal_places = 0

    for row in unique_rows:
        case_id = int(row["case_id"])
        block_id = int(row["block_id"])
        target_values = [float(v) for v in row["target_box"]]
        target = tuple(target_values)  # type: ignore[assignment]
        max_target_decimal_places = max(
            max_target_decimal_places,
            target_decimal_places(row.get("target_box", [])),
        )
        replay = replay_candidate(
            args.base_dir,
            case_id=case_id,
            block_id=block_id,
            target_box=target,
            repeats=args.repeat_count,
        )
        stored_delta = float(row["official_like_cost_delta"])
        replay_delta = float(replay["representative_delta"]["official_like_cost_delta"])
        replay["stored_official_like_cost_delta"] = stored_delta
        replay["stored_vs_replayed_abs_error"] = abs(stored_delta - replay_delta)
        replay["occurrences_in_step7q_rows"] = row.get("_occurrences")
        replay["improvement_to_strict_factor"] = (
            STRICT_EPS / (-replay_delta) if replay_delta < 0.0 else math.inf
        )
        max_replay_delta_abs_error = max(
            max_replay_delta_abs_error,
            float(replay["stored_vs_replayed_abs_error"]),
        )
        max_cost_spread = max(max_cost_spread, float(replay["cost_delta_spread"]))
        max_abs_cost = max(
            max_abs_cost,
            abs(float(row["official_before_quality"]["cost"])),
            abs(float(row["official_after_quality"]["cost"])),
        )
        replay_records.append(replay)

    gamma = gamma_m(args.m_ops_assumption)
    roundoff_guard = args.safety_multiplier * gamma * max_abs_cost
    replay_jitter_guard = args.safety_multiplier * max_cost_spread
    replay_mismatch_guard = args.safety_multiplier * max_replay_delta_abs_error
    derived_same_process_noise_guard = (
        roundoff_guard + replay_jitter_guard + replay_mismatch_guard
    )
    observed_improvements = [
        -float(record["representative_delta"]["official_like_cost_delta"])
        for record in replay_records
    ]
    min_observed_improvement = min(observed_improvements, default=0.0)

    summary = {
        "schema": "step7s_threshold_noise_audit_v1",
        "decision": (
            "same_process_noise_below_avnr_delta_and_below_strict_eps"
            if derived_same_process_noise_guard < min_observed_improvement < STRICT_EPS
            else "threshold_noise_needs_followup"
        ),
        "strict_meaningful_eps": STRICT_EPS,
        "repeat_count": args.repeat_count,
        "m_ops_assumption": args.m_ops_assumption,
        "safety_multiplier": args.safety_multiplier,
        "unique_avnr_candidate_count": len(unique_rows),
        "max_target_decimal_places": max_target_decimal_places,
        "min_target_coordinate_step_from_json": 10.0 ** (-max_target_decimal_places),
        "max_replay_cost_delta_spread": max_cost_spread,
        "max_stored_vs_replayed_delta_abs_error": max_replay_delta_abs_error,
        "gamma_m": gamma,
        "max_abs_quality_cost": max_abs_cost,
        "roundoff_guard": roundoff_guard,
        "replay_jitter_guard": replay_jitter_guard,
        "replay_mismatch_guard": replay_mismatch_guard,
        "derived_same_process_noise_guard": derived_same_process_noise_guard,
        "min_observed_avnr_improvement": min_observed_improvement,
        "observed_improvements_below_strict_eps": all(
            value < STRICT_EPS for value in observed_improvements
        ),
        "noise_guard_below_min_observed_improvement": (
            derived_same_process_noise_guard < min_observed_improvement
        ),
        "note": (
            "This same-process audit does not reclassify the project gate. "
            "It only shows that the two 1e-8 AVNR deltas replay deterministically "
            "and are not explained by measured evaluator jitter."
        ),
        "records": replay_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "unique_avnr_candidate_count": len(unique_rows),
                "derived_same_process_noise_guard": derived_same_process_noise_guard,
                "min_observed_avnr_improvement": min_observed_improvement,
                "strict_meaningful_eps": STRICT_EPS,
            }
        )
    )


if __name__ == "__main__":
    main()
