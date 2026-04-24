#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step6E diagnostic: convert multi-continuation rollout returns into "
            "pairwise consensus/advantage evidence without tuning ranker weights."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "artifacts" / "research" / "step6e_rollout_label_stability_slice.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "research" / "step6e_consensus_advantage_diagnostic.json",
    )
    return parser.parse_args()


def _policy_quality(row: dict[str, Any], policy: str) -> float:
    return float(row["rollout_return_by_policy"][policy]["quality"]["quality_cost_runtime1"])


def _pool_consensus(pool: dict[str, Any], policies: list[str]) -> dict[str, Any]:
    rows = pool["candidate_rows"]
    n = len(rows)
    pair_total = 0
    unanimous = 0
    majority = 0
    split = 0
    policy_disagreement_edges = 0
    wins = [0 for _ in rows]
    unanimous_wins = [0 for _ in rows]
    mean_quality = []
    mean_normalized_quality = []

    per_policy_values = {
        policy: [_policy_quality(row, policy) for row in rows]
        for policy in policies
    }
    for idx in range(n):
        normalized_parts = []
        for policy in policies:
            values = per_policy_values[policy]
            lo = min(values)
            hi = max(values)
            denom = max(hi - lo, 1e-9)
            normalized_parts.append((values[idx] - lo) / denom)
        mean_quality.append(sum(per_policy_values[p][idx] for p in policies) / max(len(policies), 1))
        mean_normalized_quality.append(sum(normalized_parts) / max(len(normalized_parts), 1))

    for i in range(n):
        for j in range(i + 1, n):
            pair_total += 1
            i_votes = 0
            j_votes = 0
            ties = 0
            for policy in policies:
                qi = per_policy_values[policy][i]
                qj = per_policy_values[policy][j]
                if abs(qi - qj) <= 1e-9:
                    ties += 1
                elif qi < qj:
                    i_votes += 1
                else:
                    j_votes += 1
            if i_votes == len(policies) or j_votes == len(policies):
                unanimous += 1
                winner = i if i_votes > j_votes else j
                wins[winner] += 1
                unanimous_wins[winner] += 1
            elif max(i_votes, j_votes) > len(policies) / 2:
                majority += 1
                winner = i if i_votes > j_votes else j
                wins[winner] += 1
                policy_disagreement_edges += 1
            else:
                split += 1
                policy_disagreement_edges += 1

    consensus_order = sorted(range(n), key=lambda idx: (-wins[idx], mean_normalized_quality[idx]))
    mean_order = sorted(range(n), key=lambda idx: mean_normalized_quality[idx])
    unanimous_order = sorted(range(n), key=lambda idx: (-unanimous_wins[idx], mean_normalized_quality[idx]))
    policy_oracle_indexes = pool["rollout_oracle_indexes"]
    policy_oracle_set = set(int(idx) for idx in policy_oracle_indexes.values())
    consensus_top = int(consensus_order[0]) if consensus_order else -1
    mean_top = int(mean_order[0]) if mean_order else -1
    return {
        "step": int(pool["step"]),
        "candidate_count": n,
        "pair_total": pair_total,
        "unanimous_pair_fraction": unanimous / max(pair_total, 1),
        "majority_pair_fraction": majority / max(pair_total, 1),
        "split_pair_fraction": split / max(pair_total, 1),
        "policy_disagreement_edge_fraction": policy_disagreement_edges / max(pair_total, 1),
        "consensus_top_index": consensus_top,
        "consensus_top_action_key": rows[consensus_top]["action_key"] if consensus_top >= 0 else None,
        "mean_top_index": mean_top,
        "mean_top_action_key": rows[mean_top]["action_key"] if mean_top >= 0 else None,
        "unanimous_top_index": int(unanimous_order[0]) if unanimous_order else -1,
        "consensus_top_is_any_policy_oracle": consensus_top in policy_oracle_set,
        "mean_top_is_any_policy_oracle": mean_top in policy_oracle_set,
        "rollout_unique_oracle_count": int(pool["rollout_unique_oracle_count"]),
        "all_rollout_policies_agree": bool(pool["all_rollout_policies_agree"]),
        "policy_oracle_indexes": {k: int(v) for k, v in policy_oracle_indexes.items()},
    }


def _aggregate(payload: dict[str, Any]) -> dict[str, Any]:
    policies = list(payload["continuation_policies"])
    pool_rows = []
    per_collection = []
    for result in payload["results"]:
        pools = [row for row in result["rows"] if row.get("evaluated_candidate_count", 0) > 0]
        pool_metrics = [_pool_consensus(pool, policies) for pool in pools]
        pool_rows.extend(
            {
                "case_id": result["case_id"],
                "case_index": int(result["case_index"]),
                "policy_seed": int(result["policy_seed"]),
                **metric,
            }
            for metric in pool_metrics
        )
        per_collection.append(
            {
                "case_id": result["case_id"],
                "case_index": int(result["case_index"]),
                "policy_seed": int(result["policy_seed"]),
                "pool_count": len(pool_metrics),
                "unanimous_pair_fraction": _mean([m["unanimous_pair_fraction"] for m in pool_metrics]),
                "majority_pair_fraction": _mean([m["majority_pair_fraction"] for m in pool_metrics]),
                "consensus_top_is_any_policy_oracle_fraction": _mean(
                    [float(m["consensus_top_is_any_policy_oracle"]) for m in pool_metrics]
                ),
            }
        )
    per_case = []
    case_ids = sorted({int(row["case_index"]) for row in pool_rows})
    for case_index in case_ids:
        rows = [row for row in pool_rows if int(row["case_index"]) == case_index]
        per_case.append(
            {
                "case_index": case_index,
                "case_id": rows[0]["case_id"],
                "pool_count": len(rows),
                "unanimous_pair_fraction": _mean([r["unanimous_pair_fraction"] for r in rows]),
                "majority_pair_fraction": _mean([r["majority_pair_fraction"] for r in rows]),
                "split_pair_fraction": _mean([r["split_pair_fraction"] for r in rows]),
                "consensus_top_is_any_policy_oracle_fraction": _mean(
                    [float(r["consensus_top_is_any_policy_oracle"]) for r in rows]
                ),
            }
        )
    return {
        "status": "complete",
        "purpose": "Step6E cross-continuation consensus advantage diagnostic",
        "source": str(payload.get("purpose", "")),
        "input_case_ids": payload["case_ids"],
        "input_policy_seeds": payload["policy_seeds"],
        "continuation_policies": policies,
        "pool_count": len(pool_rows),
        "unanimous_pair_fraction": _mean([r["unanimous_pair_fraction"] for r in pool_rows]),
        "majority_pair_fraction": _mean([r["majority_pair_fraction"] for r in pool_rows]),
        "split_pair_fraction": _mean([r["split_pair_fraction"] for r in pool_rows]),
        "policy_disagreement_edge_fraction": _mean(
            [r["policy_disagreement_edge_fraction"] for r in pool_rows]
        ),
        "consensus_top_is_any_policy_oracle_fraction": _mean(
            [float(r["consensus_top_is_any_policy_oracle"]) for r in pool_rows]
        ),
        "mean_top_is_any_policy_oracle_fraction": _mean(
            [float(r["mean_top_is_any_policy_oracle"]) for r in pool_rows]
        ),
        "consensus_advantage_gate": {
            "unanimous_pair_fraction_ge_0_50": _mean(
                [r["unanimous_pair_fraction"] for r in pool_rows]
            )
            >= 0.50,
            "consensus_top_hits_any_policy_oracle_ge_0_75": _mean(
                [float(r["consensus_top_is_any_policy_oracle"]) for r in pool_rows]
            )
            >= 0.75,
        },
        "per_case": per_case,
        "per_collection": per_collection,
        "pools": pool_rows,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    gate = payload["consensus_advantage_gate"]
    gate["consensus_advantage_usable"] = all(bool(v) for v in gate.values())
    rows = [
        "| {case} | {pools} | {unanimous:.4f} | {majority:.4f} | {split:.4f} | {top:.4f} |".format(
            case=row["case_id"],
            pools=row["pool_count"],
            unanimous=row["unanimous_pair_fraction"],
            majority=row["majority_pair_fraction"],
            split=row["split_pair_fraction"],
            top=row["consensus_top_is_any_policy_oracle_fraction"],
        )
        for row in payload["per_case"]
    ]
    lines = [
        "# Step6E Consensus Advantage Diagnostic",
        "",
        "- purpose: `test whether noisy multi-continuation rollout labels still contain pairwise consensus signal`",
        f"- continuation policies: `{payload['continuation_policies']}`",
        f"- pool count: `{payload['pool_count']}`",
        f"- unanimous pair fraction: `{payload['unanimous_pair_fraction']:.4f}`",
        f"- majority pair fraction: `{payload['majority_pair_fraction']:.4f}`",
        f"- split pair fraction: `{payload['split_pair_fraction']:.4f}`",
        f"- consensus top hits any policy oracle fraction: `{payload['consensus_top_is_any_policy_oracle_fraction']:.4f}`",
        f"- consensus advantage usable: `{gate['consensus_advantage_usable']}`",
        "",
        "## Per Case",
        "",
        "| Case | pools | unanimous pair | majority pair | split pair | consensus-top hits any oracle |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "Interpretation: high pairwise consensus despite unstable top1 would support training a robust pairwise advantage objective. Low consensus means the next step should shift to representation/rollout-policy design rather than ranker tuning.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    input_payload = json.loads(args.input.read_text(encoding="utf-8"))
    payload = _aggregate(input_payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(args.output.with_suffix(".md"), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "pool_count": payload["pool_count"],
                "unanimous_pair_fraction": payload["unanimous_pair_fraction"],
                "majority_pair_fraction": payload["majority_pair_fraction"],
                "split_pair_fraction": payload["split_pair_fraction"],
                "consensus_top_is_any_policy_oracle_fraction": payload[
                    "consensus_top_is_any_policy_oracle_fraction"
                ],
                "consensus_advantage_usable": payload["consensus_advantage_gate"][
                    "consensus_advantage_usable"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
