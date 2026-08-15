#!/usr/bin/env python3
"""P11.5 decisive experiment: CCRL Top-K ranking vs deterministic enumeration.

Frozen: checkpoint, masks, decoder, replay cache. Per held-out state both arms
see the identical mask-derived action set; the ONLY difference is ordering:

  deterministic: canonical index order (member, anchor, LEFT/RIGHT/TOP/BOTTOM,
                 lazy budget escalation 2 -> 16 on PATCH_BUDGET)
  ccrl:          model-scored Top-K (same lazy escalation rule)

Measured per state: recovered fraction of oracle grouping-gain at decode
budgets 1/2/4/8/16/32, success rate, decodes to first success, and model
forward latency. No model or cache change.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import gzip
import json
from pathlib import Path
import random
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.data import file_sha256  # noqa: E402
from hcfp.floorset_lite import iter_floorset_lite_with_source  # noqa: E402
from hcfp.repair.decoders.base import DecodeFailure  # noqa: E402
from hcfp.repair.decoders.contact import decode_contact_action  # noqa: E402
from hcfp.repair.model import (  # noqa: E402
    CONTACT_RELATIONS,
    PATCH_BUDGETS,
    ContactRepairModel,
    RepairModelConfig,
    topk_contact_actions,
)
from hcfp.repair.replay import repair_replay_loads  # noqa: E402
from hcfp.repair.state import build_repair_state  # noqa: E402


BUDGETS = (1, 2, 4, 8, 16, 32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--floorset-lite-root", default="artifacts/floorset-v10")
    parser.add_argument("--per-kind", type=int, default=60)
    parser.add_argument("--seed", type=int, default=5090)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    records = _load_stratified(
        Path(args.replay), args.per_kind, seed=args.seed
    )
    manifest = json.loads(Path(args.source_manifest).read_text(encoding="utf-8"))
    verifiers = _load_verifiers(
        args.floorset_lite_root,
        {r.source_id for r in records},
        seed=int(manifest["config"]["seed"]),
        max_layouts_per_file=int(manifest["config"]["max_layouts_per_file"]),
    )
    model = _load_model(Path(args.checkpoint))

    rows = []
    with torch.inference_mode():
        for record in records:
            rows.append(
                _battle_one(record, model, verifiers[record.source_id])
            )

    report = {
        "schema_version": 1,
        "purpose": "P11.5 CCRL Top-K vs deterministic contact search",
        "replay": str(Path(args.replay).resolve()),
        "replay_sha256": file_sha256(args.replay),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "per_kind": args.per_kind,
        "seed": args.seed,
        "states": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "overall": _aggregate(rows),
        "by_kind": {
            kind: _aggregate([row for row in rows if row["kind"] == kind])
            for kind in sorted({row["kind"] for row in rows})
        },
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        key: report[key] for key in ("states", "overall", "by_kind")
    }
    print(json.dumps(summary, indent=2))
    return 0


def _battle_one(record, model, verifier) -> dict:
    kind = (record.state.corruption_kind or "").upper()
    case = record.state.case
    placement = record.decoder_placement
    state_debt = int(record.outcome.debt_before)
    teacher_debt = int(record.outcome.debt_after)
    obligation = record.obligation

    # --- enumerate the mask-legal (target, anchor, side) triples ---
    members = torch.nonzero(
        case.group_membership[
            int(obligation.obligation_id.rsplit(":", 1)[1])
        ],
        as_tuple=False,
    ).reshape(-1).tolist()
    movable = (~case.preplaced_mask).tolist()
    observed = record.state.geometry_observed.tolist()
    triples = [
        (target, anchor, side)
        for target in members
        if movable[target] and observed[target]
        for anchor in members
        if anchor != target and observed[anchor]
        for side in range(len(CONTACT_RELATIONS))
    ]

    decode_cache: dict[tuple[int, int, int, int], tuple[bool, int | None]] = {}
    decode_counter = [0]

    def decode_eff(target: int, anchor: int, side: int, budget: int):
        key = (target, anchor, side, budget)
        if key not in decode_cache:
            decode_counter[0] += 1
            result = decode_contact_action(
                case,
                placement,
                _action(obligation, target, anchor, side, budget),
                verify_case=verifier,
            )
            if result.failure == DecodeFailure.PATCH_BUDGET:
                # budget is a pure size gate: escalate once to the max budget
                decode_counter[0] += 1
                escalated = decode_contact_action(
                    case,
                    placement,
                    _action(obligation, target, anchor, side, max(PATCH_BUDGETS)),
                    verify_case=verifier,
                )
                decode_cache[key] = (escalated.succeeded, escalated.debt_after)
            else:
                decode_cache[key] = (result.succeeded, result.debt_after)
        return decode_cache[key]

    # --- shared outcome scan: first pass over budget=2 for every triple ---
    outcomes = [decode_eff(t, a, s, PATCH_BUDGETS[0]) for (t, a, s) in triples]

    # --- oracle: best achievable debt over the whole action set ---
    oracle_debt = min(
        (debt for ok, debt in outcomes if ok), default=state_debt
    )
    oracle_gain = state_debt - oracle_debt

    # --- deterministic arm: canonical order, first success semantics ---
    det_seq = []
    det_first_success = None
    for index, (ok, debt) in enumerate(outcomes):
        det_seq.append((ok, debt))
        if ok and det_first_success is None:
            det_first_success = index + 1
    det_best = min((d for ok, d in det_seq if ok), default=state_debt)

    # --- ccrl arm: model-ranked Top-32 with the same lazy escalation ---
    forward_started = time.perf_counter()
    actions = topk_contact_actions(
        model(record.state, obligation), obligation, k=max(BUDGETS)
    )
    forward_seconds = time.perf_counter() - forward_started
    ccrl_seq = []
    for action in actions:
        target = action.target_ids[0]
        anchor = action.anchor_ids[0]
        side = CONTACT_RELATIONS.index(action.relation)
        ccrl_seq.append(decode_eff(target, anchor, side, action.patch_budget))

    row = {
        "kind": kind,
        "source_id": record.source_id,
        "state_debt": state_debt,
        "teacher_debt": teacher_debt,
        "oracle_debt": oracle_debt,
        "oracle_gain": oracle_gain,
        "triple_count": len(triples),
        "decode_count_total": decode_counter[0],
        "deterministic_first_success": det_first_success,
        "deterministic_best_debt": det_best,
        "ccrl_forward_seconds": forward_seconds,
        "ccrl_action_count": len(ccrl_seq),
        "budgets": {},
    }
    for budget in BUDGETS:
        det_slice = det_seq[:budget]
        ccrl_slice = ccrl_seq[:budget]
        row["budgets"][str(budget)] = {
            "deterministic_success": any(ok for ok, _ in det_slice),
            "deterministic_recovered": _recovered(det_slice, state_debt, oracle_gain),
            "ccrl_success": any(ok for ok, _ in ccrl_slice),
            "ccrl_recovered": _recovered(ccrl_slice, state_debt, oracle_gain),
        }
    return row


def _action(obligation, target, anchor, side, budget):
    from hcfp.repair.schema import ExpertKind, RepairAction

    return RepairAction(
        ExpertKind.CONTACT,
        obligation.obligation_id,
        (target,),
        (anchor,),
        CONTACT_RELATIONS[side],
        patch_budget=budget,
    )


def _recovered(seq, state_debt, oracle_gain) -> float | None:
    if oracle_gain <= 0:
        return None
    best = min((d for ok, d in seq if ok), default=state_debt)
    return (state_debt - best) / oracle_gain


def _load_stratified(path: Path, per_kind: int, *, seed: int):
    by_kind = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = _with_exact_contact_state(repair_replay_loads(line))
            by_kind[(record.state.corruption_kind or "").upper()].append(record)
    rng = random.Random(seed)
    selected = []
    for kind in sorted(by_kind):
        pool = by_kind[kind]
        rng.shuffle(pool)
        selected.extend(pool[:per_kind])
    if not selected:
        raise ValueError("no replay rows selected")
    return selected


def _with_exact_contact_state(record):
    state = record.state
    return replace(
        record,
        state=build_repair_state(
            state.case,
            state.placement,
            geometry_observed=state.geometry_observed,
            repair_target=state.repair_target,
            exact_contact_placement=record.decoder_placement,
            round_index=state.round_index,
            corruption_kind=state.corruption_kind,
            corruption_level=state.corruption_level,
        ),
    )


def _load_verifiers(root, source_ids, *, seed, max_layouts_per_file):
    found = {}
    for sample, source in iter_floorset_lite_with_source(
        root, limit=None, seed=seed, max_layouts_per_file=max_layouts_per_file
    ):
        if sample.sample_id in source_ids:
            found[sample.sample_id] = source
            if len(found) == len(source_ids):
                break
    missing = sorted(source_ids - found.keys())
    if missing:
        raise RuntimeError(f"missing official verifier sources: {missing[:8]}")
    return found


def _load_model(path: Path):
    payload = torch.load(path, map_location="cpu")
    config = RepairModelConfig(**payload["config"])
    model = ContactRepairModel(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _aggregate(rows) -> dict:
    if not rows:
        return {"states": 0}
    result = {
        "states": len(rows),
        "mean_triples": sum(r["triple_count"] for r in rows) / len(rows),
        "mean_oracle_gain": sum(r["oracle_gain"] for r in rows) / len(rows),
        "oracle_states_full": sum(
            1 for r in rows if r["oracle_gain"] > 0
        ),
        "mean_decode_count_total": sum(
            r["decode_count_total"] for r in rows
        ) / len(rows),
        "deterministic_first_success_rate": sum(
            1 for r in rows if r["deterministic_first_success"] is not None
        ) / len(rows),
        "deterministic_mean_decodes_to_success": _mean(
            [
                r["deterministic_first_success"]
                for r in rows
                if r["deterministic_first_success"] is not None
            ]
        ),
        "deterministic_median_decodes_to_success": _median(
            [
                r["deterministic_first_success"]
                for r in rows
                if r["deterministic_first_success"] is not None
            ]
        ),
        "ccrl_mean_forward_ms": 1000.0
        * sum(r["ccrl_forward_seconds"] for r in rows) / len(rows),
        "budgets": {},
    }
    for budget in BUDGETS:
        key = str(budget)
        det = [r["budgets"][key] for r in rows]
        result["budgets"][key] = {
            "deterministic_success_rate": sum(
                1 for b in det if b["deterministic_success"]
            ) / len(rows),
            "deterministic_recovered_mean": _mean(
                [b["deterministic_recovered"] for b in det]
            ),
            "ccrl_success_rate": sum(1 for b in det if b["ccrl_success"])
            / len(rows),
            "ccrl_recovered_mean": _mean([b["ccrl_recovered"] for b in det]),
        }
    return result


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


if __name__ == "__main__":
    raise SystemExit(main())
