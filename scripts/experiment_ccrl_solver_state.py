#!/usr/bin/env python3
"""P11.5b decisive experiment: CCRL ranking on real solver states.

The synthetic battle proved ordering value on structured-corruption states.
Here the ONLY variable changes: states come from the deterministic solver's
own placements (safe_shelf incumbents over training-root FloorSet sources,
same qualifying rule as the frozen P8.2 bucket). Model, masks, decoder,
action set, decode-cache semantics, and budgets are identical.

Parallel layout (same pattern as experiment_bfod_heldout.py): the parent
streams FloorSet sources; a worker pool runs safe_shelf, qualification, the
full decode scan, and the model ranking per state. Workers pin torch to one
thread; pool.map preserves examination order, so results are identical to
the serial run apart from wall-clock fields.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.data import file_sha256  # noqa: E402
from hcfp.fallback import safe_shelf  # noqa: E402
from hcfp.floorset_lite import iter_floorset_lite_with_source  # noqa: E402
from hcfp.geometry import normalize_xywh  # noqa: E402
from hcfp.repair.decoders.base import DecodeFailure  # noqa: E402
from hcfp.repair.decoders.contact import decode_contact_action  # noqa: E402
from hcfp.repair.model import (  # noqa: E402
    CONTACT_RELATIONS,
    PATCH_BUDGETS,
    ContactRepairModel,
    RepairModelConfig,
    topk_contact_actions,
)
from hcfp.repair.schema import (  # noqa: E402
    ExpertKind,
    RepairAction,
    RepairObligation,
)
from hcfp.repair.state import build_repair_state  # noqa: E402
from hcfp.verify import grouping_violation, verify_feasible  # noqa: E402


BUDGETS = (1, 2, 4, 8, 16, 32)
FORBIDDEN_ROOT_TOKENS = ("litetensordatatest", "validation", "visible", "test")

_MODEL: "ContactRepairModel | None" = None


def _init_worker(checkpoint: str) -> None:
    global _MODEL
    torch.set_num_threads(1)
    _MODEL = _load_model(Path(checkpoint))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--floorset-lite-root", default="artifacts/floorset-v10")
    parser.add_argument("--states", type=int, default=180)
    parser.add_argument("--seed", type=int, default=5090)
    parser.add_argument(
        "--jobs",
        type=_positive,
        default=max(1, min((os.cpu_count() or 2) - 2, 16)),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    layout_root = Path(args.floorset_lite_root).resolve()
    forbidden = [t for t in FORBIDDEN_ROOT_TOKENS if t in str(layout_root).lower()]
    if forbidden:
        raise ValueError(f"visible validation/test tokens in root: {forbidden}")

    started = time.perf_counter()
    rows: list[dict] = []
    examined = 0
    qualifying = 0

    from multiprocessing import Pool

    with Pool(
        args.jobs,
        initializer=_init_worker,
        initargs=(str(Path(args.checkpoint).resolve()),),
    ) as pool:
        batch: list[tuple[object, dict]] = []
        done = False
        for sample, source in iter_floorset_lite_with_source(
            layout_root, limit=None, seed=args.seed, max_layouts_per_file=2
        ):
            examined += 1
            batch.append((sample, source))
            if len(batch) >= args.jobs * 4:
                qualifying = _drain(pool, batch, rows, qualifying, args.states)
                batch = []
                if qualifying >= args.states:
                    done = True
                    break
        if batch and not done:
            _drain(pool, batch, rows, qualifying, args.states)

    if not rows:
        raise RuntimeError("no qualifying solver states found")

    report = {
        "schema_version": 1,
        "purpose": "P11.5b CCRL vs deterministic ordering on real solver states",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "floorset_lite_root": str(layout_root),
        "states": len(rows),
        "examined": examined,
        "seed": args.seed,
        "jobs": args.jobs,
        "state_source": "deterministic safe_shelf incumbent on training root",
        "elapsed_seconds": time.perf_counter() - started,
        "overall": _aggregate(rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"states": len(rows), "overall": report["overall"]}, indent=2))
    return 0


def _drain(pool, batch, rows: list[dict], qualifying: int, limit: int) -> int:
    for row in pool.map(_process_one, batch):
        if row is None:
            continue
        rows.append(row)
        qualifying += 1
        if qualifying >= limit:
            break
    return qualifying


def _process_one(args: tuple[object, dict]) -> dict | None:
    sample, source = args
    case = sample.case
    try:
        positions = safe_shelf(source)
    except (RuntimeError, ValueError):
        return None
    if not verify_feasible(source, positions):
        return None
    debt_global = int(grouping_violation(case, positions))
    if debt_global <= 0:
        return None
    groups = torch.as_tensor(case.group_membership, dtype=torch.bool)
    if not groups.any():
        return None
    # worst grouping-debt group as the repair obligation;
    # debt metric is the global grouping violation, matching the decoder
    best_group, best_group_debt = None, -1
    for gi in range(groups.shape[0]):
        members = torch.nonzero(groups[gi]).reshape(-1).tolist()
        if len(members) < 2:
            continue
        comp = _group_debt(case, positions, set(members))
        if comp > best_group_debt:
            best_group, best_group_debt = gi, comp
    if best_group is None or best_group_debt <= 0:
        return None
    return _battle_one(case, source, positions, best_group, debt_global)


def _battle_one(case, source, positions, group_index, debt_global) -> dict:
    members = (
        torch.nonzero(
            torch.as_tensor(case.group_membership, dtype=torch.bool)[group_index]
        )
        .reshape(-1)
        .tolist()
    )
    obligation = RepairObligation(
        ExpertKind.CONTACT,
        f"contact-group:{group_index}",
        tuple(members),
        debt=debt_global,
    )
    state = build_repair_state(
        case,
        normalize_xywh(case, positions),
        exact_contact_placement=positions,
    )
    state_debt = int(debt_global)

    movable = (~case.preplaced_mask).tolist()
    triples = [
        (target, anchor, side)
        for target in members
        if movable[target]
        for anchor in members
        if anchor != target
        for side in range(len(CONTACT_RELATIONS))
    ]

    decode_cache: dict[tuple[int, int, int, int], tuple[bool, int | None]] = {}
    decode_counter = [0]

    def decode_eff(target, anchor, side, budget):
        key = (target, anchor, side, budget)
        if key not in decode_cache:
            decode_counter[0] += 1
            result = decode_contact_action(
                case,
                positions,
                RepairAction(
                    ExpertKind.CONTACT,
                    obligation.obligation_id,
                    (target,),
                    (anchor,),
                    CONTACT_RELATIONS[side],
                    patch_budget=budget,
                ),
                verify_case=source,
            )
            if result.failure == DecodeFailure.PATCH_BUDGET:
                decode_counter[0] += 1
                escalated = decode_contact_action(
                    case,
                    positions,
                    RepairAction(
                        ExpertKind.CONTACT,
                        obligation.obligation_id,
                        (target,),
                        (anchor,),
                        CONTACT_RELATIONS[side],
                        patch_budget=max(PATCH_BUDGETS),
                    ),
                    verify_case=source,
                )
                decode_cache[key] = (escalated.succeeded, escalated.debt_after)
            else:
                decode_cache[key] = (result.succeeded, result.debt_after)
        return decode_cache[key]

    outcomes = [decode_eff(t, a, s, PATCH_BUDGETS[0]) for (t, a, s) in triples]
    oracle_debt = min(
        (debt for ok, debt in outcomes if ok), default=state_debt
    )
    oracle_gain = state_debt - oracle_debt

    det_seq = list(outcomes)
    det_first_success = next(
        (i + 1 for i, (ok, _) in enumerate(det_seq) if ok), None
    )

    forward_started = time.perf_counter()
    actions = topk_contact_actions(
        _MODEL(state, obligation), obligation, k=max(BUDGETS)
    )
    forward_seconds = time.perf_counter() - forward_started
    ccrl_seq = []
    for action in actions:
        ccrl_seq.append(
            decode_eff(
                action.target_ids[0],
                action.anchor_ids[0],
                CONTACT_RELATIONS.index(action.relation),
                action.patch_budget,
            )
        )

    row = {
        "source_id": _state_id(case, positions),
        "block_count": int(case.n),
        "group_index": group_index,
        "state_debt": state_debt,
        "oracle_debt": oracle_debt,
        "oracle_gain": oracle_gain,
        "triple_count": len(triples),
        "decode_count_total": decode_counter[0],
        "deterministic_first_success": det_first_success,
        "ccrl_forward_seconds": forward_seconds,
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


def _state_id(case, positions) -> str:
    import hashlib

    payload = json.dumps(
        torch.as_tensor(positions, dtype=torch.float64).tolist(),
        separators=(",", ":"),
    )
    return f"n{case.n}:{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def _group_debt(case, positions, members: set[int]) -> int:
    boxes = torch.as_tensor(positions, dtype=torch.float64)
    row = torch.zeros(boxes.shape[0], dtype=torch.bool)
    row[sorted(members)] = True
    single = type(case)(
        **{
            **_case_fields(case),
            "group_membership": row.unsqueeze(0),
        }
    )
    return int(grouping_violation(single, boxes))


def _case_fields(case) -> dict:
    import dataclasses

    return {f.name: getattr(case, f.name) for f in dataclasses.fields(case)}


def _recovered(seq, state_debt, oracle_gain) -> float | None:
    if oracle_gain <= 0:
        return None
    best = min((d for ok, d in seq if ok), default=state_debt)
    return (state_debt - best) / oracle_gain


def _load_model(path: Path):
    payload = torch.load(path, map_location="cpu")
    config = RepairModelConfig(**payload["config"])
    model = ContactRepairModel(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _aggregate(rows) -> dict:
    result = {
        "states": len(rows),
        "mean_triples": sum(r["triple_count"] for r in rows) / len(rows),
        "states_with_gain": sum(1 for r in rows if r["oracle_gain"] > 0),
        "mean_oracle_gain": sum(r["oracle_gain"] for r in rows) / len(rows),
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


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
