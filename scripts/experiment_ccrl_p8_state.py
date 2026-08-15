#!/usr/bin/env python3
"""P11.5c: CCRL ranking battle on real P8 guarded incumbent states.

Experiment B of the scaling plan: does the ordering premium that CCRL showed
on corruption states survive on the production lane's own placements?

States: P8 guarded-full100 analytic-lane incumbents (validation cases), the
placements the contact loop would actually start from. Model, masks,
decoder, action set, decode-cache semantics, and budgets are identical to
the P11.5/P11.5b battles. The obligation is the worst grouping-debt group;
debt is the global grouping violation, matching the decoder.

Parallel layout as experiment_ccrl_solver_state.py. NOTE on scope: these are
official validation cases, used read-only as solver inputs exactly like the
BFOD/P10 sidecars; no validation labels or solutions are consumed, and
nothing trains here.
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
for entry in (str(ROOT / "src"), str(ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiment_p10_case70 import _official_case  # noqa: E402
from hcfp.data import file_sha256  # noqa: E402
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

_MODEL: "ContactRepairModel | None" = None
_DATA_PATH: "Path | None" = None


def _init_worker(checkpoint: str, data_path: str) -> None:
    global _MODEL, _DATA_PATH
    torch.set_num_threads(1)
    _MODEL = _load_model(Path(checkpoint))
    _DATA_PATH = Path(data_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--incumbents",
        default="artifacts/benchmarks/hcfp5090-p8-guarded-full100.json",
    )
    parser.add_argument("--data-path", default="artifacts/floorset-v10")
    parser.add_argument("--cases", default=None, help="comma-separated test ids")
    parser.add_argument(
        "--jobs",
        type=_positive,
        default=max(1, min((os.cpu_count() or 2) - 2, 16)),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.incumbents).read_text(encoding="utf-8"))
    rows = payload["lanes"]["analytic"]
    selected = [int(row["test_id"]) for row in rows]
    if args.cases:
        wanted = {int(item) for item in args.cases.split(",") if item.strip()}
        selected = [tid for tid in selected if tid in wanted]

    started = time.perf_counter()
    from multiprocessing import Pool

    work = [(tid, rows_by_id(rows)[tid]) for tid in selected]
    with Pool(
        args.jobs,
        initializer=_init_worker,
        initargs=(str(Path(args.checkpoint).resolve()), str(Path(args.data_path).resolve())),
    ) as pool:
        results = pool.map(_process_one, work)

    battle_rows = [row for row in results if row is not None]
    skipped = len(selected) - len(battle_rows)
    if not battle_rows:
        raise RuntimeError("no qualifying P8 incumbent states")

    report = {
        "schema_version": 1,
        "purpose": "P11.5c CCRL vs deterministic ordering on P8 incumbent states",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "incumbents": str(Path(args.incumbents).resolve()),
        "incumbents_sha256": file_sha256(args.incumbents),
        "state_source": "P8 guarded-full100 analytic lane incumbents",
        "cases_selected": len(selected),
        "cases_skipped": skipped,
        "jobs": args.jobs,
        "elapsed_seconds": time.perf_counter() - started,
        "overall": _aggregate(battle_rows),
        "rows": battle_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cases": len(battle_rows), "overall": report["overall"]}, indent=2))
    return 0


def rows_by_id(rows):
    return {int(row["test_id"]): row for row in rows}


def _process_one(args: tuple[int, dict]) -> dict | None:
    case_id, incumbent = args
    (
        _module,
        case,
        raw_case,
        _metric_args,
        _visual_case,
        _edges,
        _baseline,
    ) = _official_case(_DATA_PATH, case_id)
    positions = torch.as_tensor(
        incumbent["positions"], dtype=torch.float64, device="cpu"
    )
    if not verify_feasible(raw_case, positions):
        return None
    debt_global = int(grouping_violation(case, positions))
    if debt_global <= 0:
        return None
    groups = torch.as_tensor(case.group_membership, dtype=torch.bool)
    if not groups.any():
        return None
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
    return _battle_one(
        case, raw_case, positions, best_group, debt_global, case_id
    )


def _official_case_cached(case_id: int):
    global _CASES
    cached = _CASES.get(("official", case_id))
    if cached is None:
        raise RuntimeError("official case cache not initialized")
    return cached


def _battle_one(case, raw_case, positions, group_index, debt_global, case_id) -> dict:
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
                verify_case=raw_case,
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
                    verify_case=raw_case,
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
        "case_id": case_id,
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
        "cases": len(rows),
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
