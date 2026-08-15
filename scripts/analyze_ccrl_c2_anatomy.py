#!/usr/bin/env python3
"""P11.4-D2: attribute Contact Gate D failures across native and derived factors.

D2-A: generated replay rows. For every held-out row, decode the model Top-4
and the teacher, then attribute each miss to native factors (target / anchor /
side / patch_budget) plus derived factors (same-component anchor pair,
patch-budget-only mismatch under identical decoded geometry / identical debt
outcome).

No model change, no new heads, no cache rewrite.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.data import file_sha256  # noqa: E402
from hcfp.floorset_lite import iter_floorset_lite_with_source  # noqa: E402
from hcfp.repair.actions import (  # noqa: E402
    action_from_payload,
    action_sha256,
    action_to_payload,
)
from hcfp.repair.decoders.base import DecodeFailure  # noqa: E402
from hcfp.repair.decoders.contact import decode_contact_action  # noqa: E402
from hcfp.repair.model import (  # noqa: E402
    ContactRepairModel,
    RepairModelConfig,
    topk_contact_actions,
)
from hcfp.repair.replay import repair_replay_loads  # noqa: E402
from hcfp.repair.state import build_repair_state  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--floorset-lite-root", default="artifacts/floorset-v10")
    parser.add_argument("--kinds", default="C2")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    kinds = tuple(kind.strip().upper() for kind in args.kinds.split(",") if kind.strip())
    started = time.perf_counter()
    records = _load_replay(Path(args.replay), kinds)
    manifest = json.loads(Path(args.source_manifest).read_text(encoding="utf-8"))
    verifiers = _load_verifiers(
        args.floorset_lite_root,
        {record.source_id for record in records},
        seed=int(manifest["config"]["seed"]),
        max_layouts_per_file=int(manifest["config"]["max_layouts_per_file"]),
    )
    model = _load_model(Path(args.checkpoint))

    overall = Counter()
    by_kind: dict[str, Counter] = {kind: Counter() for kind in kinds}
    examples: dict[str, list[dict]] = {}
    with torch.inference_mode():
        for record in records:
            kind = (record.state.corruption_kind or "").upper()
            bucket = by_kind[kind]
            teacher = record.action
            teacher_sha = action_sha256(teacher)
            actions = topk_contact_actions(
                model(record.state, record.obligation),
                record.obligation,
                k=args.topk,
            )
            # --- exact inverse recall (same metric as Gate D) ---
            exact_topk = any(action_sha256(a) == teacher_sha for a in actions)
            # --- budget-insensitive canonical recall ---
            canonical_topk = exact_topk or any(
                _budget_canonical_sha(a) == _budget_canonical_sha(teacher)
                for a in actions
            )
            # --- functional recall: any top-k action decodes to debt reduction ---
            decoded = [
                decode_contact_action(
                    record.state.case,
                    record.decoder_placement,
                    action,
                    verify_case=verifiers[record.source_id],
                )
                for action in actions
            ]
            component = record.state.group_component_id
            t_comp = int(component[teacher.target_ids[0]])
            a_comp = int(component[teacher.anchor_ids[0]])
            top1 = actions[0] if actions else None
            top1_component = (
                (
                    int(component[top1.target_ids[0]]),
                    int(component[top1.anchor_ids[0]]),
                )
                if top1 is not None
                else None
            )
            functional = [d for d in decoded if d.succeeded]
            functional_topk = bool(functional)
            hard_feasible_topk = functional_topk or any(
                d.failure == DecodeFailure.NO_DEBT_REDUCTION for d in decoded
            )
            best = min(
                functional,
                key=lambda d: (d.debt_after, action_sha256(d.action)),
                default=None,
            )
            teacher_decoded = decode_contact_action(
                record.state.case,
                record.decoder_placement,
                teacher,
                verify_case=verifiers[record.source_id],
            )
            teacher_debt = int(record.outcome.debt_after)
            same_geometry = bool(
                teacher_decoded.succeeded
                and teacher_decoded.placement is not None
                and any(
                    d.succeeded
                    and d.placement is not None
                    and bool(torch.equal(d.placement, teacher_decoded.placement))
                    for d in decoded
                )
            )
            same_outcome = any(
                d.succeeded
                and d.debt_after is not None
                and d.debt_after <= teacher_debt
                for d in decoded
            )
            best_debt = best.debt_after if best is not None else None
            beats_teacher = best_debt is not None and best_debt < teacher_debt
            equals_teacher = best_debt is not None and best_debt == teacher_debt
            bucket["count"] += 1
            bucket["teacher_anchor_cross_component"] += int(t_comp != a_comp)
            if not exact_topk and top1_component is not None:
                bucket["top1_anchor_same_component_as_target"] += int(
                    top1_component[0] == top1_component[1]
                )
                bucket["teacher_anchor_same_component_as_target"] += int(
                    t_comp == a_comp
                )
                factors = _miss_factors(teacher, top1)
                for name, hit in factors.items():
                    if hit:
                        bucket[f"miss:{name}"] += 1
                if canonical_topk:
                    bucket["miss_resolved_by_canonical"] += 1
                if not functional_topk:
                    bucket["miss_and_functional_fail"] += 1
                _example(
                    examples, kind, record, teacher, top1, factors,
                    canonical_topk, functional_topk,
                )
            for d in decoded:
                if d.failure is not None:
                    bucket[f"decode:{d.failure.value}"] += 1
            for name, value in (
                ("exact_topk", exact_topk),
                ("canonical_topk", canonical_topk),
                ("functional_topk", functional_topk),
                ("hard_feasible_topk", hard_feasible_topk),
                ("same_geometry_best", same_geometry),
                ("same_outcome_best", same_outcome),
                ("beats_teacher", beats_teacher),
                ("equals_teacher", equals_teacher),
            ):
                bucket[name] += int(value)
                overall[name] += int(value)
            overall["count"] += 1

    report = {
        "schema_version": 1,
        "purpose": "P11.4-D2-A Contact miss anatomy on generated replay rows",
        "replay": str(Path(args.replay).resolve()),
        "replay_sha256": file_sha256(args.replay),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "kinds": list(kinds),
        "topk": args.topk,
        "canonicalization": (
            "budget canonicalization compares action identity with patch_budget "
            "mapped to 16 (largest), keeping target/anchor/relation"
        ),
        "overall": _rates(overall),
        "by_kind": {kind: _rates(bucket) for kind, bucket in by_kind.items()},
        "miss_examples": {kind: rows[:5] for kind, rows in examples.items()},
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ("overall", "by_kind")}, indent=2))
    return 0


def _budget_canonical_sha(action) -> str:
    payload = action_to_payload(action, identity_only=True)
    payload["patch_budget"] = max(payload["patch_budget"], 16)
    return action_sha256(action_from_payload(payload))


def _miss_factors(teacher, top1) -> dict[str, bool]:
    if top1 is None:
        return {"no_prediction": True}
    same_target = top1.target_ids == teacher.target_ids
    same_anchor = top1.anchor_ids == teacher.anchor_ids
    same_side = top1.relation == teacher.relation
    role_swap = (
        top1.target_ids == teacher.anchor_ids
        and top1.anchor_ids == teacher.target_ids
    )
    return {
        "role_swap": role_swap,
        "target": not same_target and not role_swap,
        "anchor": same_target and not same_anchor,
        "side": same_target and same_anchor and not same_side,
        "patch_budget": same_target and same_anchor and same_side,
    }
def _example(examples, kind, record, teacher, top1, factors, canonical, functional):
    rows = examples.setdefault(kind, [])
    if len(rows) >= 5:
        return
    rows.append(
        {
            "source_id": record.source_id,
            "teacher": _action_summary(teacher),
            "top1": None if top1 is None else _action_summary(top1),
            "factors": {k: v for k, v in factors.items() if v},
            "resolved_by_canonical": canonical,
            "functional_topk": functional,
        }
    )


def _action_summary(action):
    return {
        "target": list(action.target_ids),
        "anchor": list(action.anchor_ids),
        "relation": action.relation,
        "patch_budget": action.patch_budget,
    }


def _rates(bucket: Counter) -> dict:
    count = bucket["count"]
    result: dict = {"count": count}
    for key, value in sorted(bucket.items()):
        if key == "count":
            continue
        if key.startswith(("miss:", "decode:")) or key in {
            "miss_exact",
            "miss_resolved_by_canonical",
            "miss_and_functional_fail",
        }:
            result[key] = value
        else:
            result[f"{key}_rate"] = value / count if count else 0.0
    return result


def _load_replay(path: Path, kinds: tuple[str, ...]):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        records = [
            _with_exact_contact_state(repair_replay_loads(line)) for line in stream
        ]
    selected = [
        record
        for record in records
        if (record.state.corruption_kind or "").upper() in kinds
    ]
    if not selected:
        raise ValueError(f"no {kinds} rows in {path}")
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


def _load_verifiers(
    root: str, source_ids: set[str], *, seed: int, max_layouts_per_file: int
):
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


if __name__ == "__main__":
    raise SystemExit(main())
