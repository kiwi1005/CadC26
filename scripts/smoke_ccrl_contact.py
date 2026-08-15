#!/usr/bin/env python3
"""Overfit a bounded C0-C2 Contact replay set before scale training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.floorset_lite import iter_floorset_lite_with_source  # noqa: E402
from hcfp.geometry import normalize_xywh  # noqa: E402
from hcfp.repair.actions import action_sha256  # noqa: E402
from hcfp.repair.corruption.contact import generate_contact_corruptions  # noqa: E402
from hcfp.repair.dataset import audit_clean_sample, source_split  # noqa: E402
from hcfp.repair.decoders.contact import decode_contact_action  # noqa: E402
from hcfp.repair.losses import contact_action_loss  # noqa: E402
from hcfp.repair.model import (  # noqa: E402
    ContactRepairModel,
    RepairModelConfig,
    topk_contact_actions,
)
from hcfp.repair.schema import ExpertKind, RepairObligation  # noqa: E402
from hcfp.repair.state import build_repair_state  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floorset-lite-root", default="artifacts/floorset-v10")
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12800)
    parser.add_argument("--scan-limit", type=int, default=600)
    parser.add_argument("--seed", type=int, default=5090)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.states <= 0 or args.steps <= 0 or args.scan_limit <= 0:
        parser.error("--states, --steps, and --scan-limit must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    records = _records(args.floorset_lite_root, args.states, args.scan_limit, args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = ContactRepairModel(
        RepairModelConfig(args.hidden_dim, args.layers, args.heads)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=0.0)

    started = time.perf_counter()
    model.train()
    for step in range(args.steps):
        record = records[step % len(records)]
        optimizer.zero_grad(set_to_none=True)
        report = contact_action_loss(
            model(record[3], record[4]), record[2].inverse_action
        )
        report.total.backward()
        optimizer.step()

    model.eval()
    report = _evaluate(model, records)
    report.update(
        config={
            "states": args.states,
            "steps": args.steps,
            "scan_limit": args.scan_limit,
            "seed": args.seed,
            "device": str(device),
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "heads": args.heads,
        },
        elapsed_seconds=time.perf_counter() - started,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _records(root: str, count: int, scan_limit: int, seed: int):
    wanted = _kind_counts(count)
    records = []
    selected = {kind: 0 for kind in wanted}
    for sample, source in iter_floorset_lite_with_source(
        root, limit=scan_limit, seed=seed, max_layouts_per_file=2
    ):
        if source_split(sample.sample_id)[0] != "train":
            continue
        if not audit_clean_sample(sample, source)["eligibility"]["contact_clean"]:
            continue
        clean = torch.as_tensor(source["fp_sol_xywh"], dtype=torch.float64)
        for corruption in generate_contact_corruptions(
            sample.case, clean, verify_case=source, kinds=("C0", "C1", "C2")
        ):
            if selected[corruption.kind] >= wanted[corruption.kind]:
                continue
            group_index = int(corruption.inverse_action.obligation_id.rsplit(":", 1)[1])
            members = tuple(
                torch.nonzero(sample.case.group_membership[group_index], as_tuple=False)
                .reshape(-1)
                .tolist()
            )
            state = build_repair_state(
                sample.case,
                normalize_xywh(sample.case, corruption.placement),
                exact_contact_placement=corruption.placement,
                corruption_kind=corruption.kind.lower(),
                corruption_level=int(corruption.kind[-1]),
            )
            obligation = RepairObligation(
                ExpertKind.CONTACT,
                corruption.inverse_action.obligation_id,
                members,
                debt=corruption.debt_after,
            )
            records.append((sample, source, corruption, state, obligation))
            selected[corruption.kind] += 1
        if selected == wanted:
            break
    if selected != wanted:
        raise RuntimeError(
            f"could not collect requested corruption mix: {selected} != {wanted}"
        )
    return records


@torch.inference_mode()
def _evaluate(model, records) -> dict:
    top1 = top4 = decoded_top4 = 0
    per_kind = {
        "C0": {"count": 0, "top1": 0, "top4": 0, "decoded_top4": 0},
        "C1": {"count": 0, "top1": 0, "top4": 0, "decoded_top4": 0},
        "C2": {"count": 0, "top1": 0, "top4": 0, "decoded_top4": 0},
    }
    losses = []
    source_ids = set()
    for sample, source, corruption, state, obligation in records:
        output = model(state, obligation)
        losses.append(
            float(contact_action_loss(output, corruption.inverse_action).total)
        )
        actions = topk_contact_actions(output, obligation, k=4)
        teacher = action_sha256(corruption.inverse_action)
        matches_top1 = bool(actions) and action_sha256(actions[0]) == teacher
        matches_top4 = any(action_sha256(action) == teacher for action in actions)
        decodes = any(
            decode_contact_action(
                sample.case, corruption.placement, action, verify_case=source
            ).succeeded
            for action in actions
        )
        kind = per_kind[corruption.kind]
        kind["count"] += 1
        kind["top1"] += int(matches_top1)
        kind["top4"] += int(matches_top4)
        kind["decoded_top4"] += int(decodes)
        top1 += int(matches_top1)
        top4 += int(matches_top4)
        decoded_top4 += int(decodes)
        source_ids.add(sample.sample_id)
    for kind in per_kind.values():
        count = max(kind["count"], 1)
        for metric in ("top1", "top4", "decoded_top4"):
            kind[metric] /= count
    count = len(records)
    return {
        "record_count": count,
        "source_count": len(source_ids),
        "source_id_sha256": hashlib.sha256(
            "\n".join(sorted(source_ids)).encode()
        ).hexdigest(),
        "mean_factorized_nll": sum(losses) / count,
        "top1_inverse_action_recall": top1 / count,
        "top4_inverse_action_recall": top4 / count,
        "decoded_top4_success_rate": decoded_top4 / count,
        "by_kind": per_kind,
    }


def _kind_counts(count: int) -> dict[str, int]:
    base, extra = divmod(count, 3)
    return {
        kind: base + int(index < extra) for index, kind in enumerate(("C0", "C1", "C2"))
    }


if __name__ == "__main__":
    raise SystemExit(main())
