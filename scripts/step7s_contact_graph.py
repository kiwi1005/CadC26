#!/usr/bin/env python3
"""Step7S Phase F2: compute active contact graph + seed closure.

Loads a case, applies an optional Step7Q-F AVNR move (e.g. block 32 -> target
box from `step7q_objective_slot_replay_rows.jsonl`), then computes pairwise
horizontal/vertical clearances, active contacts under `eps_contact`, and the
active-contact closure starting from a seed block.

Output is written to `--out` and printed as a one-line JSON summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from puzzleplace.data.schema import ConstraintColumns
from puzzleplace.experiments.step7l_learning_guided_replay import load_validation_cases
from puzzleplace.geometry.legality import positions_from_case_targets

Box = tuple[float, float, float, float]


def axis_overlap(a0: float, a1: float, b0: float, b1: float, eps: float) -> bool:
    return a1 > b0 + eps and b1 > a0 + eps


def pair_clearance(box_a: Box, box_b: Box, eps: float = 1e-9) -> dict[str, float | str]:
    """Return min positive gap, separating direction, and signed gaps in x/y."""

    ax0, ay0, aw, ah = box_a
    bx0, by0, bw, bh = box_b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    # Signed clearance: positive when separated, negative when overlapping
    gap_left = bx0 - ax1   # a is left of b
    gap_right = ax0 - bx1  # a is right of b
    gap_below = by0 - ay1  # a is below b
    gap_above = ay0 - by1  # a is above b

    x_gap = max(gap_left, gap_right)
    y_gap = max(gap_below, gap_above)

    direction: str
    binding_gap: float
    if x_gap >= -eps and y_gap >= -eps:
        # Both directions could be the separator; the smaller positive gap is binding
        if x_gap <= y_gap:
            direction = "x_left" if gap_left >= gap_right else "x_right"
            binding_gap = x_gap
        else:
            direction = "y_below" if gap_below >= gap_above else "y_above"
            binding_gap = y_gap
    elif x_gap >= -eps:
        direction = "x_left" if gap_left >= gap_right else "x_right"
        binding_gap = x_gap
    elif y_gap >= -eps:
        direction = "y_below" if gap_below >= gap_above else "y_above"
        binding_gap = y_gap
    else:
        direction = "overlap"
        binding_gap = max(x_gap, y_gap)

    return {
        "binding_gap": binding_gap,
        "direction": direction,
        "x_gap": x_gap,
        "y_gap": y_gap,
        "gap_left": gap_left,
        "gap_right": gap_right,
        "gap_below": gap_below,
        "gap_above": gap_above,
    }


def compute_closure(
    seed: int,
    block_count: int,
    boxes: list[Box],
    eps_contact: float,
    eps_overlap: float,
) -> tuple[set[int], list[dict[str, Any]]]:
    closure: set[int] = {seed}
    active_pairs: list[dict[str, Any]] = []
    grew = True
    while grew:
        grew = False
        for i in list(closure):
            for j in range(block_count):
                if j in closure:
                    continue
                pc = pair_clearance(boxes[i], boxes[j], eps=eps_overlap)
                if abs(float(pc["binding_gap"])) <= eps_contact:
                    closure.add(j)
                    active_pairs.append(
                        {"i": i, "j": j, **pc}  # type: ignore[arg-type]
                    )
                    grew = True
    # Also collect all currently active pairs across the closure (for CCQP A_sep)
    all_active_pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for i in sorted(closure):
        for j in range(block_count):
            if j == i:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            pc = pair_clearance(boxes[i], boxes[j], eps=eps_overlap)
            if abs(float(pc["binding_gap"])) <= eps_contact:
                all_active_pairs.append(
                    {"i": pair_key[0], "j": pair_key[1], **pc}  # type: ignore[arg-type]
                )
    return closure, all_active_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--case-id", type=int, required=True)
    parser.add_argument("--seed-block", type=int, required=True)
    parser.add_argument("--apply-avnr-target", type=str, default=None,
                        help="Optional 'x,y,w,h' to replace seed block's target box "
                             "with the Step7Q-F AVNR target before computing clearances.")
    parser.add_argument("--eps-contact", type=float, default=1e-3)
    parser.add_argument("--eps-overlap", type=float, default=1e-9)
    parser.add_argument("--floorset-root", type=Path, default=None)
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases = load_validation_cases(
        args.base_dir,
        [args.case_id],
        floorset_root=args.floorset_root,
        auto_download=args.auto_download,
    )
    case = cases[args.case_id]
    baseline = positions_from_case_targets(case)
    boxes = [tuple(map(float, box)) for box in baseline]

    if args.apply_avnr_target:
        avnr = tuple(float(v) for v in args.apply_avnr_target.split(","))
        if len(avnr) != 4:
            raise SystemExit(f"--apply-avnr-target must be 4 floats, got {avnr!r}")
        boxes[args.seed_block] = avnr  # type: ignore[assignment]

    constraints = case.constraints
    fixed_blocks = [
        idx
        for idx in range(case.block_count)
        if bool(constraints[idx, ConstraintColumns.FIXED].item())
        or bool(constraints[idx, ConstraintColumns.PREPLACED].item())
    ]
    mib_groups: dict[int, list[int]] = {}
    for idx in range(case.block_count):
        gid = int(constraints[idx, ConstraintColumns.MIB].item())
        if gid > 0:
            mib_groups.setdefault(gid, []).append(idx)

    closure, active_pairs_in_closure = compute_closure(
        args.seed_block,
        case.block_count,
        boxes,  # type: ignore[arg-type]
        eps_contact=args.eps_contact,
        eps_overlap=args.eps_overlap,
    )

    # Add MIB mates of any block in the closure
    closure_with_mib = set(closure)
    for members in mib_groups.values():
        if any(b in closure for b in members):
            closure_with_mib.update(members)

    boundary_blocks = [
        idx
        for idx in closure_with_mib
        if int(constraints[idx, ConstraintColumns.BOUNDARY].item()) > 0
    ]

    summary = {
        "schema": "step7s_contact_graph_v1",
        "case_id": args.case_id,
        "seed_block": args.seed_block,
        "block_count": case.block_count,
        "applied_avnr_target": list(boxes[args.seed_block]) if args.apply_avnr_target else None,
        "eps_contact": args.eps_contact,
        "eps_overlap": args.eps_overlap,
        "closure_size": len(closure),
        "closure_with_mib_size": len(closure_with_mib),
        "closure_block_ids": sorted(closure),
        "closure_with_mib_block_ids": sorted(closure_with_mib),
        "fixed_or_preplaced_in_closure": sorted(
            b for b in closure_with_mib if b in fixed_blocks
        ),
        "mib_groups_in_closure": {
            gid: members
            for gid, members in mib_groups.items()
            if any(b in closure_with_mib for b in members)
        },
        "boundary_blocks_in_closure": sorted(boundary_blocks),
        "active_pairs_in_closure": active_pairs_in_closure,
        "active_pair_count": len(active_pairs_in_closure),
        "boxes": {idx: list(boxes[idx]) for idx in sorted(closure_with_mib)},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "case_id": args.case_id,
                "seed_block": args.seed_block,
                "closure_size": len(closure),
                "closure_with_mib_size": len(closure_with_mib),
                "active_pair_count": len(active_pairs_in_closure),
                "fixed_in_closure": len([b for b in closure_with_mib if b in fixed_blocks]),
                "boundary_in_closure": len(boundary_blocks),
            }
        )
    )


if __name__ == "__main__":
    main()
