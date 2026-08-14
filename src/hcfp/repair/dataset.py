"""Clean-placement audit and leakage-safe source splits for CCRL."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable

import torch

from hcfp.btree import BStarTree
from hcfp.constraints.contact_tree import contact_tree_report
from hcfp.data import DataSample
from hcfp.verify import boundary_missing, mib_shape_keys, verify


SPLIT_VERSION = "ccrl-source-v1"
_SIDE_BITS = {"left": 1, "right": 2, "top": 4, "bottom": 8}


def validate_training_root(root: str | Path) -> Path:
    """Reject validation/test/visible path components before reading labels."""

    resolved = Path(root).resolve()
    for part in resolved.parts:
        lowered = part.lower()
        tokens = set(filter(None, re.split(r"[^a-z0-9]+", lowered)))
        if lowered == "litetensordatatest" or tokens.intersection(
            {"test", "validation", "visible"}
        ):
            raise ValueError("validation/test/visible paths are forbidden for CCRL training")
    return resolved


def source_split(sample_id: str) -> tuple[str, str]:
    """Return a stable 80/20 train/heldout split and its salted digest."""

    source_id = ":".join(sample_id.split(":")[:2])
    digest = hashlib.sha256(f"{SPLIT_VERSION}\0{source_id}".encode()).hexdigest()
    return ("heldout" if int(digest[:16], 16) % 5 == 0 else "train"), digest


def sha256_lines(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def audit_clean_sample(sample: DataSample, source: dict[str, Any]) -> dict[str, Any]:
    """Audit one exact official source placement without generating corruptions."""

    boxes = torch.as_tensor(source.get("fp_sol_xywh"), dtype=torch.float64)
    if boxes.shape != (sample.case.n, 4):
        raise ValueError("source fp_sol_xywh must have shape [N,4]")
    hard = verify(source, boxes)
    split, split_digest = source_split(sample.sample_id)
    movable = ~sample.case.preplaced_mask.to(dtype=torch.bool, device="cpu")

    contact = contact_tree_report(
        boxes,
        sample.case.group_membership,
        net_weight=sample.case.b2b_weight,
        tolerance=0.0,
        fail_on_disconnected=False,
    )
    group_rows = []
    c0 = c1 = c2 = False
    for group_id, tree in zip(sample.case.cluster_group_ids.tolist(), contact.trees, strict=True):
        members = set(tree.members)
        edges = [
            edge
            for edge in contact.contacts
            if edge.first in members and edge.second in members
        ]
        degree = Counter(index for edge in edges for index in (edge.first, edge.second))
        clean = tree.connected and len(tree.members) >= 2
        if clean:
            c0 |= any(bool(movable[index]) and degree[index] == 1 for index in tree.members)
            c1 |= any(bool(movable[index]) for index in tree.members)
            c2 |= any(bool(movable[edge.first] and movable[edge.second]) for edge in edges)
        group_rows.append(
            {
                "group_id": int(group_id),
                "member_count": len(tree.members),
                "contact_edge_count": len(edges),
                "connected": tree.connected,
            }
        )

    required = sample.case.boundary_bits.to(dtype=torch.bool, device="cpu")
    missing = boundary_missing(source, boxes)
    boundary_required = {
        side: int(required[:, index].sum())
        for index, side in enumerate(_SIDE_BITS)
    }
    boundary_missing_by_side = {
        side: int(((missing & bit) != 0).sum()) for side, bit in _SIDE_BITS.items()
    }

    shape_keys = mib_shape_keys(boxes)
    mib_rows = []
    for mib_id, row in zip(
        sample.case.mib_group_ids.tolist(), sample.case.mib_membership, strict=True
    ):
        members = torch.nonzero(row, as_tuple=False).reshape(-1).tolist()
        uniform = len({shape_keys[index] for index in members}) <= 1
        mib_rows.append(
            {
                "mib_id": int(mib_id),
                "member_count": len(members),
                "shape_uniform": uniform,
            }
        )

    tree_valid = False
    if sample.tree_edges is not None:
        try:
            BStarTree.from_edges(sample.tree_edges, sample.case.n)
        except ValueError:
            pass
        else:
            tree_valid = True

    constraints = torch.as_tensor(source["constraints"], dtype=torch.long)
    constrained = (
        (constraints[:, 0] > 0)
        | (constraints[:, 1] > 0)
        | (constraints[:, 2] > 0)
        | (constraints[:, 3] > 0)
        | (constraints[:, 4] > 0)
    )
    n = sample.case.n
    hard_feasible = bool(hard.feasible)
    clean_contact = any(row["connected"] and row["member_count"] >= 2 for row in group_rows)
    return {
        "sample_id": sample.sample_id,
        "source_id_sha256": hashlib.sha256(sample.sample_id.encode()).hexdigest(),
        "split": split,
        "split_sha256": split_digest,
        "block_count": n,
        "hard": {
            "feasible": hard_feasible,
            "overlap_pairs": len(hard.overlap_pairs),
            "area_bad": len(hard.area_bad),
            "fixed_bad": len(hard.fixed_bad),
            "preplaced_bad": len(hard.preplaced_bad),
        },
        "grouping": {
            "group_count": len(group_rows),
            "connected_group_count": sum(row["connected"] for row in group_rows),
            "groups": group_rows,
        },
        "boundary": {
            "required_block_count": int(required.any(dim=1).sum()),
            "required_by_side": boundary_required,
            "satisfied_by_side": {
                side: boundary_required[side] - boundary_missing_by_side[side]
                for side in _SIDE_BITS
            },
        },
        "mib": {
            "group_count": len(mib_rows),
            "uniform_group_count": sum(row["shape_uniform"] for row in mib_rows),
            "groups": mib_rows,
        },
        "tree_valid": tree_valid,
        "counts": {
            "fixed": int(sample.case.fixed_mask.sum()),
            "preplaced": int(sample.case.preplaced_mask.sum()),
            "constrained_blocks": int(constrained.sum()),
        },
        "densities": {
            "fixed": float(sample.case.fixed_mask.float().mean()),
            "preplaced": float(sample.case.preplaced_mask.float().mean()),
            "constrained_blocks": float(constrained.float().mean()),
        },
        "eligibility": {
            "contact_clean": hard_feasible and clean_contact,
            "contact_c0_structural": hard_feasible and c0,
            "contact_c1_structural": hard_feasible and c1,
            "contact_c2_structural": hard_feasible and c2,
            "boundary": hard_feasible
            and sum(boundary_required.values()) > sum(boundary_missing_by_side.values()),
            "mib": hard_feasible
            and any(row["member_count"] >= 2 and row["shape_uniform"] for row in mib_rows),
            "topology": hard_feasible and tree_valid,
        },
    }


def summarize_clean_pool(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one audit record is required")
    source_ids = sorted(record["sample_id"] for record in records)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate source ids are not allowed")
    split_ids = {
        split: sorted(record["sample_id"] for record in records if record["split"] == split)
        for split in ("train", "heldout")
    }
    contact_ids = {
        split: sorted(
            record["sample_id"]
            for record in records
            if record["split"] == split and record["eligibility"]["contact_clean"]
        )
        for split in ("train", "heldout")
    }
    hard_feasible = sum(record["hard"]["feasible"] for record in records)
    eligibility = {
        key: sum(record["eligibility"][key] for record in records)
        for key in records[0]["eligibility"]
    }
    group_count = sum(record["grouping"]["group_count"] for record in records)
    connected_groups = sum(
        record["grouping"]["connected_group_count"] for record in records
    )
    mib_count = sum(record["mib"]["group_count"] for record in records)
    uniform_mib = sum(record["mib"]["uniform_group_count"] for record in records)
    required_by_side = {
        side: sum(record["boundary"]["required_by_side"][side] for record in records)
        for side in _SIDE_BITS
    }
    satisfied_by_side = {
        side: sum(record["boundary"]["satisfied_by_side"][side] for record in records)
        for side in _SIDE_BITS
    }
    split_overlap = set(split_ids["train"]).intersection(split_ids["heldout"])
    gates = {
        "hard_verifier_parity": hard_feasible == len(records),
        "source_split_disjoint": not split_overlap,
        "contact_volume": len(contact_ids["train"]) >= 2000
        and len(contact_ids["heldout"]) >= 512,
        "corruption_yield": "DEFERRED_TO_ISSUE_18",
    }
    return {
        "sample_count": len(records),
        "source_id_sha256": sha256_lines(source_ids),
        "hard_feasible_count": hard_feasible,
        "hard_feasible_rate": hard_feasible / len(records),
        "splits": {
            split: {
                "source_count": len(split_ids[split]),
                "source_id_sha256": sha256_lines(split_ids[split]),
                "source_ids": split_ids[split],
                "contact_eligible_count": len(contact_ids[split]),
                "contact_eligible_id_sha256": sha256_lines(contact_ids[split]),
            }
            for split in ("train", "heldout")
        },
        "split_overlap_count": len(split_overlap),
        "eligibility_counts": eligibility,
        "grouping": {
            "group_count": group_count,
            "connected_group_count": connected_groups,
            "connected_group_rate": connected_groups / max(group_count, 1),
        },
        "boundary": {
            "required_by_side": required_by_side,
            "satisfied_by_side": satisfied_by_side,
        },
        "mib": {
            "group_count": mib_count,
            "uniform_group_count": uniform_mib,
            "uniform_group_rate": uniform_mib / max(mib_count, 1),
        },
        "tree_valid_count": sum(record["tree_valid"] for record in records),
        "distributions": {
            "block_count": _distribution(record["block_count"] for record in records),
            "constraint_block_density": _distribution(
                record["densities"]["constrained_blocks"] for record in records
            ),
            "fixed_density": _distribution(record["densities"]["fixed"] for record in records),
            "preplaced_density": _distribution(
                record["densities"]["preplaced"] for record in records
            ),
        },
        "gates": gates,
        "decision": "KEEP"
        if all(value is True for key, value in gates.items() if key != "corruption_yield")
        else "MODIFY",
    }


def render_clean_pool_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    gates = summary["gates"]
    eligible = summary["eligibility_counts"]
    return f"""# CCRL clean-placement pool audit

Date: {report['date']}  
Source: `{report['config']['floorset_lite_root']}`  
Samples: {summary['sample_count']}

## Result

| Metric | Value |
| --- | ---: |
| Exact hard-feasible `fp_sol` | {summary['hard_feasible_count']} / {summary['sample_count']} |
| Contact-clean train sources | {summary['splits']['train']['contact_eligible_count']} |
| Contact-clean held-out sources | {summary['splits']['heldout']['contact_eligible_count']} |
| Structural Contact C0 / C1 / C2 | {eligible['contact_c0_structural']} / {eligible['contact_c1_structural']} / {eligible['contact_c2_structural']} |
| Connected groups | {summary['grouping']['connected_group_count']} / {summary['grouping']['group_count']} |
| Uniform MIB groups | {summary['mib']['uniform_group_count']} / {summary['mib']['group_count']} |
| Valid `tree_sol` | {summary['tree_valid_count']} / {summary['sample_count']} |
| Split overlap | {summary['split_overlap_count']} |

## Gates

- hard verifier parity: `{gates['hard_verifier_parity']}`
- source split disjoint: `{gates['source_split_disjoint']}`
- 2K train / 512 held-out Contact volume: `{gates['contact_volume']}`
- actual C0/C1 corruption yield: `{gates['corruption_yield']}`

## Decision

`{summary['decision']}` for the P11.1 data foundation. Actual corruption success remains owned by issue #18.
"""


def _distribution(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "min": ordered[0],
        "p25": _percentile(ordered, 0.25),
        "median": _percentile(ordered, 0.5),
        "p75": _percentile(ordered, 0.75),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    position = quantile * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction
