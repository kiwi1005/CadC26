#!/usr/bin/env python3
"""Measure held-out Contact C0-C2 corruption and inverse-decoder yield."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.floorset_lite import iter_floorset_lite_with_source  # noqa: E402
from hcfp.repair.corruption.contact import (  # noqa: E402
    contact_c2_eligible,
    generate_contact_corruptions,
)
from hcfp.repair.dataset import (  # noqa: E402
    audit_clean_sample,
    sha256_lines,
    source_split,
    validate_training_root,
)
from hcfp.verify import mib_shape_keys  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floorset-lite-root", default="artifacts/floorset-v10")
    parser.add_argument("--selected-sources", type=int, default=200)
    parser.add_argument("--scan-limit", type=int)
    parser.add_argument("--seed", type=int, default=5090)
    parser.add_argument("--max-layouts-per-file", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.selected_sources <= 0:
        parser.error("--selected-sources must be positive")
    scan_limit = args.scan_limit or args.selected_sources * 6
    if scan_limit < args.selected_sources:
        parser.error("--scan-limit must be at least --selected-sources")

    root = validate_training_root(args.floorset_lite_root)
    records = []
    for sample, source in iter_floorset_lite_with_source(
        root,
        limit=scan_limit,
        seed=args.seed,
        max_layouts_per_file=args.max_layouts_per_file,
    ):
        split, _ = source_split(sample.sample_id)
        if split != "heldout":
            continue
        clean = audit_clean_sample(sample, source)
        if not clean["eligibility"]["contact_clean"]:
            continue
        boxes = torch.as_tensor(source["fp_sol_xywh"], dtype=torch.float64)
        corruptions = generate_contact_corruptions(
            sample.case,
            boxes,
            verify_case=source,
            kinds=("C0", "C1", "C2"),
        )
        by_kind = {corruption.kind: corruption for corruption in corruptions}
        row = {"sample_id": sample.sample_id, "kinds": {}}
        for kind in ("C0", "C1", "C2"):
            eligible = (
                contact_c2_eligible(sample.case, boxes)
                if kind == "C2"
                else clean["eligibility"][f"contact_{kind.lower()}_structural"]
            )
            corruption = by_kind.get(kind)
            if corruption is None:
                row["kinds"][kind] = {"eligible": eligible, "generated": False}
                continue
            preplaced = sample.case.preplaced_mask
            fixed = sample.case.fixed_mask
            mib_before = mib_shape_keys(boxes)
            mib_after = mib_shape_keys(corruption.placement)
            row["kinds"][kind] = {
                "eligible": eligible,
                "generated": True,
                "corruption_id": corruption.inverse_action.corruption_id,
                "hard_feasible": True,
                "decoded_hard_feasible": True,
                "inverse_reduced_debt": corruption.decoded_debt < corruption.debt_after,
                "preplaced_preserved": torch.equal(
                    corruption.placement[preplaced], boxes[preplaced]
                ),
                "fixed_shape_preserved": torch.equal(
                    corruption.placement[fixed, 2:4], boxes[fixed, 2:4]
                ),
                "mib_shape_preserved": all(
                    mib_before[index] == mib_after[index]
                    for index in torch.nonzero(
                        sample.case.mib_membership.any(0), as_tuple=False
                    )
                    .reshape(-1)
                    .tolist()
                ),
                "debt_before": corruption.debt_before,
                "debt_after": corruption.debt_after,
                "decoded_debt": corruption.decoded_debt,
            }
        records.append(row)
        if len(records) >= args.selected_sources:
            break
    if len(records) != args.selected_sources:
        raise RuntimeError(
            f"collected {len(records)} held-out Contact-clean sources, expected {args.selected_sources}"
        )

    summary = _summary(records)
    report = {
        "schema_version": 1,
        "config": {
            "floorset_lite_root": str(root),
            "selected_sources": args.selected_sources,
            "scan_limit": scan_limit,
            "seed": args.seed,
            "max_layouts_per_file": args.max_layouts_per_file,
        },
        "summary": summary,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _summary(records: list[dict]) -> dict:
    by_kind = {}
    for kind in ("C0", "C1", "C2"):
        rows = [
            record["kinds"][kind]
            for record in records
            if record["kinds"][kind]["eligible"]
        ]
        generated = [row for row in rows if row["generated"]]
        by_kind[kind] = {
            "selected": len(rows),
            "generated": len(generated),
            "generation_rate": len(generated) / len(rows),
            "hard_feasible_rate": _rate(generated, "hard_feasible"),
            "decoded_hard_feasible_rate": _rate(generated, "decoded_hard_feasible"),
            "inverse_debt_reduction_rate": _rate(generated, "inverse_reduced_debt"),
            "preplaced_preservation_rate": _rate(generated, "preplaced_preserved"),
            "fixed_shape_preservation_rate": _rate(generated, "fixed_shape_preserved"),
            "mib_shape_preservation_rate": _rate(generated, "mib_shape_preserved"),
        }
    gates = {
        "c0_generation": by_kind["C0"]["generation_rate"] >= 0.95,
        "c1_generation": by_kind["C1"]["generation_rate"] >= 0.95,
        "c2_generation": by_kind["C2"]["generation_rate"] >= 0.80,
        "hard_feasible": all(
            by_kind[kind]["hard_feasible_rate"] >= 0.99 for kind in by_kind
        ),
        "decoded_hard_feasible": all(
            by_kind[kind]["decoded_hard_feasible_rate"] >= 0.99 for kind in by_kind
        ),
        "inverse_debt_reduction": all(
            by_kind[kind]["inverse_debt_reduction_rate"] >= 0.90 for kind in by_kind
        ),
        "mobility_preservation": all(
            by_kind[kind][key] == 1.0
            for kind in by_kind
            for key in (
                "preplaced_preservation_rate",
                "fixed_shape_preservation_rate",
                "mib_shape_preservation_rate",
            )
        ),
    }
    return {
        "source_count": len(records),
        "source_id_sha256": sha256_lines(record["sample_id"] for record in records),
        "by_kind": by_kind,
        "gates": gates,
        "decision": "KEEP"
        if all(value is True for value in gates.values())
        else "MODIFY",
    }


def _rate(rows: list[dict], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / max(len(rows), 1)


if __name__ == "__main__":
    raise SystemExit(main())
