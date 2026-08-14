#!/usr/bin/env python3
"""Audit official FloorSet-Lite placements for CCRL source eligibility."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.floorset_lite import iter_floorset_lite_with_source  # noqa: E402
from hcfp.repair.dataset import (  # noqa: E402
    SPLIT_VERSION,
    audit_clean_sample,
    render_clean_pool_markdown,
    sha256_lines,
    summarize_clean_pool,
    validate_training_root,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floorset-lite-root", default="artifacts/floorset-v10")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=5090)
    parser.add_argument("--max-layouts-per-file", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output")
    args = parser.parse_args(argv)
    if args.limit <= 0 or args.max_layouts_per_file <= 0:
        parser.error("--limit and --max-layouts-per-file must be positive")

    root = validate_training_root(args.floorset_lite_root)
    records = [
        audit_clean_sample(sample, source)
        for sample, source in iter_floorset_lite_with_source(
            root,
            limit=args.limit,
            seed=args.seed,
            max_layouts_per_file=args.max_layouts_per_file,
        )
    ]
    if len(records) != args.limit:
        raise RuntimeError(f"collected {len(records)} samples, expected {args.limit}")
    source_files = sorted({record["sample_id"].rsplit(":", 1)[0] for record in records})
    report = {
        "schema_version": 1,
        "date": date.today().isoformat(),
        "config": {
            "floorset_lite_root": str(root),
            "limit": args.limit,
            "seed": args.seed,
            "max_layouts_per_file": args.max_layouts_per_file,
            "split_version": SPLIT_VERSION,
        },
        "provenance": {
            "source_file_count": len(source_files),
            "source_file_sha256": sha256_lines(source_files),
            "torch_version": torch.__version__,
        },
        "summary": summarize_clean_pool(records),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        markdown = Path(args.markdown_output)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_clean_pool_markdown(report), encoding="utf-8")
    compact_summary = dict(report["summary"])
    compact_summary["splits"] = {
        split: {key: value for key, value in data.items() if key != "source_ids"}
        for split, data in report["summary"]["splits"].items()
    }
    print(json.dumps(compact_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
