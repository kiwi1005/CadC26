#!/usr/bin/env python3
"""Build auditable HCFP tar shards from JSON or torch fixture lists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.data import file_sha256, sample_from_fixture, write_shard  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help=".json, .pt, or .pth containing a sample list")
    parser.add_argument("-o", "--output", required=True, help="output .tar path")
    parser.add_argument("--source", required=True, help="source dataset name")
    parser.add_argument("--source-version", required=True, help="source dataset version/checksum tag")
    parser.add_argument("--split", required=True, choices=("train", "dev", "internal-test"))
    parser.add_argument("--denylist", required=True, help="official validation sample_id/test_id denylist")
    args = parser.parse_args(argv)

    deny = _denylist(args.denylist)
    payload = _load(Path(args.input))
    if not isinstance(payload, list):
        raise ValueError("input must contain a list of sample fixtures")
    samples = []
    for item in payload:
        sample_id = str(item.get("sample_id", item.get("test_id", "")))
        test_id = item.get("test_id")
        identifiers = {sample_id}
        if test_id is not None:
            identifiers.update((str(test_id), f"validation-{test_id}"))
        if identifiers & deny:
            continue
        samples.append(sample_from_fixture(item))
    manifest = write_shard(
        samples,
        args.output,
        provenance={
            "source": args.source,
            "source_version": args.source_version,
            "split": args.split,
            "denylist_sha256": file_sha256(args.denylist),
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _load(path: Path):
    if path.suffix.lower() in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu", weights_only=True)
    return json.loads(path.read_text(encoding="utf-8"))


def _denylist(path: str | None) -> set[str]:
    if not path:
        return set()
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return set()
    if text[0] in "[{":
        payload = json.loads(text)
        values = payload.values() if isinstance(payload, dict) else payload
        return {str(value) for value in values}
    return {line.strip() for line in text.splitlines() if line.strip()}


if __name__ == "__main__":
    raise SystemExit(main())
