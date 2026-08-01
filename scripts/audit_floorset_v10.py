#!/usr/bin/env python3
"""Audit a pinned FloorSet v10 reference checkout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_reference_module():
    module_path = REPO_ROOT / "src" / "hcfp" / "reference.py"
    spec = importlib.util.spec_from_file_location("_hcfp_reference", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reference = _load_reference_module()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkout", type=Path, help="Existing FloorSet checkout to verify")
    source.add_argument(
        "--fetch-cache",
        action="store_true",
        help="Clone/fetch the pinned reference into a repo-local cache before auditing",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/floorset-v10"),
        help="Repo-local cache directory used with --fetch-cache (default: artifacts/floorset-v10)",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        checkout = args.checkout
        if args.fetch_cache:
            checkout = reference.fetch_reference_checkout(_repo_local_cache_dir(args.cache_dir))
        audit = reference.audit_reference_checkout(checkout)
    except reference.ReferenceAuditError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    payload = {
        "ok": True,
        "repo_url": reference.OFFICIAL_FLOORSET_V10.repo_url,
        "commit": audit.head,
        "root": str(audit.root),
        "clean": audit.clean,
        "evaluator": str(audit.evaluator),
        "evaluator_sha256": audit.evaluator_sha256,
        "import_paths": list(audit.import_paths),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"ok: {payload['ok']}")
        print(f"repo_url: {payload['repo_url']}")
        print(f"commit: {payload['commit']}")
        print(f"root: {payload['root']}")
        print(f"evaluator_sha256: {payload['evaluator_sha256']}")
        print("import_paths:")
        for import_path in payload["import_paths"]:
            print(f"  - {import_path}")
    return 0


def _repo_local_cache_dir(cache_dir: Path) -> Path:
    resolved = (REPO_ROOT / cache_dir).resolve() if not cache_dir.is_absolute() else cache_dir.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise reference.ReferenceAuditError(f"cache dir must stay inside this repo: {resolved}") from exc
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
