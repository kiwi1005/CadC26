#!/usr/bin/env python3
"""Cache exact Contact C0-C2 inverse-action replay from a frozen source manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.data import file_sha256  # noqa: E402
from hcfp.floorset_lite import iter_floorset_lite_with_source  # noqa: E402
from hcfp.geometry import normalize_xywh  # noqa: E402
from hcfp.repair.corruption.contact import generate_contact_corruptions  # noqa: E402
from hcfp.repair.dataset import audit_clean_sample, source_split, validate_training_root  # noqa: E402
from hcfp.repair.decoders.contact import decode_contact_action  # noqa: E402
from hcfp.repair.replay import (  # noqa: E402
    candidate_sha256,
    repair_generation_dumps,
    repair_replay_dumps,
)
from hcfp.repair.schema import (  # noqa: E402
    ExpertKind,
    RepairCandidate,
    RepairGenerationRecord,
    RepairObligation,
    RepairOutcome,
    RepairReplayRecord,
)
from hcfp.repair.state import build_repair_state  # noqa: E402


_KINDS = ("C0", "C1", "C2")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--floorset-lite-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-limit", type=_positive)
    parser.add_argument("--heldout-limit", type=_positive)
    parser.add_argument(
        "--known-c2-failures",
        default="artifacts/experiments/p11_ccrl_contact_corruption/c2_failure_buckets_512.json",
    )
    args = parser.parse_args(argv)

    manifest = _load_manifest(Path(args.source_manifest))
    root = validate_training_root(
        args.floorset_lite_root or manifest["config"]["resolved_floorset_lite_root"]
    )
    selected = _selected_sources(
        manifest,
        train_limit=args.train_limit,
        heldout_limit=args.heldout_limit,
    )
    c2_failures = _known_c2_failures(Path(args.known_c2_failures), selected["heldout"])
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing replay directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    paths = {
        split: {
            "generation": output_dir / f"{split}.generation.jsonl.gz",
            "replay": output_dir / f"{split}.replay.jsonl.gz",
        }
        for split in selected
    }
    counters = {split: Counter() for split in selected}
    found = {split: set() for split in selected}
    with _GzipWriters(paths) as writers:
        for sample, source in iter_floorset_lite_with_source(
            root,
            limit=None,
            seed=int(manifest["config"]["seed"]),
            max_layouts_per_file=int(manifest["config"]["max_layouts_per_file"]),
        ):
            split = source_split(sample.sample_id)[0]
            if sample.sample_id not in selected.get(split, set()):
                continue
            if sample.sample_id in found[split]:
                raise RuntimeError(f"duplicate selected source: {sample.sample_id}")
            clean = audit_clean_sample(sample, source)
            if not clean["eligibility"]["contact_clean"]:
                raise RuntimeError(
                    f"manifest source lost Contact eligibility: {sample.sample_id}"
                )
            found[split].add(sample.sample_id)
            _write_source(
                sample=sample,
                source=source,
                split=split,
                split_version=str(manifest["config"]["split_version"]),
                clean=clean,
                c2_failures=c2_failures,
                generation_writer=writers[split]["generation"],
                replay_writer=writers[split]["replay"],
                counters=counters[split],
            )
            if all(found[name] == selected[name] for name in selected):
                break

    missing = {
        split: sorted(selected[split] - found[split])
        for split in selected
        if selected[split] - found[split]
    }
    if missing:
        raise RuntimeError(f"frozen manifest sources were not found: {missing}")

    report = {
        "schema_version": 1,
        "purpose": "P11.4 Gate D Contact-only cached inverse replay",
        "source_manifest": {
            "path": str(Path(args.source_manifest).resolve()),
            "file_sha256": file_sha256(args.source_manifest),
            "artifact_sha256": manifest["integrity"]["artifact_sha256"],
        },
        "floorset_lite_root": str(root),
        "model_scope": "Contact-only inverse-action learning; no oracle relabeling",
        "splits": {
            split: {
                "source_count": len(found[split]),
                "source_id_sha256": _sha256_lines(found[split]),
                "counts": dict(sorted(counters[split].items())),
                "files": {
                    kind: {
                        "path": str(path.resolve()),
                        "sha256": file_sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for kind, path in paths[split].items()
                },
            }
            for split in selected
        },
        "known_heldout_c2_failure_categories": dict(
            sorted(Counter(c2_failures.values()).items())
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _write_source(
    *,
    sample,
    source,
    split: str,
    split_version: str,
    clean: dict,
    c2_failures: dict[str, str],
    generation_writer,
    replay_writer,
    counters: Counter,
) -> None:
    boxes = torch.as_tensor(source["fp_sol_xywh"], dtype=torch.float64)
    corruptions = {
        corruption.kind: corruption
        for corruption in generate_contact_corruptions(
            sample.case, boxes, verify_case=source, kinds=_KINDS
        )
    }
    for kind in _KINDS:
        counters["generation_requested"] += 1
        corruption = corruptions.get(kind)
        if corruption is None:
            generation_writer.write(
                repair_generation_dumps(
                    RepairGenerationRecord(
                        source_id=sample.sample_id,
                        source_split=split,
                        split_version=split_version,
                        corruption_kind=kind,
                        corruption_requested=True,
                        corruption_generated=False,
                        generation_failure_reason=_failure_reason(
                            sample.sample_id, kind, clean, c2_failures
                        ),
                    )
                )
                + "\n"
            )
            counters[f"{kind}_generation_failed"] += 1
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
            corruption_kind=kind.lower(),
            corruption_level=int(kind[-1]),
        )
        obligation = RepairObligation(
            ExpertKind.CONTACT,
            corruption.inverse_action.obligation_id,
            members,
            debt=corruption.debt_after,
        )
        decoded = decode_contact_action(
            sample.case,
            corruption.placement,
            corruption.inverse_action,
            verify_case=source,
        )
        if (
            not decoded.succeeded
            or decoded.placement is None
            or decoded.debt_after is None
        ):
            raise RuntimeError(
                f"stored inverse action stopped decoding: {sample.sample_id} {kind}"
            )
        candidate = RepairCandidate(
            corruption.inverse_action, decoded.placement.float(), "contact-inverse-v1"
        )
        outcome = RepairOutcome(
            candidate_sha256(candidate),
            accepted=True,
            hard_feasible=True,
            debt_before=corruption.debt_after,
            debt_after=decoded.debt_after,
        )
        replay_writer.write(
            repair_replay_dumps(
                RepairReplayRecord(
                    source_id=sample.sample_id,
                    source_split=split,
                    split_version=split_version,
                    state=state,
                    decoder_placement=corruption.placement.clone(),
                    obligation=obligation,
                    action=corruption.inverse_action,
                    candidate=candidate,
                    outcome=outcome,
                )
            )
            + "\n"
        )
        generation_writer.write(
            repair_generation_dumps(
                RepairGenerationRecord(
                    source_id=sample.sample_id,
                    source_split=split,
                    split_version=split_version,
                    corruption_kind=kind,
                    corruption_requested=True,
                    corruption_generated=True,
                    generation_failure_reason=None,
                    inverse_action=corruption.inverse_action,
                    inverse_decode_success=True,
                )
            )
            + "\n"
        )
        counters[f"{kind}_generated"] += 1
        counters["replay_rows"] += 1


def _failure_reason(
    source_id: str,
    kind: str,
    clean: dict,
    c2_failures: dict[str, str],
) -> str:
    if kind == "C2" and source_id in c2_failures:
        return c2_failures[source_id]
    key = f"contact_{kind.lower()}_structural"
    if not clean["eligibility"].get(key, False):
        return "not_structurally_eligible"
    return "no_exact_inverse_action"


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("integrity", {}).get("artifact_sha256")
    canonical = copy.deepcopy(payload)
    canonical["integrity"]["artifact_sha256"] = None
    actual = hashlib.sha256(
        (json.dumps(canonical, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    if expected != actual:
        raise ValueError("source manifest canonical SHA-256 mismatch")
    if payload.get("overlap", {}).get("disjoint") is not True:
        raise ValueError("source manifest splits are not disjoint")
    return payload


def _selected_sources(
    manifest: dict,
    *,
    train_limit: int | None,
    heldout_limit: int | None,
) -> dict[str, set[str]]:
    limits = {"train": train_limit, "heldout": heldout_limit}
    selected = {}
    for split in ("train", "heldout"):
        records = manifest["selected"][split]["records"]
        limit = limits[split]
        if limit is not None:
            records = records[:limit]
        ids = [str(record["source_id"]) for record in records]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"invalid selected {split} source IDs")
        selected[split] = set(ids)
    if selected["train"].intersection(selected["heldout"]):
        raise ValueError("selected source splits overlap")
    return selected


def _known_c2_failures(path: Path, heldout: set[str]) -> dict[str, str]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {
        source_id: category
        for category, source_ids in payload.get("by_category", {}).items()
        for source_id in source_ids
        if source_id in heldout
    }
    return result


class _GzipWriters:
    def __init__(self, paths: dict[str, dict[str, Path]]) -> None:
        self.paths = paths
        self._streams = []

    def __enter__(self):
        writers = {}
        for split, kinds in self.paths.items():
            writers[split] = {}
            for kind, path in kinds.items():
                raw = path.open("xb")
                compressed = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0)
                text = io.TextIOWrapper(compressed, encoding="utf-8")
                writers[split][kind] = text
                self._streams.append((text, compressed, raw))
        return writers

    def __exit__(self, exc_type, exc, traceback) -> None:
        for text, compressed, raw in reversed(self._streams):
            text.flush()
            text.detach()
            compressed.close()
            raw.close()


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _sha256_lines(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
