#!/usr/bin/env python3
"""Generate repair-aware ranker replay from the exact HCFP tail."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.analytic import AnalyticConfig, select_device  # noqa: E402
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint  # noqa: E402
from hcfp.data import file_sha256, sample_to_payload  # noqa: E402
from hcfp.dynamics import DynamicsConfig  # noqa: E402
from hcfp.floorset_lite import iter_floorset_lite_with_source  # noqa: E402
from hcfp.learned import (  # noqa: E402
    LearnedConfig,
    analyze_case_with_checkpoint,
    effective_collective_steps,
    effective_flow_steps,
)
from hcfp.ranker_features import RANKER_FEATURE_DIM, RANKER_FEATURE_VERSION  # noqa: E402
from hcfp.replay import (  # noqa: E402
    OFFICIAL_TARGET_KIND,
    ReplayRecord,
    iter_replay,
    records_from_learned_analysis,
    write_replay_v3,
)


REPLAY_MANIFEST_SCHEMA = 1
NEAR_CAP_ABS_MARGIN = 0.25


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floorset-lite-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=_positive_int, default=32)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--dynamics-steps", type=int, default=4)
    parser.add_argument("--projection-steps", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=0)
    parser.add_argument("--collective-steps", type=_non_negative_int, default=0)
    parser.add_argument("--topology-seeds", type=_non_negative_int, default=0)
    parser.add_argument("--constraint-seeds", type=_non_negative_int, default=0)
    parser.add_argument("--flow-seed", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--score-aware", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--exclude-replay", action="append", default=[])
    parser.add_argument(
        "--shard-dir",
        help="write resumable replay shards into this directory; --output becomes the manifest path",
    )
    parser.add_argument(
        "--shard-sample-size",
        type=_positive_int,
        default=512,
        help="accepted samples per shard when --shard-dir is set",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a sharded replay run after validating the existing manifest fingerprint",
    )
    parser.add_argument(
        "--record-stage",
        choices=("both", "initial", "post_relax"),
        default="both",
    )
    args = parser.parse_args(argv)
    if args.score_aware and args.seed is None:
        parser.error("--score-aware requires an explicit --seed")

    device = select_device(args.device)
    model, checkpoint_metadata = load_checkpoint(
        args.checkpoint,
        expected_normalization=RUNTIME_NORMALIZATION,
        map_location="cpu",
    )
    flow_steps = effective_flow_steps(args.flow_steps, checkpoint_metadata)
    collective_steps = effective_collective_steps(
        args.collective_steps,
        checkpoint_metadata,
        getattr(model, "config", checkpoint_metadata.get("config", {})),
    )
    analytic = AnalyticConfig(
        dynamics=DynamicsConfig(population=args.population, steps=args.dynamics_steps),
        projection_iterations=args.projection_steps,
        direction_beam=2,
    )
    config = LearnedConfig(
        analytic=analytic,
        flow_steps=flow_steps,
        collective_steps=collective_steps,
        seed=args.flow_seed,
        topology_seeds=args.topology_seeds,
        constraint_seeds=args.constraint_seeds,
    )
    excluded_ids = {
        record.sample.sample_id
        for replay_path in args.exclude_replay
        for record in iter_replay(replay_path)
    }
    exclusion_stats = {"skipped": 0, "accepted": 0}
    candidate_config = {
        "population": args.population,
        "dynamics_steps": args.dynamics_steps,
        "projection_steps": args.projection_steps,
        "direction_beam": analytic.direction_beam,
        "requested_flow_steps": args.flow_steps,
        "flow_steps": flow_steps,
        "requested_collective_steps": args.collective_steps,
        "collective_steps": collective_steps,
        "topology_seeds": args.topology_seeds,
        "constraint_seeds": args.constraint_seeds,
        "flow_seed": args.flow_seed,
        "record_stage": args.record_stage,
    }
    checkpoint_report = {
        "path": args.checkpoint,
        "sha256": file_sha256(args.checkpoint),
        "state_hash": checkpoint_metadata["state_hash"],
        "capabilities": checkpoint_metadata["capabilities"],
        "trained_heads": checkpoint_metadata["trained_heads"],
        "config": checkpoint_metadata.get("config", {}),
    }
    dataset_report = {
        "root": str(Path(args.floorset_lite_root).resolve()),
        "seed": args.seed,
        "score_aware": args.score_aware,
        "exclusions": {
            "sample_count": len(excluded_ids),
            "replays": [
                {
                    "path": str(Path(path).resolve()),
                    "sha256": file_sha256(path),
                }
                for path in args.exclude_replay
            ],
        },
    }
    run_fingerprint = _run_fingerprint(
        dataset=dataset_report,
        checkpoint=checkpoint_report,
        candidate_config=candidate_config,
        shard_sample_size=args.shard_sample_size,
        target_kind=OFFICIAL_TARGET_KIND,
    )

    if args.shard_dir:
        return _run_sharded(
            args=args,
            device=device,
            config=config,
            excluded_ids=excluded_ids,
            exclusion_stats=exclusion_stats,
            dataset_report=dataset_report,
            checkpoint_report=checkpoint_report,
            candidate_config=candidate_config,
            run_fingerprint=run_fingerprint,
        )

    def records():
        for sample, source in iter_floorset_lite_with_source(
            args.floorset_lite_root,
            limit=None,
            seed=args.seed,
            score_aware=args.score_aware,
        ):
            if sample.sample_id in excluded_ids:
                exclusion_stats["skipped"] += 1
                continue
            device_sample = sample.case.to(device=device, dtype=None)
            analysis = analyze_case_with_checkpoint(device_sample, args.checkpoint, config)
            if not analysis.result.used_checkpoint or analysis.result.checkpoint_hash is None:
                raise RuntimeError(analysis.result.failure_reason or "checkpoint was not used")
            stage_records = records_from_learned_analysis(
                sample,
                source,
                analysis.result.checkpoint_hash,
                analysis,
                analytic_population=args.population,
                population_seed=args.flow_seed,
                stages=(
                    ("initial", "post_relax")
                    if args.record_stage == "both"
                    else (args.record_stage,)
                ),
            )
            yield from stage_records
            exclusion_stats["accepted"] += 1
            if exclusion_stats["accepted"] >= args.limit:
                return

    value_count = 0
    value_sum = 0.0
    value_square_sum = 0.0
    value_min = math.inf
    value_max = -math.inf
    unique_values: set[float] = set()
    records_with_signal = 0
    sample_ids: list[str] = []
    seen_samples: set[str] = set()
    stage_counts: dict[str, int] = {}

    def observed_records():
        nonlocal value_count, value_sum, value_square_sum, value_min, value_max
        nonlocal records_with_signal
        for record in records():
            values = [float(value) for value in record.target_score]
            value_count += len(values)
            value_sum += sum(values)
            value_square_sum += sum(value * value for value in values)
            value_min = min(value_min, *values)
            value_max = max(value_max, *values)
            unique_values.update(values)
            records_with_signal += float(record.target_score.max() - record.target_score.min()) > 1.0e-8
            sample_id = record.sample.sample_id
            if sample_id not in seen_samples:
                seen_samples.add(sample_id)
                sample_ids.append(sample_id)
            stage = str(record.candidate_stage)
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            yield record

    count = write_replay_v3(observed_records(), args.output)
    if not count or not value_count:
        raise ValueError("replay generation produced no records")
    sample_id_sha256 = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
    value_mean = value_sum / value_count
    value_variance = max(0.0, value_square_sum / value_count - value_mean * value_mean)
    report = {
        "schema_version": 3,
        "records": count,
        "target_kind": OFFICIAL_TARGET_KIND,
        "stages": stage_counts,
        "mid_flow_state_recorded": False,
        "mid_flow_state_note": "not available in current LearnedAnalysis; replay records initial and post_relax only",
        "dataset": {
            **dataset_report,
            "samples": len(sample_ids),
            "sample_id_sha256": sample_id_sha256,
            "exclusions": {
                **dataset_report["exclusions"],
                "skipped": exclusion_stats["skipped"],
            },
        },
        "checkpoint": checkpoint_report,
        "candidate_config": candidate_config,
        "ranker_feature_views": {
            "stored": "stored_candidate_features_v1",
            "repair_aware": RANKER_FEATURE_VERSION,
            "repair_aware_dim": RANKER_FEATURE_DIM,
        },
        "target_distribution": {
            "count": value_count,
            "unique": len(unique_values),
            "minimum": value_min,
            "maximum": value_max,
            "mean": value_mean,
            "standard_deviation": math.sqrt(value_variance),
            "records_with_signal": records_with_signal,
        },
        "device": str(device),
        "output": args.output,
        "output_sha256": file_sha256(args.output),
    }
    Path(f"{args.output}.report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _run_sharded(
    *,
    args: argparse.Namespace,
    device,
    config: LearnedConfig,
    excluded_ids: set[str],
    exclusion_stats: dict[str, int],
    dataset_report: dict[str, object],
    checkpoint_report: dict[str, object],
    candidate_config: dict[str, object],
    run_fingerprint: str,
) -> int:
    shard_dir = Path(args.shard_dir)
    manifest_path = Path(args.output)
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing_manifest = _load_resume_manifest(
        manifest_path,
        resume=bool(args.resume),
        expected_fingerprint=run_fingerprint,
    )
    completed_shards = list(existing_manifest.get("shards", [])) if existing_manifest else []
    completed_sample_ids = {
        sample_id
        for shard in completed_shards
        for sample_id in shard.get("sample_ids", [])
    }
    _validate_completed_shards(shard_dir, completed_shards)
    _reconcile_untracked_shards(
        shard_dir,
        completed_shards,
        resume=bool(args.resume),
    )
    _drop_part_files(shard_dir)

    state = _ManifestState.from_existing(
        manifest=existing_manifest,
        run_fingerprint=run_fingerprint,
        completed_sample_ids=completed_sample_ids,
    )
    if int(args.limit) < state.samples:
        raise ValueError("resume limit cannot be smaller than completed sample count")
    current_records: list[ReplayRecord] = []
    current_sample_ids: list[str] = []
    current_sample_hashes: list[str] = []
    next_shard_index = len(completed_shards)
    target_new_samples = max(0, int(args.limit) - len(completed_sample_ids))

    def persist_manifest(status: str) -> None:
        report = _sharded_manifest(
            state=state,
            dataset_report=dataset_report,
            checkpoint_report=checkpoint_report,
            candidate_config=candidate_config,
            run_fingerprint=run_fingerprint,
            limit=args.limit,
            shard_dir=shard_dir,
            shard_sample_size=args.shard_sample_size,
            device=str(device),
            exclusions={
                **dataset_report["exclusions"],
                "skipped": exclusion_stats["skipped"],
            },
            resumed_from=str(manifest_path.resolve()) if existing_manifest else None,
            status=status,
        )
        _atomic_write_text(
            manifest_path,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )

    persist_manifest("in_progress")

    for sample, source in iter_floorset_lite_with_source(
        args.floorset_lite_root,
        limit=None,
        seed=args.seed,
        score_aware=args.score_aware,
    ):
        if state.new_samples >= target_new_samples:
            break
        sample_id = sample.sample_id
        if sample_id in completed_sample_ids:
            continue
        if sample_id in excluded_ids:
            exclusion_stats["skipped"] += 1
            continue
        records = _records_for_sample(
            sample=sample,
            source=source,
            device=device,
            checkpoint_path=args.checkpoint,
            config=config,
            analytic_population=args.population,
            population_seed=args.flow_seed,
            record_stage=args.record_stage,
        )
        sample_hash = _sample_sha256(sample)
        current_records.extend(records)
        current_sample_ids.append(sample_id)
        current_sample_hashes.append(sample_hash)
        state.observe_sample(sample, sample_hash, records)
        exclusion_stats["accepted"] += 1
        if len(current_sample_ids) >= int(args.shard_sample_size):
            shard = _write_replay_shard(
                shard_dir=shard_dir,
                index=next_shard_index,
                records=current_records,
                sample_ids=current_sample_ids,
                sample_hashes=current_sample_hashes,
            )
            state.shards.append(shard)
            persist_manifest("in_progress")
            next_shard_index += 1
            current_records = []
            current_sample_ids = []
            current_sample_hashes = []

    if current_records:
        shard = _write_replay_shard(
            shard_dir=shard_dir,
            index=next_shard_index,
            records=current_records,
            sample_ids=current_sample_ids,
            sample_hashes=current_sample_hashes,
        )
        state.shards.append(shard)
        persist_manifest("in_progress")

    if not state.records:
        raise ValueError("replay generation produced no records")
    if state.samples < int(args.limit):
        persist_manifest("source_exhausted")
        raise RuntimeError(
            f"replay source exhausted after {state.samples} of {int(args.limit)} samples"
        )

    persist_manifest("complete")
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _records_for_sample(
    *,
    sample,
    source,
    device,
    checkpoint_path: str,
    config: LearnedConfig,
    analytic_population: int,
    population_seed: int,
    record_stage: str,
) -> tuple[ReplayRecord, ...]:
    device_sample = sample.case.to(device=device, dtype=None)
    analysis = analyze_case_with_checkpoint(device_sample, checkpoint_path, config)
    if not analysis.result.used_checkpoint or analysis.result.checkpoint_hash is None:
        raise RuntimeError(analysis.result.failure_reason or "checkpoint was not used")
    return records_from_learned_analysis(
        sample,
        source,
        analysis.result.checkpoint_hash,
        analysis,
        analytic_population=analytic_population,
        population_seed=population_seed,
        stages=(
            ("initial", "post_relax")
            if record_stage == "both"
            else (record_stage,)
        ),
    )


class _ManifestState:
    def __init__(self, *, run_fingerprint: str) -> None:
        self.run_fingerprint = run_fingerprint
        self.records = 0
        self.samples = 0
        self.new_samples = 0
        self.sample_ids: list[str] = []
        self.sample_hashes: list[str] = []
        self.shards: list[dict[str, object]] = []
        self.stages: dict[str, int] = {}
        self.composition = _empty_composition()

    @classmethod
    def from_existing(
        cls,
        *,
        manifest: dict[str, object] | None,
        run_fingerprint: str,
        completed_sample_ids: set[str],
    ) -> "_ManifestState":
        state = cls(run_fingerprint=run_fingerprint)
        if not manifest:
            return state
        state.records = int(manifest.get("records", 0))
        state.samples = int(manifest.get("samples", 0))
        state.sample_ids = list(manifest.get("sample_ids", []))
        state.sample_hashes = list(manifest.get("sample_hashes", []))
        state.shards = list(manifest.get("shards", []))
        state.stages = dict(manifest.get("stages", {}))
        state.composition = dict(manifest.get("composition", _empty_composition()))
        if state.samples != len(completed_sample_ids):
            raise ValueError("resume manifest sample count does not match completed shards")
        flattened_ids = [
            str(sample_id)
            for shard in state.shards
            for sample_id in shard.get("sample_ids", [])
        ]
        flattened_hashes = [
            str(sample_hash)
            for shard in state.shards
            for sample_hash in shard.get("sample_hashes", [])
        ]
        if len(set(flattened_ids)) != len(flattened_ids):
            raise ValueError("resume manifest contains duplicate sample ids")
        if state.sample_ids != flattened_ids or state.sample_hashes != flattened_hashes:
            raise ValueError("resume manifest sample catalog does not match completed shards")
        if state.records != sum(int(shard.get("records", 0)) for shard in state.shards):
            raise ValueError("resume manifest record count does not match completed shards")
        shard_stages: dict[str, int] = {}
        for shard in state.shards:
            for stage, count in dict(shard.get("stages", {})).items():
                shard_stages[str(stage)] = shard_stages.get(str(stage), 0) + int(count)
        if state.stages != shard_stages:
            raise ValueError("resume manifest stage counts do not match completed shards")
        return state

    def observe_sample(
        self,
        sample,
        sample_hash: str,
        records: Sequence[ReplayRecord],
    ) -> None:
        self.samples += 1
        self.new_samples += 1
        self.sample_ids.append(sample.sample_id)
        self.sample_hashes.append(sample_hash)
        for record in records:
            self.records += 1
            stage = str(record.candidate_stage)
            self.stages[stage] = self.stages.get(stage, 0) + 1
            _update_composition(self.composition, record)


def _empty_composition() -> dict[str, int]:
    return {
        "records_hard_negative": 0,
        "records_near_cap": 0,
        "records_large_106_120": 0,
        "records_successful_positive": 0,
        "candidates_hard_negative": 0,
        "candidates_near_cap": 0,
        "candidates_large_106_120": 0,
        "candidates_successful_positive": 0,
    }


def _update_composition(composition: dict[str, int], record: ReplayRecord) -> None:
    candidate_count = int(record.target_score.numel())
    large = 106 <= int(record.sample.case.n) <= 120
    hard_negative = _bool_mask(record.feasibility_tier, candidate_count, positive_when_nonzero=True)
    if record.post_repair_hard_feasible is not None:
        hard_negative |= ~record.post_repair_hard_feasible.detach().cpu().bool()
    near_cap = _bool_mask(record.post_repair_cap_margin, candidate_count, near_cap=True)
    successful = torch.zeros(candidate_count, dtype=torch.bool)
    if record.post_repair_cap_margin is not None:
        successful = record.post_repair_cap_margin.detach().cpu().float() > 0.0
        if record.post_repair_hard_feasible is not None:
            successful &= record.post_repair_hard_feasible.detach().cpu().bool()
    composition["records_hard_negative"] += int(bool(hard_negative.any()))
    composition["records_near_cap"] += int(bool(near_cap.any()))
    composition["records_large_106_120"] += int(large)
    composition["records_successful_positive"] += int(bool(successful.any()))
    composition["candidates_hard_negative"] += int(hard_negative.sum().item())
    composition["candidates_near_cap"] += int(near_cap.sum().item())
    composition["candidates_large_106_120"] += candidate_count if large else 0
    composition["candidates_successful_positive"] += int(successful.sum().item())


def _bool_mask(
    values,
    candidate_count: int,
    *,
    positive_when_nonzero: bool = False,
    near_cap: bool = False,
) -> torch.Tensor:
    if values is None:
        return torch.zeros(candidate_count, dtype=torch.bool)
    tensor = values.detach().cpu()
    if positive_when_nonzero:
        return tensor.long() != 0
    if near_cap:
        return tensor.float().abs() <= NEAR_CAP_ABS_MARGIN
    raise AssertionError("unsupported mask mode")


def _write_replay_shard(
    *,
    shard_dir: Path,
    index: int,
    records: Sequence[ReplayRecord],
    sample_ids: Sequence[str],
    sample_hashes: Sequence[str],
) -> dict[str, object]:
    if not records or not sample_ids:
        raise ValueError("cannot write an empty replay shard")
    filename = f"replay-{index:05d}.jsonl"
    final_path = shard_dir / filename
    part_path = shard_dir / f"{filename}.part"
    if final_path.exists():
        raise ValueError(f"refusing to overwrite existing shard: {final_path}")
    if part_path.exists():
        part_path.unlink()
    count = write_replay_v3(records, part_path)
    os.replace(part_path, final_path)
    stages: dict[str, int] = {}
    for record in records:
        stage = str(record.candidate_stage)
        stages[stage] = stages.get(stage, 0) + 1
    return {
        "path": filename,
        "sha256": file_sha256(final_path),
        "records": count,
        "samples": len(sample_ids),
        "sample_ids": list(sample_ids),
        "sample_hashes": list(sample_hashes),
        "stages": stages,
    }


def _load_resume_manifest(
    manifest_path: Path,
    *,
    resume: bool,
    expected_fingerprint: str,
) -> dict[str, object] | None:
    if not manifest_path.exists():
        if resume:
            return None
        return None
    if not resume:
        raise ValueError(f"manifest already exists; pass --resume to continue: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != REPLAY_MANIFEST_SCHEMA:
        raise ValueError("resume manifest schema mismatch")
    if manifest.get("run_fingerprint") != expected_fingerprint:
        raise ValueError("resume manifest fingerprint mismatch")
    return manifest


def _validate_completed_shards(shard_dir: Path, shards: Sequence[dict[str, object]]) -> None:
    for index, shard in enumerate(shards):
        expected_name = f"replay-{index:05d}.jsonl"
        if str(shard.get("path")) != expected_name:
            raise ValueError("resume manifest shard sequence is invalid")
        path = shard_dir / str(shard["path"])
        if not path.is_file():
            raise ValueError(f"resume shard is missing: {path}")
        if file_sha256(path) != shard.get("sha256"):
            raise ValueError(f"resume shard checksum mismatch: {path}")


def _reconcile_untracked_shards(
    shard_dir: Path,
    shards: Sequence[dict[str, object]],
    *,
    resume: bool,
) -> None:
    tracked = {str(shard["path"]) for shard in shards}
    untracked = sorted(
        path
        for path in shard_dir.glob("replay-*.jsonl")
        if path.name not in tracked
    )
    if untracked and not resume:
        raise ValueError(f"shard directory contains untracked replay files: {untracked[0]}")
    if not untracked:
        return
    expected_crash_shard = shard_dir / f"replay-{len(shards):05d}.jsonl"
    if untracked != [expected_crash_shard]:
        raise ValueError(
            "resume shard directory contains unexpected untracked replay files"
        )
    expected_crash_shard.unlink()


def _drop_part_files(shard_dir: Path) -> None:
    for path in shard_dir.glob("replay-*.jsonl.part"):
        path.unlink()


def _sharded_manifest(
    *,
    state: _ManifestState,
    dataset_report: dict[str, object],
    checkpoint_report: dict[str, object],
    candidate_config: dict[str, object],
    run_fingerprint: str,
    limit: int,
    shard_dir: Path,
    shard_sample_size: int,
    device: str,
    exclusions: dict[str, object],
    resumed_from: str | None,
    status: str,
) -> dict[str, object]:
    sample_id_sha256 = hashlib.sha256("\n".join(state.sample_ids).encode()).hexdigest()
    return {
        "schema_version": REPLAY_MANIFEST_SCHEMA,
        "replay_schema_version": 3,
        "target_kind": OFFICIAL_TARGET_KIND,
        "run_fingerprint": run_fingerprint,
        "mode": "sharded",
        "status": status,
        "limit": limit,
        "records": state.records,
        "samples": state.samples,
        "new_samples": state.new_samples,
        "sample_ids": state.sample_ids,
        "sample_hashes": state.sample_hashes,
        "sample_id_sha256": sample_id_sha256,
        "stages": state.stages,
        "composition": state.composition,
        "dataset": {**dataset_report, "samples": state.samples, "sample_id_sha256": sample_id_sha256, "exclusions": exclusions},
        "checkpoint": checkpoint_report,
        "candidate_config": candidate_config,
        "ranker_feature_views": {
            "stored": "stored_candidate_features_v1",
            "repair_aware": RANKER_FEATURE_VERSION,
            "repair_aware_dim": RANKER_FEATURE_DIM,
        },
        "shard_dir": str(shard_dir.resolve()),
        "shard_sample_size": shard_sample_size,
        "shards": state.shards,
        "device": device,
        "resumed_from": resumed_from,
    }


def _run_fingerprint(
    *,
    dataset: dict[str, object],
    checkpoint: dict[str, object],
    candidate_config: dict[str, object],
    shard_sample_size: int,
    target_kind: str,
) -> str:
    payload = {
        "dataset": {
            "root": dataset["root"],
            "seed": dataset["seed"],
            "score_aware": dataset["score_aware"],
            "exclusions": dataset["exclusions"],
        },
        "checkpoint": checkpoint,
        "candidate_config": candidate_config,
        "shard_sample_size": shard_sample_size,
        "target_kind": target_kind,
        "ranker_feature_version": RANKER_FEATURE_VERSION,
        "ranker_feature_dim": RANKER_FEATURE_DIM,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sample_sha256(sample) -> str:
    return hashlib.sha256(
        json.dumps(sample_to_payload(sample), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    part_path = Path(f"{path}.part")
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path.write_text(content, encoding="utf-8")
    os.replace(part_path, path)


if __name__ == "__main__":
    raise SystemExit(main())
