from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from hcfp.candidates import candidate_features
from hcfp.checkpoint import RUNTIME_NORMALIZATION, save_checkpoint
from hcfp.data import DataSample, extract_labels
from hcfp.fallback import safe_shelf
from hcfp.model import HCFPModel, ModelConfig
from hcfp.profile import synthetic_case
from hcfp.replay import (
    ReplayRecord,
    _candidate_geometry_hashes,
    _candidate_row_ids,
    _target_rank,
    iter_replay,
)


ROOT = Path(__file__).resolve().parents[1]
REPLAY_SCRIPT = ROOT / "scripts/generate_hcfp_replay.py"
REPLAY_SPEC = importlib.util.spec_from_file_location("generate_hcfp_replay_generation", REPLAY_SCRIPT)
assert REPLAY_SPEC is not None and REPLAY_SPEC.loader is not None
generate_replay = importlib.util.module_from_spec(REPLAY_SPEC)
REPLAY_SPEC.loader.exec_module(generate_replay)


def test_sharded_replay_writes_atomic_shards_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = _samples("s", 3)
    checkpoint = _checkpoint(tmp_path)
    _patch_generator(monkeypatch, samples)
    manifest = tmp_path / "manifest.json"
    shard_dir = tmp_path / "shards"

    assert generate_replay.main(
        [
            "--floorset-lite-root",
            str(tmp_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(manifest),
            "--limit",
            "3",
            "--population",
            "2",
            "--shard-dir",
            str(shard_dir),
            "--shard-sample-size",
            "2",
            "--device",
            "cpu",
            "--record-stage",
            "initial",
        ]
    ) == 0

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["mode"] == "sharded"
    assert payload["status"] == "complete"
    assert payload["samples"] == 3
    assert payload["records"] == 3
    assert payload["stages"] == {"initial": 3}
    assert [shard["samples"] for shard in payload["shards"]] == [2, 1]
    assert not list(shard_dir.glob("*.part"))
    assert [record.sample.sample_id for shard in payload["shards"] for record in iter_replay(shard_dir / shard["path"])] == [
        "s0",
        "s1",
        "s2",
    ]
    assert payload["composition"]["records_hard_negative"] == 3
    assert payload["composition"]["candidates_successful_positive"] == 3


def test_sharded_replay_resume_skips_completed_samples_and_ignores_part_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = _samples("resume", 3)
    checkpoint = _checkpoint(tmp_path)
    calls: list[str] = []
    _patch_generator(monkeypatch, samples, calls=calls)
    manifest = tmp_path / "manifest.json"
    shard_dir = tmp_path / "shards"
    common_args = [
        "--floorset-lite-root",
        str(tmp_path),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(manifest),
        "--population",
        "2",
        "--shard-dir",
        str(shard_dir),
        "--shard-sample-size",
        "1",
        "--device",
        "cpu",
        "--record-stage",
        "initial",
    ]

    assert generate_replay.main([*common_args, "--limit", "2"]) == 0
    (shard_dir / "replay-00002.jsonl.part").write_text("incomplete\n", encoding="utf-8")
    unrelated_part = shard_dir / "notes.part"
    unrelated_part.write_text("keep me\n", encoding="utf-8")
    calls.clear()

    assert generate_replay.main([*common_args, "--limit", "3", "--resume"]) == 0

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert calls == ["resume2"]
    assert payload["samples"] == 3
    assert payload["new_samples"] == 1
    assert [shard["path"] for shard in payload["shards"]] == [
        "replay-00000.jsonl",
        "replay-00001.jsonl",
        "replay-00002.jsonl",
    ]
    assert not (shard_dir / "replay-00002.jsonl.part").exists()
    assert unrelated_part.read_text(encoding="utf-8") == "keep me\n"


def test_sharded_replay_resume_fails_closed_on_fingerprint_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = _samples("bad", 2)
    checkpoint = _checkpoint(tmp_path)
    _patch_generator(monkeypatch, samples)
    manifest = tmp_path / "manifest.json"
    shard_dir = tmp_path / "shards"
    base = [
        "--floorset-lite-root",
        str(tmp_path),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(manifest),
        "--limit",
        "1",
        "--population",
        "2",
        "--shard-dir",
        str(shard_dir),
        "--device",
        "cpu",
        "--record-stage",
        "initial",
    ]
    assert generate_replay.main(base) == 0

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        generate_replay.main([*base, "--flow-seed", "99", "--resume"])


def test_sharded_replay_recovers_untracked_atomic_shard_after_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = _samples("crash", 2)
    checkpoint = _checkpoint(tmp_path)
    calls: list[str] = []
    _patch_generator(monkeypatch, samples, calls=calls)
    manifest = tmp_path / "manifest.json"
    shard_dir = tmp_path / "shards"
    args = [
        "--floorset-lite-root",
        str(tmp_path),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(manifest),
        "--limit",
        "2",
        "--population",
        "2",
        "--shard-dir",
        str(shard_dir),
        "--shard-sample-size",
        "1",
        "--device",
        "cpu",
        "--record-stage",
        "initial",
    ]
    original = generate_replay._write_replay_shard
    interrupted = False

    def crash_after_rename(**kwargs):
        nonlocal interrupted
        shard = original(**kwargs)
        if not interrupted:
            interrupted = True
            raise RuntimeError("simulated interruption")
        return shard

    monkeypatch.setattr(generate_replay, "_write_replay_shard", crash_after_rename)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        generate_replay.main(args)

    partial = json.loads(manifest.read_text(encoding="utf-8"))
    assert partial["status"] == "in_progress"
    assert partial["samples"] == 0
    assert partial["shards"] == []
    assert (shard_dir / "replay-00000.jsonl").exists()

    monkeypatch.setattr(generate_replay, "_write_replay_shard", original)
    calls.clear()
    assert generate_replay.main([*args, "--resume"]) == 0

    completed = json.loads(manifest.read_text(encoding="utf-8"))
    assert completed["status"] == "complete"
    assert completed["samples"] == 2
    assert calls == ["crash0", "crash1"]
    assert [shard["path"] for shard in completed["shards"]] == [
        "replay-00000.jsonl",
        "replay-00001.jsonl",
    ]


def test_sharded_replay_resume_rejects_unexpected_untracked_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = _samples("unexpected", 2)
    checkpoint = _checkpoint(tmp_path)
    _patch_generator(monkeypatch, samples)
    manifest = tmp_path / "manifest.json"
    shard_dir = tmp_path / "shards"
    args = [
        "--floorset-lite-root",
        str(tmp_path),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(manifest),
        "--limit",
        "1",
        "--population",
        "2",
        "--shard-dir",
        str(shard_dir),
        "--shard-sample-size",
        "1",
        "--device",
        "cpu",
        "--record-stage",
        "initial",
    ]
    assert generate_replay.main(args) == 0
    unexpected = shard_dir / "replay-00099.jsonl"
    unexpected.write_text("do not delete\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected untracked"):
        generate_replay.main([*args, "--resume"])

    assert unexpected.read_text(encoding="utf-8") == "do not delete\n"


def _patch_generator(
    monkeypatch: pytest.MonkeyPatch,
    samples: list[DataSample],
    *,
    calls: list[str] | None = None,
) -> None:
    monkeypatch.setattr(
        generate_replay,
        "iter_floorset_lite_with_source",
        lambda *_args, **_kwargs: iter((sample, {}) for sample in samples),
    )

    def records_for_sample(**kwargs):
        sample = kwargs["sample"]
        if calls is not None:
            calls.append(sample.sample_id)
        return (_record(sample),)

    monkeypatch.setattr(generate_replay, "_records_for_sample", records_for_sample)


def _samples(prefix: str, count: int) -> list[DataSample]:
    case = synthetic_case(32, device="cpu")
    labels = extract_labels(case, safe_shelf(case), normalized=True)
    return [
        DataSample(f"{prefix}{index}", case, labels)
        for index in range(count)
    ]


def _checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "model.pt"
    save_checkpoint(HCFPModel(ModelConfig(hidden_dim=16)), path, RUNTIME_NORMALIZATION)
    return path


def _record(sample: DataSample) -> ReplayRecord:
    case = sample.case
    base = safe_shelf(case)
    boxes = torch.stack((base, _shifted(base, dx=0.01)), dim=0)
    features = candidate_features(case, boxes, base)
    score = torch.tensor([1.0, 3.0], dtype=torch.float32)
    tiers = torch.tensor([0, 1], dtype=torch.long)
    kinds = ("learned", "constraint")
    geometry_hashes = _candidate_geometry_hashes(boxes)
    row_ids = _candidate_row_ids(
        sample_id=sample.sample_id,
        stage="initial",
        kinds=kinds,
        source_types=kinds,
        geometry_hashes=geometry_hashes,
    )
    return ReplayRecord(
        sample=sample,
        checkpoint_hash="a" * 64,
        candidate_features=features,
        target_score=score,
        candidate_row_ids=row_ids,
        candidate_source_indices=torch.tensor([3, 4], dtype=torch.long),
        candidate_kinds=kinds,
        candidate_source_types=kinds,
        candidate_geometry_sha256=geometry_hashes,
        feasibility_tier=tiers,
        target_rank=_target_rank(score, tiers, row_ids),
        candidate_stage="initial",
        candidate_population=2,
        population_seed=0,
        candidate_geometry=boxes,
        post_bdp_geometry=boxes,
        post_repair_geometry=boxes,
        teacher_delta_xy=torch.zeros((2, case.n, 2), dtype=torch.float32),
        repair_displacement=torch.zeros(2, dtype=torch.float32),
        post_repair_hard_feasible=tiers == 0,
        post_repair_log_uncapped_cost=score,
        post_repair_cap_margin=torch.full((2,), math.log(10.0)) - score,
        boundary_violations=torch.tensor([0, 1], dtype=torch.long),
        grouping_violations=torch.tensor([0, 0], dtype=torch.long),
        mib_violations=torch.tensor([0, 0], dtype=torch.long),
    )


def _shifted(boxes: torch.Tensor, *, dx: float) -> torch.Tensor:
    shifted = boxes.clone()
    shifted[:, 0] += dx
    return shifted
