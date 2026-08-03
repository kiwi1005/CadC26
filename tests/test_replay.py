from __future__ import annotations

import importlib.util
from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from hcfp.analytic import AnalyticConfig, solve_case_with_telemetry
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint, save_checkpoint
from hcfp.data import DataSample, extract_labels, sample_to_payload
from hcfp.dynamics import DynamicsConfig
from hcfp.fallback import safe_shelf
from hcfp.model import HCFPModel, ModelConfig
from hcfp.profile import synthetic_case
from hcfp.replay import (
    OFFICIAL_TARGET_KIND,
    iter_replay,
    official_replay_scores,
    record_from_analysis,
    train_ranker_steps,
    write_replay,
    write_replay_v3,
)


ROOT = Path(__file__).resolve().parents[1]
REPLAY_SCRIPT = ROOT / "scripts/generate_hcfp_replay.py"
REPLAY_SPEC = importlib.util.spec_from_file_location("generate_hcfp_replay_test", REPLAY_SCRIPT)
assert REPLAY_SPEC is not None and REPLAY_SPEC.loader is not None
generate_replay = importlib.util.module_from_spec(REPLAY_SPEC)
REPLAY_SPEC.loader.exec_module(generate_replay)
ACTIVATION_REPLAY_SCRIPT = ROOT / "scripts/generate_hcfp_activation_replay.py"
ACTIVATION_REPLAY_SPEC = importlib.util.spec_from_file_location(
    "generate_hcfp_activation_replay_test",
    ACTIVATION_REPLAY_SCRIPT,
)
assert ACTIVATION_REPLAY_SPEC is not None and ACTIVATION_REPLAY_SPEC.loader is not None
generate_activation_replay = importlib.util.module_from_spec(ACTIVATION_REPLAY_SPEC)
ACTIVATION_REPLAY_SPEC.loader.exec_module(generate_activation_replay)


def test_replay_roundtrip_and_ranker_update(tmp_path: Path) -> None:
    case = synthetic_case(32, device="cpu")
    sample = DataSample("train-0", case, extract_labels(case, safe_shelf(case), normalized=True))
    config = AnalyticConfig(
        dynamics=DynamicsConfig(population=2, steps=0),
        projection_iterations=2,
        direction_beam=1,
    )
    analysis = solve_case_with_telemetry(case, config)
    record = record_from_analysis(
        sample,
        "a" * 64,
        analysis.raw_candidates,
        analysis.telemetry,
        population=2,
    )
    path = tmp_path / "replay.jsonl"
    assert write_replay([record], path) == 1
    loaded = list(iter_replay(path))

    assert len(loaded) == 1
    assert loaded[0].candidate_features.shape == (2, 8)
    assert loaded[0].target_score.shape == (2,)
    assert loaded[0].target_kind == OFFICIAL_TARGET_KIND
    assert torch.isfinite(record.target_score).all()

    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1))
    before = {name: value.detach().clone() for name, value in model.ranker.named_parameters()}
    optimizer = torch.optim.AdamW(model.ranker.parameters(), lr=1.0e-3)
    history = train_ranker_steps(model, loaded, optimizer, steps=2)

    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(value)) for value in history)
    assert any(not torch.equal(before[name], value.detach()) for name, value in model.ranker.named_parameters())


def test_v3_replay_roundtrip_adds_stable_candidate_provenance(tmp_path: Path) -> None:
    case = synthetic_case(32, device="cpu")
    sample = DataSample("train-v3", case, extract_labels(case, safe_shelf(case), normalized=True))
    config = AnalyticConfig(
        dynamics=DynamicsConfig(population=3, steps=0),
        projection_iterations=2,
        direction_beam=1,
    )
    analysis = solve_case_with_telemetry(case, config)
    record = record_from_analysis(
        sample,
        "c" * 64,
        analysis.raw_candidates,
        analysis.telemetry,
        population=3,
    )
    legacy_path = tmp_path / "legacy-v2.jsonl"
    v3_path = tmp_path / "replay-v3.jsonl"

    assert write_replay([record], legacy_path) == 1
    assert write_replay_v3([record], v3_path) == 1
    legacy_payload = json.loads(legacy_path.read_text(encoding="utf-8"))
    v3_payload = json.loads(v3_path.read_text(encoding="utf-8"))
    loaded = next(iter_replay(v3_path))

    assert legacy_payload["schema_version"] == 2
    assert "candidate_row_ids" not in legacy_payload
    assert v3_payload["schema_version"] == 3
    assert loaded.target_kind == OFFICIAL_TARGET_KIND
    assert loaded.candidate_row_ids == record.candidate_row_ids
    assert torch.equal(loaded.candidate_source_indices, record.candidate_source_indices)
    assert loaded.candidate_kinds == ("learned", "learned", "learned")
    assert loaded.candidate_source_types == loaded.candidate_kinds
    assert loaded.candidate_geometry_sha256 == record.candidate_geometry_sha256
    assert loaded.population_seed == 0
    assert torch.equal(loaded.feasibility_tier, record.feasibility_tier)
    assert torch.equal(loaded.target_rank, record.target_rank)


def test_v3_row_id_tracks_geometry_not_slot_metadata(tmp_path: Path) -> None:
    record = _v3_record()
    # Build a second record from the same analysis path by perturbing one raw candidate slot.
    case = record.sample.case
    config = AnalyticConfig(
        dynamics=DynamicsConfig(population=3, steps=0),
        projection_iterations=2,
        direction_beam=1,
    )
    analysis = solve_case_with_telemetry(case, config)
    raw = analysis.raw_candidates.clone()
    raw[4, 0] = raw[4, 0] + 0.123
    changed = record_from_analysis(
        record.sample,
        "e" * 64,
        raw,
        analysis.telemetry,
        population=3,
        population_seed=7,
    )

    assert changed.candidate_source_indices.tolist() == record.candidate_source_indices.tolist()
    assert changed.candidate_row_ids[0] != record.candidate_row_ids[0]
    assert changed.candidate_row_ids[1:] == record.candidate_row_ids[1:]


def test_v3_row_id_ignores_checkpoint_source_index_population_seed(tmp_path: Path) -> None:
    payload = _v3_payload(tmp_path)
    original_ids = tuple(payload["candidate_row_ids"])
    payload["checkpoint_hash"] = "e" * 64
    payload["candidate_source_indices"] = [101, 102, 103]
    payload["candidate_population"] = 99
    payload["population_seed"] = 12345
    path = tmp_path / "audit-metadata-changed.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    loaded = next(iter_replay(path))
    assert loaded.candidate_row_ids == original_ids
    assert loaded.candidate_source_indices.tolist() == [101, 102, 103]
    assert loaded.candidate_population == 99
    assert loaded.population_seed == 12345


def test_v3_target_order_is_tie_stable_and_row_permutation_safe(tmp_path: Path) -> None:
    payload = _v3_payload(tmp_path)
    count = len(payload["target_score"])
    payload["target_score"] = [1.0] * count
    payload["feasibility_tier"] = [0] * count
    ordered_ids = sorted(range(count), key=lambda index: payload["candidate_row_ids"][index])
    ranks = [0] * count
    for rank, row in enumerate(ordered_ids):
        ranks[row] = rank
    payload["target_rank"] = ranks
    path = tmp_path / "tie.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    loaded = next(iter_replay(path))
    assert loaded.target_rank.tolist() == ranks
    before_mapping = {
        row_id: {
            "target_rank": payload["target_rank"][index],
            "target_score": payload["target_score"][index],
            "feasibility_tier": payload["feasibility_tier"][index],
            "candidate_source_indices": payload["candidate_source_indices"][index],
            "candidate_kind": payload["candidate_kinds"][index],
            "candidate_source_type": payload["candidate_source_types"][index],
            "candidate_geometry_sha256": payload["candidate_geometry_sha256"][index],
        }
        for index, row_id in enumerate(payload["candidate_row_ids"])
    }

    permutation = [2, 0, 1]
    for key in (
        "candidate_features",
        "target_score",
        "candidate_row_ids",
        "candidate_source_indices",
        "candidate_kinds",
        "candidate_source_types",
        "candidate_geometry_sha256",
        "feasibility_tier",
        "target_rank",
    ):
        payload[key] = [payload[key][index] for index in permutation]
    permuted = tmp_path / "tie-permuted.jsonl"
    permuted.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    loaded_permuted = next(iter_replay(permuted))
    after_mapping = {
        row_id: {
            "target_rank": payload["target_rank"][index],
            "target_score": payload["target_score"][index],
            "feasibility_tier": payload["feasibility_tier"][index],
            "candidate_source_indices": payload["candidate_source_indices"][index],
            "candidate_kind": payload["candidate_kinds"][index],
            "candidate_source_type": payload["candidate_source_types"][index],
            "candidate_geometry_sha256": payload["candidate_geometry_sha256"][index],
        }
        for index, row_id in enumerate(payload["candidate_row_ids"])
    }
    assert loaded_permuted.candidate_row_ids == tuple(payload["candidate_row_ids"])
    assert after_mapping == before_mapping


@pytest.mark.parametrize(
    ("field", "mutate", "message"),
    (
        ("candidate_row_ids", lambda value: [value[0], value[0], value[2]], "unique"),
        ("target_rank", lambda _value: [0, 0, 1], "target_rank"),
        ("candidate_kinds", lambda value: [*value[:1], "bad-kind", *value[2:]], "candidate kind"),
        ("feasibility_tier", lambda value: [*value[:1], 9, *value[2:]], "feasibility tier"),
        ("candidate_source_indices", lambda value: [*value[:1], -1, *value[2:]], "source_indices"),
        ("candidate_source_types", lambda value: [*value[:1], "bad-source", *value[2:]], "source type"),
        ("candidate_geometry_sha256", lambda value: [*value[:1], "not-a-sha", *value[2:]], "geometry_sha256"),
        ("target_score", lambda value: value[:-1], "align"),
        ("target_kind", lambda _value: "wrong_target", "target_kind"),
        ("checkpoint_hash", lambda _value: "not-a-sha", "checkpoint_hash"),
        ("candidate_stage", lambda _value: "", "candidate_stage"),
        ("population_seed", lambda _value: 1.5, "population_seed"),
        ("population_seed", lambda _value: True, "population_seed"),
    ),
)
def test_v3_replay_validation_fails_closed_on_tamper(
    tmp_path: Path,
    field: str,
    mutate,
    message: str,
) -> None:
    payload = _v3_payload(tmp_path)
    payload[field] = mutate(payload[field])
    path = tmp_path / f"bad-{field}.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        next(iter_replay(path))


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("candidate_source_indices", [1.5, 5, 6]),
        ("candidate_source_indices", [True, 5, 6]),
        ("feasibility_tier", [0, 1.5, 2]),
        ("feasibility_tier", [0, False, 2]),
        ("target_rank", [0, 1.5, 2]),
        ("target_rank", [0, True, 2]),
    ),
)
def test_v3_replay_rejects_non_exact_integer_lists(
    tmp_path: Path,
    field: str,
    replacement: list[object],
) -> None:
    payload = _v3_payload(tmp_path)
    payload[field] = replacement
    path = tmp_path / f"bad-int-{field}.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact integers"):
        next(iter_replay(path))


def test_v3_row_id_includes_candidate_source_type(tmp_path: Path) -> None:
    payload = _v3_payload(tmp_path)
    payload["candidate_source_types"] = ["constraint", "constraint", "constraint"]
    path = tmp_path / "source-type-tamper.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="row_ids"):
        next(iter_replay(path))


def test_v3_row_id_includes_candidate_geometry_hash(tmp_path: Path) -> None:
    payload = _v3_payload(tmp_path)
    payload["candidate_geometry_sha256"][1] = "f" * 64
    path = tmp_path / "geometry-hash-tamper.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="row_ids"):
        next(iter_replay(path))


def test_legacy_proxy_replay_remains_readable_but_cannot_train(tmp_path: Path) -> None:
    case = synthetic_case(32, device="cpu")
    sample = DataSample("legacy-0", case, extract_labels(case, safe_shelf(case), normalized=True))
    path = tmp_path / "legacy.jsonl"
    payload = {
        "schema_version": 1,
        "checkpoint_hash": "b" * 64,
        "sample": sample_to_payload(sample),
        "candidate_features": [[0.0] * 8],
        "target_score": [1.0],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    record = next(iter_replay(path))

    assert record.target_kind == "legacy_proxy_v1"
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1))
    optimizer = torch.optim.AdamW(model.ranker.parameters(), lr=1.0e-3)
    with pytest.raises(ValueError, match="official v10 replay targets"):
        train_ranker_steps(model, [record], optimizer, steps=1)


def _v3_payload(tmp_path: Path) -> dict[str, object]:
    record = _v3_record()
    path = tmp_path / "helper-v3.jsonl"
    write_replay_v3([record], path)
    return json.loads(path.read_text(encoding="utf-8"))


def _v3_record():
    case = synthetic_case(32, device="cpu")
    sample = DataSample("helper-v3", case, extract_labels(case, safe_shelf(case), normalized=True))
    config = AnalyticConfig(
        dynamics=DynamicsConfig(population=3, steps=0),
        projection_iterations=2,
        direction_beam=1,
    )
    analysis = solve_case_with_telemetry(case, config)
    record = record_from_analysis(
        sample,
        "d" * 64,
        analysis.raw_candidates,
        analysis.telemetry,
        population=3,
    )
    return record


def test_official_replay_scores_are_lexicographic_without_cost_cap_ties() -> None:
    telemetry = SimpleNamespace(
        hpwl=torch.tensor([5.0, 10.0, 5.0]),
        bbox_area=torch.tensor([1.0, 2.0, 1.0]),
        soft_violation=torch.tensor([0.0, 0.0, 0.0]),
        hard_feasible=torch.tensor([True, True, False]),
        projected_overlap=torch.tensor([0.0, 0.0, 0.5]),
        overlap_components=torch.tensor([0, 0, 1]),
        projection_ok=torch.tensor([True, True, False]),
        projection_displacement=torch.tensor([0.0, 0.0, 2.0]),
    )
    case = SimpleNamespace(scale=10.0)

    scores = official_replay_scores(case, telemetry, baseline_area=100.0, baseline_hpwl=50.0)

    assert float(scores[0]) == pytest.approx(0.0)
    assert float(scores[1]) == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert float(scores[2]) > float(scores[:2].max())


def test_ranker_training_preserves_capabilities_and_declares_ranker(tmp_path: Path) -> None:
    case = synthetic_case(32, device="cpu")
    sample = DataSample("ranker-0", case, extract_labels(case, safe_shelf(case), normalized=True))
    checkpoint = tmp_path / "source.pt"
    source_hash = save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16)),
        checkpoint,
        RUNTIME_NORMALIZATION,
        metadata={
            "capabilities": {"flow": True},
            "trained_heads": ["encoder", "flow"],
            "training_objective_version": "supervised_loss_v1",
        },
    )
    replay = tmp_path / "ranker.jsonl"
    replay.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "checkpoint_hash": source_hash,
                "target_kind": OFFICIAL_TARGET_KIND,
                "sample": sample_to_payload(sample),
                "candidate_features": [[0.0] * 8, [1.0] * 8],
                "target_score": [0.0, 1.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ranked = tmp_path / "ranked.pt"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_hcfp_ranker.py",
            str(replay),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(ranked),
            "--steps",
            "1",
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    _, metadata = load_checkpoint(ranked, expected_normalization=RUNTIME_NORMALIZATION)
    assert metadata["capabilities"] == {"flow": True}
    assert metadata["trained_heads"] == ["encoder", "flow", "ranker"]
    assert metadata["training_objective_version"] == "ranker_official_v10_v1"
    assert metadata["parent_state_hash"] == source_hash


def test_ranker_clis_reject_checkpoint_mismatch_and_split_leakage(tmp_path: Path) -> None:
    case = synthetic_case(32, device="cpu")
    sample = DataSample("guard-0", case, extract_labels(case, safe_shelf(case), normalized=True))
    replay = tmp_path / "guard.jsonl"
    replay.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "checkpoint_hash": "b" * 64,
                "target_kind": OFFICIAL_TARGET_KIND,
                "sample": sample_to_payload(sample),
                "candidate_features": [[0.0] * 8, [1.0] * 8],
                "target_score": [0.0, 1.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "source.pt"
    save_checkpoint(HCFPModel(ModelConfig(hidden_dim=16)), checkpoint, RUNTIME_NORMALIZATION)
    root = Path(__file__).resolve().parents[1]

    train = subprocess.run(
        [
            sys.executable,
            "scripts/train_hcfp_ranker.py",
            str(replay),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "ranked.pt"),
            "--steps",
            "1",
            "--device",
            "cpu",
        ],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert train.returncode != 0
    assert "replay checkpoint hash does not match" in train.stderr

    evaluate = subprocess.run(
        [
            sys.executable,
            "scripts/eval_hcfp_ranker.py",
            "--replay",
            f"first={replay}",
            "--replay",
            f"second={replay}",
            "--checkpoint",
            f"model={checkpoint}",
            "--output",
            str(tmp_path / "report.json"),
            "--device",
            "cpu",
        ],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert evaluate.returncode != 0
    assert "replay sample overlap" in evaluate.stderr


@pytest.mark.parametrize("module", (generate_replay, generate_activation_replay))
def test_replay_generators_reject_negative_collective_steps(module) -> None:
    with pytest.raises(module.argparse.ArgumentTypeError):
        module._non_negative_int("-1")
