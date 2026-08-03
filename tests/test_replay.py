from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import json
import math
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from hcfp.analytic import AnalyticConfig, solve_case_with_telemetry
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint, save_checkpoint
from hcfp.data import DataSample, extract_labels, file_sha256, sample_to_payload
from hcfp.dynamics import DynamicsConfig
from hcfp.fallback import safe_shelf
from hcfp.geometry import centers_from_xywh
from hcfp.model import HCFPModel, ModelConfig
from hcfp.profile import synthetic_case
from hcfp.replay import (
    OFFICIAL_TARGET_KIND,
    RANKER_OBJECTIVES,
    ReplayRecord,
    iter_replay,
    official_replay_scores,
    ranker_features_for_record,
    ranker_loss_report,
    ranker_training_schedule,
    record_from_analysis,
    records_from_learned_analysis,
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


def test_v3_ranker_loss_prefers_correct_low_cost_ordering() -> None:
    record = _feature_prediction_record(
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor([0, 1, 2], dtype=torch.long),
    )
    reversed_record = _feature_prediction_record(
        torch.tensor([2.0, 1.0, 0.0]),
        torch.tensor([0, 1, 2], dtype=torch.long),
    )
    model = _feature_ranker_model()

    correct = ranker_loss_report(model, record)
    reversed_report = ranker_loss_report(model, reversed_record)

    assert correct.combined < reversed_report.combined
    assert correct.listwise < reversed_report.listwise


def test_v3_ranker_loss_is_row_permutation_invariant() -> None:
    prediction = torch.tensor([0.25, 1.0, -0.5, 2.0])
    rank = torch.tensor([1, 2, 0, 3], dtype=torch.long)
    cap_margin = torch.tensor([0.5, -0.2, 0.1, -0.4])
    record = _feature_prediction_record(prediction, rank, cap_margin=cap_margin)
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    permuted = replace(
        record,
        candidate_features=record.candidate_features[permutation],
        target_score=record.target_score[permutation],
        target_rank=record.target_rank[permutation],
        post_repair_cap_margin=record.post_repair_cap_margin[permutation],
    )
    model = _feature_ranker_model()

    original = ranker_loss_report(model, record)
    changed = ranker_loss_report(model, permuted)

    assert original.combined == pytest.approx(float(changed.combined))
    assert original.listwise == pytest.approx(float(changed.listwise))
    assert original.listwise_weight_max > original.listwise_weight_mean > 0.0


def test_v2_ranker_loss_keeps_pointwise_fallback() -> None:
    record = replace(
        _feature_prediction_record(
            torch.tensor([0.0, 1.0]),
            torch.tensor([0, 1], dtype=torch.long),
        ),
        target_rank=None,
        post_repair_cap_margin=None,
    )

    report = ranker_loss_report(_feature_ranker_model(), record)

    assert report.listwise == pytest.approx(0.0)
    assert report.combined == pytest.approx(float(report.pointwise))


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
    v3_record = _v3_record()
    legacy_path = tmp_path / "legacy-v2.jsonl"
    v3_path = tmp_path / "replay-v3.jsonl"

    assert write_replay([record], legacy_path) == 1
    assert write_replay_v3([v3_record], v3_path) == 1
    legacy_payload = json.loads(legacy_path.read_text(encoding="utf-8"))
    v3_payload = json.loads(v3_path.read_text(encoding="utf-8"))
    loaded = next(iter_replay(v3_path))

    assert legacy_payload["schema_version"] == 2
    assert "candidate_row_ids" not in legacy_payload
    assert v3_payload["schema_version"] == 3
    assert loaded.target_kind == OFFICIAL_TARGET_KIND
    assert loaded.candidate_row_ids == v3_record.candidate_row_ids
    assert torch.equal(loaded.candidate_source_indices, v3_record.candidate_source_indices)
    assert loaded.candidate_kinds == v3_record.candidate_kinds
    assert loaded.candidate_source_types == v3_record.candidate_source_types
    assert loaded.candidate_geometry_sha256 == v3_record.candidate_geometry_sha256
    assert loaded.population_seed == 0
    assert torch.equal(loaded.feasibility_tier, v3_record.feasibility_tier)
    assert torch.equal(loaded.target_rank, v3_record.target_rank)


def test_v3_replay_derives_repair_aware_ranker_feature_view() -> None:
    record = _v3_record()

    stored = ranker_features_for_record(
        record,
        expected_dim=8,
        expected_version="stored_candidate_features_v1",
    )
    repair_aware = ranker_features_for_record(
        record,
        expected_dim=26,
        expected_version="repair_aware_ranker_features_v4_device_parity",
    )

    assert torch.equal(stored, record.candidate_features)
    assert repair_aware.shape == (len(record.target_score), 26)
    assert torch.isfinite(repair_aware).all()
    torch.testing.assert_close(repair_aware[:, 18:21], torch.eye(3))


def test_repair_aware_features_do_not_read_post_repair_targets() -> None:
    record = _v3_record()
    changed_targets = replace(
        record,
        feasibility_tier=torch.flip(record.feasibility_tier, dims=(0,)),
        post_repair_hard_feasible=~record.post_repair_hard_feasible,
        post_repair_cap_margin=record.post_repair_cap_margin + 10.0,
    )

    original = ranker_features_for_record(
        record,
        expected_dim=26,
        expected_version="repair_aware_ranker_features_v4_device_parity",
    )
    changed = ranker_features_for_record(
        changed_targets,
        expected_dim=26,
        expected_version="repair_aware_ranker_features_v4_device_parity",
    )

    torch.testing.assert_close(changed, original, rtol=0.0, atol=0.0)


def test_ranker_loss_reuses_prepared_repair_aware_features() -> None:
    record = _v3_record()
    prepared = ranker_features_for_record(
        record,
        expected_dim=26,
        expected_version="repair_aware_ranker_features_v4_device_parity",
    )
    prepared_record = replace(
        record,
        candidate_features=prepared,
        candidate_geometry=None,
        post_bdp_geometry=None,
    )
    model = HCFPModel(
        ModelConfig(
            hidden_dim=16,
            encoder_layers=1,
            candidate_metric_dim=26,
            ranker_feature_version="repair_aware_ranker_features_v4_device_parity",
        )
    )

    report = ranker_loss_report(model, prepared_record)

    assert torch.isfinite(report.combined)


def test_candidate_only_ranker_loss_skips_scene_encoder() -> None:
    record = _v3_record()
    prepared = ranker_features_for_record(
        record,
        expected_dim=26,
        expected_version="repair_aware_ranker_features_v4_device_parity",
    )
    model = HCFPModel(
        ModelConfig(
            hidden_dim=16,
            encoder_layers=1,
            candidate_metric_dim=26,
            ranker_feature_version="repair_aware_ranker_features_v4_device_parity",
            ranker_use_scene_embedding=False,
        )
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("candidate-only ranker must not call the scene encoder")

    model.encoder.forward = fail_if_called

    report = ranker_loss_report(model, replace(record, candidate_features=prepared))

    assert torch.isfinite(report.combined)


def test_v3_row_id_tracks_geometry_not_slot_metadata(tmp_path: Path) -> None:
    record = _v3_record()
    sample, source, analysis = _merged_replay_fixture()
    analysis.analytic.raw_candidates[3, 0, 0] += 0.123
    changed = records_from_learned_analysis(
        sample,
        source,
        "e" * 64,
        analysis,
        analytic_population=2,
        population_seed=7,
    )[0]

    assert changed.candidate_source_indices.tolist() == record.candidate_source_indices.tolist()
    assert changed.candidate_row_ids[0] != record.candidate_row_ids[0]
    assert changed.candidate_row_ids[1] == record.candidate_row_ids[1]
    assert changed.candidate_row_ids[2] == record.candidate_row_ids[2]


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


def test_records_from_learned_analysis_emits_initial_and_post_relax_stages() -> None:
    sample, source, analysis = _merged_replay_fixture()

    initial, post_relax = records_from_learned_analysis(
        sample,
        source,
        "a" * 64,
        analysis,
        analytic_population=2,
        population_seed=17,
    )

    assert (initial.candidate_stage, post_relax.candidate_stage) == ("initial", "post_relax")
    assert initial.candidate_geometry.shape == post_relax.candidate_geometry.shape == (3, sample.case.n, 4)
    assert initial.candidate_source_indices.tolist() == [3, 4, 5]
    assert post_relax.candidate_source_indices.tolist() == [8, 9, 10]
    assert initial.population_seed == post_relax.population_seed == 17
    assert initial.candidate_kinds == ("learned", "constraint", "topology")
    assert post_relax.candidate_kinds == ("learned", "constraint", "topology")
    assert torch.equal(initial.post_repair_log_uncapped_cost, initial.target_score)
    assert torch.equal(post_relax.post_repair_log_uncapped_cost, post_relax.target_score)


def test_records_from_learned_analysis_constraint_repair_and_teacher_delta() -> None:
    sample, source, analysis = _merged_replay_fixture()
    initial, _post_relax = records_from_learned_analysis(
        sample,
        source,
        "b" * 64,
        analysis,
        analytic_population=2,
        population_seed=0,
    )

    assert initial.candidate_kinds[1] == "constraint"
    assert not torch.equal(initial.post_repair_geometry[1], initial.post_bdp_geometry[1])
    expected = centers_from_xywh(initial.post_repair_geometry) - centers_from_xywh(initial.candidate_geometry)
    assert torch.allclose(initial.teacher_delta_xy, expected)
    assert torch.all(initial.repair_displacement >= 0.0)


def test_records_from_learned_analysis_post_relax_hard_negative() -> None:
    sample, source, analysis = _merged_replay_fixture()
    _initial, post_relax = records_from_learned_analysis(
        sample,
        source,
        "c" * 64,
        analysis,
        analytic_population=2,
        population_seed=0,
    )

    assert post_relax.post_repair_hard_feasible.dtype == torch.bool
    assert bool((~post_relax.post_repair_hard_feasible).any())
    assert bool((post_relax.feasibility_tier > 0).any())


def test_cli_generate_replay_emits_two_v3_stage_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample, source, analysis = _merged_replay_fixture()
    checkpoint = tmp_path / "model.pt"
    checkpoint_hash = save_checkpoint(HCFPModel(ModelConfig(hidden_dim=16)), checkpoint, RUNTIME_NORMALIZATION)
    analysis = SimpleNamespace(
        result=SimpleNamespace(
            used_checkpoint=True,
            checkpoint_hash=checkpoint_hash,
            failure_reason=None,
            candidate_count=5,
            topology_seed_count=1,
            constraint_seed_count=1,
        ),
        analytic=analysis.analytic,
    )
    monkeypatch.setattr(generate_replay, "iter_floorset_lite_with_source", lambda *_args, **_kwargs: iter(((sample, source),)))
    monkeypatch.setattr(generate_replay, "analyze_case_with_checkpoint", lambda *_args, **_kwargs: analysis)
    output = tmp_path / "replay.jsonl"

    assert generate_replay.main(
        [
            "--floorset-lite-root",
            str(tmp_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--limit",
            "1",
            "--population",
            "2",
            "--dynamics-steps",
            "1",
            "--projection-steps",
            "2",
            "--flow-seed",
            "123",
            "--topology-seeds",
            "1",
            "--constraint-seeds",
            "1",
            "--device",
            "cpu",
        ]
    ) == 0

    loaded = list(iter_replay(output))
    report = json.loads(Path(f"{output}.report.json").read_text(encoding="utf-8"))
    assert [record.candidate_stage for record in loaded] == ["initial", "post_relax"]
    assert report["schema_version"] == 3
    assert report["stages"] == {"initial": 1, "post_relax": 1}
    assert report["dataset"]["samples"] == 1
    assert report["mid_flow_state_recorded"] is False


def test_cli_generate_replay_skips_excluded_sample_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample, source, analysis = _merged_replay_fixture()
    excluded_sample = replace(sample, sample_id="excluded")
    fresh_sample = replace(sample, sample_id="fresh")
    checkpoint = tmp_path / "model.pt"
    checkpoint_hash = save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16)),
        checkpoint,
        RUNTIME_NORMALIZATION,
    )
    merged = SimpleNamespace(
        result=SimpleNamespace(
            used_checkpoint=True,
            checkpoint_hash=checkpoint_hash,
            failure_reason=None,
            candidate_count=5,
            topology_seed_count=1,
            constraint_seed_count=1,
        ),
        analytic=analysis.analytic,
    )
    monkeypatch.setattr(
        generate_replay,
        "iter_floorset_lite_with_source",
        lambda *_args, **_kwargs: iter(((excluded_sample, source), (fresh_sample, source))),
    )
    monkeypatch.setattr(
        generate_replay,
        "iter_replay",
        lambda _path: iter((SimpleNamespace(sample=excluded_sample),)),
    )
    monkeypatch.setattr(
        generate_replay,
        "analyze_case_with_checkpoint",
        lambda *_args, **_kwargs: merged,
    )
    exclusion = tmp_path / "exclude.jsonl"
    exclusion.write_text("excluded\n", encoding="utf-8")
    output = tmp_path / "replay.jsonl"

    assert generate_replay.main(
        [
            "--floorset-lite-root",
            str(tmp_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--exclude-replay",
            str(exclusion),
            "--record-stage",
            "initial",
            "--limit",
            "1",
            "--population",
            "2",
            "--dynamics-steps",
            "1",
            "--projection-steps",
            "2",
            "--topology-seeds",
            "1",
            "--constraint-seeds",
            "1",
            "--device",
            "cpu",
        ]
    ) == 0

    loaded = list(iter_replay(output))
    report = json.loads(Path(f"{output}.report.json").read_text(encoding="utf-8"))
    assert [(record.sample.sample_id, record.candidate_stage) for record in loaded] == [
        ("fresh", "initial")
    ]
    assert report["dataset"]["exclusions"]["sample_count"] == 1
    assert report["dataset"]["exclusions"]["skipped"] == 1
    assert report["dataset"]["exclusions"]["replays"][0]["sha256"] == file_sha256(exclusion)


def test_v3_target_order_is_tie_stable_and_row_permutation_safe(tmp_path: Path) -> None:
    payload = _v3_payload(tmp_path)
    count = len(payload["target_score"])
    payload["target_score"] = [1.0] * count
    payload["post_repair_log_uncapped_cost"] = [1.0] * count
    payload["post_repair_cap_margin"] = [math.log(10.0) - 1.0] * count
    payload["post_repair_hard_feasible"] = [True] * count
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
            "candidate_geometry": payload["candidate_geometry"][index],
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
        "candidate_geometry",
        "post_bdp_geometry",
        "post_repair_geometry",
        "teacher_delta_xy",
        "repair_displacement",
        "post_repair_hard_feasible",
        "post_repair_log_uncapped_cost",
        "post_repair_cap_margin",
        "boundary_violations",
        "grouping_violations",
        "mib_violations",
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
            "candidate_geometry": payload["candidate_geometry"][index],
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
        ("post_repair_log_uncapped_cost", lambda value: [value[0] + 1.0, *value[1:]], "target_score"),
        ("post_repair_cap_margin", lambda value: [value[0] + 1.0, *value[1:]], "cap_margin"),
        ("post_repair_hard_feasible", lambda value: [not value[0], *value[1:]], "hard_feasible"),
        ("teacher_delta_xy", lambda value: [[[entry + 1.0 for entry in pair] for pair in value[0]], *value[1:]], "teacher_delta"),
        ("repair_displacement", lambda value: [value[0] + 1.0, *value[1:]], "repair_displacement"),
        ("boundary_violations", lambda value: [*value[:1], -1, *value[2:]], "boundary_violations"),
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
        ("boundary_violations", [0, 1.5, 2]),
        ("grouping_violations", [0, True, 2]),
        ("mib_violations", [0, 1, False]),
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


def test_v3_replay_rejects_non_exact_boolean_hard_feasible(tmp_path: Path) -> None:
    payload = _v3_payload(tmp_path)
    payload["post_repair_hard_feasible"] = [1, False, True]
    path = tmp_path / "bad-bool.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact booleans"):
        next(iter_replay(path))


def test_records_from_learned_analysis_fails_on_tampered_provenance_hash() -> None:
    sample, source, analysis = _merged_replay_fixture()
    bad = dict(analysis.analytic.incumbent_snapshot)
    records = [dict(record) for record in bad["constraint_seed_provenance"]]
    records[0]["candidate_sha256"] = "0" * 64
    bad["constraint_seed_provenance"] = tuple(records)
    analysis.analytic.incumbent_snapshot = bad

    with pytest.raises(ValueError, match="candidate hash mismatch"):
        records_from_learned_analysis(
            sample,
            source,
            "a" * 64,
            analysis,
            analytic_population=2,
            population_seed=0,
        )


@pytest.mark.parametrize("missing", ("count", "source", "record"))
def test_records_from_learned_analysis_fails_on_missing_seed_provenance(missing: str) -> None:
    sample, source, analysis = _merged_replay_fixture()
    bad = dict(analysis.analytic.incumbent_snapshot)
    if missing == "count":
        bad.pop("constraint_seed_count")
    elif missing == "source":
        bad["constraint_seed_sources"] = tuple(bad["constraint_seed_sources"][:-1])
    else:
        bad["constraint_seed_provenance"] = tuple(bad["constraint_seed_provenance"][:-1])
    analysis.analytic.incumbent_snapshot = bad

    with pytest.raises(ValueError, match="constraint_seed"):
        records_from_learned_analysis(
            sample,
            source,
            "a" * 64,
            analysis,
            analytic_population=2,
            population_seed=0,
        )


def test_v3_replay_rejects_candidate_features_not_derived_from_geometry(tmp_path: Path) -> None:
    payload = _v3_payload(tmp_path)
    payload["candidate_features"][0][0] += 1.0
    path = tmp_path / "bad-features.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="candidate_features"):
        next(iter_replay(path))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("candidate_source_indices", torch.tensor([3.0, 4.0, 5.0]), "exact integers"),
        ("feasibility_tier", torch.tensor([0.0, 1.0, 2.0]), "exact integers"),
        ("target_rank", torch.tensor([0.0, 1.0, 2.0]), "exact integers"),
        ("post_repair_hard_feasible", torch.tensor([1, 0, 1]), "exact booleans"),
        ("candidate_population", 3.0, "must be an integer"),
    ),
)
def test_v3_writer_rejects_silent_type_coercion(
    tmp_path: Path,
    field: str,
    value,
    message: str,
) -> None:
    record = replace(_v3_record(), **{field: value})

    with pytest.raises(ValueError, match=message):
        write_replay_v3([record], tmp_path / f"bad-writer-{field}.jsonl")


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

    with pytest.raises(ValueError, match="candidate_geometry"):
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
    sample, source, analysis = _merged_replay_fixture()
    record = records_from_learned_analysis(
        sample,
        source,
        "d" * 64,
        analysis,
        analytic_population=2,
        population_seed=0,
    )[0]
    return record


def _merged_replay_fixture():
    case = synthetic_case(32, device="cpu")
    sample = DataSample("merged-v3", case, extract_labels(case, safe_shelf(case), normalized=True))
    config = AnalyticConfig(
        dynamics=DynamicsConfig(population=5, steps=1),
        projection_iterations=2,
        direction_beam=1,
    )
    analytic = solve_case_with_telemetry(case, config)
    raw = analytic.raw_candidates.clone()
    raw[9, 0, 0] += 0.05
    source = {
        "normalized": False,
        "area_targets": case.area,
        "b2b_weight": case.b2b_weight,
        "pins": case.pins,
        "p2b_edges": case.p2b_edges,
        "boundary_bits": case.boundary_bits,
        "group_membership": case.group_membership,
        "mib_membership": case.mib_membership,
    }
    snapshot = {
        "constraint_seed_count": 1,
        "constraint_seed_sources": ("candidate_4", "candidate_9"),
        "constraint_seed_provenance": (
            {
                "source": "candidate_4",
                "candidate_type": "constraint",
                "stage": "initial",
                "candidate_sha256": _lineage_sha256(raw[4]),
                "details": {"boundary": {"placed": [0]}},
            },
            {
                "source": "candidate_9",
                "candidate_type": "constraint",
                "stage": "post_relax",
                "parent_candidate_sha256": _lineage_sha256(raw[4]),
                "candidate_sha256": _lineage_sha256(raw[9]),
                "transform": (
                    "identity"
                    if _lineage_sha256(raw[9]) == _lineage_sha256(raw[4])
                    else "population_relaxation"
                ),
                "details": {"boundary": {"placed": [0]}},
            },
        ),
        "topology_seed_count": 1,
        "topology_seed_sources": ("candidate_5", "candidate_10"),
        "topology_seed_provenance": (
            {
                "source": "candidate_5",
                "candidate_type": "topology",
                "stage": "initial",
                "candidate_sha256": _lineage_sha256(raw[5]),
            },
            {
                "source": "candidate_10",
                "candidate_type": "topology",
                "stage": "post_relax",
                "parent_candidate_sha256": _lineage_sha256(raw[5]),
                "candidate_sha256": _lineage_sha256(raw[10]),
                "transform": (
                    "identity"
                    if _lineage_sha256(raw[10]) == _lineage_sha256(raw[5])
                    else "population_relaxation"
                ),
            },
        ),
    }
    merged = SimpleNamespace(
        result=SimpleNamespace(
            candidate_count=5,
            topology_seed_count=1,
            constraint_seed_count=1,
        ),
        analytic=SimpleNamespace(
            raw_candidates=raw,
            projected_candidates=analytic.projected_candidates.clone(),
            telemetry=analytic.telemetry,
            incumbent_snapshot=snapshot,
        ),
    )
    return sample, source, merged


def _lineage_sha256(tensor: torch.Tensor) -> str:
    raw = torch.as_tensor(tensor).detach().cpu().contiguous().view(torch.uint8)
    import hashlib

    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def _feature_prediction_record(
    prediction: torch.Tensor,
    target_rank: torch.Tensor,
    *,
    cap_margin: torch.Tensor | None = None,
) -> ReplayRecord:
    sample, _source, _analysis = _merged_replay_fixture()
    count = int(prediction.numel())
    features = torch.zeros((count, 8), dtype=torch.float32)
    features[:, 0] = prediction
    return ReplayRecord(
        sample,
        "a" * 64,
        features,
        target_rank.to(dtype=torch.float32),
        candidate_row_ids=tuple(f"row-{index}" for index in range(count)),
        candidate_source_indices=torch.arange(count, dtype=torch.long),
        candidate_kinds=tuple("learned" for _ in range(count)),
        candidate_source_types=tuple("learned" for _ in range(count)),
        candidate_geometry_sha256=tuple("b" * 64 for _ in range(count)),
        feasibility_tier=torch.zeros(count, dtype=torch.long),
        target_rank=target_rank,
        candidate_stage="test",
        candidate_population=count,
        population_seed=0,
        post_repair_cap_margin=cap_margin,
    )


def _feature_ranker_model() -> HCFPModel:
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1))

    def feature_ranker(_embedding, _population, features):
        return features[:, 0]

    model.ranker.forward = feature_ranker  # type: ignore[method-assign]
    return model


def _sampling_record(
    sample_id: str,
    *,
    block_count: int,
    stage: str = "initial",
    tiers: tuple[int, ...] = (0, 0, 0),
    cap_margin: tuple[float, ...] = (1.0, 1.0, 1.0),
    hard_feasible: tuple[bool, ...] = (True, True, True),
) -> ReplayRecord:
    case = synthetic_case(block_count, device="cpu")
    sample = DataSample(sample_id, case, extract_labels(case, safe_shelf(case), normalized=True))
    count = len(tiers)
    features = torch.arange(count * 8, dtype=torch.float32).reshape(count, 8)
    return ReplayRecord(
        sample,
        "a" * 64,
        features,
        torch.arange(count, dtype=torch.float32),
        candidate_row_ids=tuple(f"{sample_id}:row-{index}" for index in range(count)),
        feasibility_tier=torch.tensor(tiers, dtype=torch.long),
        target_rank=torch.arange(count, dtype=torch.long),
        candidate_stage=stage,
        post_repair_hard_feasible=torch.tensor(hard_feasible, dtype=torch.bool),
        post_repair_cap_margin=torch.tensor(cap_margin, dtype=torch.float32),
    )


def test_ranker_sampling_legacy_matches_modulo_order() -> None:
    records = [_sampling_record(f"r{index}", block_count=32) for index in range(3)]

    plan = ranker_training_schedule(records, steps=8, seed=17, preset="legacy")

    assert plan.indices == (0, 1, 2, 0, 1, 2, 0, 1)
    assert plan.metadata["preset"] == "legacy"
    assert plan.metadata["epoch_shuffle"] is False


def test_ranker_sampling_epoch_shuffle_is_deterministic_per_epoch() -> None:
    records = [_sampling_record(f"r{index}", block_count=32) for index in range(5)]

    first = ranker_training_schedule(records, steps=12, seed=17, preset="epoch_shuffle")
    second = ranker_training_schedule(records, steps=12, seed=17, preset="epoch_shuffle")
    changed = ranker_training_schedule(records, steps=12, seed=18, preset="epoch_shuffle")

    assert first.indices == second.indices
    assert first.indices != changed.indices
    assert sorted(first.indices[:5]) == list(range(5))
    assert sorted(first.indices[5:10]) == list(range(5))
    assert first.metadata["epoch_shuffle"] is True


def test_q5_dagger_sampling_uses_overlapping_pools_and_requested_quota() -> None:
    records = [
        _sampling_record(
            "overlap",
            block_count=120,
            tiers=(0, 1, 0),
            cap_margin=(0.1, 1.0, 1.0),
            hard_feasible=(True, False, True),
        ),
        _sampling_record("hard", block_count=32, tiers=(1, 1, 1), hard_feasible=(False, False, False)),
        _sampling_record(
            "near",
            block_count=64,
            cap_margin=(-0.2, -1.0, -1.0),
        ),
        _sampling_record("positive", block_count=96, cap_margin=(0.5, 1.0, 1.0)),
    ]

    plan = ranker_training_schedule(records, steps=10, seed=17, preset="q5_dagger_v1")

    assert len(plan.indices) == 10
    assert plan.metadata["bucket_targets"] == {
        "hard_negative": 4,
        "near_cap": 3,
        "large_106_120": 2,
        "successful_positive": 1,
    }
    assert plan.metadata["bucket_draws"] == plan.metadata["bucket_targets"]
    eligible = plan.metadata["bucket_eligible"]
    assert eligible == {
        "hard_negative": 2,
        "near_cap": 2,
        "large_106_120": 1,
        "successful_positive": 2,
    }
    assert sum(eligible.values()) > len(records)
    assert plan.metadata["fallback_draws"] == 0


def test_q5_dagger_sampling_reports_deterministic_fallback_shortfall() -> None:
    records = [
        _sampling_record(
            "plain-a",
            block_count=32,
            tiers=(1, 1, 1),
            cap_margin=(1.0, 1.0, 1.0),
            hard_feasible=(False, False, False),
        ),
        _sampling_record(
            "plain-b",
            block_count=64,
            tiers=(1, 1, 1),
            cap_margin=(1.0, 1.0, 1.0),
            hard_feasible=(False, False, False),
        ),
    ]

    first = ranker_training_schedule(records, steps=10, seed=17, preset="q5_dagger_v1")
    second = ranker_training_schedule(records, steps=10, seed=17, preset="q5_dagger_v1")

    assert first.indices == second.indices
    assert first.metadata["bucket_eligible"] == {
        "hard_negative": 2,
        "near_cap": 0,
        "large_106_120": 0,
        "successful_positive": 0,
    }
    assert first.metadata["bucket_shortfall"] == {
        "hard_negative": 0,
        "near_cap": 3,
        "large_106_120": 2,
        "successful_positive": 1,
    }
    assert first.metadata["fallback_draws"] == 6
    assert len(first.indices) == 10


def test_ranker_sampler_target_telemetry_does_not_change_feature_view() -> None:
    base = _sampling_record("same", block_count=120, cap_margin=(1.0, 1.0, 1.0))
    changed = replace(
        base,
        feasibility_tier=torch.tensor([1, 1, 1], dtype=torch.long),
        post_repair_cap_margin=torch.tensor([0.0, -0.1, 0.2], dtype=torch.float32),
    )

    base_features = ranker_features_for_record(
        base,
        expected_dim=8,
        expected_version="stored_candidate_features_v1",
    )
    changed_features = ranker_features_for_record(
        changed,
        expected_dim=8,
        expected_version="stored_candidate_features_v1",
    )
    base_schedule = ranker_training_schedule([base], steps=4, seed=17, preset="q5_dagger_v1")
    changed_schedule = ranker_training_schedule([changed], steps=4, seed=17, preset="q5_dagger_v1")

    assert torch.equal(base_features, changed_features)
    assert base_schedule.metadata["bucket_eligible"] != changed_schedule.metadata["bucket_eligible"]


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


def test_ranker_training_upgrades_features_but_keeps_runtime_shadowed(tmp_path: Path) -> None:
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
    source_record = _v3_record()
    scores = source_record.target_rank.to(dtype=torch.float32)
    training_record = replace(
        source_record,
        checkpoint_hash=source_hash,
        target_score=scores,
        post_repair_log_uncapped_cost=scores,
        post_repair_cap_margin=math.log(10.0) - scores,
    )
    write_replay_v3([training_record], replay)
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
            "--stage",
            "initial",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    loaded, metadata = load_checkpoint(ranked, expected_normalization=RUNTIME_NORMALIZATION)
    report = json.loads(Path(f"{ranked}.training.json").read_text(encoding="utf-8"))
    assert metadata["capabilities"] == {"flow": True, "ranker": False}
    assert metadata["trained_heads"] == ["encoder", "flow", "ranker"]
    assert metadata["training_objective_version"] == "ranker_post_repair_listwise_v3_feasibility_shadow"
    assert metadata["parent_state_hash"] == source_hash
    assert loaded.config.candidate_metric_dim == 26
    assert report["listwise_records"] == 1
    assert report["candidate_stage_filter"] == "initial"
    assert report["replays"] == [
        {
            "path": str(replay),
            "sha256": file_sha256(replay),
            "records": 1,
            "selected_records": 1,
            "samples": 1,
        }
    ]
    assert report["candidate_feature_dim"] == 26
    assert report["ranker_use_scene_embedding"] is False
    assert report["ranker_initialization"] == "reset_from_non_ranker_source"
    assert report["candidate_feature_version"] == "repair_aware_ranker_features_v4_device_parity"
    assert (
        report["candidate_feature_normalization"]["kind"]
        == "global_zscore_constant_identity_v2"
    )
    assert report["candidate_feature_normalization"]["source"] == "training_replay"
    assert len(report["candidate_feature_normalization"]["mean"]) == 26
    assert len(report["candidate_feature_normalization"]["scale"]) == 26
    assert min(report["candidate_feature_normalization"]["scale"]) > 1.0e-6
    assert report["loss_window_records"] == 1
    assert report["first_window_mean_loss"] == report["first_loss"]
    assert report["last_window_mean_loss"] == report["last_loss"]
    assert report["objective_preset"] == "default"
    assert report["sampling"]["preset"] == "legacy"
    assert report["sampling"]["epoch_shuffle"] is False
    assert report["sampling"]["steps"] == 1
    assert report["sampling"]["record_count"] == 1
    assert report["training_objective_version"] == metadata["training_objective_version"]
    assert report["training_objective_weights"] == {
        "name": "ranker_post_repair_listwise_v3_feasibility_shadow",
        "listwise": 1.0,
        "feasibility_order": 0.25,
        "pointwise": 0.05,
        "top_one": 0.0,
    }
    assert metadata["training_objective_weights"] == report["training_objective_weights"]
    assert set(report["last_loss_components"]) == {
        "combined",
        "feasibility_order",
        "listwise",
        "listwise_weight_max",
        "listwise_weight_mean",
        "pointwise",
        "top_one",
    }

    continuation_replay = tmp_path / "ranker-continuation.jsonl"
    write_replay_v3(
        [replace(training_record, checkpoint_hash=metadata["state_hash"])],
        continuation_replay,
    )
    continued = tmp_path / "ranked-continued.pt"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_hcfp_ranker.py",
            str(continuation_replay),
            "--checkpoint",
            str(ranked),
            "--output",
            str(continued),
            "--steps",
            "1",
            "--device",
            "cpu",
            "--stage",
            "initial",
            "--sampling-preset",
            "epoch_shuffle",
            "--sampling-seed",
            "7",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    _, continued_metadata = load_checkpoint(
        continued,
        expected_normalization=RUNTIME_NORMALIZATION,
    )
    continued_report = json.loads(
        Path(f"{continued}.training.json").read_text(encoding="utf-8")
    )
    assert continued_metadata["parent_state_hash"] == metadata["state_hash"]
    assert continued_report["ranker_initialization"] == "continued_from_source_checkpoint"
    assert continued_report["sampling"]["preset"] == "epoch_shuffle"
    assert continued_report["sampling"]["seed"] == 7
    assert continued_report["sampling"]["epoch_shuffle"] is True
    assert continued_report["candidate_feature_normalization"]["source"] == "source_checkpoint"
    assert (
        continued_report["candidate_feature_normalization"]["mean"]
        == report["candidate_feature_normalization"]["mean"]
    )
    assert (
        continued_report["candidate_feature_normalization"]["scale"]
        == report["candidate_feature_normalization"]["scale"]
    )

    mismatch = tmp_path / "ranked-mismatch.pt"
    mismatched = subprocess.run(
        [
            sys.executable,
            "scripts/train_hcfp_ranker.py",
            str(continuation_replay),
            "--checkpoint",
            str(ranked),
            "--output",
            str(mismatch),
            "--steps",
            "1",
            "--device",
            "cpu",
            "--stage",
            "initial",
            "--objective-preset",
            "v4b",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    assert mismatched.returncode != 0
    assert "same training objective" in mismatched.stderr


def test_v4b_ranker_objective_changes_only_requested_weights() -> None:
    record = _v3_record()
    model = _feature_ranker_model()

    default = ranker_loss_report(model, record, objective=RANKER_OBJECTIVES["default"])
    v4b = ranker_loss_report(model, record, objective=RANKER_OBJECTIVES["v4b"])

    expected = v4b.listwise + 0.50 * v4b.feasibility_order
    assert v4b.combined.detach().item() == pytest.approx(float(expected.detach()))
    assert v4b.pointwise.detach().item() == pytest.approx(default.pointwise.detach().item())
    assert v4b.top_one.detach().item() == pytest.approx(default.top_one.detach().item())
    assert v4b.combined.detach().item() != pytest.approx(default.combined.detach().item())


@pytest.mark.parametrize("bad_weight", (-1.0, float("inf"), float("nan")))
def test_ranker_objective_rejects_invalid_weights(bad_weight: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        type(RANKER_OBJECTIVES["default"])(
            name="bad",
            listwise=bad_weight,
            feasibility_order=0.0,
            pointwise=0.0,
            top_one=0.0,
        )


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
