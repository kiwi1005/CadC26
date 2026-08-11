from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hcfp.analytic import AnalyticConfig, AnalyticResult, CandidateTelemetry, solve_case_with_telemetry
from hcfp.case import from_official
from hcfp.checkpoint import RUNTIME_NORMALIZATION, _payload_hash, save_checkpoint
from hcfp.dynamics import DynamicsConfig
from hcfp.fallback import safe_shelf
from hcfp.learned import (
    LearnedConfig,
    _attach_ranker_shadow_snapshot,
    _merge_energy_history,
    _tensor_sha256,
    analyze_case_with_checkpoint,
    effective_collective_steps,
    effective_flow_steps,
    effective_tail_topk,
    solve_case_with_checkpoint,
)
from hcfp.model import HCFPModel, ModelConfig
from hcfp.ranker_features import RANKER_FEATURE_DIM, RANKER_FEATURE_VERSION
from hcfp.verify import verify_feasible


FLOW_METADATA = {
    "capabilities": {"flow": True},
    "trained_heads": ["flow"],
    "training_objective_version": "supervised_loss_v1",
}

COLLECTIVE_METADATA = {
    "capabilities": {"collective": True},
    "trained_heads": ["collective"],
    "training_objective_version": "collective_loss_v1",
}

RANKER_METADATA = {
    "capabilities": {"ranker": True},
    "trained_heads": ["ranker"],
    "training_objective_version": "ranker_loss_v1",
}


def _case():
    return from_official(
        4,
        [4.0, 9.0, 16.0, 25.0],
        [[0, 1, 2.0], [1, 2, 3.0]],
        [],
        [],
        [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0.0, 0.0, 2.0, 2.0], [4.0, 0.0, 3.0, 3.0], [-1.0] * 4, [-1.0] * 4],
    )


def _config() -> AnalyticConfig:
    return AnalyticConfig(
        dynamics=DynamicsConfig(population=2, steps=0),
        projection_iterations=4,
        direction_beam=1,
    )


def _source() -> SimpleNamespace:
    return SimpleNamespace(
        block_count=2,
        area_targets=[4.0, 4.0],
        b2b_connectivity=[],
        p2b_connectivity=[],
        pins_pos=[],
        constraints=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        target_positions=None,
    )


def _analysis(*, used_checkpoint: bool = True) -> SimpleNamespace:
    candidates = torch.tensor(
        [
            [[0.0, 0.0, 2.0, 2.0], [3.0, 0.0, 2.0, 2.0]],
            [[20.0, 0.0, 2.0, 2.0], [23.0, 0.0, 2.0, 2.0]],
            [[10.0, 0.0, 2.0, 2.0], [13.0, 0.0, 2.0, 2.0]],
            [[30.0, 0.0, 2.0, 2.0], [33.0, 0.0, 2.0, 2.0]],
        ]
    )
    return SimpleNamespace(
        result=SimpleNamespace(
            selected=candidates[0],
            used_checkpoint=used_checkpoint,
            failure_reason=None if used_checkpoint else "checkpoint unavailable",
        ),
        analytic=SimpleNamespace(
            projected_candidates=candidates,
            telemetry=SimpleNamespace(
                hard_feasible=torch.tensor([True, True, True, False]),
                soft_violation=torch.tensor([0.0, 0.0, 0.0, -1.0]),
                bbox_area=torch.tensor([1.0, 3.0, 2.0, 0.0]),
                hpwl=torch.zeros(4),
            ),
        ),
    )


def _pareto_analysis(
    *,
    analytic_metrics: tuple[float, float, float] = (1.0, 10.0, 10.0),
    analytic_source: str = "candidate_2",
    analytic_exact_source: str = "candidate_2",
    analytic_fast_source: str = "candidate_1",
) -> SimpleNamespace:
    candidates = torch.stack(
        [
            torch.tensor([[x, 0.0, 2.0, 2.0], [x + 3.0, 0.0, 2.0, 2.0]])
            for x in (0.0, 10.0, 20.0, 30.0, 40.0)
        ]
    )
    soft, area, hpwl = analytic_metrics
    return SimpleNamespace(
        result=SimpleNamespace(
            selected=candidates[2],
            used_checkpoint=True,
            failure_reason=None,
            candidate_count=2,
        ),
        analytic=SimpleNamespace(
            projected_candidates=candidates,
            incumbent_snapshot={
                "exact_source": analytic_source,
                "analytic_exact_source": analytic_exact_source,
                "analytic_fast_source": analytic_fast_source,
            },
            telemetry=SimpleNamespace(
                hard_feasible=torch.tensor([True, False, True, True, True]),
                soft_violation=torch.tensor([4.0, soft, 2.0, 3.0, 0.0]),
                bbox_area=torch.tensor([40.0, area, 20.0, 5.0, 1.0]),
                hpwl=torch.tensor([40.0, hpwl, 20.0, 5.0, 1.0]),
            ),
        ),
    )


def _pareto_config() -> AnalyticConfig:
    return AnalyticConfig(
        dynamics=DynamicsConfig(population=1, steps=0),
        projection_iterations=4,
        direction_beam=1,
    )


def _constant_candidates(values: tuple[float, ...]) -> torch.Tensor:
    return torch.stack(
        [
            torch.full((4, 4), value, dtype=torch.float32)
            for value in values
        ]
    )


def _box_candidates(count: int) -> torch.Tensor:
    rows = []
    for index in range(count):
        offset = float(index * 10)
        rows.append(
            torch.tensor(
                [
                    [offset, 0.0, 2.0, 2.0],
                    [offset + 3.0, 0.0, 3.0, 3.0],
                    [offset, 4.0, 4.0, 4.0],
                    [offset + 5.0, 4.0, 5.0, 5.0],
                ],
                dtype=torch.float32,
            )
        )
    return torch.stack(rows)


def _telemetry(count: int) -> CandidateTelemetry:
    zeros = torch.zeros(count, dtype=torch.float32)
    bools = torch.ones(count, dtype=torch.bool)
    xywh = torch.zeros((count, 4, 4), dtype=torch.float32)
    return CandidateTelemetry(
        hard_feasible=bools,
        raw_overlap=zeros,
        projected_overlap=zeros,
        overlap_components=zeros,
        projection_ok=bools,
        projection_active_pairs=zeros,
        hpwl=zeros,
        bbox_area=zeros,
        soft_violation=zeros,
        projection_displacement=zeros,
        projection_failure_reasons=tuple("" for _ in range(count)),
        projection_initial_pairs=zeros,
        projection_final_pairs=zeros,
        projection_component_rebuilds=zeros,
        projection_new_pairs=zeros,
        projection_resets=zeros,
        projection_beam_states=zeros,
        projection_max_component_size=zeros,
        component_proposal_available=torch.zeros(count, dtype=torch.bool),
        component_proposal_xywh=xywh,
        component_proposal_hard_ok=torch.zeros(count, dtype=torch.bool),
        component_proposal_structure_ok=torch.zeros(count, dtype=torch.bool),
        component_proposal_final_pair_count=zeros,
        component_proposal_displacement=zeros,
        component_proposal_rollback_reason=tuple("" for _ in range(count)),
    )


def _shadow_metadata() -> dict[str, object]:
    return {
        "capabilities": {"ranker": False},
        "trained_heads": ["ranker"],
        "training_objective_version": "ranker_post_repair_listwise_v3",
    }


class _ScoreRanker:
    def __init__(self, scores: tuple[float, ...]):
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.calls = 0

    def __call__(
        self,
        _embedding: torch.Tensor,
        population: int,
        features: torch.Tensor,
    ) -> torch.Tensor:
        self.calls += 1
        assert population == len(self.scores)
        assert features.shape == (population, RANKER_FEATURE_DIM)
        return self.scores.to(device=features.device)


def _shadow_model(scores: tuple[float, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        config=ModelConfig(
            hidden_dim=16,
            candidate_metric_dim=RANKER_FEATURE_DIM,
            ranker_feature_version=RANKER_FEATURE_VERSION,
            ranker_use_scene_embedding=False,
        ),
        ranker=_ScoreRanker(scores),
    )


def _shadow_analysis(*, eligible: tuple[bool, ...] = (True, True, True, True)) -> AnalyticResult:
    analytic_count = 2
    learned_count = 4
    count = 1 + analytic_count + learned_count + analytic_count + learned_count
    raw = _box_candidates(count)
    projected = raw.clone()
    projected[3:7] = projected[3:7] + torch.tensor([0.25, 0.0, 0.0, 0.0])
    telemetry = _telemetry(count)
    hard = telemetry.hard_feasible.clone()
    projection_ok = telemetry.projection_ok.clone()
    initial_start = 1 + analytic_count
    for offset, ok in enumerate(eligible):
        hard[initial_start + offset] = bool(ok)
        projection_ok[initial_start + offset] = bool(ok)
    return replace(
        AnalyticResult(
            selected=projected[5].clone(),
            raw_candidates=raw,
            projected_candidates=projected,
            telemetry=telemetry,
            energy_history=torch.zeros((1, 0, 3), dtype=torch.float32),
            projection_status="ok",
            incumbent_snapshot={"exact_source": "candidate_5"},
        ),
        telemetry=replace(
            telemetry,
            hard_feasible=hard,
            projection_ok=projection_ok,
        ),
    )


def _synthetic_result(values: tuple[float, ...]) -> AnalyticResult:
    candidates = _constant_candidates(values)
    return AnalyticResult(
        selected=candidates[0],
        raw_candidates=candidates,
        projected_candidates=candidates.clone(),
        telemetry=_telemetry(len(values)),
        energy_history=torch.zeros((1, 0, 3), dtype=torch.float32),
        projection_status="ok",
        incumbent_snapshot={"exact_source": "fallback", "fast_source": "fallback"},
    )


def _provenance_for_synthetic_merge(
    learned: AnalyticResult,
    *,
    tamper_constraint_initial: bool = False,
    omit_constraint_hash: bool = False,
) -> dict[str, object]:
    constraint_records = []
    for index in range(2):
        digest = _tensor_sha256(learned.raw_candidates[1 + index])
        record = {
            "kind": "combined",
            "topology_seed_index": index,
            "details": {"moves": ()},
        }
        if not (index == 0 and omit_constraint_hash):
            record["candidate_sha256"] = (
                "bad" if index == 0 and tamper_constraint_initial else digest
            )
        constraint_records.append(record)
    topology_records = []
    for index in range(2):
        topology_records.append(
            {
                "order_variant": f"variant_{index}",
                "candidate_sha256": _tensor_sha256(learned.raw_candidates[3 + index]),
            }
        )
    return {
        "topology_seed_attempted": True,
        "topology_seed_count": 2,
        "constraint_seed_count": 2,
        "constraint_seed_records": tuple(constraint_records),
        "topology_seed_orders": tuple(topology_records),
    }


def _merged_synthetic_provenance(
    *,
    tamper_constraint_initial: bool = False,
    omit_constraint_hash: bool = False,
    changed_post_relax: bool = True,
) -> AnalyticResult:
    import hcfp.learned as learned_module

    analytic = _synthetic_result((0.0, 1.0, 2.0))
    post = (110.0, 120.0, 130.0, 140.0) if changed_post_relax else (10.0, 20.0, 30.0, 40.0)
    learned = _synthetic_result((0.0, 10.0, 20.0, 30.0, 40.0, *post))
    return learned_module._merge_tail_analyses(
        _case(),
        analytic,
        learned,
        topology_provenance=_provenance_for_synthetic_merge(
            learned,
            tamper_constraint_initial=tamper_constraint_initial,
            omit_constraint_hash=omit_constraint_hash,
        ),
    )


def _assert_sources_match_raw_hashes(
    snapshot: dict[str, object],
    raw_candidates: torch.Tensor,
    key: str,
) -> None:
    records = tuple(snapshot[key])
    sources_key = key.replace("_provenance", "_sources")
    assert tuple(snapshot[sources_key]) == tuple(record["source"] for record in records)
    for record in records:
        index = int(str(record["source"]).removeprefix("candidate_"))
        assert record["candidate_sha256"] == _tensor_sha256(raw_candidates[index])
        if record["stage"] == "post_relax":
            assert record["parent_candidate_sha256"]
            assert record["transform"] in {"identity", "population_relaxation"}


def _solve_pareto(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    analysis: SimpleNamespace,
    *,
    feasible_x: set[float],
    quality: dict[float, tuple[float, float, float]],
):
    import hcfp.learned as learned

    conversions: list[float] = []

    def to_official(_source, _case, candidate):
        x = float(candidate[0, 0])
        conversions.append(x)
        return [(x, 0.0, 2.0, 2.0), (x + 3.0, 0.0, 2.0, 2.0)]

    monkeypatch.setattr(learned, "analyze_case_with_checkpoint", lambda *_args: analysis)
    monkeypatch.setattr(learned, "to_official_placements", to_official)
    monkeypatch.setattr(learned, "verify_feasible", lambda _source, rows: rows[0][0] in feasible_x)
    monkeypatch.setattr(learned, "_raw_quality", lambda _source, _case, rows: quality[rows[0][0]])
    result = learned.solve(
        _source(),
        checkpoint=tmp_path / "model.pt",
        config=_pareto_config(),
        require_checkpoint=True,
    )
    return result, conversions


def _select_with_ranker_counterfactual(
    monkeypatch: pytest.MonkeyPatch,
    analysis: SimpleNamespace,
    *,
    feasible_x: set[float],
    quality: dict[float, tuple[float, float, float]],
    experiment: bool,
) -> list[tuple[float, float, float, float]]:
    import hcfp.learned as learned

    source = _source()
    case = from_official(
        source.block_count,
        source.area_targets,
        source.b2b_connectivity,
        source.p2b_connectivity,
        source.pins_pos,
        source.constraints,
        source.target_positions,
    )

    def to_official(_source, _case, candidate):
        x = float(candidate[0, 0])
        return [(x, 0.0, 2.0, 2.0), (x + 3.0, 0.0, 2.0, 2.0)]

    monkeypatch.setattr(learned, "to_official_placements", to_official)
    monkeypatch.setattr(learned, "verify_feasible", lambda _source, rows: rows[0][0] in feasible_x)
    monkeypatch.setattr(learned, "_raw_quality", lambda _source, _case, rows: quality[rows[0][0]])
    return learned.select_official_from_analysis(
        source,
        case,
        analysis,
        config=LearnedConfig(
            analytic=_pareto_config(),
            ranker_selection_experiment=experiment,
        ),
    )


def test_checkpoint_lane_runs_through_exact_safe_tail(tmp_path: Path) -> None:
    torch.manual_seed(5)
    checkpoint = tmp_path / "model.pt"
    saved_hash = save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16)),
        checkpoint,
        RUNTIME_NORMALIZATION,
    )

    result = solve_case_with_checkpoint(_case(), checkpoint, _config())

    assert result.used_checkpoint is True
    assert result.checkpoint_hash == saved_hash
    assert result.failure_reason is None
    assert verify_feasible(_case(), result.selected)
    assert result.flow_steps == 0
    assert result.candidate_count == 4


def test_multistep_flow_population_preserves_exact_safe_output(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16)),
        checkpoint,
        RUNTIME_NORMALIZATION,
        metadata=FLOW_METADATA,
    )
    config = LearnedConfig(analytic=_config(), flow_steps=3, flow_fraction=1.0)

    result = solve_case_with_checkpoint(_case(), checkpoint, config)

    assert result.used_checkpoint is True
    assert result.flow_steps == 3
    assert verify_feasible(_case(), result.selected)


def test_effective_collective_steps_requires_all_checkpoint_and_model_gates() -> None:
    enabled = ModelConfig(hidden_dim=16, collective_enabled=True)

    assert effective_collective_steps(2, COLLECTIVE_METADATA, enabled) == 2
    assert (
        effective_collective_steps(
            2,
            {**COLLECTIVE_METADATA, "capabilities": {"collective": False}},
            enabled,
        )
        == 0
    )
    assert (
        effective_collective_steps(
            2,
            {**COLLECTIVE_METADATA, "trained_heads": []},
            enabled,
        )
        == 0
    )
    assert effective_collective_steps(2, COLLECTIVE_METADATA, ModelConfig(hidden_dim=16)) == 0


def test_effective_tail_topk_requires_ranker_capability_and_head() -> None:
    assert effective_tail_topk(None, {}) is None
    assert effective_tail_topk(1, RANKER_METADATA) == 1
    assert (
        effective_tail_topk(
            1,
            {**RANKER_METADATA, "capabilities": {"ranker": False}},
        )
        is None
    )
    assert (
        effective_tail_topk(
            1,
            {**RANKER_METADATA, "trained_heads": []},
        )
        is None
    )


def test_effective_flow_steps_rejects_negative_requests_before_capability_gate() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        effective_flow_steps(-1, {"capabilities": {"flow": False}})


def test_effective_collective_steps_rejects_negative_requests_before_capability_gate() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        effective_collective_steps(-1, {"capabilities": {"collective": False}}, {})


def test_learned_config_rejects_negative_collective_steps() -> None:
    with pytest.raises(ValueError, match="collective_steps must be non-negative"):
        LearnedConfig(collective_steps=-1)


def test_effective_tail_topk_rejects_nonpositive_requests() -> None:
    with pytest.raises(ValueError, match="tail_topk must be positive"):
        effective_tail_topk(0, RANKER_METADATA)


def test_energy_history_merge_repeats_last_observation_for_shorter_tail() -> None:
    analytic = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    learned = torch.tensor([[[7.0, 8.0, 9.0]]])

    merged = _merge_energy_history(analytic, learned)

    assert merged.shape == (2, 2, 3)
    assert torch.equal(merged[1, 1], learned[0, 0])


def test_post_relax_seed_provenance_keeps_derived_hashes_for_changed_rows() -> None:
    merged = _merged_synthetic_provenance(changed_post_relax=True)
    snapshot = merged.incumbent_snapshot

    assert len(snapshot["constraint_seed_sources"]) == 4
    assert len(snapshot["topology_seed_sources"]) == 4
    _assert_sources_match_raw_hashes(
        snapshot,
        merged.raw_candidates,
        "constraint_seed_provenance",
    )
    _assert_sources_match_raw_hashes(
        snapshot,
        merged.raw_candidates,
        "topology_seed_provenance",
    )
    assert {
        record["transform"]
        for record in snapshot["constraint_seed_provenance"]
        if record["stage"] == "post_relax"
    } == {"population_relaxation"}
    assert {
        record["transform"]
        for record in snapshot["topology_seed_provenance"]
        if record["stage"] == "post_relax"
    } == {"population_relaxation"}


def test_post_relax_seed_provenance_marks_tampered_initial_stale() -> None:
    merged = _merged_synthetic_provenance(
        changed_post_relax=True,
        tamper_constraint_initial=True,
    )
    snapshot = merged.incumbent_snapshot

    assert "candidate_2" in snapshot["constraint_seed_stale_sources"]
    assert "candidate_2" not in snapshot["constraint_seed_sources"]
    assert "candidate_7" not in snapshot["constraint_seed_sources"]
    assert len(snapshot["constraint_seed_sources"]) == 2
    _assert_sources_match_raw_hashes(
        snapshot,
        merged.raw_candidates,
        "constraint_seed_provenance",
    )
    assert len(snapshot["topology_seed_sources"]) == 4


def test_constraint_seed_provenance_missing_initial_hash_fails_closed() -> None:
    merged = _merged_synthetic_provenance(
        changed_post_relax=True,
        omit_constraint_hash=True,
    )
    snapshot = merged.incumbent_snapshot

    assert "candidate_2" in snapshot["constraint_seed_stale_sources"]
    assert "candidate_2" not in snapshot["constraint_seed_sources"]
    assert "candidate_7" not in snapshot["constraint_seed_sources"]
    assert tuple(snapshot["constraint_seed_sources"]) == ("candidate_3", "candidate_8")
    assert len(snapshot["constraint_seed_provenance"]) == 2
    _assert_sources_match_raw_hashes(
        snapshot,
        merged.raw_candidates,
        "constraint_seed_provenance",
    )
    assert len(snapshot["topology_seed_sources"]) == 4


def test_post_relax_seed_provenance_keeps_identity_hash_semantics() -> None:
    merged = _merged_synthetic_provenance(changed_post_relax=False)
    snapshot = merged.incumbent_snapshot

    _assert_sources_match_raw_hashes(
        snapshot,
        merged.raw_candidates,
        "constraint_seed_provenance",
    )
    _assert_sources_match_raw_hashes(
        snapshot,
        merged.raw_candidates,
        "topology_seed_provenance",
    )
    post_records = (
        tuple(snapshot["constraint_seed_provenance"])[1::2]
        + tuple(snapshot["topology_seed_provenance"])[1::2]
    )
    assert {record["transform"] for record in post_records} == {"identity"}
    assert all(
        record["candidate_sha256"] == record["parent_candidate_sha256"]
        for record in post_records
    )


def test_ranker_shadow_uses_exact_replay_initial_slice_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hcfp.learned as learned

    captured: dict[str, object] = {}

    def features(case, raw, post_bdp, anchor, kinds, stage):
        captured["raw"] = raw.detach().clone()
        captured["post_bdp"] = post_bdp.detach().clone()
        captured["anchor"] = anchor.detach().clone()
        captured["kinds"] = tuple(kinds)
        captured["stage"] = stage
        return torch.zeros((raw.shape[0], RANKER_FEATURE_DIM), dtype=torch.float32)

    monkeypatch.setattr(learned, "repair_aware_ranker_features", features)
    analysis = _shadow_analysis()

    shadowed = _attach_ranker_shadow_snapshot(
        _case(),
        analysis,
        model=_shadow_model((0.4, 0.3, 0.2, 0.1)),
        metadata=_shadow_metadata(),
        analytic_count=2,
        learned_count=4,
        residual_count=1,
        constraint_count=1,
        topology_count=2,
    )

    snapshot = shadowed.incumbent_snapshot
    assert torch.equal(captured["raw"], analysis.raw_candidates[3:7])
    assert torch.equal(captured["post_bdp"], analysis.projected_candidates[3:7])
    assert torch.equal(captured["anchor"], safe_shelf(_case()))
    assert captured["kinds"] == ("learned", "constraint", "topology", "topology")
    assert captured["stage"] == "initial"
    assert snapshot["ranker_shadow_candidate_kinds"] == captured["kinds"]
    assert tuple(row["source"] for row in snapshot["ranker_shadow_top4"]) == (
        "candidate_6",
        "candidate_5",
        "candidate_4",
        "candidate_3",
    )


def test_ranker_shadow_does_not_change_selection_or_exact_source() -> None:
    analysis = _shadow_analysis()

    shadowed = _attach_ranker_shadow_snapshot(
        _case(),
        analysis,
        model=_shadow_model((0.1, 0.2, 0.3, 0.4)),
        metadata=_shadow_metadata(),
        analytic_count=2,
        learned_count=4,
        residual_count=2,
        constraint_count=1,
        topology_count=1,
    )

    assert torch.equal(shadowed.selected, analysis.selected)
    assert shadowed.incumbent_snapshot["exact_source"] == "candidate_5"
    assert shadowed.incumbent_snapshot["ranker_shadow_source"] == "merged_learned_initial"


def test_ranker_shadow_ignores_ineligible_best_score() -> None:
    analysis = _shadow_analysis(eligible=(True, False, True, True))

    shadowed = _attach_ranker_shadow_snapshot(
        _case(),
        analysis,
        model=_shadow_model((0.9, -9.0, 0.2, 0.3)),
        metadata=_shadow_metadata(),
        analytic_count=2,
        learned_count=4,
        residual_count=2,
        constraint_count=1,
        topology_count=1,
    )

    snapshot = shadowed.incumbent_snapshot
    assert snapshot["ranker_shadow_eligible_count"] == 3
    assert "candidate_4" not in {
        row["source"] for row in snapshot["ranker_shadow_top4"]
    }
    assert snapshot["ranker_shadow_top4"][0]["source"] == "candidate_5"


def test_ranker_shadow_records_empty_reason_for_zero_eligible_candidates() -> None:
    analysis = _shadow_analysis(eligible=(False, False, False, False))

    shadowed = _attach_ranker_shadow_snapshot(
        _case(),
        analysis,
        model=_shadow_model((0.1, 0.2, 0.3, 0.4)),
        metadata=_shadow_metadata(),
        analytic_count=2,
        learned_count=4,
        residual_count=2,
        constraint_count=1,
        topology_count=1,
    )

    snapshot = shadowed.incumbent_snapshot
    assert torch.equal(shadowed.selected, analysis.selected)
    assert snapshot["exact_source"] == "candidate_5"
    assert snapshot["ranker_shadow_eligible_count"] == 0
    assert snapshot["ranker_shadow_empty_reason"] == "no_exact_eligible_candidates"
    assert snapshot["ranker_shadow_top4"] == ()


def test_ranker_shadow_untrained_or_incompatible_checkpoint_is_neutral() -> None:
    analysis = _shadow_analysis()
    untrained = _attach_ranker_shadow_snapshot(
        _case(),
        analysis,
        model=_shadow_model((0.1, 0.2, 0.3, 0.4)),
        metadata={"capabilities": {"ranker": False}, "trained_heads": []},
        analytic_count=2,
        learned_count=4,
        residual_count=2,
        constraint_count=1,
        topology_count=1,
    )
    incompatible = _attach_ranker_shadow_snapshot(
        _case(),
        analysis,
        model=SimpleNamespace(
            config=ModelConfig(hidden_dim=16),
            ranker=_ScoreRanker((0.1, 0.2, 0.3, 0.4)),
        ),
        metadata=_shadow_metadata(),
        analytic_count=2,
        learned_count=4,
        residual_count=2,
        constraint_count=1,
        topology_count=1,
    )

    assert torch.equal(untrained.selected, analysis.selected)
    assert untrained.incumbent_snapshot["ranker_shadow_skipped_reason"] == "ranker_not_trained"
    assert "ranker_shadow_top4" not in untrained.incumbent_snapshot
    assert torch.equal(incompatible.selected, analysis.selected)
    assert (
        incompatible.incumbent_snapshot["ranker_shadow_skipped_reason"]
        == "ranker_feature_dim_mismatch"
    )


def test_ranker_shadow_uses_source_index_for_deterministic_ties() -> None:
    shadowed = _attach_ranker_shadow_snapshot(
        _case(),
        _shadow_analysis(),
        model=_shadow_model((0.5, 0.5, 0.5, 0.5)),
        metadata=_shadow_metadata(),
        analytic_count=2,
        learned_count=4,
        residual_count=2,
        constraint_count=1,
        topology_count=1,
    )

    assert tuple(row["source"] for row in shadowed.incumbent_snapshot["ranker_shadow_top4"]) == (
        "candidate_3",
        "candidate_4",
        "candidate_5",
        "candidate_6",
    )


def test_ranker_shadow_failure_is_recorded_without_changing_selection() -> None:
    analysis = _shadow_analysis()

    shadowed = _attach_ranker_shadow_snapshot(
        _case(),
        analysis,
        model=_shadow_model((float("nan"), 0.2, 0.3, 0.4)),
        metadata=_shadow_metadata(),
        analytic_count=2,
        learned_count=4,
        residual_count=2,
        constraint_count=1,
        topology_count=1,
    )

    assert torch.equal(shadowed.selected, analysis.selected)
    assert shadowed.incumbent_snapshot["exact_source"] == "candidate_5"
    assert (
        shadowed.incumbent_snapshot["ranker_shadow_failure_reason"]
        == "ValueError: ranker shadow scores must be finite"
    )
    assert "ranker_shadow_top4" not in shadowed.incumbent_snapshot


def test_ranker_selection_experiment_default_off_records_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _pareto_analysis()
    analysis.analytic.incumbent_snapshot["ranker_shadow_top4"] = (
        {"source": "candidate_4", "score": -1.0, "kind": "learned"},
    )

    result = _select_with_ranker_counterfactual(
        monkeypatch,
        analysis,
        feasible_x={20.0, 40.0},
        quality={20.0: (2.0, 20.0, 20.0), 40.0: (0.0, 1.0, 1.0)},
        experiment=False,
    )

    assert result[0][0] == 20.0
    assert "ranker_selection_counterfactual" not in analysis.analytic.incumbent_snapshot


def test_ranker_selection_experiment_records_passing_counterfactual_without_selecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _pareto_analysis()
    analysis.analytic.incumbent_snapshot["ranker_shadow_top4"] = (
        {"source": "candidate_4", "score": -1.0, "kind": "learned"},
    )

    result = _select_with_ranker_counterfactual(
        monkeypatch,
        analysis,
        feasible_x={20.0, 40.0},
        quality={20.0: (2.0, 20.0, 20.0), 40.0: (0.0, 1.0, 1.0)},
        experiment=True,
    )

    snapshot = analysis.analytic.incumbent_snapshot
    assert result[0][0] == 20.0
    assert snapshot["exact_source"] == "candidate_2"
    assert snapshot["ranker_selection_experiment_mode"] == "counterfactual_only"
    assert snapshot["ranker_selection_counterfactual"] == {
        "would_accept": True,
        "source": "candidate_4",
        "shadow_rank": 0,
        "metrics": (0.0, 1.0, 1.0),
        "current_metrics": (2.0, 20.0, 20.0),
        "rejection_reason": None,
    }


def test_ranker_selection_experiment_rejects_hard_infeasible_shadow_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _pareto_analysis()
    analysis.analytic.incumbent_snapshot["ranker_shadow_top4"] = (
        {"source": "candidate_4", "score": -1.0, "kind": "learned"},
    )

    result = _select_with_ranker_counterfactual(
        monkeypatch,
        analysis,
        feasible_x={20.0},
        quality={20.0: (2.0, 20.0, 20.0)},
        experiment=True,
    )

    snapshot = analysis.analytic.incumbent_snapshot
    assert result[0][0] == 20.0
    assert snapshot["ranker_selection_counterfactual"]["would_accept"] is False
    assert (
        snapshot["ranker_selection_counterfactual"]["rejection_reason"]
        == "hard_infeasible"
    )
    assert (
        snapshot["ranker_selection_evaluated_top4"][0]["rejection_reason"]
        == "hard_infeasible"
    )


def test_ranker_counterfactual_rejects_zero_eligible_without_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _pareto_analysis()
    analysis.analytic.incumbent_snapshot["ranker_shadow_top4"] = ()
    analysis.analytic.incumbent_snapshot["ranker_shadow_eligible_count"] = 0
    analysis.analytic.incumbent_snapshot["ranker_shadow_empty_reason"] = (
        "no_exact_eligible_candidates"
    )

    result = _select_with_ranker_counterfactual(
        monkeypatch,
        analysis,
        feasible_x={20.0},
        quality={20.0: (2.0, 20.0, 20.0)},
        experiment=True,
    )

    snapshot = analysis.analytic.incumbent_snapshot
    assert result[0][0] == 20.0
    assert snapshot["exact_source"] == "candidate_2"
    assert snapshot["ranker_selection_counterfactual"] == {
        "would_accept": False,
        "rejection_reason": "no_exact_eligible_ranker_candidates",
    }


def test_ranker_selection_experiment_rejects_non_dominating_shadow_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _pareto_analysis()
    analysis.analytic.incumbent_snapshot["ranker_shadow_top4"] = (
        {"source": "candidate_4", "score": -1.0, "kind": "learned"},
    )

    result = _select_with_ranker_counterfactual(
        monkeypatch,
        analysis,
        feasible_x={20.0, 40.0},
        quality={20.0: (2.0, 20.0, 20.0), 40.0: (3.0, 1.0, 1.0)},
        experiment=True,
    )

    snapshot = analysis.analytic.incumbent_snapshot
    assert result[0][0] == 20.0
    assert snapshot["ranker_selection_counterfactual"]["would_accept"] is False
    assert (
        snapshot["ranker_selection_counterfactual"]["rejection_reason"]
        == "not_pareto_dominating"
    )
    assert (
        snapshot["ranker_selection_evaluated_top4"][0]["rejection_reason"]
        == "not_pareto_dominating"
    )


def test_legacy_checkpoint_disables_requested_flow_without_falling_back(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    save_checkpoint(
        HCFPModel(
            ModelConfig(
                hidden_dim=16,
                topology_enabled=True,
                constraint_enabled=True,
            )
        ),
        checkpoint,
        RUNTIME_NORMALIZATION,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["schema_version"] = 1
    for field in (
        "capabilities",
        "trained_heads",
        "training_objective_version",
        "parent_state_hash",
    ):
        payload.pop(field)
    payload["state_hash"] = _payload_hash(payload)
    torch.save(payload, checkpoint)

    result = solve_case_with_checkpoint(
        _case(),
        checkpoint,
        LearnedConfig(analytic=_config(), flow_steps=3, topology_seeds=1),
    )

    assert result.used_checkpoint is True
    assert result.failure_reason is None
    assert result.flow_steps == 0
    assert result.topology_seed_attempted is True


def test_collective_capability_false_disables_requested_steps_without_falling_back(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16, collective_enabled=True)),
        checkpoint,
        RUNTIME_NORMALIZATION,
        metadata={
            "capabilities": {"collective": False},
            "trained_heads": ["collective"],
            "training_objective_version": "collective_loss_v1",
        },
    )

    result = solve_case_with_checkpoint(
        _case(),
        checkpoint,
        LearnedConfig(analytic=_config(), collective_steps=2),
    )

    assert result.used_checkpoint is True
    assert result.failure_reason is None
    assert result.collective_steps == 0
    assert result.collective_used is False
    assert result.collective_calls == 0
    assert verify_feasible(_case(), result.selected)


def test_collective_tail_runs_when_checkpoint_and_model_gates_pass(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16, collective_enabled=True)),
        checkpoint,
        RUNTIME_NORMALIZATION,
        metadata=COLLECTIVE_METADATA,
    )

    result = solve_case_with_checkpoint(
        _case(),
        checkpoint,
        LearnedConfig(analytic=_config(), collective_steps=2),
    )

    assert result.used_checkpoint is True
    assert result.collective_steps == 2
    assert result.collective_used is True
    assert result.collective_calls == 2
    assert verify_feasible(_case(), result.selected)


def test_collective_tail_keeps_topology_and_constraint_provenance_with_bdp_disabled(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        HCFPModel(
            ModelConfig(
                hidden_dim=16,
                topology_enabled=True,
                constraint_enabled=True,
                collective_enabled=True,
            )
        ),
        checkpoint,
        RUNTIME_NORMALIZATION,
        metadata=COLLECTIVE_METADATA,
    )

    analysis = analyze_case_with_checkpoint(
        _case(),
        checkpoint,
        LearnedConfig(
            analytic=_config(),
            topology_seeds=1,
            constraint_seeds=1,
            collective_steps=1,
        ),
    )

    assert analysis.result.used_checkpoint is True
    assert analysis.result.collective_calls == 1
    assert analysis.result.topology_seed_attempted is True
    assert analysis.result.topology_seed_accepted is True
    assert analysis.result.constraint_seed_attempted is True
    assert "topology_seed_provenance" in analysis.analytic.incumbent_snapshot
    assert "constraint_seed_failure_reason" in analysis.analytic.incumbent_snapshot


def test_collective_default_off_preserves_selected_and_candidate_pool(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16, collective_enabled=True)),
        checkpoint,
        RUNTIME_NORMALIZATION,
        metadata=COLLECTIVE_METADATA,
    )

    legacy = analyze_case_with_checkpoint(_case(), checkpoint, _config())
    default_off = analyze_case_with_checkpoint(
        _case(),
        checkpoint,
        LearnedConfig(analytic=_config(), collective_steps=0),
    )

    assert default_off.result.collective_steps == 0
    assert default_off.result.collective_used is False
    assert torch.equal(default_off.result.selected, legacy.result.selected)
    assert torch.equal(default_off.analytic.raw_candidates, legacy.analytic.raw_candidates)
    assert torch.equal(
        default_off.analytic.projected_candidates,
        legacy.analytic.projected_candidates,
    )


def test_untrained_ranker_request_does_not_prune_learned_candidates(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(HCFPModel(ModelConfig(hidden_dim=16)), checkpoint, RUNTIME_NORMALIZATION)
    config = LearnedConfig(analytic=_config(), tail_topk=1)

    result = solve_case_with_checkpoint(_case(), checkpoint, config)

    assert result.used_checkpoint is True
    assert result.candidate_count == 4
    assert verify_feasible(_case(), result.selected)


def test_trained_ranker_capability_prunes_learned_sidecar_candidates(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16)),
        checkpoint,
        RUNTIME_NORMALIZATION,
        metadata=RANKER_METADATA,
    )
    config = LearnedConfig(analytic=_config(), tail_topk=1)

    first = solve_case_with_checkpoint(_case(), checkpoint, config)
    second = solve_case_with_checkpoint(_case(), checkpoint, config)

    assert first.used_checkpoint is True
    assert first.candidate_count == 3
    assert torch.equal(first.selected, second.selected)
    assert second.candidate_count == 3
    assert verify_feasible(_case(), first.selected)


def test_ranker_capability_requires_matching_trained_head(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="matching trained heads"):
        save_checkpoint(
            HCFPModel(ModelConfig(hidden_dim=16)),
            tmp_path / "bad-ranker.pt",
            RUNTIME_NORMALIZATION,
            metadata={
                "capabilities": {"ranker": True},
                "trained_heads": [],
                "training_objective_version": "ranker_loss_v1",
            },
        )


def test_split_tail_preserves_standalone_analytic_candidates(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(HCFPModel(ModelConfig(hidden_dim=16)), checkpoint, RUNTIME_NORMALIZATION)

    analysis = analyze_case_with_checkpoint(_case(), checkpoint, _config())
    standalone = solve_case_with_telemetry(_case(), _config())

    assert torch.equal(analysis.analytic.raw_candidates[:3], standalone.raw_candidates[:3])
    assert torch.equal(analysis.analytic.raw_candidates[5:7], standalone.raw_candidates[3:5])
    assert torch.equal(analysis.analytic.projected_candidates[:3], standalone.projected_candidates[:3])
    assert torch.equal(analysis.analytic.projected_candidates[5:7], standalone.projected_candidates[3:5])
    assert analysis.analytic.incumbent_snapshot["analytic_exact_source"] is not None
    assert analysis.analytic.incumbent_snapshot["analytic_fast_source"] is not None


def test_split_tail_carries_normalized_infeasible_analytic_fast_source() -> None:
    import hcfp.learned as learned

    standalone = solve_case_with_telemetry(_case(), _config())
    hard = standalone.telemetry.hard_feasible.clone()
    projection_ok = standalone.telemetry.projection_ok.clone()
    hard[1] = False
    projection_ok[1:3] = True
    analytic = replace(
        standalone,
        telemetry=replace(standalone.telemetry, hard_feasible=hard, projection_ok=projection_ok),
        incumbent_snapshot={"exact_source": "candidate_2", "fast_source": "candidate_1"},
    )

    merged = learned._merge_tail_analyses(_case(), analytic, standalone)

    assert merged.incumbent_snapshot["analytic_exact_source"] == "candidate_2"
    assert merged.incumbent_snapshot["analytic_fast_source"] == "candidate_1"
    assert not bool(merged.telemetry.hard_feasible[1])


def test_ranker_only_checkpoint_change_does_not_resample_candidate_pool(tmp_path: Path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    model = HCFPModel(ModelConfig(hidden_dim=16))
    save_checkpoint(model, first, RUNTIME_NORMALIZATION, metadata=FLOW_METADATA)
    with torch.no_grad():
        for parameter in model.ranker.parameters():
            parameter.add_(0.25)
    save_checkpoint(model, second, RUNTIME_NORMALIZATION, metadata=FLOW_METADATA)
    config = LearnedConfig(analytic=_config(), flow_steps=1, seed=17)

    first_analysis = analyze_case_with_checkpoint(_case(), first, config)
    second_analysis = analyze_case_with_checkpoint(_case(), second, config)

    assert first_analysis.result.checkpoint_hash != second_analysis.result.checkpoint_hash
    assert torch.equal(first_analysis.analytic.raw_candidates, second_analysis.analytic.raw_candidates)


def test_missing_checkpoint_fails_closed_to_analytic_lane(tmp_path: Path) -> None:
    result = solve_case_with_checkpoint(_case(), tmp_path / "missing.pt", _config())

    assert result.used_checkpoint is False
    assert result.checkpoint_hash is None
    assert result.failure_reason is not None
    assert verify_feasible(_case(), result.selected)


def test_normalization_mismatch_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(HCFPModel(ModelConfig(hidden_dim=16)), checkpoint, {"coordinate_scale": "wrong"})

    result = solve_case_with_checkpoint(_case(), checkpoint, _config())

    assert result.used_checkpoint is False
    assert result.failure_reason is not None and "normalization mismatch" in result.failure_reason


def test_raw_infeasible_learned_output_replays_analytic_incumbent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hcfp.learned as learned

    checkpoint = tmp_path / "model.pt"
    save_checkpoint(HCFPModel(ModelConfig(hidden_dim=16)), checkpoint, RUNTIME_NORMALIZATION)
    source = SimpleNamespace(
        block_count=2,
        area_targets=[4.0, 4.0],
        b2b_connectivity=[],
        p2b_connectivity=[],
        pins_pos=[],
        constraints=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        target_positions=None,
    )
    analytic = [(0.0, 0.0, 2.0, 2.0), (3.0, 0.0, 2.0, 2.0)]
    monkeypatch.setattr(
        learned,
        "to_official_placements",
        lambda *_args: [(0.0, 0.0, 2.0, 2.0), (1.0, 0.0, 2.0, 2.0)],
    )
    monkeypatch.setattr(learned, "solve_analytic", lambda *_args, **_kwargs: analytic)

    result = learned.solve(source, checkpoint=checkpoint, config=_config(), require_checkpoint=True)

    assert result == analytic


def test_raw_infeasible_analytic_replay_uses_safe_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hcfp.learned as learned

    checkpoint = tmp_path / "model.pt"
    save_checkpoint(HCFPModel(ModelConfig(hidden_dim=16)), checkpoint, RUNTIME_NORMALIZATION)
    source = SimpleNamespace(
        block_count=2,
        area_targets=[4.0, 4.0],
        b2b_connectivity=[],
        p2b_connectivity=[],
        pins_pos=[],
        constraints=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        target_positions=None,
    )
    overlap = [(0.0, 0.0, 2.0, 2.0), (1.0, 0.0, 2.0, 2.0)]
    safe = [(0.0, 0.0, 2.0, 2.0), (3.0, 0.0, 2.0, 2.0)]
    monkeypatch.setattr(learned, "to_official_placements", lambda *_args: overlap)
    monkeypatch.setattr(learned, "solve_analytic", lambda *_args, **_kwargs: overlap)
    monkeypatch.setattr(learned, "safe_fallback", lambda *_args, **_kwargs: safe)

    result = learned.solve(source, checkpoint=checkpoint, config=_config(), require_checkpoint=True)

    assert result == safe


def test_raw_infeasible_selected_uses_next_raw_feasible_pool_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hcfp.learned as learned

    source = _source()
    converted: list[float] = []

    def to_official(_source, _case, candidate):
        x = float(candidate[0, 0])
        converted.append(x)
        return [(x, 0.0, 2.0, 2.0), (x + 3.0, 0.0, 2.0, 2.0)]

    monkeypatch.setattr(learned, "analyze_case_with_checkpoint", lambda *_args: _analysis())
    monkeypatch.setattr(learned, "to_official_placements", to_official)
    monkeypatch.setattr(learned, "verify_feasible", lambda _source, rows: rows[0][0] == 10.0)
    monkeypatch.setattr(
        learned,
        "solve_analytic",
        lambda *_args, **_kwargs: pytest.fail("analytic replay must not run"),
    )

    result = learned.solve(source, checkpoint=tmp_path / "model.pt", require_checkpoint=True)

    assert result[0][0] == 10.0
    assert converted[-1] == 10.0
    assert 20.0 not in converted
    assert 30.0 not in converted


def test_raw_infeasible_pool_still_fails_closed_to_analytic_then_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hcfp.learned as learned

    source = _source()
    analytic_calls = 0
    overlap = [(0.0, 0.0, 2.0, 2.0), (1.0, 0.0, 2.0, 2.0)]
    safe = [(0.0, 0.0, 2.0, 2.0), (3.0, 0.0, 2.0, 2.0)]

    def replay_analytic(*_args, **_kwargs):
        nonlocal analytic_calls
        analytic_calls += 1
        return overlap

    monkeypatch.setattr(learned, "analyze_case_with_checkpoint", lambda *_args: _analysis())
    monkeypatch.setattr(learned, "to_official_placements", lambda *_args: overlap)
    monkeypatch.setattr(learned, "verify_feasible", lambda *_args: False)
    monkeypatch.setattr(learned, "solve_analytic", replay_analytic)
    monkeypatch.setattr(learned, "safe_fallback", lambda *_args: safe)

    result = learned.solve(source, checkpoint=tmp_path / "model.pt", require_checkpoint=True)

    assert analytic_calls == 1
    assert result == safe


def test_raw_feasible_pool_fallback_does_not_replace_analytic_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hcfp.learned as learned

    source = _source()
    analytic = [(40.0, 0.0, 2.0, 2.0), (43.0, 0.0, 2.0, 2.0)]

    def to_official(_source, _case, candidate):
        x = float(candidate[0, 0])
        return [(x, 0.0, 2.0, 2.0), (x + 3.0, 0.0, 2.0, 2.0)]

    analysis = _analysis()
    analysis.result.selected = analysis.analytic.projected_candidates[1]
    monkeypatch.setattr(learned, "analyze_case_with_checkpoint", lambda *_args: analysis)
    monkeypatch.setattr(learned, "to_official_placements", to_official)
    monkeypatch.setattr(learned, "verify_feasible", lambda _source, rows: rows[0][0] in (0.0, 40.0))
    monkeypatch.setattr(learned, "solve_analytic", lambda *_args, **_kwargs: analytic)

    result = learned.solve(source, checkpoint=tmp_path / "model.pt", require_checkpoint=True)

    assert result == analytic


def test_raw_feasible_selected_keeps_fast_path_without_pool_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hcfp.learned as learned

    source = _source()
    expected = [(0.0, 0.0, 2.0, 2.0), (3.0, 0.0, 2.0, 2.0)]
    conversions = 0

    def to_official(*_args):
        nonlocal conversions
        conversions += 1
        return expected

    monkeypatch.setattr(learned, "analyze_case_with_checkpoint", lambda *_args: _analysis())
    monkeypatch.setattr(learned, "to_official_placements", to_official)
    monkeypatch.setattr(learned, "verify_feasible", lambda *_args: True)
    monkeypatch.setattr(
        learned,
        "solve_analytic",
        lambda *_args, **_kwargs: pytest.fail("analytic replay must not run"),
    )

    result = learned.solve(source, checkpoint=tmp_path / "model.pt", require_checkpoint=True)

    assert result == expected
    assert conversions == 1


def test_raw_feasible_analytic_dominator_replaces_selected_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _solve_pareto(
        monkeypatch,
        tmp_path,
        _pareto_analysis(),
        feasible_x={10.0, 20.0},
        quality={10.0: (1.0, 10.0, 10.0), 20.0: (2.0, 20.0, 20.0)},
    )
    assert result[0][0] == 10.0


def test_raw_repaired_constraint_dominator_replaces_selected_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hcfp.learned as learned

    analysis = _pareto_analysis(
        analytic_source="candidate_2",
        analytic_exact_source=None,
        analytic_fast_source=None,
    )
    analysis.analytic.incumbent_snapshot["constraint_seed_provenance"] = (
        {"source": "candidate_3", "details": {"moves": ()}},
    )

    def to_official(_source, _case, candidate):
        x = float(candidate[0, 0])
        return [(x, 0.0, 2.0, 2.0), (x + 3.0, 0.0, 2.0, 2.0)]

    monkeypatch.setattr(
        learned,
        "analyze_case_with_checkpoint",
        lambda *_args: analysis,
    )
    monkeypatch.setattr(learned, "to_official_placements", to_official)
    monkeypatch.setattr(learned, "verify_feasible", lambda *_args: True)
    monkeypatch.setattr(
        learned,
        "repair_raw_constraints",
        lambda _source, _rows, _record: SimpleNamespace(
            placements=((10.0, 0.0, 2.0, 2.0), (13.0, 0.0, 2.0, 2.0))
        ),
    )
    monkeypatch.setattr(
        learned,
        "_raw_quality",
        lambda _source, _case, rows: {
            10.0: (1.0, 10.0, 10.0),
            20.0: (2.0, 20.0, 20.0),
        }[rows[0][0]],
    )

    result = learned.solve(
        _source(),
        checkpoint=tmp_path / "model.pt",
        config=_pareto_config(),
        require_checkpoint=True,
    )

    assert result[0][0] == 10.0


def test_raw_constraint_branch_survives_a_worse_projected_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hcfp.learned as learned

    analysis = _pareto_analysis(
        analytic_source="candidate_2",
        analytic_exact_source=None,
        analytic_fast_source=None,
    )
    raw = analysis.analytic.projected_candidates.clone()
    raw[3, :, 0] -= 25.0
    analysis.analytic.raw_candidates = raw
    analysis.analytic.incumbent_snapshot["constraint_seed_provenance"] = (
        {"source": "candidate_3", "details": {"moves": ()}},
    )

    monkeypatch.setattr(
        learned,
        "to_official_placements",
        lambda _source, _case, candidate: [
            tuple(float(value) for value in row) for row in candidate.tolist()
        ],
    )
    monkeypatch.setattr(
        learned,
        "repair_raw_constraints",
        lambda _source, rows, _record: SimpleNamespace(placements=tuple(rows)),
    )
    monkeypatch.setattr(learned, "verify_feasible", lambda *_args: True)
    monkeypatch.setattr(
        learned,
        "_raw_quality",
        lambda _source, _case, rows: {
            5.0: (1.0, 10.0, 10.0),
            20.0: (2.0, 20.0, 20.0),
            30.0: (3.0, 30.0, 30.0),
        }[rows[0][0]],
    )

    selected = learned._raw_constraint_pareto_guard(
        _source(),
        object(),
        analysis,
        [(20.0, 0.0, 2.0, 2.0), (23.0, 0.0, 2.0, 2.0)],
    )

    assert selected[0][0] == 5.0


def test_raw_constraint_guard_can_admit_exact_dominating_component_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hcfp.learned as learned

    analysis = _pareto_analysis(
        analytic_source="candidate_2",
        analytic_exact_source=None,
        analytic_fast_source=None,
    )
    proposal = analysis.analytic.projected_candidates.clone()
    proposal[3, :, 0] = torch.tensor([5.0, 8.0])
    analysis.analytic.telemetry.component_proposal_xywh = proposal
    analysis.analytic.telemetry.component_proposal_available = torch.tensor(
        [False, False, False, True, False]
    )
    analysis.analytic.incumbent_snapshot["constraint_seed_provenance"] = (
        {"source": "candidate_3", "details": {"moves": ()}},
    )

    monkeypatch.setattr(
        learned,
        "to_official_placements",
        lambda _source, _case, candidate: [
            tuple(float(value) for value in row) for row in candidate.tolist()
        ],
    )
    monkeypatch.setattr(
        learned,
        "repair_raw_constraints",
        lambda _source, rows, _record: SimpleNamespace(placements=tuple(rows)),
    )
    monkeypatch.setattr(learned, "verify_feasible", lambda _source, rows: rows[0][0] == 5.0)
    monkeypatch.setattr(
        learned,
        "_raw_quality",
        lambda _source, _case, rows: {
            5.0: (1.0, 10.0, 10.0),
            20.0: (2.0, 20.0, 20.0),
            30.0: (3.0, 30.0, 30.0),
        }[rows[0][0]],
    )

    selected = learned._raw_constraint_pareto_guard(
        _source(),
        object(),
        analysis,
        [(20.0, 0.0, 2.0, 2.0), (23.0, 0.0, 2.0, 2.0)],
    )

    assert selected[0][0] == 5.0


def test_raw_constraint_guard_ignores_proposal_without_constraint_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hcfp.learned as learned

    analysis = _pareto_analysis(
        analytic_source="candidate_2",
        analytic_exact_source=None,
        analytic_fast_source=None,
    )
    proposal = analysis.analytic.projected_candidates.clone()
    proposal[3, :, 0] = torch.tensor([5.0, 8.0])
    analysis.analytic.telemetry.component_proposal_xywh = proposal
    analysis.analytic.telemetry.component_proposal_available = torch.tensor(
        [False, False, False, True, False]
    )

    monkeypatch.setattr(
        learned,
        "to_official_placements",
        lambda _source, _case, candidate: [
            tuple(float(value) for value in row) for row in candidate.tolist()
        ],
    )
    monkeypatch.setattr(learned, "verify_feasible", lambda *_args: True)
    monkeypatch.setattr(
        learned,
        "_raw_quality",
        lambda _source, _case, rows: {
            5.0: (1.0, 10.0, 10.0),
            20.0: (2.0, 20.0, 20.0),
        }[rows[0][0]],
    )

    selected = learned._raw_constraint_pareto_guard(
        _source(),
        object(),
        analysis,
        [(20.0, 0.0, 2.0, 2.0), (23.0, 0.0, 2.0, 2.0)],
    )

    assert selected[0][0] == 20.0


def test_raw_constraint_guard_rejects_infeasible_component_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hcfp.learned as learned

    analysis = _pareto_analysis(
        analytic_source="candidate_2",
        analytic_exact_source=None,
        analytic_fast_source=None,
    )
    proposal = analysis.analytic.projected_candidates.clone()
    proposal[3, :, 0] = torch.tensor([5.0, 8.0])
    analysis.analytic.telemetry.component_proposal_xywh = proposal
    analysis.analytic.telemetry.component_proposal_available = torch.tensor(
        [False, False, False, True, False]
    )
    analysis.analytic.incumbent_snapshot["constraint_seed_provenance"] = (
        {"source": "candidate_3", "details": {"moves": ()}},
    )

    monkeypatch.setattr(
        learned,
        "to_official_placements",
        lambda _source, _case, candidate: [
            tuple(float(value) for value in row) for row in candidate.tolist()
        ],
    )
    monkeypatch.setattr(
        learned,
        "repair_raw_constraints",
        lambda _source, rows, _record: SimpleNamespace(placements=tuple(rows)),
    )
    monkeypatch.setattr(learned, "verify_feasible", lambda _source, rows: rows[0][0] != 5.0)
    monkeypatch.setattr(
        learned,
        "_raw_quality",
        lambda _source, _case, rows: {
            5.0: (1.0, 10.0, 10.0),
            20.0: (2.0, 20.0, 20.0),
            30.0: (3.0, 30.0, 30.0),
        }[rows[0][0]],
    )

    selected = learned._raw_constraint_pareto_guard(
        _source(),
        object(),
        analysis,
        [(20.0, 0.0, 2.0, 2.0), (23.0, 0.0, 2.0, 2.0)],
    )

    assert selected[0][0] == 20.0


def test_post_tail_group_repair_accepts_only_exact_pareto_improvement() -> None:
    import hcfp.learned as learned

    source = SimpleNamespace(
        block_count=4,
        area_targets=[1.0] * 4,
        b2b_connectivity=[],
        p2b_connectivity=[],
        pins_pos=[],
        constraints=[
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        target_positions=None,
    )
    case = from_official(
        source.block_count,
        source.area_targets,
        source.b2b_connectivity,
        source.p2b_connectivity,
        source.pins_pos,
        source.constraints,
    )
    placements = [
        (0.0, 0.0, 1.0, 1.0),
        (3.0, 0.0, 1.0, 1.0),
        (6.0, 0.0, 1.0, 1.0),
        (9.0, 0.0, 1.0, 1.0),
    ]

    repaired = learned._post_tail_group_repair(source, case, placements)

    assert verify_feasible(source, repaired)
    assert learned._raw_quality(source, case, repaired) < learned._raw_quality(
        source, case, placements
    )

    source.b2b_connectivity = [[1, 3, 100.0], [2, 3, 100.0]]
    protected_case = from_official(
        source.block_count,
        source.area_targets,
        source.b2b_connectivity,
        source.p2b_connectivity,
        source.pins_pos,
        source.constraints,
    )

    assert learned._post_tail_group_repair(
        source, protected_case, placements
    ) == placements


def test_legacy_mib_challenger_uses_observable_anchor_signature() -> None:
    import hcfp.learned as learned

    n = 84
    constraints = torch.zeros((n, 5), dtype=torch.long)
    constraints[:16, 0] = 1
    constraints[[0, 20, 21], 2] = 1
    targets = torch.full((n, 4), -1.0)
    targets[:16, 2:4] = 1.0
    case = from_official(n, torch.ones(n), [], [], [], constraints, targets)

    assert learned._needs_legacy_mib_challenger(case)

    constraints[10:16, 0] = 0
    targets[10:16, 2:4] = -1.0
    sparse_anchor_case = from_official(
        n, torch.ones(n), [], [], [], constraints, targets
    )
    assert not learned._needs_legacy_mib_challenger(sparse_anchor_case)


def test_legacy_portfolio_merge_preserves_candidate_stage_order() -> None:
    import hcfp.learned as learned

    primary = torch.arange(3.0).view(3, 1, 1).expand(-1, 2, 4)
    legacy = torch.arange(3.0, 6.0).view(3, 1, 1).expand(-1, 2, 4)
    primary_provenance = {
        "topology_seed_count": 1,
        "constraint_seed_count": 1,
        "topology_seed_orders": ({"name": "primary"},),
        "constraint_seed_records": ({"topology_seed_index": 0},),
    }
    legacy_provenance = {
        "topology_seed_count": 1,
        "constraint_seed_count": 1,
        "topology_seed_orders": ({"name": "legacy"},),
        "constraint_seed_records": ({"topology_seed_index": 0},),
    }

    population, provenance = learned._merge_legacy_mib_challenger(
        primary,
        primary_provenance,
        legacy,
        legacy_provenance,
    )

    assert population[:, 0, 0].tolist() == [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
    assert provenance["topology_seed_count"] == 2
    assert provenance["constraint_seed_count"] == 2
    assert provenance["topology_seed_orders"] == (
        {"name": "primary"},
        {"name": "legacy"},
    )
    assert provenance["constraint_seed_records"][-1]["topology_seed_index"] == 1
    assert provenance["constraint_seed_records"][-1]["challenger"] == "legacy_mib"


def test_legacy_mib_challenger_uses_exact_incumbent_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hcfp.learned as learned

    analysis = _pareto_analysis()
    analysis.analytic.incumbent_snapshot["constraint_seed_provenance"] = (
        {
            "source": "candidate_3",
            "challenger": "legacy_mib",
            "stage": "initial",
        },
    )
    monkeypatch.setattr(
        learned,
        "to_official_placements",
        lambda _source, _case, candidate: [
            tuple(float(value) for value in row) for row in candidate.tolist()
        ],
    )
    monkeypatch.setattr(
        learned,
        "_repair_constraint_candidate",
        lambda _source, _case, rows, _snapshot, _candidate: rows,
    )
    monkeypatch.setattr(
        learned,
        "_post_tail_group_repair",
        lambda _source, _case, rows: rows,
    )
    monkeypatch.setattr(learned, "verify_feasible", lambda *_args: True)
    monkeypatch.setattr(
        learned,
        "_raw_quality",
        lambda _source, _case, rows: {
            20.0: (2.0, 10.0, 10.0),
            30.0: (1.0, 30.0, 30.0),
        }[rows[0][0]],
    )

    selected = learned._legacy_mib_challenger_guard(
        _source(),
        object(),
        analysis,
        [(20.0, 0.0, 2.0, 2.0), (23.0, 0.0, 2.0, 2.0)],
    )

    assert selected[0][0] == 30.0


def test_raw_infeasible_or_tradeoff_candidate_cannot_trigger_pareto_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _solve_pareto(
        monkeypatch,
        tmp_path,
        _pareto_analysis(),
        feasible_x={20.0},
        quality={10.0: (1.0, 10.0, 10.0), 20.0: (2.0, 20.0, 20.0)},
    )
    assert result[0][0] == 20.0

    result, _ = _solve_pareto(
        monkeypatch,
        tmp_path,
        _pareto_analysis(),
        feasible_x={10.0, 20.0},
        quality={10.0: (1.0, 30.0, 10.0), 20.0: (2.0, 20.0, 20.0)},
    )
    assert result[0][0] == 20.0


def test_unknown_incumbent_source_fails_closed_to_selected_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _solve_pareto(
        monkeypatch,
        tmp_path,
        _pareto_analysis(
            analytic_source="unknown",
            analytic_exact_source="unknown",
            analytic_fast_source="unknown",
        ),
        feasible_x={10.0, 20.0},
        quality={},
    )
    assert result[0][0] == 20.0


def test_selected_analytic_incumbent_skips_extra_pareto_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, conversions = _solve_pareto(
        monkeypatch,
        tmp_path,
        _pareto_analysis(
            analytic_exact_source="candidate_2",
            analytic_fast_source="candidate_2",
        ),
        feasible_x={20.0},
        quality={},
    )
    assert result[0][0] == 20.0
    assert conversions == [20.0]


def test_require_checkpoint_failure_does_not_scan_pool_or_replay_analytic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hcfp.learned as learned

    monkeypatch.setattr(
        learned,
        "analyze_case_with_checkpoint",
        lambda *_args: _analysis(used_checkpoint=False),
    )
    monkeypatch.setattr(
        learned,
        "to_official_placements",
        lambda *_args: pytest.fail("checkpoint gate must run before conversion"),
    )
    monkeypatch.setattr(
        learned,
        "solve_analytic",
        lambda *_args, **_kwargs: pytest.fail("checkpoint gate must fail closed"),
    )

    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        learned.solve(_source(), checkpoint=tmp_path / "model.pt", require_checkpoint=True)
