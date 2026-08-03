from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hcfp.analytic import AnalyticConfig, solve_case_with_telemetry
from hcfp.case import from_official
from hcfp.checkpoint import RUNTIME_NORMALIZATION, save_checkpoint
from hcfp.dynamics import DynamicsConfig
from hcfp.learned import LearnedConfig, analyze_case_with_checkpoint, solve_case_with_checkpoint
from hcfp.model import HCFPModel, ModelConfig
from hcfp.verify import verify_feasible


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
    assert result.flow_steps == 6
    assert result.candidate_count == 4


def test_multistep_flow_population_preserves_exact_safe_output(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(HCFPModel(ModelConfig(hidden_dim=16)), checkpoint, RUNTIME_NORMALIZATION)
    config = LearnedConfig(analytic=_config(), flow_steps=3, flow_fraction=1.0)

    result = solve_case_with_checkpoint(_case(), checkpoint, config)

    assert result.used_checkpoint is True
    assert result.flow_steps == 3
    assert verify_feasible(_case(), result.selected)


def test_ranker_prunes_only_learned_sidecar_candidates(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(HCFPModel(ModelConfig(hidden_dim=16)), checkpoint, RUNTIME_NORMALIZATION)
    config = LearnedConfig(analytic=_config(), tail_topk=1)

    result = solve_case_with_checkpoint(_case(), checkpoint, config)

    assert result.used_checkpoint is True
    assert result.candidate_count == 3
    assert verify_feasible(_case(), result.selected)


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
    save_checkpoint(model, first, RUNTIME_NORMALIZATION)
    with torch.no_grad():
        for parameter in model.ranker.parameters():
            parameter.add_(0.25)
    save_checkpoint(model, second, RUNTIME_NORMALIZATION)
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
