from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hcfp.analytic import AnalyticConfig
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
