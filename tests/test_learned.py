from __future__ import annotations

from pathlib import Path

import torch

from hcfp.analytic import AnalyticConfig
from hcfp.case import from_official
from hcfp.checkpoint import RUNTIME_NORMALIZATION, save_checkpoint
from hcfp.dynamics import DynamicsConfig
from hcfp.learned import LearnedConfig, solve_case_with_checkpoint
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
    assert result.candidate_count == 2


def test_multistep_flow_population_preserves_exact_safe_output(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(HCFPModel(ModelConfig(hidden_dim=16)), checkpoint, RUNTIME_NORMALIZATION)
    config = LearnedConfig(analytic=_config(), flow_steps=3, flow_fraction=1.0)

    result = solve_case_with_checkpoint(_case(), checkpoint, config)

    assert result.used_checkpoint is True
    assert result.flow_steps == 3
    assert verify_feasible(_case(), result.selected)


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
