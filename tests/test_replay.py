from __future__ import annotations

from pathlib import Path

import torch

from hcfp.analytic import AnalyticConfig, solve_case_with_telemetry
from hcfp.data import DataSample, extract_labels
from hcfp.dynamics import DynamicsConfig
from hcfp.fallback import safe_shelf
from hcfp.model import HCFPModel, ModelConfig
from hcfp.profile import synthetic_case
from hcfp.replay import iter_replay, record_from_analysis, train_ranker_steps, write_replay


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

    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1))
    before = {name: value.detach().clone() for name, value in model.ranker.named_parameters()}
    optimizer = torch.optim.AdamW(model.ranker.parameters(), lr=1.0e-3)
    history = train_ranker_steps(model, loaded, optimizer, steps=2)

    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(value)) for value in history)
    assert any(not torch.equal(before[name], value.detach()) for name, value in model.ranker.named_parameters())
