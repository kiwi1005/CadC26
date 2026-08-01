from __future__ import annotations

import torch

from hcfp.case import from_official
from hcfp.model import HCFPModel, ModelConfig


def _case():
    return from_official(
        4,
        [4.0, 9.0, 16.0, 25.0],
        [[0, 1, 2.0], [1, 2, 3.0], [2, 3, 1.0]],
        [[0, 0, 2.0], [1, 3, 4.0]],
        [[1.0, 2.0], [8.0, 1.0]],
        [[0, 1, 0, 1, 1], [1, 0, 0, 1, 0], [0, 0, 7, 0, 2], [0, 0, 7, 0, 4]],
        [[0.0, 0.0, 2.0, 2.0], [4.0, 0.0, 3.0, 3.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]],
    )


def test_model_outputs_are_bounded_and_honor_hard_masks() -> None:
    torch.manual_seed(7)
    case = _case()
    cfg = ModelConfig(hidden_dim=32, residual_bound=0.125, aspect_residual_bound=0.25, force_channels=7)
    output = HCFPModel(cfg)(case, population=5, candidate_metrics=torch.zeros(5, cfg.candidate_metric_dim))

    assert output.embedding.shape == (4, 32)
    assert output.precedence_logits.shape == (4, 4, 5)
    assert output.outline.shape == (4,)
    assert output.center_residual.shape == (5, 4, 2)
    assert output.log_aspect_residual.shape == (5, 4)
    assert output.flow_velocity.shape == (5, 4, 3)
    assert output.force_gates.shape == (5, 4, 7)
    assert output.rank_score.shape == (5,)
    assert output.center_residual.dtype == torch.float32
    assert output.log_aspect_residual.dtype == torch.float32
    assert float(output.center_residual.detach().abs().amax()) <= cfg.residual_bound
    assert float(output.log_aspect_residual.detach().abs().amax()) <= cfg.aspect_residual_bound
    assert torch.equal(output.center_residual[:, case.preplaced_mask], torch.zeros(5, 1, 2))
    assert torch.equal(output.log_aspect_residual[:, case.fixed_mask | case.preplaced_mask], torch.zeros(5, 2))
    assert torch.equal(output.flow_velocity[:, case.preplaced_mask, :2], torch.zeros(5, 1, 2))
    assert torch.equal(output.flow_velocity[:, case.fixed_mask | case.preplaced_mask, 2], torch.zeros(5, 2))
    assert torch.all(output.force_gates > 0.0)
    assert 0.45 < float(output.outline[2].detach()) < 0.95


def test_model_takes_one_optimizer_step() -> None:
    torch.manual_seed(11)
    case = _case()
    model = HCFPModel(ModelConfig(hidden_dim=24))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    output = model(case, population=3)
    loss = (
        output.precedence_logits.square().mean()
        + output.outline.square().mean()
        + output.center_residual.square().mean()
        + output.log_aspect_residual.square().mean()
        + output.flow_velocity.square().mean()
        + output.rank_score.mean()
    )
    loss.backward()
    optimizer.step()

    assert any(not torch.equal(before[name], value.detach()) for name, value in model.named_parameters())
