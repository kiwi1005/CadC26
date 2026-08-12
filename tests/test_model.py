from __future__ import annotations

import torch

from hcfp.case import from_official
from hcfp.collective import dynamic_pair_features
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


def _free_case():
    return from_official(
        3,
        [4.0, 9.0, 16.0],
        [[0, 1, 2.0], [1, 2, 3.0]],
        [],
        [],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
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
    assert output.contact_logits is None
    assert output.boundary_order_scores is None
    assert output.mib_log_aspect is None
    assert not any(name.startswith("constraints.") for name in HCFPModel(cfg).state_dict())


def test_constraint_heads_are_optional_and_candidate_independent() -> None:
    torch.manual_seed(13)
    case = _case()
    cfg = ModelConfig(hidden_dim=24, encoder_layers=1, constraint_enabled=True)
    output = HCFPModel(cfg)(case, population=3)

    assert output.contact_logits is not None
    assert output.boundary_order_scores is not None
    assert output.mib_log_aspect is not None
    assert output.contact_logits.shape == (case.n, case.n, 5)
    assert output.boundary_order_scores.shape == (case.n, 4)
    assert output.mib_log_aspect.shape == (case.mib_membership.shape[0],)


def test_btree_heads_are_optional_and_score_parent_branches() -> None:
    case = _case()
    output = HCFPModel(
        ModelConfig(hidden_dim=24, encoder_layers=1, btree_enabled=True)
    )(case, population=2)

    assert output.btree_root_logits is not None
    assert output.btree_edge_logits is not None
    assert output.btree_root_logits.shape == (case.n,)
    assert output.btree_edge_logits.shape == (case.n, case.n, 2)
    assert torch.all(output.btree_edge_logits.diagonal(dim1=0, dim2=1) < -1.0e20)


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


def test_constraint_heads_receive_gradients_when_enabled() -> None:
    torch.manual_seed(17)
    case = _case()
    model = HCFPModel(ModelConfig(hidden_dim=24, encoder_layers=1, constraint_enabled=True))
    output = model(case, population=2)
    assert output.contact_logits is not None
    assert output.boundary_order_scores is not None
    assert output.mib_log_aspect is not None

    loss = (
        output.contact_logits.square().mean()
        + output.boundary_order_scores.square().mean()
        + output.mib_log_aspect.square().mean()
    )
    loss.backward()

    grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("constraints.")
    ]
    assert grads
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_legacy_state_loads_into_constraint_model_with_only_missing_head_keys() -> None:
    legacy = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1))
    upgraded = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1, constraint_enabled=True))

    incompatible = upgraded.load_state_dict(legacy.state_dict(), strict=False)

    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(name.startswith("constraints.") for name in incompatible.missing_keys)


def test_collective_head_is_optional_neutral_and_honors_hard_masks() -> None:
    torch.manual_seed(23)
    case = _case()
    disabled = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1))
    assert not hasattr(disabled, "collective")
    assert not any(name.startswith("collective.") for name in disabled.state_dict())

    model = HCFPModel(
        ModelConfig(
            hidden_dim=16,
            encoder_layers=1,
            collective_enabled=True,
            collective_message_dim=12,
            collective_passes=2,
        )
    )
    center = torch.tensor(
        [
            [[1.0, 1.0], [5.0, 1.0], [2.0, 5.0], [7.0, 6.0]],
            [[2.0, 2.0], [5.0, 2.0], [3.0, 5.0], [8.0, 6.0]],
        ]
    )
    dimensions = torch.tensor(
        [
            [[2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
            [[2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
        ]
    )
    pairs = dynamic_pair_features(case, center, dimensions)
    geometry = torch.cat(
        (torch.log(dimensions[..., :1] / dimensions[..., 1:]), dimensions),
        dim=-1,
    )
    output = model.collective(
        case,
        model.encoder(case),
        geometry,
        pairs.features,
        pairs.pair_mask,
        0.25,
    )

    assert output.velocity.shape == (2, case.n, 3)
    assert output.force_gates.shape == (2, case.n, 7)
    assert torch.equal(output.velocity, torch.zeros_like(output.velocity))
    assert torch.equal(output.force_gates, torch.ones_like(output.force_gates))
    assert torch.equal(output.velocity[:, case.preplaced_mask, :2], torch.zeros(2, 1, 2))
    assert torch.equal(
        output.velocity[:, case.fixed_mask | case.preplaced_mask, 2],
        torch.zeros(2, 2),
    )


def test_collective_message_and_gate_parameters_receive_gradients() -> None:
    torch.manual_seed(29)
    case = _free_case()
    model = HCFPModel(
        ModelConfig(
            hidden_dim=12,
            encoder_layers=1,
            collective_enabled=True,
            collective_message_dim=10,
            collective_passes=2,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2)
    center = torch.tensor([[[1.0, 1.0], [4.0, 1.0], [2.0, 5.0]]])
    dimensions = torch.tensor([[[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]])
    pairs = dynamic_pair_features(case, center, dimensions)
    geometry = torch.cat(
        (torch.log(dimensions[..., :1] / dimensions[..., 1:]), dimensions),
        dim=-1,
    )

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model.collective(
            case,
            model.encoder(case),
            geometry,
            pairs.features,
            pairs.pair_mask,
            0.5,
        )
        loss = (output.velocity - 0.01).square().mean()
        loss = loss + (output.force_gates - 1.20).square().mean()
        loss.backward()
        optimizer.step()

    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("collective.")
    }
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients.values())
    assert gradients["collective.pair.weight"].abs().sum() > 0.0
    assert gradients["collective.force_gates.3.weight"].abs().sum() > 0.0
