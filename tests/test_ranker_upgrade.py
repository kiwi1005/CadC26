from __future__ import annotations

import pytest
import torch

from hcfp.model import HCFPModel, ModelConfig
from hcfp.ranker_upgrade import upgrade_candidate_metric_dim


def test_upgrade_candidate_metric_dim_preserves_non_ranker_state_bitwise() -> None:
    torch.manual_seed(1)
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1, candidate_metric_dim=8))
    source_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    upgraded = upgrade_candidate_metric_dim(model, 14)

    assert upgraded is not model
    assert upgraded.config.candidate_metric_dim == 14
    assert upgraded.ranker.net[0].in_features == model.config.hidden_dim + 14
    assert model.ranker.net[0].in_features == model.config.hidden_dim + 8
    upgraded_state = upgraded.state_dict()
    for key, value in source_state.items():
        if key.startswith("ranker."):
            continue
        assert torch.equal(upgraded_state[key], value), key


def test_upgrade_candidate_metric_dim_reinitializes_only_ranker_shape() -> None:
    torch.manual_seed(2)
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1, candidate_metric_dim=8))

    upgraded = upgrade_candidate_metric_dim(model, 12)

    assert upgraded.ranker.net[0].weight.shape == (16, 28)
    assert upgraded.ranker.net[0].bias.shape == model.ranker.net[0].bias.shape
    assert upgraded.ranker.net[2].weight.shape == model.ranker.net[2].weight.shape
    assert upgraded.ranker.net[2].bias.shape == model.ranker.net[2].bias.shape


def test_upgrade_candidate_metric_dim_same_dim_returns_same_model() -> None:
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1, candidate_metric_dim=8))

    assert upgrade_candidate_metric_dim(model, 8) is model


def test_upgrade_configures_ranker_normalization_without_changing_weights() -> None:
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1, candidate_metric_dim=8))
    source_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    upgraded = upgrade_candidate_metric_dim(
        model,
        8,
        feature_mean=[1.0] * 8,
        feature_scale=[2.0] * 8,
    )

    assert upgraded is not model
    assert upgraded.config.ranker_feature_mean == (1.0,) * 8
    assert upgraded.config.ranker_feature_scale == (2.0,) * 8
    for key, value in source_state.items():
        assert torch.equal(upgraded.state_dict()[key], value), key


def test_ranker_applies_configured_feature_normalization() -> None:
    torch.manual_seed(3)
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1, candidate_metric_dim=8))
    normalized = upgrade_candidate_metric_dim(
        model,
        8,
        feature_mean=[1.0] * 8,
        feature_scale=[2.0] * 8,
    )
    embedding = torch.randn(4, 16)
    raw = torch.randn(3, 8)

    expected = model.ranker(embedding, 3, (raw - 1.0) / 2.0)
    actual = normalized.ranker(embedding, 3, raw)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_upgrade_can_remove_scene_embedding_from_ranker_only() -> None:
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1, candidate_metric_dim=8))

    upgraded = upgrade_candidate_metric_dim(
        model,
        8,
        use_scene_embedding=False,
    )

    assert upgraded.config.ranker_use_scene_embedding is False
    assert upgraded.ranker.net[0].in_features == 8
    for key, value in model.state_dict().items():
        if not key.startswith("ranker."):
            assert torch.equal(upgraded.state_dict()[key], value), key


@pytest.mark.parametrize("bad_dim", (0, -1, 1.5, True))
def test_upgrade_candidate_metric_dim_rejects_invalid_dim(bad_dim: object) -> None:
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1, candidate_metric_dim=8))

    with pytest.raises(ValueError, match="candidate_metric_dim"):
        upgrade_candidate_metric_dim(model, bad_dim)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "metadata",
    (
        {"capabilities": {"ranker": True}, "trained_heads": []},
        {"capabilities": {}, "trained_heads": ["ranker"]},
    ),
)
def test_upgrade_candidate_metric_dim_rejects_trained_ranker_metadata(metadata: dict[str, object]) -> None:
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1, candidate_metric_dim=8))

    with pytest.raises(ValueError, match="trained ranker"):
        upgrade_candidate_metric_dim(model, 12, source_metadata=metadata)


def test_upgrade_reuses_trained_shadow_ranker_when_feature_contract_is_identical() -> None:
    model = HCFPModel(
        ModelConfig(
            hidden_dim=16,
            encoder_layers=1,
            candidate_metric_dim=8,
            ranker_feature_mean=(1.0,) * 8,
            ranker_feature_scale=(2.0,) * 8,
            ranker_feature_version="test_features_v1",
            ranker_use_scene_embedding=False,
        )
    )

    continued = upgrade_candidate_metric_dim(
        model,
        8,
        source_metadata={"capabilities": {"ranker": False}, "trained_heads": ["ranker"]},
        feature_mean=(1.0,) * 8,
        feature_scale=(2.0,) * 8,
        feature_version="test_features_v1",
        use_scene_embedding=False,
    )

    assert continued is model


def test_upgrade_candidate_metric_dim_rejects_call_level_preserve_ranker() -> None:
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1, candidate_metric_dim=8))

    with pytest.raises(ValueError, match="trained ranker"):
        upgrade_candidate_metric_dim(model, 12, preserve_trained_ranker=True)
