from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hcfp.checkpoint import load_checkpoint, save_checkpoint
from hcfp.model import HCFPModel, ModelConfig


def test_checkpoint_roundtrip_preserves_config_metadata_and_weights(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = HCFPModel(ModelConfig(hidden_dim=16))
    path = tmp_path / "model.pt"
    normalization = {"area_mean": 0.2, "area_std": 0.7}

    saved_hash = save_checkpoint(model, path, normalization)
    loaded, metadata = load_checkpoint(path, expected_config=model.config, expected_normalization=normalization)

    assert metadata["state_hash"] == saved_hash
    assert metadata["normalization"] == normalization
    assert loaded.config == model.config
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_checkpoint_fails_closed_on_hash_and_config_mismatch(tmp_path: Path) -> None:
    model = HCFPModel(ModelConfig(hidden_dim=16))
    path = tmp_path / "model.pt"
    save_checkpoint(model, path, {"scale": 1.0})

    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["state_hash"] = "0" * 64
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)

    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(broken, expected_config=model.config, expected_normalization={"scale": 1.0})
    with pytest.raises(ValueError, match="config mismatch"):
        load_checkpoint(path, expected_config=ModelConfig(hidden_dim=24), expected_normalization={"scale": 1.0})
    with pytest.raises(ValueError, match="normalization mismatch"):
        load_checkpoint(path, expected_config=model.config, expected_normalization={"scale": 2.0})
