from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import torch

from hcfp.checkpoint import SCHEMA_VERSION, load_checkpoint, save_checkpoint
from hcfp.model import HCFPModel, ModelConfig


def test_checkpoint_roundtrip_preserves_config_metadata_and_weights(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = HCFPModel(ModelConfig(hidden_dim=16))
    path = tmp_path / "model.pt"
    normalization = {"area_mean": 0.2, "area_std": 0.7}

    checkpoint_metadata = {
        "capabilities": {"flow": True},
        "trained_heads": ["structure", "flow"],
        "training_objective_version": "supervised_loss_v1",
        "parent_state_hash": "a" * 64,
    }
    saved_hash = save_checkpoint(model, path, normalization, metadata=checkpoint_metadata)
    loaded, metadata = load_checkpoint(path, expected_config=model.config, expected_normalization=normalization)

    assert metadata["schema_version"] == SCHEMA_VERSION == 2
    assert metadata["state_hash"] == saved_hash
    assert metadata["normalization"] == normalization
    assert metadata["capabilities"] == {"flow": True}
    assert metadata["trained_heads"] == ["flow", "structure"]
    assert metadata["training_objective_version"] == "supervised_loss_v1"
    assert metadata["parent_state_hash"] == "a" * 64
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


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("capabilities", {"flow": False}),
        ("trained_heads", ["flow", "ranker"]),
        ("training_objective_version", "other_objective_v1"),
        ("parent_state_hash", "b" * 64),
    ],
)
def test_v2_hash_covers_capability_metadata(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    path = tmp_path / "model.pt"
    save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16)),
        path,
        metadata={
            "capabilities": {"flow": True},
            "trained_heads": ["flow"],
            "training_objective_version": "supervised_loss_v1",
            "parent_state_hash": "a" * 64,
        },
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload[field] = replacement
    broken = tmp_path / f"broken-{field}.pt"
    torch.save(payload, broken)

    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(broken)


def test_schema_v1_load_preserves_legacy_hash_and_disables_capabilities(tmp_path: Path) -> None:
    model = HCFPModel(ModelConfig(hidden_dim=16))
    payload = {
        "schema_version": 1,
        "config": asdict(model.config),
        "normalization": {"scale": 1.0},
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    payload["state_hash"] = _legacy_hash(payload)
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)

    loaded, metadata = load_checkpoint(path, expected_normalization={"scale": 1.0})

    assert loaded.config == model.config
    assert metadata["schema_version"] == 1
    assert metadata["state_hash"] == payload["state_hash"]
    assert metadata["capabilities"] == {"flow": False}
    assert metadata["trained_heads"] == []
    assert metadata["training_objective_version"] is None
    assert metadata["parent_state_hash"] is None


def _legacy_hash(payload: dict[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(payload["config"], sort_keys=True, separators=(",", ":")).encode())
    digest.update(
        json.dumps(payload.get("normalization", {}), sort_keys=True, separators=(",", ":")).encode()
    )
    state_dict = payload["state_dict"]
    assert isinstance(state_dict, dict)
    for key, value in sorted(state_dict.items()):
        assert isinstance(key, str) and isinstance(value, torch.Tensor)
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        for chunk in tensor.view(torch.uint8).reshape(-1).split(1024 * 1024):
            digest.update(bytes(chunk.tolist()))
    return digest.hexdigest()
