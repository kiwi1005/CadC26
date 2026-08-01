"""Versioned checkpoint helpers for HCFP models."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from hcfp.model import HCFPModel, ModelConfig


SCHEMA_VERSION = 1
RUNTIME_NORMALIZATION = {"coordinate_scale": "sqrt_total_area_v1", "geometry_dtype": "float32"}


def save_checkpoint(model: HCFPModel, path: str | Path, normalization: dict[str, Any] | None = None) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(model.config),
        "normalization": normalization or {},
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    payload["state_hash"] = _payload_hash(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return str(payload["state_hash"])


def load_checkpoint(
    path: str | Path,
    *,
    expected_config: ModelConfig | None = None,
    expected_normalization: dict[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[HCFPModel, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint schema mismatch")
    if payload.get("state_hash") != _payload_hash(payload):
        raise ValueError("checkpoint hash mismatch")
    config = ModelConfig(**payload["config"])
    if expected_config is not None and asdict(expected_config) != payload["config"]:
        raise ValueError("checkpoint config mismatch")
    normalization = payload.get("normalization", {})
    if expected_normalization is not None and normalization != expected_normalization:
        raise ValueError("checkpoint normalization mismatch")
    model = HCFPModel(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, {
        "schema_version": payload["schema_version"],
        "config": payload["config"],
        "normalization": normalization,
        "state_hash": payload["state_hash"],
    }


def _payload_hash(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(payload["config"], sort_keys=True, separators=(",", ":")).encode())
    digest.update(json.dumps(payload.get("normalization", {}), sort_keys=True, separators=(",", ":")).encode())
    for key, value in sorted(payload["state_dict"].items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        raw = tensor.view(torch.uint8).reshape(-1)
        for chunk in raw.split(1024 * 1024):
            digest.update(bytes(chunk.tolist()))
    return digest.hexdigest()
