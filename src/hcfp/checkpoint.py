"""Versioned checkpoint helpers for HCFP models."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from hcfp.model import HCFPModel, ModelConfig


SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
RUNTIME_NORMALIZATION = {"coordinate_scale": "sqrt_total_area_v1", "geometry_dtype": "float32"}
_METADATA_FIELDS = (
    "capabilities",
    "trained_heads",
    "training_objective_version",
    "training_objective_weights",
    "parent_state_hash",
)
_REQUIRED_METADATA_FIELDS = (
    "capabilities",
    "trained_heads",
    "training_objective_version",
    "parent_state_hash",
)
_DEFAULT_CAPABILITIES = {"flow": False}


def save_checkpoint(
    model: HCFPModel,
    path: str | Path,
    normalization: dict[str, Any] | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    checkpoint_metadata = _normalize_metadata(metadata)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(model.config),
        "normalization": normalization or {},
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        **checkpoint_metadata,
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
    schema_version = payload.get("schema_version")
    if schema_version not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}:
        raise ValueError("checkpoint schema mismatch")
    checkpoint_metadata = _metadata_from_payload(payload, int(schema_version))
    if payload.get("state_hash") != _payload_hash(payload):
        raise ValueError("checkpoint hash mismatch")
    config = ModelConfig(**payload["config"])
    if expected_config is not None and expected_config != config:
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
        **checkpoint_metadata,
    }


def _payload_hash(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    schema_version = payload.get("schema_version")
    if schema_version == SCHEMA_VERSION:
        metadata = _metadata_from_payload(payload, SCHEMA_VERSION)
        if "training_objective_weights" not in payload:
            metadata = dict(metadata)
            metadata.pop("training_objective_weights", None)
        digest.update(
            json.dumps(
                {"schema_version": SCHEMA_VERSION, **metadata},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    elif schema_version != LEGACY_SCHEMA_VERSION:
        raise ValueError("checkpoint schema mismatch")
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


def _metadata_from_payload(payload: dict[str, Any], schema_version: int) -> dict[str, Any]:
    if schema_version == LEGACY_SCHEMA_VERSION:
        return _normalize_metadata(None)
    missing = [name for name in _REQUIRED_METADATA_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"checkpoint metadata missing: {missing}")
    return _normalize_metadata({name: payload.get(name) for name in _METADATA_FIELDS})


def _normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    supplied = dict(metadata or {})
    unknown = sorted(set(supplied) - set(_METADATA_FIELDS))
    if unknown:
        raise ValueError(f"unsupported checkpoint metadata: {unknown}")

    raw_capabilities = supplied.get("capabilities", {})
    if not isinstance(raw_capabilities, dict):
        raise ValueError("checkpoint capabilities must be a mapping")
    capabilities = dict(_DEFAULT_CAPABILITIES)
    for name, enabled in raw_capabilities.items():
        if not isinstance(name, str) or not name or type(enabled) is not bool:
            raise ValueError("checkpoint capabilities must map non-empty names to booleans")
        capabilities[name] = enabled

    raw_heads = supplied.get("trained_heads", [])
    if not isinstance(raw_heads, (list, tuple)) or any(
        not isinstance(name, str) or not name for name in raw_heads
    ):
        raise ValueError("checkpoint trained_heads must be a sequence of non-empty names")
    trained_heads = sorted(set(raw_heads))
    missing_capability_heads = sorted(
        name for name, enabled in capabilities.items() if enabled and name not in trained_heads
    )
    if missing_capability_heads:
        raise ValueError(
            "enabled capabilities require matching trained heads: "
            f"{missing_capability_heads}"
        )

    objective = supplied.get("training_objective_version")
    if objective is not None and (not isinstance(objective, str) or not objective):
        raise ValueError("training_objective_version must be a non-empty string or null")
    weights = supplied.get("training_objective_weights")
    if weights is not None:
        if not isinstance(weights, dict):
            raise ValueError("training_objective_weights must be a mapping or null")
        required = {"name", "listwise", "feasibility_order", "pointwise", "top_one"}
        if set(weights) != required:
            raise ValueError("training_objective_weights has an unsupported schema")
        if not isinstance(weights["name"], str) or not weights["name"]:
            raise ValueError("training_objective_weights name must be a non-empty string")
        for name in ("listwise", "feasibility_order", "pointwise", "top_one"):
            if type(weights[name]) not in {float, int}:
                raise ValueError("training_objective_weights values must be numeric")
            if not math.isfinite(float(weights[name])) or float(weights[name]) < 0.0:
                raise ValueError(
                    "training_objective_weights values must be finite and non-negative"
                )
        weights = {
            "name": weights["name"],
            "listwise": float(weights["listwise"]),
            "feasibility_order": float(weights["feasibility_order"]),
            "pointwise": float(weights["pointwise"]),
            "top_one": float(weights["top_one"]),
        }
        if not any(weights[name] > 0.0 for name in required - {"name"}):
            raise ValueError("training_objective_weights must enable at least one loss term")
        if objective is not None and weights["name"] != objective:
            raise ValueError(
                "training_objective_weights name must match training_objective_version"
            )
    parent_hash = supplied.get("parent_state_hash")
    if parent_hash is not None and (
        not isinstance(parent_hash, str)
        or len(parent_hash) != 64
        or any(character not in "0123456789abcdef" for character in parent_hash.lower())
    ):
        raise ValueError("parent_state_hash must be a SHA-256 hex digest or null")
    return {
        "capabilities": capabilities,
        "trained_heads": trained_heads,
        "training_objective_version": objective,
        "training_objective_weights": weights,
        "parent_state_hash": parent_hash,
    }
