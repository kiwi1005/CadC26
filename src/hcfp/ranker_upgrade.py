"""Utilities for ranker-only model shape upgrades."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Sequence
from typing import Any

from hcfp.model import HCFPModel


def upgrade_candidate_metric_dim(
    model: HCFPModel,
    candidate_metric_dim: int,
    *,
    source_metadata: dict[str, Any] | None = None,
    preserve_trained_ranker: bool = False,
    feature_mean: Sequence[float] = (),
    feature_scale: Sequence[float] = (),
    feature_version: str | None = None,
    use_scene_embedding: bool | None = None,
) -> HCFPModel:
    """Return a model with a new ranker metric width.

    Non-ranker weights are loaded from ``model`` exactly. The ranker is
    reinitialized because its first linear layer depends on
    ``candidate_metric_dim``.
    """

    if type(candidate_metric_dim) is not int or candidate_metric_dim <= 0:
        raise ValueError("candidate_metric_dim must be a positive integer")
    target_config = replace(
        model.config,
        candidate_metric_dim=candidate_metric_dim,
        ranker_feature_mean=tuple(float(value) for value in feature_mean),
        ranker_feature_scale=tuple(float(value) for value in feature_scale),
        ranker_feature_version=(
            model.config.ranker_feature_version
            if feature_version is None
            else feature_version
        ),
        ranker_use_scene_embedding=(
            model.config.ranker_use_scene_embedding
            if use_scene_embedding is None
            else use_scene_embedding
        ),
    )
    if target_config == model.config:
        return model
    if preserve_trained_ranker or _metadata_has_trained_ranker(source_metadata):
        raise ValueError("cannot change candidate_metric_dim while preserving a trained ranker")

    upgraded = HCFPModel(target_config)
    ranker_shape_changed = (
        candidate_metric_dim != model.config.candidate_metric_dim
        or target_config.ranker_use_scene_embedding
        != model.config.ranker_use_scene_embedding
    )
    source_state = {
        key: value
        for key, value in model.state_dict().items()
        if not ranker_shape_changed or not key.startswith("ranker.")
    }
    incompatible = upgraded.load_state_dict(source_state, strict=False)
    expected_missing = (
        {key for key in upgraded.state_dict() if key.startswith("ranker.")}
        if ranker_shape_changed
        else set()
    )
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != expected_missing or unexpected:
        raise RuntimeError(
            "candidate_metric_dim upgrade touched non-ranker state: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return upgraded


def _metadata_has_trained_ranker(metadata: dict[str, Any] | None) -> bool:
    if metadata is None:
        return False
    capabilities = metadata.get("capabilities", {})
    if isinstance(capabilities, dict) and capabilities.get("ranker") is True:
        return True
    trained_heads = metadata.get("trained_heads", ())
    return isinstance(trained_heads, (list, tuple, set)) and "ranker" in trained_heads
