"""Small learned ranker for bounded deterministic contact-patch proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from hcfp.verify import bbox, boundary_missing


Tensor = torch.Tensor
CONTACT_POLICY_SCHEMA_VERSION = 1
CONTACT_FEATURE_VERSION = "bfod_contact_patch_v1"
CONTACT_FEATURE_NAMES = (
    "block_fraction",
    "group_fraction",
    "group_size_fraction",
    "patch_fraction",
    "grouping_before_fraction",
    "grouping_delta_fraction",
    "boundary_delta_fraction",
    "pair_dx_fraction",
    "pair_dy_fraction",
    "pair_l1_fraction",
    "pair_mid_x_fraction",
    "pair_mid_y_fraction",
    "bridge_degree_relative",
    "anchor_degree_relative",
    "pair_weight_relative",
    "bridge_area_relative",
    "anchor_area_relative",
    "bridge_log_aspect",
    "anchor_log_aspect",
    "patch_occupancy",
    "side_left",
    "side_right",
    "side_bottom",
    "side_top",
)


@dataclass(frozen=True)
class ContactPolicyConfig:
    hidden_dim: int = 32
    feature_dim: int = len(CONTACT_FEATURE_NAMES)

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.feature_dim != len(CONTACT_FEATURE_NAMES):
            raise ValueError("contact policy dimensions do not match the feature schema")


class ContactPolicy(nn.Module):
    """Rank already constructed contact patches before exact scoring."""

    def __init__(self, config: ContactPolicyConfig | None = None):
        super().__init__()
        self.config = config or ContactPolicyConfig()
        self.register_buffer("feature_mean", torch.zeros(self.config.feature_dim))
        self.register_buffer("feature_scale", torch.ones(self.config.feature_dim))
        self.net = nn.Sequential(
            nn.Linear(self.config.feature_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def set_normalization(self, mean: Tensor, scale: Tensor) -> None:
        expected = (self.config.feature_dim,)
        mean = torch.as_tensor(mean, dtype=torch.float32, device=self.feature_mean.device)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=self.feature_scale.device)
        if mean.shape != expected or scale.shape != expected:
            raise ValueError("contact policy normalization has the wrong feature shape")
        if not bool(torch.isfinite(mean).all() and torch.isfinite(scale).all()):
            raise ValueError("contact policy normalization must be finite")
        self.feature_mean.copy_(mean)
        self.feature_scale.copy_(scale.clamp_min(1.0e-6))

    def forward(self, features: Tensor) -> Tensor:
        values = torch.as_tensor(
            features, dtype=torch.float32, device=self.feature_mean.device
        )
        if values.ndim != 2 or values.shape[1] != self.config.feature_dim:
            raise ValueError("contact policy features must have shape [K,F]")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("contact policy features must be finite")
        normalized = (values - self.feature_mean) / self.feature_scale
        return self.net(normalized).reshape(-1)


def contact_candidate_features(
    case: Any,
    raw_case: Any,
    placements: Any,
    candidate: Any,
) -> Tensor:
    """Return runtime-visible features for one local contact proposal.

    This intentionally excludes exact scorer outputs.  The policy can only
    rank geometry and connectivity facts known after local repacking but before
    the expensive official score is requested.
    """

    boxes = torch.as_tensor(placements, dtype=torch.float64, device="cpu")
    proposal = torch.as_tensor(candidate.placement, dtype=torch.float64, device="cpu")
    if boxes.ndim != 2 or boxes.shape != proposal.shape or boxes.shape[1] != 4:
        raise ValueError("contact policy placements must have matching [N,4] shapes")
    n = int(boxes.shape[0])
    bridge = int(candidate.bridge_member)
    anchor = int(candidate.anchor_member)
    if not 0 <= bridge < n or not 0 <= anchor < n:
        raise ValueError("contact policy candidate member index is out of range")

    groups = torch.as_tensor(_field(case, "group_membership"), dtype=torch.bool)
    if groups.ndim != 2 or groups.shape[1] != n:
        raise ValueError("contact policy requires group_membership with shape [G,N]")
    group_index = int(candidate.group_index)
    if not 0 <= group_index < groups.shape[0]:
        raise ValueError("contact policy candidate group index is out of range")
    weights = torch.as_tensor(_field(case, "b2b_weight"), dtype=torch.float64)
    if weights.shape != (n, n):
        raise ValueError("contact policy requires b2b_weight with shape [N,N]")

    left, bottom, right, top = bbox(boxes)
    width = max(right - left, 1.0e-9)
    height = max(top - bottom, 1.0e-9)
    diagonal = max((width * width + height * height) ** 0.5, 1.0e-9)
    centers = boxes[:, :2] + 0.5 * boxes[:, 2:4]
    first = centers[bridge]
    second = centers[anchor]
    midpoint = 0.5 * (first + second)
    areas = boxes[:, 2] * boxes[:, 3]
    mean_area = max(float(areas.mean()), 1.0e-12)
    degree = weights.sum(dim=1)
    mean_degree = max(float(degree.mean()), 1.0e-12)
    boundary_before = int(torch.count_nonzero(boundary_missing(raw_case, boxes)))
    boundary_after = int(torch.count_nonzero(boundary_missing(raw_case, proposal)))
    boundary_bits = torch.as_tensor(_field(raw_case, "boundary_bits"))
    required_boundary_blocks = (
        boundary_bits.to(dtype=torch.bool).any(1)
        if boundary_bits.ndim == 2
        else boundary_bits.to(dtype=torch.int64) != 0
    )
    boundary_total = max(int(required_boundary_blocks.sum()), 1)
    members = tuple(int(member) for member in candidate.members)
    patch = boxes[list(members)]
    patch_left, patch_bottom, patch_right, patch_top = bbox(patch)
    patch_area = max((patch_right - patch_left) * (patch_top - patch_bottom), 1.0e-12)
    occupancy = float(areas[list(members)].sum()) / patch_area
    side = str(candidate.side)
    side_features = tuple(float(side == name) for name in ("left", "right", "bottom", "top"))
    result = torch.tensor(
        (
            n / 120.0,
            group_index / max(int(groups.shape[0]) - 1, 1),
            float(groups[group_index].sum()) / n,
            len(members) / n,
            float(candidate.grouping_before) / n,
            float(candidate.grouping_before - candidate.grouping_after) / n,
            (boundary_before - boundary_after) / boundary_total,
            abs(float(first[0] - second[0])) / width,
            abs(float(first[1] - second[1])) / height,
            float(torch.abs(first - second).sum()) / diagonal,
            (float(midpoint[0]) - left) / width,
            (float(midpoint[1]) - bottom) / height,
            float(degree[bridge]) / mean_degree,
            float(degree[anchor]) / mean_degree,
            float(weights[bridge, anchor]) / mean_degree,
            float(areas[bridge]) / mean_area,
            float(areas[anchor]) / mean_area,
            float(torch.log(boxes[bridge, 2] / boxes[bridge, 3])),
            float(torch.log(boxes[anchor, 2] / boxes[anchor, 3])),
            occupancy,
            *side_features,
        ),
        dtype=torch.float32,
    )
    if result.shape != (len(CONTACT_FEATURE_NAMES),) or not bool(torch.isfinite(result).all()):
        raise ValueError("contact policy produced invalid features")
    return result


def rank_contact_candidates(
    policy: ContactPolicy,
    case: Any,
    raw_case: Any,
    placements: Any,
    candidates: Iterable[Any],
) -> tuple[tuple[Any, float], ...]:
    """Score candidates stably; the caller retains only its bounded prefix."""

    items = tuple(candidates)
    if not items:
        return ()
    features = torch.stack(
        [contact_candidate_features(case, raw_case, placements, item) for item in items]
    )
    policy = policy.to(device="cpu").eval()
    with torch.inference_mode():
        scores = policy(features).detach().cpu().tolist()
    ranked = list(zip(items, (float(score) for score in scores), strict=True))
    ranked.sort(key=lambda item: (-item[1], _candidate_tie_key(item[0])))
    return tuple(ranked)


def save_contact_policy(
    policy: ContactPolicy,
    path: str | Path,
    *,
    metadata: dict[str, Any],
) -> str:
    """Save a compact, self-validating experiment-only policy checkpoint."""

    _validate_metadata(metadata)
    payload = {
        "schema_version": CONTACT_POLICY_SCHEMA_VERSION,
        "feature_version": CONTACT_FEATURE_VERSION,
        "config": asdict(policy.config),
        "metadata": metadata,
        "state_dict": {
            key: value.detach().cpu() for key, value in policy.state_dict().items()
        },
    }
    payload["state_hash"] = _state_hash(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return file_sha256(destination)


def load_contact_policy(path: str | Path) -> tuple[ContactPolicy, dict[str, Any]]:
    """Load a policy only when its feature and tensor payload are exact."""

    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != CONTACT_POLICY_SCHEMA_VERSION:
        raise ValueError("contact policy schema mismatch")
    if payload.get("feature_version") != CONTACT_FEATURE_VERSION:
        raise ValueError("contact policy feature schema mismatch")
    if payload.get("state_hash") != _state_hash(payload):
        raise ValueError("contact policy state hash mismatch")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("contact policy metadata is missing")
    _validate_metadata(metadata)
    policy = ContactPolicy(ContactPolicyConfig(**payload["config"]))
    policy.load_state_dict(payload["state_dict"], strict=True)
    return policy.eval(), {**metadata, "checkpoint_sha256": file_sha256(source)}


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _candidate_tie_key(candidate: Any) -> tuple[Any, ...]:
    return (
        int(candidate.group_index),
        int(candidate.bridge_member),
        int(candidate.anchor_member),
        str(candidate.side),
        tuple(int(member) for member in candidate.members),
    )


def _field(source: Any, name: str) -> Any:
    value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
    if value is None:
        raise ValueError(f"contact policy case is missing {name}")
    return value


def _state_hash(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema_version": payload.get("schema_version"),
                "feature_version": payload.get("feature_version"),
                "config": payload.get("config"),
                "metadata": payload.get("metadata"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for key, value in sorted(payload["state_dict"].items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _validate_metadata(metadata: dict[str, Any]) -> None:
    try:
        json.dumps(metadata, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("contact policy metadata must be JSON serializable") from exc


__all__ = [
    "CONTACT_FEATURE_NAMES",
    "CONTACT_FEATURE_VERSION",
    "ContactPolicy",
    "ContactPolicyConfig",
    "contact_candidate_features",
    "file_sha256",
    "load_contact_policy",
    "rank_contact_candidates",
    "save_contact_policy",
]
