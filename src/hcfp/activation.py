"""Training-only data contract for learned-tail activation decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Iterator

import torch

from hcfp.analytic import AnalyticResult
from hcfp.candidates import candidate_features
from hcfp.case import FloorplanCase
from hcfp.fallback import safe_shelf
from hcfp.verify import exact_metrics


Tensor = torch.Tensor
ACTIVATION_SCHEMA_VERSION = 2
ACTIVATION_POLICY_SCHEMA_VERSION = 1
ACTIVATION_FEATURE_VERSION = "pre_tail_v2"
ACTIVATION_FEATURE_DIM = 55


@dataclass(frozen=True)
class ActivationOutcome:
    feasible: bool
    soft_violation: float
    area_gap: float
    hpwl_gap: float
    objective: float
    runtime_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.feasible, bool):
            raise ValueError("activation outcome feasible must be boolean")
        values = (
            self.soft_violation,
            self.area_gap,
            self.hpwl_gap,
            self.objective,
            self.runtime_seconds,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("activation outcome values must be finite")
        if self.runtime_seconds < 0.0:
            raise ValueError("activation outcome runtime must be non-negative")


@dataclass(frozen=True)
class ActivationRecord:
    sample_id: str
    block_count: int
    checkpoint_hash: str
    config_hash: str
    features: Tensor
    tail_needed: bool
    quality_margin: float
    analytic: ActivationOutcome
    learned: ActivationOutcome
    failure_reason: str | None = None
    feature_version: str = ACTIVATION_FEATURE_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("activation sample_id must be a non-empty string")
        if (
            not isinstance(self.block_count, int)
            or isinstance(self.block_count, bool)
            or not 1 <= self.block_count <= 120
        ):
            raise ValueError("activation block_count must be in [1, 120]")
        _require_sha256("checkpoint_hash", self.checkpoint_hash)
        _require_sha256("config_hash", self.config_hash)
        if self.feature_version != ACTIVATION_FEATURE_VERSION:
            raise ValueError("activation feature version mismatch")
        if not isinstance(self.tail_needed, bool):
            raise ValueError("activation tail_needed must be boolean")
        if not math.isfinite(float(self.quality_margin)):
            raise ValueError("activation quality margin must be finite")
        expected_margin = self.learned.objective - self.analytic.objective
        if not math.isclose(self.quality_margin, expected_margin, rel_tol=1.0e-7, abs_tol=1.0e-7):
            raise ValueError("activation quality margin does not match outcomes")
        expected_tail_needed = self.learned.feasible and expected_margin < -1.0e-6
        if self.tail_needed != expected_tail_needed:
            raise ValueError("activation tail_needed does not match outcomes")
        if self.failure_reason is not None:
            if not isinstance(self.failure_reason, str) or not self.failure_reason.strip():
                raise ValueError("activation failure_reason must be a non-empty string")
            if self.learned.feasible or self.tail_needed:
                raise ValueError("activation failure_reason requires an infeasible learned outcome")
        features = torch.as_tensor(self.features, dtype=torch.float32, device="cpu").reshape(-1)
        if features.shape != (ACTIVATION_FEATURE_DIM,):
            raise ValueError(f"activation features must have shape [{ACTIVATION_FEATURE_DIM}]")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("activation features must be finite")
        object.__setattr__(self, "features", features)


@dataclass(frozen=True)
class ActivationPolicy:
    checkpoint_hash: str
    config_hash: str
    feature_mean: Tensor
    feature_scale: Tensor
    weight: Tensor
    bias: float
    threshold: float
    feature_version: str = ACTIVATION_FEATURE_VERSION

    def __post_init__(self) -> None:
        _require_sha256("checkpoint_hash", self.checkpoint_hash)
        _require_sha256("config_hash", self.config_hash)
        if self.feature_version != ACTIVATION_FEATURE_VERSION:
            raise ValueError("activation feature version mismatch")
        for name in ("feature_mean", "feature_scale", "weight"):
            value = torch.as_tensor(getattr(self, name), dtype=torch.float32, device="cpu").reshape(-1)
            if value.shape != (ACTIVATION_FEATURE_DIM,) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"activation policy {name} is invalid")
            object.__setattr__(self, name, value)
        if not bool((self.feature_scale > 0.0).all()):
            raise ValueError("activation policy feature_scale must be positive")
        if not math.isfinite(float(self.bias)):
            raise ValueError("activation policy bias must be finite")
        if not math.isfinite(float(self.threshold)) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("activation policy threshold must be in [0, 1]")

    def probability(self, features: Tensor) -> Tensor:
        values = torch.as_tensor(features, dtype=torch.float32, device="cpu")
        if values.shape[-1:] != (ACTIVATION_FEATURE_DIM,) or not bool(torch.isfinite(values).all()):
            raise ValueError("activation policy features are invalid")
        normalized = (values - self.feature_mean) / self.feature_scale
        return torch.sigmoid(normalized @ self.weight + self.bias)


def activation_features(
    case: FloorplanCase,
    analytic: AnalyticResult,
    learned_initial_boxes: Tensor,
    learned_rank_scores: Tensor,
) -> Tensor:
    """Return deterministic pre-learned-tail features on CPU FP32."""

    device = case.area.device
    learned = torch.as_tensor(learned_initial_boxes, dtype=torch.float32, device=device)
    if learned.ndim != 3 or learned.shape[1:] != (case.n, 4) or learned.shape[0] == 0:
        raise ValueError("learned_initial_boxes must have shape [K,N,4] with K > 0")
    scores = torch.as_tensor(learned_rank_scores, dtype=torch.float32, device=device).reshape(-1)
    if scores.shape != (learned.shape[0],) or not bool(torch.isfinite(scores).all()):
        raise ValueError("learned_rank_scores must be finite with shape [K]")

    selected = analytic.selected.to(device=device, dtype=torch.float32).unsqueeze(0)
    anchor = safe_shelf(case).to(device=device, dtype=torch.float32)
    metrics = candidate_features(case, torch.cat((selected, learned), dim=0), anchor)
    analytic_metric = metrics[0]
    learned_metric = metrics[1:]
    learned_min = learned_metric.amin(dim=0)

    static = torch.stack(
        (
            case.area.new_tensor(case.n / 120.0),
            case.fixed_mask.float().mean(),
            case.preplaced_mask.float().mean(),
            case.boundary_bits.any(dim=1).float().mean(),
            (case.cluster_id > 0).float().mean(),
            (case.mib_id > 0).float().mean(),
            (case.b2b_weight > 0).float().mean(),
            case.area.new_tensor(min(1.0, float(case.p2b_edges.shape[0]) / max(case.n, 1) / 8.0)),
        )
    ).float()

    exact_index = _source_index(analytic.incumbent_snapshot.get("exact_source"))
    if exact_index is None or not 0 <= exact_index < analytic.telemetry.hard_feasible.numel():
        raise ValueError("analytic exact source is missing or out of range")
    telemetry = torch.stack(
        (
            analytic.telemetry.hard_feasible[exact_index].float(),
            analytic.telemetry.projection_ok[exact_index].float(),
            analytic.telemetry.soft_violation[exact_index].float(),
        )
    ).to(device=device)

    ordered_scores = torch.sort(scores).values
    top2_margin = ordered_scores[1] - ordered_scores[0] if len(ordered_scores) > 1 else scores.new_zeros(())
    rank_stats = torch.stack(
        (
            ordered_scores[0],
            scores.mean(),
            scores.std(unbiased=False),
            top2_margin,
        )
    )
    features = torch.cat(
        (
            static,
            analytic_metric,
            learned_min,
            learned_metric.mean(dim=0),
            learned_metric.std(dim=0, unbiased=False),
            learned_min - analytic_metric,
            telemetry,
            rank_stats,
        )
    ).detach().to(device="cpu", dtype=torch.float32)
    if features.shape != (ACTIVATION_FEATURE_DIM,) or not bool(torch.isfinite(features).all()):
        raise ValueError("activation features are invalid")
    return features


def activation_outcome(
    case,
    placements,
    *,
    baseline_area: float,
    baseline_hpwl: float,
    runtime_seconds: float,
) -> ActivationOutcome:
    """Measure one raw placement with the runtime-independent v10 objective."""

    metrics = exact_metrics(
        case,
        placements,
        baseline_area=float(baseline_area),
        baseline_hpwl=float(baseline_hpwl),
        runtime_seconds=float(runtime_seconds),
    )
    quality = 1.0 + 0.5 * (max(0.0, metrics.area_gap) + max(0.0, metrics.hpwl_gap))
    objective = math.log(quality) + 2.0 * metrics.soft.total if metrics.verification.feasible else 10.0
    return ActivationOutcome(
        metrics.verification.feasible,
        metrics.soft.total,
        metrics.area_gap,
        metrics.hpwl_gap,
        objective,
        float(runtime_seconds),
    )


def write_activation_replay(records: Iterable[ActivationRecord], path: str | Path) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            checked = _validated_record(record)
            payload = {
                "schema_version": ACTIVATION_SCHEMA_VERSION,
                "feature_version": checked.feature_version,
                "sample_id": checked.sample_id,
                "block_count": checked.block_count,
                "checkpoint_hash": checked.checkpoint_hash,
                "config_hash": checked.config_hash,
                "features": checked.features.tolist(),
                "tail_needed": checked.tail_needed,
                "quality_margin": checked.quality_margin,
                "analytic": asdict(checked.analytic),
                "learned": asdict(checked.learned),
                "failure_reason": checked.failure_reason,
            }
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def iter_activation_replay(path: str | Path) -> Iterator[ActivationRecord]:
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            if payload.get("schema_version") != ACTIVATION_SCHEMA_VERSION:
                raise ValueError("activation replay schema mismatch")
            if payload.get("feature_version") != ACTIVATION_FEATURE_VERSION:
                raise ValueError("activation feature version mismatch")
            yield ActivationRecord(
                sample_id=payload.get("sample_id"),
                block_count=payload.get("block_count"),
                checkpoint_hash=payload.get("checkpoint_hash"),
                config_hash=payload.get("config_hash"),
                features=payload.get("features"),
                tail_needed=payload.get("tail_needed"),
                quality_margin=payload.get("quality_margin"),
                analytic=_outcome(payload.get("analytic")),
                learned=_outcome(payload.get("learned")),
                failure_reason=payload.get("failure_reason"),
                feature_version=payload.get("feature_version"),
            )


def fit_activation_policy(
    train_records: Iterable[ActivationRecord],
    calibration_records: Iterable[ActivationRecord],
    *,
    steps: int = 1000,
    learning_rate: float = 1.0e-2,
    device: str | torch.device = "cpu",
) -> tuple[ActivationPolicy, list[float]]:
    """Fit a deterministic class-weighted linear policy and calibrate recall."""

    train = list(train_records)
    calibration = list(calibration_records)
    checkpoint_hash, config_hash = _compatible_splits(train, calibration)
    if steps <= 0 or learning_rate <= 0.0:
        raise ValueError("activation training steps and learning rate must be positive")
    x_train, y_train = _record_matrix(train, device)
    x_calibration, y_calibration = _record_matrix(calibration, device)
    positives = int(y_train.sum().item())
    if positives == 0 or positives == len(train):
        raise ValueError("activation training requires positive and negative records")
    calibration_positives = int(y_calibration.sum().item())
    if calibration_positives == 0:
        raise ValueError("activation calibration requires positive records")

    mean = x_train.mean(dim=0)
    scale = x_train.std(dim=0, unbiased=False).clamp_min(1.0e-5)
    weight = torch.zeros(ACTIVATION_FEATURE_DIM, device=x_train.device, requires_grad=True)
    bias = torch.zeros((), device=x_train.device, requires_grad=True)
    optimizer = torch.optim.AdamW((weight, bias), lr=learning_rate, weight_decay=1.0e-3)
    positive_weight = x_train.new_tensor((len(train) - positives) / positives)
    history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = ((x_train - mean) / scale) @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            y_train,
            pos_weight=positive_weight,
        )
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    with torch.inference_mode():
        probabilities = torch.sigmoid(((x_calibration - mean) / scale) @ weight + bias)
        threshold = float(probabilities[y_calibration.bool()].min().detach().cpu())
    policy = ActivationPolicy(
        checkpoint_hash,
        config_hash,
        mean.detach().cpu(),
        scale.detach().cpu(),
        weight.detach().cpu(),
        float(bias.detach().cpu()),
        threshold,
    )
    return policy, history


def activation_policy_metrics(
    policy: ActivationPolicy,
    records: Iterable[ActivationRecord],
    *,
    force_large_min: int = 106,
) -> dict[str, object]:
    materialized = list(records)
    _policy_compatible(policy, materialized)
    x, y = _record_matrix(materialized, "cpu")
    probabilities = policy.probability(x)
    large = torch.tensor(
        [record.block_count >= force_large_min for record in materialized],
        dtype=torch.bool,
    )
    active = large | (probabilities >= policy.threshold)
    positive = y.bool()
    true_positive = active & positive
    false_skip = positive & ~active
    return {
        "records": len(materialized),
        "positives": int(positive.sum().item()),
        "activated": int(active.sum().item()),
        "activation_rate": float(active.float().mean()),
        "recall": float(true_positive.sum() / positive.sum().clamp_min(1)),
        "precision": float(true_positive.sum() / active.sum().clamp_min(1)),
        "false_skip_sample_ids": [
            record.sample_id for record, skipped in zip(materialized, false_skip.tolist()) if skipped
        ],
        "probabilities": [float(value) for value in probabilities.tolist()],
    }


def save_activation_policy(policy: ActivationPolicy, path: str | Path) -> str:
    payload = _policy_payload(policy)
    payload_hash = _json_hash(payload)
    payload["payload_hash"] = payload_hash
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload_hash


def load_activation_policy(path: str | Path) -> ActivationPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload_hash = payload.pop("payload_hash", None)
    if payload_hash != _json_hash(payload):
        raise ValueError("activation policy hash mismatch")
    if payload.get("schema_version") != ACTIVATION_POLICY_SCHEMA_VERSION:
        raise ValueError("activation policy schema mismatch")
    return ActivationPolicy(
        checkpoint_hash=payload.get("checkpoint_hash"),
        config_hash=payload.get("config_hash"),
        feature_mean=payload.get("feature_mean"),
        feature_scale=payload.get("feature_scale"),
        weight=payload.get("weight"),
        bias=payload.get("bias"),
        threshold=payload.get("threshold"),
        feature_version=payload.get("feature_version"),
    )


def _source_index(source: object) -> int | None:
    value = str(source)
    if value == "fallback":
        return 0
    if not value.startswith("candidate_"):
        return None
    try:
        return int(value.removeprefix("candidate_"))
    except ValueError:
        return None


def _require_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"activation {name} must be a SHA256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"activation {name} must be a SHA256 hex digest") from exc


def _validated_record(record: ActivationRecord) -> ActivationRecord:
    if not isinstance(record, ActivationRecord):
        raise ValueError("activation replay accepts ActivationRecord values")
    return record


def _outcome(payload: object) -> ActivationOutcome:
    if not isinstance(payload, dict):
        raise ValueError("activation outcome payload must be an object")
    try:
        return ActivationOutcome(**payload)
    except TypeError as exc:
        raise ValueError("activation outcome payload is invalid") from exc


def _compatible_splits(
    train: list[ActivationRecord],
    calibration: list[ActivationRecord],
) -> tuple[str, str]:
    if not train or not calibration:
        raise ValueError("activation train and calibration splits must be non-empty")
    for name, records in (("train", train), ("calibration", calibration)):
        ids = [record.sample_id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"activation {name} split contains duplicate samples")
    overlap = {record.sample_id for record in train} & {record.sample_id for record in calibration}
    if overlap:
        raise ValueError("activation train and calibration sample overlap")
    hashes = {(record.checkpoint_hash, record.config_hash) for record in train + calibration}
    if len(hashes) != 1:
        raise ValueError("activation replay checkpoint or config mismatch")
    return next(iter(hashes))


def _record_matrix(
    records: list[ActivationRecord],
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    if not records:
        raise ValueError("activation records must be non-empty")
    x = torch.stack([record.features for record in records]).to(device=device, dtype=torch.float32)
    y = torch.tensor([record.tail_needed for record in records], device=device, dtype=torch.float32)
    return x, y


def _policy_compatible(policy: ActivationPolicy, records: list[ActivationRecord]) -> None:
    if not records:
        raise ValueError("activation records must be non-empty")
    if any(
        record.checkpoint_hash != policy.checkpoint_hash
        or record.config_hash != policy.config_hash
        or record.feature_version != policy.feature_version
        for record in records
    ):
        raise ValueError("activation policy and replay mismatch")


def _policy_payload(policy: ActivationPolicy) -> dict[str, object]:
    return {
        "schema_version": ACTIVATION_POLICY_SCHEMA_VERSION,
        "feature_version": policy.feature_version,
        "checkpoint_hash": policy.checkpoint_hash,
        "config_hash": policy.config_hash,
        "feature_mean": policy.feature_mean.tolist(),
        "feature_scale": policy.feature_scale.tolist(),
        "weight": policy.weight.tolist(),
        "bias": policy.bias,
        "threshold": policy.threshold,
    }


def _json_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
