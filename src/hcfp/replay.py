"""Exact-tail replay records for repair-aware candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
from torch.nn import functional as F

from hcfp.candidates import candidate_features
from hcfp.data import DataSample, sample_from_payload, sample_to_payload
from hcfp.fallback import safe_shelf
from hcfp.model import HCFPModel
from hcfp.verify import ALPHA, BETA


Tensor = torch.Tensor
OFFICIAL_TARGET_KIND = "official_v10_lexicographic_v1"
LEGACY_TARGET_KIND = "legacy_proxy_v1"
V3_TARGET_KIND = OFFICIAL_TARGET_KIND
V3_SCHEMA_VERSION = 3
_LEGACY_SCHEMA_VERSIONS = (1, 2)
_VALID_CANDIDATE_KINDS = {
    "safe",
    "analytic",
    "topology",
    "flow",
    "collective",
    "reconfigured",
    "projected",
    "learned",
}
_VALID_CANDIDATE_SOURCE_TYPES = {
    "safe",
    "analytic",
    "topology",
    "constraint",
    "flow",
    "collective",
    "reconfigured",
    "projected",
    "learned",
    "post_repair",
}
_VALID_FEASIBILITY_TIERS = {0, 1, 2}
_HEX = set("0123456789abcdef")


@dataclass(frozen=True)
class ReplayRecord:
    sample: DataSample
    checkpoint_hash: str
    candidate_features: Tensor
    target_score: Tensor
    target_kind: str = OFFICIAL_TARGET_KIND
    candidate_row_ids: tuple[str, ...] | None = None
    candidate_source_indices: Tensor | None = None
    candidate_kinds: tuple[str, ...] | None = None
    candidate_source_types: tuple[str, ...] | None = None
    candidate_geometry_sha256: tuple[str, ...] | None = None
    feasibility_tier: Tensor | None = None
    target_rank: Tensor | None = None
    candidate_stage: str | None = None
    candidate_population: int | None = None
    population_seed: int | None = None


def official_replay_scores(
    case,
    telemetry,
    *,
    baseline_area: Tensor | float,
    baseline_hpwl: Tensor | float,
) -> Tensor:
    """Rank feasible candidates by v10 quality and exact-tail failures by repair residual."""

    hpwl = telemetry.hpwl.float() * float(case.scale)
    area = telemetry.bbox_area.float() * float(case.scale) ** 2
    soft = telemetry.soft_violation.float()
    area_base = torch.as_tensor(baseline_area, dtype=torch.float32, device=area.device).reshape(())
    hpwl_base = torch.as_tensor(baseline_hpwl, dtype=torch.float32, device=hpwl.device).reshape(())
    overlap = telemetry.projected_overlap.float()
    displacement = telemetry.projection_displacement.float()
    values = torch.cat(
        (
            hpwl.reshape(-1),
            area.reshape(-1),
            soft.reshape(-1),
            overlap.reshape(-1),
            displacement.reshape(-1),
            area_base[None],
            hpwl_base[None],
        )
    )
    if not bool(torch.isfinite(values).all()) or float(area_base) < 0.0 or float(hpwl_base) < 0.0:
        raise ValueError("official replay metrics must be finite and baselines non-negative")
    hpwl_gap = (hpwl - hpwl_base) / hpwl_base.clamp_min(1.0e-6)
    area_gap = (area - area_base) / area_base.clamp_min(1.0e-6)
    quality = 1.0 + ALPHA * (hpwl_gap.clamp_min(0.0) + area_gap.clamp_min(0.0))
    feasible_score = torch.log(quality) + BETA * soft
    feasible = telemetry.hard_feasible.to(device=feasible_score.device, dtype=torch.bool)
    infeasible_floor = feasible_score[feasible].max() + 1.0 if bool(feasible.any()) else feasible_score.new_tensor(1.0)
    repair_score = (
        torch.log1p(overlap.clamp_min(0.0))
        + 0.1 * telemetry.overlap_components.to(device=overlap.device, dtype=torch.float32)
        + 0.1 * (~telemetry.projection_ok.to(device=overlap.device, dtype=torch.bool)).float()
        + 0.01 * displacement
    )
    return torch.where(feasible, feasible_score, infeasible_floor + repair_score)


def record_from_analysis(
    sample: DataSample,
    checkpoint_hash: str,
    raw_candidates: Tensor,
    telemetry,
    *,
    population: int,
    population_seed: int = 0,
) -> ReplayRecord:
    """Label learned initial candidates with their exact projected outcomes."""

    start, stop = population + 1, 2 * population + 1
    boxes = raw_candidates[start:stop]
    features = candidate_features(sample.case.to(device=boxes.device), boxes, safe_shelf(sample.case).to(boxes.device))
    score = official_replay_scores(
        sample.case,
        telemetry,
        baseline_area=sample.labels.baseline_area,
        baseline_hpwl=sample.labels.baseline_hpwl,
    )
    score_slice = score[start:stop].detach().cpu()
    source_indices = torch.arange(start, stop, dtype=torch.long)
    kinds = tuple("learned" for _ in range(population))
    geometry_hashes = _candidate_geometry_hashes(boxes)
    tiers = _telemetry_feasibility_tier(telemetry, start, stop)
    row_ids = _candidate_row_ids(
        sample_id=sample.sample_id,
        stage="learned_initial",
        kinds=kinds,
        source_types=kinds,
        geometry_hashes=geometry_hashes,
    )
    return ReplayRecord(
        sample,
        checkpoint_hash,
        features.detach().cpu(),
        score_slice,
        candidate_row_ids=row_ids,
        candidate_source_indices=source_indices,
        candidate_kinds=kinds,
        candidate_source_types=kinds,
        candidate_geometry_sha256=geometry_hashes,
        feasibility_tier=tiers,
        target_rank=_target_rank(score_slice, tiers, row_ids),
        candidate_stage="learned_initial",
        candidate_population=population,
        population_seed=population_seed,
    )


def write_replay(records: Iterable[ReplayRecord], path: str | Path) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            payload = {
                "schema_version": 2,
                "checkpoint_hash": record.checkpoint_hash,
                "target_kind": record.target_kind,
                "sample": sample_to_payload(record.sample),
                "candidate_features": record.candidate_features.tolist(),
                "target_score": record.target_score.tolist(),
            }
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def write_replay_v3(records: Iterable[ReplayRecord], path: str | Path) -> int:
    """Write schema v3 replay rows with stable candidate identity metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            payload = _v3_payload(record)
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def iter_replay(path: str | Path) -> Iterator[ReplayRecord]:
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            schema_version = payload.get("schema_version")
            if schema_version in _LEGACY_SCHEMA_VERSIONS:
                yield ReplayRecord(
                    sample_from_payload(payload["sample"]),
                    str(payload["checkpoint_hash"]),
                    torch.as_tensor(payload["candidate_features"], dtype=torch.float32),
                    torch.as_tensor(payload["target_score"], dtype=torch.float32),
                    LEGACY_TARGET_KIND if schema_version == 1 else str(payload["target_kind"]),
                )
                continue
            if schema_version != V3_SCHEMA_VERSION:
                raise ValueError("replay schema mismatch")
            yield _record_from_v3_payload(payload)


def ranker_loss(model: HCFPModel, record: ReplayRecord) -> Tensor:
    device = next(model.parameters()).device
    case = record.sample.case.to(device=device, dtype=torch.float32)
    features = record.candidate_features.to(device=device)
    target = record.target_score.to(device=device)
    target = (target - target.mean()) / target.std(unbiased=False).clamp_min(1.0e-6)
    with torch.no_grad():
        embedding = model.encoder(case)
    prediction = model.ranker(embedding, len(features), features)
    return F.smooth_l1_loss(prediction, target)


def train_ranker_steps(
    model: HCFPModel,
    records: Iterable[ReplayRecord],
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
) -> list[float]:
    materialized = list(records)
    if not materialized or steps <= 0:
        raise ValueError("ranker training requires records and positive steps")
    if any(record.target_kind != OFFICIAL_TARGET_KIND for record in materialized):
        raise ValueError("ranker training requires official v10 replay targets")
    history = []
    model.train()
    for index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = ranker_loss(model, materialized[index % len(materialized)])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.ranker.parameters(), max_norm=5.0)
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def _v3_payload(record: ReplayRecord) -> dict[str, object]:
    _validate_replay_tensors(record.candidate_features, record.target_score)
    if record.target_kind != V3_TARGET_KIND:
        raise ValueError("schema v3 target_kind mismatch")
    (
        row_ids,
        source_indices,
        kinds,
        source_types,
        geometry_hashes,
        tiers,
        ranks,
        stage,
        population,
        population_seed,
    ) = _require_v3_fields(record)
    _validate_v3_alignment(
        sample_id=record.sample.sample_id,
        checkpoint_hash=record.checkpoint_hash,
        target_score=record.target_score,
        row_ids=row_ids,
        source_indices=source_indices,
        kinds=kinds,
        source_types=source_types,
        geometry_hashes=geometry_hashes,
        tiers=tiers,
        ranks=ranks,
        stage=stage,
        population=population,
        population_seed=population_seed,
    )
    return {
        "schema_version": V3_SCHEMA_VERSION,
        "checkpoint_hash": record.checkpoint_hash,
        "target_kind": record.target_kind,
        "sample": sample_to_payload(record.sample),
        "candidate_features": record.candidate_features.tolist(),
        "target_score": record.target_score.tolist(),
        "candidate_population": population,
        "population_seed": population_seed,
        "candidate_stage": stage,
        "candidate_row_ids": list(row_ids),
        "candidate_source_indices": source_indices.tolist(),
        "candidate_kinds": list(kinds),
        "candidate_source_types": list(source_types),
        "candidate_geometry_sha256": list(geometry_hashes),
        "feasibility_tier": tiers.tolist(),
        "target_rank": ranks.tolist(),
    }


def _record_from_v3_payload(payload: dict[str, object]) -> ReplayRecord:
    sample = sample_from_payload(payload["sample"])
    checkpoint_hash = str(payload["checkpoint_hash"])
    target_kind = str(payload.get("target_kind"))
    if target_kind != V3_TARGET_KIND:
        raise ValueError("schema v3 target_kind mismatch")
    features = torch.as_tensor(payload["candidate_features"], dtype=torch.float32)
    score = torch.as_tensor(payload["target_score"], dtype=torch.float32)
    _validate_replay_tensors(features, score)
    row_ids = _string_tuple(payload.get("candidate_row_ids"), "candidate_row_ids")
    source_indices = _int_tensor(payload.get("candidate_source_indices"), "candidate_source_indices")
    kinds = _string_tuple(payload.get("candidate_kinds"), "candidate_kinds")
    source_types = _string_tuple(payload.get("candidate_source_types"), "candidate_source_types")
    geometry_hashes = _string_tuple(payload.get("candidate_geometry_sha256"), "candidate_geometry_sha256")
    tiers = _int_tensor(payload.get("feasibility_tier"), "feasibility_tier")
    ranks = _int_tensor(payload.get("target_rank"), "target_rank")
    stage = _required_string(payload.get("candidate_stage"), "candidate_stage")
    population = _required_int(payload.get("candidate_population"), "candidate_population")
    population_seed = _required_int(payload.get("population_seed"), "population_seed")
    _validate_v3_alignment(
        sample_id=sample.sample_id,
        checkpoint_hash=checkpoint_hash,
        target_score=score,
        row_ids=row_ids,
        source_indices=source_indices,
        kinds=kinds,
        source_types=source_types,
        geometry_hashes=geometry_hashes,
        tiers=tiers,
        ranks=ranks,
        stage=stage,
        population=population,
        population_seed=population_seed,
    )
    return ReplayRecord(
        sample,
        checkpoint_hash,
        features,
        score,
        target_kind,
        candidate_row_ids=row_ids,
        candidate_source_indices=source_indices,
        candidate_kinds=kinds,
        candidate_source_types=source_types,
        candidate_geometry_sha256=geometry_hashes,
        feasibility_tier=tiers,
        target_rank=ranks,
        candidate_stage=stage,
        candidate_population=population,
        population_seed=population_seed,
    )


def _validate_replay_tensors(features: Tensor, score: Tensor) -> None:
    if features.ndim != 2:
        raise ValueError("candidate_features must have shape [K,F]")
    if score.ndim != 1 or score.shape[0] != features.shape[0]:
        raise ValueError("target_score must align with candidate_features")
    if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(score).all()):
        raise ValueError("replay tensors must be finite")


def _require_v3_fields(
    record: ReplayRecord,
) -> tuple[
    tuple[str, ...],
    Tensor,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    Tensor,
    Tensor,
    str,
    int,
    int,
]:
    if (
        record.candidate_row_ids is None
        or record.candidate_source_indices is None
        or record.candidate_kinds is None
        or record.candidate_source_types is None
        or record.candidate_geometry_sha256 is None
        or record.feasibility_tier is None
        or record.target_rank is None
        or record.candidate_stage is None
        or record.candidate_population is None
        or record.population_seed is None
    ):
        raise ValueError("schema v3 replay requires candidate provenance")
    if type(record.population_seed) is not int:
        raise ValueError("population_seed must be an integer")
    return (
        tuple(record.candidate_row_ids),
        record.candidate_source_indices.to(dtype=torch.long).cpu(),
        tuple(record.candidate_kinds),
        tuple(record.candidate_source_types),
        tuple(record.candidate_geometry_sha256),
        record.feasibility_tier.to(dtype=torch.long).cpu(),
        record.target_rank.to(dtype=torch.long).cpu(),
        record.candidate_stage,
        int(record.candidate_population),
        int(record.population_seed),
    )


def _validate_v3_alignment(
    *,
    sample_id: str,
    checkpoint_hash: str,
    target_score: Tensor,
    row_ids: Sequence[str],
    source_indices: Tensor,
    kinds: Sequence[str],
    source_types: Sequence[str],
    geometry_hashes: Sequence[str],
    tiers: Tensor,
    ranks: Tensor,
    stage: str,
    population: int,
    population_seed: int,
) -> None:
    count = int(target_score.numel())
    lengths = {
        len(row_ids),
        int(source_indices.numel()),
        len(kinds),
        len(source_types),
        len(geometry_hashes),
        int(tiers.numel()),
        int(ranks.numel()),
        count,
    }
    if len(lengths) != 1:
        raise ValueError("schema v3 candidate metadata length mismatch")
    if len(set(row_ids)) != count:
        raise ValueError("schema v3 candidate_row_ids must be unique")
    if not _is_sha256_hex(checkpoint_hash):
        raise ValueError("schema v3 checkpoint_hash must be a SHA-256 hex digest")
    if population <= 0:
        raise ValueError("candidate_population must be positive")
    if type(population_seed) is not int:
        raise ValueError("population_seed must be an integer")
    if not stage:
        raise ValueError("candidate_stage must be non-empty")
    if any(not _is_sha256_hex(value) for value in geometry_hashes):
        raise ValueError("candidate_geometry_sha256 must contain SHA-256 hex digests")
    if any(kind not in _VALID_CANDIDATE_KINDS for kind in kinds):
        raise ValueError("invalid schema v3 candidate kind")
    if any(source_type not in _VALID_CANDIDATE_SOURCE_TYPES for source_type in source_types):
        raise ValueError("invalid schema v3 candidate source type")
    if bool((source_indices < 0).any()):
        raise ValueError("candidate_source_indices must be non-negative")
    if any(int(value) not in _VALID_FEASIBILITY_TIERS for value in tiers.tolist()):
        raise ValueError("invalid schema v3 feasibility tier")
    expected_ids = _candidate_row_ids(
        sample_id=sample_id,
        stage=stage,
        kinds=tuple(kinds),
        source_types=tuple(source_types),
        geometry_hashes=tuple(geometry_hashes),
    )
    if tuple(row_ids) != expected_ids:
        raise ValueError("schema v3 candidate_row_ids do not match provenance")
    expected_rank = _target_rank(target_score, tiers, row_ids)
    if not torch.equal(ranks, expected_rank):
        raise ValueError("schema v3 target_rank does not match target order")


def _candidate_row_ids(
    *,
    sample_id: str,
    stage: str,
    kinds: tuple[str, ...],
    source_types: tuple[str, ...],
    geometry_hashes: tuple[str, ...],
) -> tuple[str, ...]:
    ids = []
    for kind, source_type, geometry_hash in zip(kinds, source_types, geometry_hashes, strict=True):
        raw = {
            "sample_id": sample_id,
            "candidate_geometry_sha256": geometry_hash,
            "stage": stage,
            "kind": kind,
            "source_type": source_type,
        }
        digest = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        ids.append(digest)
    return tuple(ids)


def _candidate_geometry_hashes(boxes: Tensor) -> tuple[str, ...]:
    canonical = torch.as_tensor(boxes, dtype=torch.float32).detach().cpu().contiguous()
    if canonical.ndim != 3 or canonical.shape[2] != 4:
        raise ValueError("candidate geometry must have shape [K,N,4]")
    if not bool(torch.isfinite(canonical).all()):
        raise ValueError("candidate geometry must be finite")
    hashes = []
    for row in canonical:
        digest = hashlib.sha256()
        digest.update(str(tuple(row.shape)).encode())
        digest.update(str(row.dtype).encode())
        digest.update(row.contiguous().view(torch.uint8).numpy().tobytes())
        hashes.append(digest.hexdigest())
    return tuple(hashes)


def _target_rank(target_score: Tensor, tiers: Tensor, row_ids: Sequence[str]) -> Tensor:
    count = int(target_score.numel())
    order = sorted(
        range(count),
        key=lambda index: (int(tiers[index]), float(target_score[index]), str(row_ids[index])),
    )
    ranks = torch.empty(count, dtype=torch.long)
    for rank, row in enumerate(order):
        ranks[row] = rank
    return ranks


def _telemetry_feasibility_tier(telemetry, start: int, stop: int) -> Tensor:
    hard = telemetry.hard_feasible.to(dtype=torch.bool).detach().cpu()[start:stop]
    projected = telemetry.projection_ok.to(dtype=torch.bool).detach().cpu()[start:stop]
    return torch.where(hard, torch.zeros_like(hard, dtype=torch.long), torch.where(projected, 1, 2))


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return tuple(value)


def _is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in _HEX for character in value.lower())


def _int_tensor(value: object, name: str) -> Tensor:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"{name} must be a list of exact integers")
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return tensor


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return int(value)
