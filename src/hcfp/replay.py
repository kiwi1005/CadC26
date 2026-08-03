"""Exact-tail replay records for repair-aware candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
from torch.nn import functional as F

from hcfp.analytic import to_official_placements
from hcfp.candidates import candidate_features
from hcfp.data import DataSample, sample_from_payload, sample_to_payload
from hcfp.fallback import safe_shelf
from hcfp.geometry import centers_from_xywh, normalize_xywh
from hcfp.listwise import listmle_loss
from hcfp.model import HCFPModel
from hcfp.ranker_features import (
    RANKER_FEATURE_DIM,
    RANKER_FEATURE_VERSION,
    STORED_RANKER_FEATURE_VERSION,
    repair_aware_ranker_features,
)
from hcfp.score_attribution import CAP_LOG, attribute_score
from hcfp.constraints.raw_repair import repair_raw_constraints
from hcfp.verify import ALPHA, BETA, exact_metrics


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
    "constraint",
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
    candidate_geometry: Tensor | None = None
    post_bdp_geometry: Tensor | None = None
    post_repair_geometry: Tensor | None = None
    teacher_delta_xy: Tensor | None = None
    repair_displacement: Tensor | None = None
    post_repair_hard_feasible: Tensor | None = None
    post_repair_log_uncapped_cost: Tensor | None = None
    post_repair_cap_margin: Tensor | None = None
    boundary_violations: Tensor | None = None
    grouping_violations: Tensor | None = None
    mib_violations: Tensor | None = None


@dataclass(frozen=True)
class RankerLossReport:
    combined: Tensor
    listwise: Tensor
    top_one: Tensor
    feasibility_order: Tensor
    pointwise: Tensor
    listwise_weight_mean: Tensor
    listwise_weight_max: Tensor

    def scalars(self) -> dict[str, float]:
        return {
            "combined": float(self.combined.detach()),
            "listwise": float(self.listwise.detach()),
            "top_one": float(self.top_one.detach()),
            "feasibility_order": float(self.feasibility_order.detach()),
            "pointwise": float(self.pointwise.detach()),
            "listwise_weight_mean": float(self.listwise_weight_mean.detach()),
            "listwise_weight_max": float(self.listwise_weight_max.detach()),
        }


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


def records_from_learned_analysis(
    sample: DataSample,
    source,
    checkpoint_hash: str,
    analysis,
    analytic_population: int,
    population_seed: int,
    stages: Sequence[str] = ("initial", "post_relax"),
) -> tuple[ReplayRecord, ...]:
    """Build paired v3 records for learned initial and post-relax candidates."""

    if not isinstance(analytic_population, int) or analytic_population <= 0:
        raise ValueError("analytic_population must be a positive integer")
    if type(population_seed) is not int:
        raise ValueError("population_seed must be an integer")
    requested_stages = tuple(stages)
    if (
        not requested_stages
        or len(set(requested_stages)) != len(requested_stages)
        or not set(requested_stages) <= {"initial", "post_relax"}
    ):
        raise ValueError("stages must be a non-empty unique subset of initial/post_relax")
    result = analysis.result
    analytic = analysis.analytic
    learned_count = int(result.candidate_count) - int(analytic_population)
    if learned_count <= 0:
        raise ValueError("learned analysis contains no learned candidates")
    topology_count = _result_seed_count(result, "topology_seed_count")
    constraint_count = _result_seed_count(result, "constraint_seed_count")
    if topology_count + constraint_count > learned_count:
        raise ValueError("learned seed counts exceed learned candidate count")
    expected = 1 + 2 * int(analytic_population) + 2 * learned_count
    if int(analytic.raw_candidates.shape[0]) != expected or int(analytic.projected_candidates.shape[0]) != expected:
        raise ValueError("learned analysis candidate count does not match merged layout")
    initial_start = 1 + int(analytic_population)
    final_start = 1 + int(analytic_population) + learned_count + int(analytic_population)
    stage_starts = {
        "initial": initial_start,
        "post_relax": final_start,
    }
    return tuple(
        _record_from_stage(
            sample,
            source,
            checkpoint_hash,
            analytic,
            start=stage_starts[stage],
            stop=stage_starts[stage] + learned_count,
            stage=stage,
            population_seed=population_seed,
            analytic_population=analytic_population,
            learned_count=learned_count,
            topology_count=topology_count,
            constraint_count=constraint_count,
        )
        for stage in requested_stages
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
    return ranker_loss_report(model, record).combined


def ranker_loss_report(model: HCFPModel, record: ReplayRecord) -> RankerLossReport:
    device = next(model.parameters()).device
    if record.candidate_features.shape[1] == model.config.candidate_metric_dim:
        features = record.candidate_features.detach().to(device=device, dtype=torch.float32)
    else:
        features = ranker_features_for_record(
            record,
            expected_dim=model.config.candidate_metric_dim,
            expected_version=model.config.ranker_feature_version,
        ).to(device=device)
    if model.ranker.use_scene_embedding:
        case = record.sample.case.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            embedding = model.encoder(case)
    else:
        embedding = features.new_empty((0, 0))
    prediction = model.ranker(embedding, len(features), features)
    return _ranker_loss_from_prediction(prediction, record)


def _ranker_loss_from_prediction(prediction: Tensor, record: ReplayRecord) -> RankerLossReport:
    device = prediction.device
    target = record.target_score.to(device=device)
    target = (target - target.mean()) / target.std(unbiased=False).clamp_min(1.0e-6)
    pointwise = F.smooth_l1_loss(prediction, target)
    if record.target_rank is None:
        zero = pointwise * 0.0
        one = pointwise.detach().new_tensor(1.0)
        return RankerLossReport(pointwise, zero, zero, zero, pointwise, one, one)

    weight = _listwise_weight(record, device=device)
    target_rank = record.target_rank.to(device=device)
    listwise = listmle_loss(
        prediction,
        target_rank,
        weight=weight,
    )
    oracle = torch.argmin(target_rank).reshape(1)
    top_one = F.cross_entropy(-prediction.unsqueeze(0), oracle)
    feasibility_order = _feasibility_order_loss(prediction, record, device=device)
    combined = listwise + 0.25 * feasibility_order + 0.05 * pointwise
    return RankerLossReport(
        combined,
        listwise,
        top_one,
        feasibility_order,
        pointwise,
        weight.mean(),
        weight.max(),
    )


def _feasibility_order_loss(
    prediction: Tensor,
    record: ReplayRecord,
    *,
    device: torch.device,
) -> Tensor:
    if record.feasibility_tier is None:
        return prediction.sum() * 0.0
    tiers = record.feasibility_tier.to(device=device, dtype=torch.long)
    preferred = tiers[:, None] < tiers[None, :]
    if not bool(preferred.any()):
        return prediction.sum() * 0.0
    margin = 0.25 + prediction[:, None] - prediction[None, :]
    return F.softplus(margin[preferred]).mean()


def _listwise_weight(record: ReplayRecord, *, device: torch.device) -> Tensor:
    if record.target_rank is None:
        return torch.ones_like(record.target_score, dtype=torch.float32, device=device)
    rank = record.target_rank.to(device=device, dtype=torch.float32)
    weight = torch.reciprocal(torch.log2(rank + 2.0))
    weight = weight / weight.mean().clamp_min(1.0e-6)
    if record.post_repair_cap_margin is None:
        return weight
    cap_margin = record.post_repair_cap_margin.to(device=device, dtype=torch.float32)
    uncapped = cap_margin > 0.0
    if record.post_repair_hard_feasible is not None:
        uncapped &= record.post_repair_hard_feasible.to(device=device, dtype=torch.bool)
    if bool(uncapped.any()) and bool((~uncapped).any()):
        weight = weight * (1.0 + 0.25 * uncapped.to(dtype=torch.float32))
    return weight


def ranker_features_for_record(
    record: ReplayRecord,
    *,
    expected_dim: int,
    expected_version: str,
) -> Tensor:
    """Return the feature view required by a ranker checkpoint."""

    stored = record.candidate_features.detach().to(device="cpu", dtype=torch.float32)
    if expected_version == STORED_RANKER_FEATURE_VERSION:
        if expected_dim != int(stored.shape[1]):
            raise ValueError("stored ranker feature width does not match checkpoint")
        return stored
    if expected_version != RANKER_FEATURE_VERSION:
        raise ValueError(f"unsupported ranker feature version {expected_version!r}")
    if expected_dim != RANKER_FEATURE_DIM:
        raise ValueError(
            f"ranker expects unsupported candidate feature dimension {expected_dim}"
        )
    if (
        record.candidate_geometry is None
        or record.post_bdp_geometry is None
        or record.candidate_kinds is None
    ):
        raise ValueError("repair-aware ranker features require schema v3 geometry and provenance")
    case = record.sample.case.to(device="cpu", dtype=torch.float32)
    return repair_aware_ranker_features(
        case,
        record.candidate_geometry.to(device="cpu", dtype=torch.float32),
        record.post_bdp_geometry.to(device="cpu", dtype=torch.float32),
        safe_shelf(case).to(device="cpu", dtype=torch.float32),
        record.candidate_kinds,
        str(record.candidate_stage),
    ).detach()


def train_ranker_steps(
    model: HCFPModel,
    records: Iterable[ReplayRecord],
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    report_components: bool = False,
) -> list[float] | list[dict[str, float]]:
    materialized = list(records)
    if not materialized or steps <= 0:
        raise ValueError("ranker training requires records and positive steps")
    if any(record.target_kind != OFFICIAL_TARGET_KIND for record in materialized):
        raise ValueError("ranker training requires official v10 replay targets")
    history = []
    model.train()
    for index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        report = ranker_loss_report(model, materialized[index % len(materialized)])
        report.combined.backward()
        torch.nn.utils.clip_grad_norm_(model.ranker.parameters(), max_norm=5.0)
        optimizer.step()
        history.append(report.scalars() if report_components else float(report.combined.detach()))
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
    candidate_geometry = _required_tensor(record.candidate_geometry, "candidate_geometry").float()
    if candidate_geometry.ndim != 3 or candidate_geometry.shape[1:] != (record.sample.case.n, 4):
        raise ValueError("candidate_geometry must have shape [K,sample.case.n,4]")
    _validate_candidate_features(record.sample, candidate_geometry, record.candidate_features)
    if _candidate_geometry_hashes(candidate_geometry) != tuple(geometry_hashes):
        raise ValueError("candidate_geometry does not match candidate_geometry_sha256")
    aligned = {
        "post_bdp_geometry": _required_tensor(record.post_bdp_geometry, "post_bdp_geometry").float(),
        "post_repair_geometry": _required_tensor(record.post_repair_geometry, "post_repair_geometry").float(),
        "teacher_delta_xy": _required_tensor(record.teacher_delta_xy, "teacher_delta_xy").float(),
        "repair_displacement": _required_tensor(record.repair_displacement, "repair_displacement").float(),
        "post_repair_hard_feasible": _required_bool_tensor(
            record.post_repair_hard_feasible, "post_repair_hard_feasible"
        ),
        "post_repair_log_uncapped_cost": _required_tensor(
            record.post_repair_log_uncapped_cost, "post_repair_log_uncapped_cost"
        ).float(),
        "post_repair_cap_margin": _required_tensor(record.post_repair_cap_margin, "post_repair_cap_margin").float(),
        "boundary_violations": _required_int_tensor(record.boundary_violations, "boundary_violations"),
        "grouping_violations": _required_int_tensor(record.grouping_violations, "grouping_violations"),
        "mib_violations": _required_int_tensor(record.mib_violations, "mib_violations"),
    }
    _validate_v3_aligned_values(candidate_geometry, record.target_score, tiers, aligned)
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
        "candidate_geometry": candidate_geometry.tolist(),
        "post_bdp_geometry": aligned["post_bdp_geometry"].tolist(),
        "post_repair_geometry": aligned["post_repair_geometry"].tolist(),
        "teacher_delta_xy": aligned["teacher_delta_xy"].tolist(),
        "repair_displacement": aligned["repair_displacement"].tolist(),
        "post_repair_hard_feasible": aligned["post_repair_hard_feasible"].tolist(),
        "post_repair_log_uncapped_cost": aligned["post_repair_log_uncapped_cost"].tolist(),
        "post_repair_cap_margin": aligned["post_repair_cap_margin"].tolist(),
        "boundary_violations": aligned["boundary_violations"].tolist(),
        "grouping_violations": aligned["grouping_violations"].tolist(),
        "mib_violations": aligned["mib_violations"].tolist(),
    }


def _record_from_v3_payload(payload: dict[str, object]) -> ReplayRecord:
    sample = sample_from_payload(payload["sample"])
    checkpoint_hash = str(payload["checkpoint_hash"])
    target_kind = str(payload.get("target_kind"))
    if target_kind != V3_TARGET_KIND:
        raise ValueError("schema v3 target_kind mismatch")
    features = torch.as_tensor(payload["candidate_features"], dtype=torch.float32)
    score = torch.as_tensor(payload["target_score"], dtype=torch.float32)
    candidate_geometry = torch.as_tensor(payload["candidate_geometry"], dtype=torch.float32)
    _validate_replay_tensors(features, score)
    if candidate_geometry.ndim != 3 or candidate_geometry.shape[1:] != (sample.case.n, 4):
        raise ValueError("candidate_geometry must have shape [K,sample.case.n,4]")
    _validate_candidate_features(sample, candidate_geometry, features)
    geometry_hashes = _string_tuple(payload.get("candidate_geometry_sha256"), "candidate_geometry_sha256")
    if _candidate_geometry_hashes(candidate_geometry) != geometry_hashes:
        raise ValueError("candidate_geometry does not match candidate_geometry_sha256")
    row_ids = _string_tuple(payload.get("candidate_row_ids"), "candidate_row_ids")
    source_indices = _int_tensor(payload.get("candidate_source_indices"), "candidate_source_indices")
    kinds = _string_tuple(payload.get("candidate_kinds"), "candidate_kinds")
    source_types = _string_tuple(payload.get("candidate_source_types"), "candidate_source_types")
    tiers = _int_tensor(payload.get("feasibility_tier"), "feasibility_tier")
    ranks = _int_tensor(payload.get("target_rank"), "target_rank")
    stage = _required_string(payload.get("candidate_stage"), "candidate_stage")
    population = _required_int(payload.get("candidate_population"), "candidate_population")
    population_seed = _required_int(payload.get("population_seed"), "population_seed")
    aligned = _read_aligned_tensors(payload, int(score.numel()), int(candidate_geometry.shape[1]))
    _validate_v3_aligned_values(candidate_geometry, score, tiers, aligned)
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
        candidate_geometry=candidate_geometry,
        post_bdp_geometry=aligned["post_bdp_geometry"],
        post_repair_geometry=aligned["post_repair_geometry"],
        teacher_delta_xy=aligned["teacher_delta_xy"],
        repair_displacement=aligned["repair_displacement"],
        post_repair_hard_feasible=aligned["post_repair_hard_feasible"],
        post_repair_log_uncapped_cost=aligned["post_repair_log_uncapped_cost"],
        post_repair_cap_margin=aligned["post_repair_cap_margin"],
        boundary_violations=aligned["boundary_violations"],
        grouping_violations=aligned["grouping_violations"],
        mib_violations=aligned["mib_violations"],
    )


def _validate_replay_tensors(features: Tensor, score: Tensor) -> None:
    if features.ndim != 2:
        raise ValueError("candidate_features must have shape [K,F]")
    if score.ndim != 1 or score.shape[0] != features.shape[0]:
        raise ValueError("target_score must align with candidate_features")
    if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(score).all()):
        raise ValueError("replay tensors must be finite")


def _validate_candidate_features(sample: DataSample, candidate_geometry: Tensor, features: Tensor) -> None:
    expected = candidate_features(
        sample.case.to(device="cpu", dtype=torch.float32),
        candidate_geometry.to(device="cpu", dtype=torch.float32),
        safe_shelf(sample.case).to(device="cpu", dtype=torch.float32),
    )
    if not torch.allclose(features.to(device="cpu", dtype=torch.float32), expected, rtol=0.0, atol=1.0e-6):
        raise ValueError("candidate_features do not match candidate_geometry")


def _read_aligned_tensors(payload: dict[str, object], count: int, blocks: int) -> dict[str, Tensor]:
    tensors = {
        "post_bdp_geometry": torch.as_tensor(payload["post_bdp_geometry"], dtype=torch.float32),
        "post_repair_geometry": torch.as_tensor(payload["post_repair_geometry"], dtype=torch.float32),
        "teacher_delta_xy": torch.as_tensor(payload["teacher_delta_xy"], dtype=torch.float32),
        "repair_displacement": torch.as_tensor(payload["repair_displacement"], dtype=torch.float32),
        "post_repair_hard_feasible": _bool_tensor(payload["post_repair_hard_feasible"], "post_repair_hard_feasible"),
        "post_repair_log_uncapped_cost": torch.as_tensor(payload["post_repair_log_uncapped_cost"], dtype=torch.float32),
        "post_repair_cap_margin": torch.as_tensor(payload["post_repair_cap_margin"], dtype=torch.float32),
        "boundary_violations": _int_tensor(payload["boundary_violations"], "boundary_violations"),
        "grouping_violations": _int_tensor(payload["grouping_violations"], "grouping_violations"),
        "mib_violations": _int_tensor(payload["mib_violations"], "mib_violations"),
    }
    for name in ("post_bdp_geometry", "post_repair_geometry"):
        if tensors[name].shape != (count, blocks, 4):
            raise ValueError(f"{name} must have shape [K,N,4]")
        if not bool(torch.isfinite(tensors[name]).all()):
            raise ValueError(f"{name} must be finite")
    if tensors["teacher_delta_xy"].shape != (count, blocks, 2):
        raise ValueError("teacher_delta_xy must have shape [K,N,2]")
    if not bool(torch.isfinite(tensors["teacher_delta_xy"]).all()):
        raise ValueError("teacher_delta_xy must be finite")
    for name in (
        "repair_displacement",
        "post_repair_hard_feasible",
        "post_repair_log_uncapped_cost",
        "post_repair_cap_margin",
        "boundary_violations",
        "grouping_violations",
        "mib_violations",
    ):
        if tensors[name].shape != (count,):
            raise ValueError(f"{name} must have shape [K]")
    for name in ("repair_displacement", "post_repair_log_uncapped_cost", "post_repair_cap_margin"):
        if not bool(torch.isfinite(tensors[name]).all()):
            raise ValueError(f"{name} must be finite")
    return tensors


def _validate_v3_aligned_values(
    candidate_geometry: Tensor,
    target_score: Tensor,
    tiers: Tensor,
    tensors: dict[str, Tensor],
) -> None:
    if not torch.allclose(target_score, tensors["post_repair_log_uncapped_cost"], rtol=0.0, atol=1.0e-6):
        raise ValueError("target_score must match post_repair_log_uncapped_cost")
    expected_margin = CAP_LOG - tensors["post_repair_log_uncapped_cost"]
    if not torch.allclose(tensors["post_repair_cap_margin"], expected_margin, rtol=0.0, atol=1.0e-6):
        raise ValueError("post_repair_cap_margin does not match post_repair_log_uncapped_cost")
    if not torch.equal(tensors["post_repair_hard_feasible"], tiers == 0):
        raise ValueError("post_repair_hard_feasible does not match feasibility_tier")
    expected_delta = centers_from_xywh(tensors["post_repair_geometry"]) - centers_from_xywh(candidate_geometry)
    if not torch.allclose(tensors["teacher_delta_xy"], expected_delta, rtol=0.0, atol=1.0e-6):
        raise ValueError("teacher_delta_xy does not match repair geometry")
    expected_displacement = torch.linalg.vector_norm(expected_delta, dim=-1).sum(dim=1)
    if not torch.allclose(tensors["repair_displacement"], expected_displacement, rtol=0.0, atol=1.0e-6):
        raise ValueError("repair_displacement does not match teacher_delta_xy")
    for name in ("repair_displacement", "boundary_violations", "grouping_violations", "mib_violations"):
        if bool((tensors[name] < 0).any()):
            raise ValueError(f"{name} must be non-negative")


def _required_tensor(value: Tensor | None, name: str) -> Tensor:
    if value is None:
        raise ValueError(f"schema v3 replay requires {name}")
    tensor = torch.as_tensor(value)
    if not bool(torch.isfinite(tensor.float()).all()):
        raise ValueError(f"{name} must be finite")
    return tensor.detach().cpu()


def _required_bool_tensor(value: Tensor | None, name: str) -> Tensor:
    tensor = _required_tensor(value, name)
    if tensor.dtype != torch.bool:
        raise ValueError(f"{name} must contain exact booleans")
    return tensor


def _required_int_tensor(value: Tensor | None, name: str) -> Tensor:
    tensor = _required_tensor(value, name)
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise ValueError(f"{name} must contain exact integers")
    return tensor.to(dtype=torch.long)


def _record_from_stage(
    sample: DataSample,
    source,
    checkpoint_hash: str,
    analytic,
    *,
    start: int,
    stop: int,
    stage: str,
    population_seed: int,
    analytic_population: int,
    learned_count: int,
    topology_count: int,
    constraint_count: int,
) -> ReplayRecord:
    candidate_geometry = analytic.raw_candidates[start:stop].detach().to(device="cpu", dtype=torch.float32)
    post_bdp = analytic.projected_candidates[start:stop].detach().to(device="cpu", dtype=torch.float32)
    telemetry = analytic.telemetry
    source_indices = torch.arange(start, stop, dtype=torch.long)
    kinds, source_types, provenance = _stage_provenance(
        analytic.incumbent_snapshot,
        source_indices,
        stage,
        analytic.raw_candidates.detach().to(device="cpu", dtype=torch.float32),
        analytic_population=analytic_population,
        learned_count=learned_count,
        topology_count=topology_count,
        constraint_count=constraint_count,
    )
    post_repair = _post_repair_geometry(sample, source, post_bdp, source_indices, kinds, provenance, stage)
    metrics = [_post_repair_metrics(sample, source, row) for row in post_repair]
    scores = torch.tensor([item[0] for item in metrics], dtype=torch.float32)
    cap_margin = torch.tensor([item[1] for item in metrics], dtype=torch.float32)
    boundary = torch.tensor([item[2] for item in metrics], dtype=torch.long)
    grouping = torch.tensor([item[3] for item in metrics], dtype=torch.long)
    mib = torch.tensor([item[4] for item in metrics], dtype=torch.long)
    hard = torch.tensor([item[5] for item in metrics], dtype=torch.bool)
    projected_ok = telemetry.projection_ok.detach().to(device="cpu", dtype=torch.bool)[start:stop]
    tiers = torch.where(hard, torch.zeros_like(projected_ok, dtype=torch.long), torch.where(projected_ok, 1, 2))
    geometry_hashes = _candidate_geometry_hashes(candidate_geometry)
    row_ids = _candidate_row_ids(
        sample_id=sample.sample_id,
        stage=stage,
        kinds=kinds,
        source_types=source_types,
        geometry_hashes=geometry_hashes,
    )
    teacher_delta = centers_from_xywh(post_repair) - centers_from_xywh(candidate_geometry)
    repair_displacement = torch.linalg.vector_norm(teacher_delta, dim=-1).sum(dim=1)
    return ReplayRecord(
        sample,
        checkpoint_hash,
        candidate_features(
            sample.case.to(device=candidate_geometry.device),
            candidate_geometry,
            safe_shelf(sample.case).to(candidate_geometry.device),
        ).detach().cpu(),
        scores,
        candidate_row_ids=row_ids,
        candidate_source_indices=source_indices,
        candidate_kinds=kinds,
        candidate_source_types=source_types,
        candidate_geometry_sha256=geometry_hashes,
        feasibility_tier=tiers,
        target_rank=_target_rank(scores, tiers, row_ids),
        candidate_stage=stage,
        candidate_population=int(stop - start),
        population_seed=population_seed,
        candidate_geometry=candidate_geometry,
        post_bdp_geometry=post_bdp,
        post_repair_geometry=post_repair,
        teacher_delta_xy=teacher_delta,
        repair_displacement=repair_displacement,
        post_repair_hard_feasible=hard,
        post_repair_log_uncapped_cost=scores,
        post_repair_cap_margin=cap_margin,
        boundary_violations=boundary,
        grouping_violations=grouping,
        mib_violations=mib,
    )


def _stage_provenance(
    snapshot: dict[str, object],
    source_indices: Tensor,
    stage: str,
    raw_candidates: Tensor,
    *,
    analytic_population: int,
    learned_count: int,
    topology_count: int,
    constraint_count: int,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[tuple[str, str], dict[str, object]]]:
    records = _validated_stage_catalog(
        snapshot,
        raw_candidates,
        analytic_population=analytic_population,
        learned_count=learned_count,
        topology_count=topology_count,
        constraint_count=constraint_count,
    )
    kinds: list[str] = []
    source_types: list[str] = []
    for index in source_indices.tolist():
        source = f"candidate_{int(index)}"
        record = records.get((stage, source))
        if record is None:
            kinds.append("learned")
            source_types.append("learned")
            continue
        candidate_type = str(record.get("candidate_type", ""))
        if candidate_type not in _VALID_CANDIDATE_KINDS:
            raise ValueError("malformed candidate provenance kind")
        kinds.append(candidate_type)
        source_types.append(candidate_type)
    return tuple(kinds), tuple(source_types), records


def _validated_stage_catalog(
    snapshot: dict[str, object],
    raw_candidates: Tensor,
    *,
    analytic_population: int,
    learned_count: int,
    topology_count: int,
    constraint_count: int,
) -> dict[tuple[str, str], dict[str, object]]:
    catalog: dict[tuple[str, str], dict[str, object]] = {}
    residual_count = learned_count - topology_count - constraint_count
    initial_base = 1 + analytic_population
    post_relax_base = initial_base + learned_count + analytic_population
    family_specs = (
        ("constraint_seed", "constraint", constraint_count, residual_count),
        ("topology_seed", "topology", topology_count, residual_count + constraint_count),
    )
    for prefix, candidate_type, count, offset in family_specs:
        snapshot_count = snapshot.get(f"{prefix}_count")
        if count and snapshot_count is None:
            raise ValueError(f"missing {prefix} count")
        if snapshot_count is not None and (type(snapshot_count) is not int or int(snapshot_count) != count):
            raise ValueError(f"malformed {prefix} count")
        sources_raw = snapshot.get(f"{prefix}_sources", ())
        records_raw = snapshot.get(f"{prefix}_provenance", ())
        if not isinstance(sources_raw, (tuple, list)) or not isinstance(records_raw, (tuple, list)):
            raise ValueError(f"malformed {prefix} provenance")
        sources = tuple(str(value) for value in sources_raw)
        expected_sources = tuple(
            source
            for index in range(count)
            for source in (
                f"candidate_{initial_base + offset + index}",
                f"candidate_{post_relax_base + offset + index}",
            )
        )
        if sources != expected_sources or len(records_raw) != len(expected_sources):
            raise ValueError(f"malformed {prefix} provenance length")
        record_sources: list[str] = []
        initial_hashes: dict[int, str] = {}
        for expected_source, raw in zip(expected_sources, records_raw, strict=True):
            if not isinstance(raw, dict):
                raise ValueError(f"malformed {prefix} provenance record")
            source = str(raw.get("source", ""))
            index = _candidate_index(source)
            stage = str(raw.get("stage", ""))
            expected_stage = "initial" if len(record_sources) % 2 == 0 else "post_relax"
            if (
                source != expected_source
                or index is None
                or not 0 <= index < int(raw_candidates.shape[0])
                or stage != expected_stage
                or raw.get("candidate_type") != candidate_type
            ):
                raise ValueError(f"malformed {prefix} provenance record")
            expected = raw.get("candidate_sha256")
            if expected != _lineage_tensor_sha256(raw_candidates[index]):
                raise ValueError(f"{prefix} provenance candidate hash mismatch")
            pair = len(record_sources) // 2
            if stage == "initial":
                initial_hashes[pair] = str(expected)
            else:
                parent = initial_hashes.get(pair)
                transform = "identity" if expected == parent else "population_relaxation"
                if raw.get("parent_candidate_sha256") != parent or raw.get("transform") != transform:
                    raise ValueError(f"malformed {prefix} post-relax lineage")
            record_sources.append(source)
            key = (stage, source)
            if key in catalog:
                raise ValueError("duplicate candidate provenance source")
            catalog[key] = raw
        if tuple(record_sources) != sources:
            raise ValueError(f"malformed {prefix} provenance sources")
    return catalog


def _result_seed_count(result, name: str) -> int:
    value = getattr(result, name, 0)
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _post_repair_geometry(
    sample: DataSample,
    source,
    post_bdp: Tensor,
    source_indices: Tensor,
    kinds: tuple[str, ...],
    provenance: dict[tuple[str, str], dict[str, object]],
    stage: str,
) -> Tensor:
    rows = []
    for row, (index, kind) in enumerate(zip(source_indices.tolist(), kinds, strict=True)):
        projected = post_bdp[row]
        if kind != "constraint":
            rows.append(projected)
            continue
        record = provenance.get((stage, f"candidate_{int(index)}"))
        if record is None:
            raise ValueError("constraint candidate is missing provenance")
        placements = to_official_placements(source, sample.case, projected)
        repaired = repair_raw_constraints(source, placements, record).placements
        rows.append(normalize_xywh(sample.case, torch.tensor(repaired, dtype=torch.float32)))
    return torch.stack(rows).to(dtype=torch.float32)


def _candidate_index(source: object) -> int | None:
    value = str(source)
    if not value.startswith("candidate_"):
        return None
    try:
        return int(value.removeprefix("candidate_"))
    except ValueError:
        return None


def _post_repair_metrics(sample: DataSample, source, normalized_xywh: Tensor) -> tuple[float, float, int, int, int, bool]:
    raw = to_official_placements(source, sample.case, normalized_xywh)
    metrics = exact_metrics(
        source,
        raw,
        baseline_hpwl=float(sample.labels.baseline_hpwl),
        baseline_area=float(sample.labels.baseline_area),
        runtime_seconds=1.0,
        median_runtime=1.0,
    )
    attribution = attribute_score(
        metrics.hpwl_gap,
        metrics.area_gap,
        boundary_violations=metrics.soft.raw_boundary,
        grouping_violations=metrics.soft.raw_grouping,
        mib_violations=metrics.soft.raw_mib,
        max_possible_violations=metrics.soft.maximum,
        hard_feasible=metrics.verification.feasible,
        runtime_factor=1.0,
    )
    return (
        float(attribution.log_uncapped_cost),
        float(attribution.cap_margin),
        int(metrics.soft.raw_boundary),
        int(metrics.soft.raw_grouping),
        int(metrics.soft.raw_mib),
        bool(metrics.verification.feasible),
    )


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
    source_indices = _required_int_tensor(record.candidate_source_indices, "candidate_source_indices")
    tiers = _required_int_tensor(record.feasibility_tier, "feasibility_tier")
    ranks = _required_int_tensor(record.target_rank, "target_rank")
    stage = _required_string(record.candidate_stage, "candidate_stage")
    population = _required_int(record.candidate_population, "candidate_population")
    population_seed = _required_int(record.population_seed, "population_seed")
    return (
        tuple(record.candidate_row_ids),
        source_indices,
        tuple(record.candidate_kinds),
        tuple(record.candidate_source_types),
        tuple(record.candidate_geometry_sha256),
        tiers,
        ranks,
        stage,
        population,
        population_seed,
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


def _lineage_tensor_sha256(tensor: Tensor) -> str:
    raw = torch.as_tensor(tensor).detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


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


def _bool_tensor(value: object, name: str) -> Tensor:
    if not isinstance(value, list) or any(type(item) is not bool for item in value):
        raise ValueError(f"{name} must be a list of exact booleans")
    return torch.as_tensor(value, dtype=torch.bool)


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return int(value)
