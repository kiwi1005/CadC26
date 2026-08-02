"""Exact-tail replay records for repair-aware candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator

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


@dataclass(frozen=True)
class ReplayRecord:
    sample: DataSample
    checkpoint_hash: str
    candidate_features: Tensor
    target_score: Tensor
    target_kind: str = OFFICIAL_TARGET_KIND


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
    return ReplayRecord(
        sample,
        checkpoint_hash,
        features.detach().cpu(),
        score[start:stop].detach().cpu(),
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


def iter_replay(path: str | Path) -> Iterator[ReplayRecord]:
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            schema_version = payload.get("schema_version")
            if schema_version not in (1, 2):
                raise ValueError("replay schema mismatch")
            yield ReplayRecord(
                sample_from_payload(payload["sample"]),
                str(payload["checkpoint_hash"]),
                torch.as_tensor(payload["candidate_features"], dtype=torch.float32),
                torch.as_tensor(payload["target_score"], dtype=torch.float32),
                LEGACY_TARGET_KIND if schema_version == 1 else str(payload["target_kind"]),
            )


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
