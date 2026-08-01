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


Tensor = torch.Tensor


@dataclass(frozen=True)
class ReplayRecord:
    sample: DataSample
    checkpoint_hash: str
    candidate_features: Tensor
    target_score: Tensor


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
    feasible = telemetry.hard_feasible[start:stop].float()
    score = (
        (1.0 - feasible) * 10.0
        + telemetry.soft_violation[start:stop]
        + 0.01 * torch.log1p(telemetry.hpwl[start:stop])
        + 0.01 * torch.log1p(telemetry.bbox_area[start:stop])
        + 0.10 * telemetry.projection_displacement[start:stop]
    )
    return ReplayRecord(
        sample,
        checkpoint_hash,
        features.detach().cpu(),
        score.detach().cpu(),
    )


def write_replay(records: Iterable[ReplayRecord], path: str | Path) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            payload = {
                "schema_version": 1,
                "checkpoint_hash": record.checkpoint_hash,
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
            if payload.get("schema_version") != 1:
                raise ValueError("replay schema mismatch")
            yield ReplayRecord(
                sample_from_payload(payload["sample"]),
                str(payload["checkpoint_hash"]),
                torch.as_tensor(payload["candidate_features"], dtype=torch.float32),
                torch.as_tensor(payload["target_score"], dtype=torch.float32),
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
