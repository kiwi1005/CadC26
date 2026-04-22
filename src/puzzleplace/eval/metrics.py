from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CandidateCoverageSummary:
    case_id: str | None
    trace_count: int | None
    total_steps: int
    heuristic_hits: int
    heuristic_coverage: float
    augmented_hits: int
    augmented_coverage: float


@dataclass(slots=True)
class BCSummary:
    dataset_size: int
    epochs: int
    initial_loss: float
    final_loss: float
    primitive_accuracy: float
    block_accuracy: float

    @property
    def loss_delta(self) -> float:
        return self.final_loss - self.initial_loss

    @property
    def loss_ratio(self) -> float:
        if self.initial_loss == 0.0:
            return 0.0
        return self.final_loss / self.initial_loss


@dataclass(slots=True)
class RolloutModeSummary:
    name: str
    case_count: int
    mean_placed_count: float
    min_placed_count: int
    max_placed_count: int
    completion_rate: float
    feasible_rate: float | None
    stopped_reasons: dict[str, int]


@dataclass(slots=True)
class RolloutSummary:
    case_count: int
    greedy: RolloutModeSummary
    beam: RolloutModeSummary
    beam_mean_advantage: float
    best_mode: str


@dataclass(slots=True)
class MilestoneSnapshot:
    candidate_coverage: CandidateCoverageSummary
    bc: BCSummary
    rollout: RolloutSummary
    inferred_checks: dict[str, bool]


def load_json_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def summarize_candidate_coverage(payload: dict[str, Any]) -> CandidateCoverageSummary:
    return CandidateCoverageSummary(
        case_id=payload.get("case_id"),
        trace_count=payload.get("trace_count"),
        total_steps=int(payload["total_steps"]),
        heuristic_hits=int(payload["heuristic_hits"]),
        heuristic_coverage=float(payload["heuristic_coverage"]),
        augmented_hits=int(payload["augmented_hits"]),
        augmented_coverage=float(payload["augmented_coverage"]),
    )


def summarize_bc_training(payload: dict[str, Any]) -> BCSummary:
    return BCSummary(
        dataset_size=int(payload.get("dataset_size", 0)),
        epochs=int(payload["epochs"]),
        initial_loss=float(payload["initial_loss"]),
        final_loss=float(payload["final_loss"]),
        primitive_accuracy=float(payload["primitive_accuracy"]),
        block_accuracy=float(payload["block_accuracy"]),
    )


def _summarize_rollout_mode(name: str, cases: list[dict[str, Any]]) -> RolloutModeSummary:
    placed_counts = [int(case["placed_count"]) for case in cases]
    completion_hits = sum(1 for case in cases if bool(case["all_blocks_placed"]))
    known_feasibility = [
        bool(case["feasible"]) for case in cases if case.get("feasible") is not None
    ]
    stopped_reasons = Counter(str(case["stopped_reason"]) for case in cases)
    case_count = len(cases)
    return RolloutModeSummary(
        name=name,
        case_count=case_count,
        mean_placed_count=sum(placed_counts) / max(case_count, 1),
        min_placed_count=min(placed_counts) if placed_counts else 0,
        max_placed_count=max(placed_counts) if placed_counts else 0,
        completion_rate=completion_hits / max(case_count, 1),
        feasible_rate=(
            sum(known_feasibility) / len(known_feasibility) if known_feasibility else None
        ),
        stopped_reasons=dict(stopped_reasons),
    )


def summarize_rollout_results(payload: dict[str, Any]) -> RolloutSummary:
    results = payload["results"]
    greedy_cases = [dict(item["greedy"]) for item in results]
    beam_cases = [dict(item["beam"]) for item in results]
    greedy = _summarize_rollout_mode("greedy", greedy_cases)
    beam = _summarize_rollout_mode("beam", beam_cases)
    return RolloutSummary(
        case_count=len(results),
        greedy=greedy,
        beam=beam,
        beam_mean_advantage=beam.mean_placed_count - greedy.mean_placed_count,
        best_mode="beam" if beam.mean_placed_count >= greedy.mean_placed_count else "greedy",
    )


def build_milestone_snapshot(
    candidate_payload: dict[str, Any],
    bc_payload: dict[str, Any],
    rollout_payload: dict[str, Any],
) -> MilestoneSnapshot:
    candidate = summarize_candidate_coverage(candidate_payload)
    bc = summarize_bc_training(bc_payload)
    rollout = summarize_rollout_results(rollout_payload)
    inferred_checks = {
        "candidate_augmented_full_coverage": candidate.augmented_coverage >= 0.999,
        "candidate_heuristic_ge_0_50": candidate.heuristic_coverage >= 0.50,
        "bc_loss_improved": bc.final_loss < bc.initial_loss,
        "bc_primitive_accuracy_ge_0_80": bc.primitive_accuracy >= 0.80,
        "bc_block_accuracy_ge_0_20": bc.block_accuracy >= 0.20,
        "rollout_places_any_block": (
            rollout.beam.mean_placed_count >= 1.0 or rollout.greedy.mean_placed_count >= 1.0
        ),
        "beam_outperforms_greedy": (
            rollout.beam.mean_placed_count > rollout.greedy.mean_placed_count
        ),
        "rollout_completes_any_case": (
            rollout.beam.completion_rate > 0.0 or rollout.greedy.completion_rate > 0.0
        ),
    }
    return MilestoneSnapshot(
        candidate_coverage=candidate,
        bc=bc,
        rollout=rollout,
        inferred_checks=inferred_checks,
    )
