#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.actions import compute_expert_candidate_coverage
from puzzleplace.eval import build_milestone_snapshot, render_milestone_report
from puzzleplace.rollout import beam_rollout, greedy_rollout
from puzzleplace.train import build_bc_dataset_from_cases, load_validation_cases, run_bc_overfit
from puzzleplace.trajectory import generate_pseudo_traces


def _parse_epoch_list(raw: str) -> list[int]:
    epochs = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return epochs or [5, 20]


def _load_or_compute_candidate_coverage(case) -> dict[str, object]:
    artifact_path = (
        ROOT / "artifacts" / "reports" / "agent6_candidate_coverage_validation0.json"
    )
    if artifact_path.exists():
        return json.loads(artifact_path.read_text())

    traces = generate_pseudo_traces(case, max_traces=4)
    report = compute_expert_candidate_coverage(case, traces)
    return {
        "case_id": str(case.case_id),
        "trace_count": len(traces),
        "total_steps": report.total_steps,
        "heuristic_hits": report.heuristic_hits,
        "heuristic_coverage": report.heuristic_coverage,
        "augmented_hits": report.augmented_hits,
        "augmented_coverage": report.augmented_coverage,
        "note": (
            "generated during agent10 ablation because "
            "no precomputed agent6 artifact was found"
        ),
    }


def _evaluate_rollout(
    case,
    policy,
    *,
    beam_width: int,
    per_state_candidates: int,
) -> dict[str, object]:
    greedy = greedy_rollout(case, policy)
    beam = beam_rollout(
        case,
        policy,
        beam_width=beam_width,
        per_state_candidates=per_state_candidates,
    )
    return {
        "case_id": str(case.case_id),
        "greedy": {
            "placed_count": greedy.placed_count,
            "all_blocks_placed": greedy.all_blocks_placed,
            "stopped_reason": greedy.stopped_reason,
            "feasible": greedy.feasible,
        },
        "beam": {
            "placed_count": beam.placed_count,
            "all_blocks_placed": beam.all_blocks_placed,
            "stopped_reason": beam.stopped_reason,
            "feasible": beam.feasible,
        },
    }


def main() -> None:
    case_limit = int(os.environ.get("AGENT10_CASE_LIMIT", "5"))
    max_traces = int(os.environ.get("AGENT10_MAX_TRACES", "2"))
    hidden_dim = int(os.environ.get("AGENT10_HIDDEN_DIM", "64"))
    lr = float(os.environ.get("AGENT10_LR", "1e-3"))
    seed = int(os.environ.get("AGENT10_SEED", "0"))
    beam_width = int(os.environ.get("AGENT10_BEAM_WIDTH", "4"))
    per_state_candidates = int(os.environ.get("AGENT10_PER_STATE_CANDIDATES", "3"))
    epoch_list = _parse_epoch_list(
        os.environ.get("AGENT10_ABLATION_EPOCHS", "5,20")
    )

    cases = load_validation_cases(case_limit=case_limit)
    candidate_payload = _load_or_compute_candidate_coverage(cases[0])
    ablations: list[dict[str, Any]] = []
    best_snapshot = None
    best_name = None
    best_key = None

    for epochs in epoch_list:
        dataset = build_bc_dataset_from_cases(cases, max_traces_per_case=max_traces)
        policy, summary = run_bc_overfit(
            dataset,
            hidden_dim=hidden_dim,
            lr=lr,
            epochs=epochs,
            seed=seed,
        )
        rollout_results = [
            _evaluate_rollout(
                case,
                policy,
                beam_width=beam_width,
                per_state_candidates=per_state_candidates,
            )
            for case in cases
        ]
        bc_payload = {
            "dataset_size": summary.dataset_size,
            "epochs": summary.epochs,
            "initial_loss": summary.initial_loss,
            "final_loss": summary.final_loss,
            "primitive_accuracy": summary.primitive_accuracy,
            "block_accuracy": summary.block_accuracy,
        }
        rollout_payload = {"bc_summary": bc_payload, "results": rollout_results}
        snapshot = build_milestone_snapshot(candidate_payload, bc_payload, rollout_payload)
        item: dict[str, Any] = {
            "name": f"epochs_{epochs}",
            "bc_summary": bc_payload,
            "rollout_results": rollout_results,
            "rollout_summary": asdict(snapshot.rollout),
            "snapshot": asdict(snapshot),
        }
        ablations.append(item)
        candidate_key = (
            snapshot.rollout.beam.mean_placed_count,
            snapshot.rollout.beam.completion_rate,
            snapshot.bc.block_accuracy,
            snapshot.bc.primitive_accuracy,
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_snapshot = snapshot
            best_name = item["name"]

    for item in ablations:
        item["selected"] = item["name"] == best_name

    if best_snapshot is None or best_name is None:
        raise RuntimeError("agent10 ablation produced no results")

    output_dir = ROOT / "artifacts" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "agent10_ablation_validation0_4.json"
    md_path = output_dir / "agent10_summary.md"

    payload = {
        "case_limit": case_limit,
        "max_traces_per_case": max_traces,
        "beam_width": beam_width,
        "per_state_candidates": per_state_candidates,
        "candidate_coverage": candidate_payload,
        "selected_ablation": best_name,
        "ablations": ablations,
        "selected_snapshot": asdict(best_snapshot),
    }
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(render_milestone_report(best_snapshot, ablations=ablations))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
