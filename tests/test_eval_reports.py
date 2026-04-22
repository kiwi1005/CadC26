from __future__ import annotations

from puzzleplace.eval import build_milestone_snapshot, render_milestone_report


def _candidate_payload() -> dict[str, float | int | str]:
    return {
        "case_id": "validation-0",
        "trace_count": 4,
        "total_steps": 96,
        "heuristic_hits": 16,
        "heuristic_coverage": 16 / 96,
        "augmented_hits": 96,
        "augmented_coverage": 1.0,
    }


def _bc_payload() -> dict[str, float | int]:
    return {
        "dataset_size": 252,
        "epochs": 20,
        "initial_loss": 10.0,
        "final_loss": 4.0,
        "primitive_accuracy": 0.92,
        "block_accuracy": 0.25,
    }


def _rollout_payload() -> dict[str, object]:
    return {
        "results": [
            {
                "case_id": "validation-0",
                "greedy": {
                    "placed_count": 1,
                    "all_blocks_placed": False,
                    "stopped_reason": "step_limit",
                    "feasible": None,
                },
                "beam": {
                    "placed_count": 2,
                    "all_blocks_placed": False,
                    "stopped_reason": "beam_exhausted_or_step_limit",
                    "feasible": None,
                },
            },
            {
                "case_id": "validation-1",
                "greedy": {
                    "placed_count": 1,
                    "all_blocks_placed": False,
                    "stopped_reason": "step_limit",
                    "feasible": None,
                },
                "beam": {
                    "placed_count": 3,
                    "all_blocks_placed": False,
                    "stopped_reason": "beam_exhausted_or_step_limit",
                    "feasible": None,
                },
            },
        ]
    }


def test_milestone_snapshot_aggregates_candidate_bc_and_rollout_metrics() -> None:
    snapshot = build_milestone_snapshot(_candidate_payload(), _bc_payload(), _rollout_payload())
    assert snapshot.rollout.case_count == 2
    assert snapshot.rollout.greedy.mean_placed_count == 1.0
    assert snapshot.rollout.beam.mean_placed_count == 2.5
    assert snapshot.rollout.beam_mean_advantage == 1.5
    assert snapshot.inferred_checks["candidate_augmented_full_coverage"] is True
    assert snapshot.inferred_checks["beam_outperforms_greedy"] is True
    assert snapshot.inferred_checks["rollout_completes_any_case"] is False


def test_render_milestone_report_contains_metrics_and_ablation_section() -> None:
    snapshot = build_milestone_snapshot(_candidate_payload(), _bc_payload(), _rollout_payload())
    report = render_milestone_report(
        snapshot,
        ablations=[
            {
                "name": "epochs_5",
                "selected": True,
                "bc_summary": {"epochs": 5, "primitive_accuracy": 0.91, "block_accuracy": 0.22},
                "rollout_summary": {
                    "greedy": {"mean_placed_count": 1.0},
                    "beam": {"mean_placed_count": 2.0},
                },
            }
        ],
    )
    assert "# Agent 10 Summary" in report
    assert "heuristic coverage" in report
    assert "beam advantage over greedy" in report
    assert "## Ablations" in report
