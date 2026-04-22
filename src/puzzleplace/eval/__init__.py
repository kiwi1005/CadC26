from .metrics import (
    BCSummary,
    CandidateCoverageSummary,
    MilestoneSnapshot,
    RolloutModeSummary,
    RolloutSummary,
    build_milestone_snapshot,
    load_json_report,
    summarize_bc_training,
    summarize_candidate_coverage,
    summarize_rollout_results,
)
from .reports import render_milestone_report
from .violation import ViolationProfile, summarize_violation_profile

try:
    from .official import (
        OfficialEvaluatorWrapper,
        evaluate_positions,
        extract_validation_baseline_metrics,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent fallback
    _OFFICIAL_IMPORT_ERROR = exc

    class OfficialEvaluatorWrapper:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "Official evaluator import failed; ensure iccad2026_evaluate is importable."
            ) from _OFFICIAL_IMPORT_ERROR

    def evaluate_positions(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(
            "Official evaluator import failed; ensure iccad2026_evaluate is importable."
        ) from _OFFICIAL_IMPORT_ERROR

    def extract_validation_baseline_metrics(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(
            "Official evaluator import failed; ensure iccad2026_evaluate is importable."
        ) from _OFFICIAL_IMPORT_ERROR


__all__ = [
    "BCSummary",
    "CandidateCoverageSummary",
    "MilestoneSnapshot",
    "OfficialEvaluatorWrapper",
    "RolloutModeSummary",
    "RolloutSummary",
    "ViolationProfile",
    "build_milestone_snapshot",
    "extract_validation_baseline_metrics",
    "evaluate_positions",
    "load_json_report",
    "render_milestone_report",
    "summarize_violation_profile",
    "summarize_bc_training",
    "summarize_candidate_coverage",
    "summarize_rollout_results",
]
