from __future__ import annotations

from typing import Any

from .metrics import MilestoneSnapshot


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.1f}%"


def _fmt_float(value: float) -> str:
    return f"{value:.4f}"


def render_milestone_report(
    snapshot: MilestoneSnapshot,
    *,
    ablations: list[dict[str, Any]] | None = None,
) -> str:
    candidate = snapshot.candidate_coverage
    bc = snapshot.bc
    rollout = snapshot.rollout
    lines = [
        "# Agent 10 Summary",
        "",
        "## Candidate coverage",
        f"- case: `{candidate.case_id or 'unknown'}`",
        f"- traces: `{candidate.trace_count if candidate.trace_count is not None else 'n/a'}`",
        (
            f"- heuristic coverage: `{candidate.heuristic_coverage:.4f}` "
            f"({candidate.heuristic_hits}/{candidate.total_steps})"
        ),
        (
            f"- augmented coverage: `{candidate.augmented_coverage:.4f}` "
            f"({candidate.augmented_hits}/{candidate.total_steps})"
        ),
        "",
        "## Behavior cloning",
        f"- dataset size: `{bc.dataset_size}`",
        f"- epochs: `{bc.epochs}`",
        f"- loss: `{_fmt_float(bc.initial_loss)}` -> `{_fmt_float(bc.final_loss)}`",
        f"- primitive accuracy: `{_fmt_pct(bc.primitive_accuracy)}`",
        f"- block accuracy: `{_fmt_pct(bc.block_accuracy)}`",
        "",
        "## Rollout",
        f"- cases: `{rollout.case_count}`",
        (
            f"- greedy mean placed blocks: `{rollout.greedy.mean_placed_count:.2f}` "
            f"(completion `{_fmt_pct(rollout.greedy.completion_rate)}`)"
        ),
        (
            f"- beam mean placed blocks: `{rollout.beam.mean_placed_count:.2f}` "
            f"(completion `{_fmt_pct(rollout.beam.completion_rate)}`)"
        ),
        f"- beam advantage over greedy: `{rollout.beam_mean_advantage:.2f}`",
        f"- best mode in this snapshot: `{rollout.best_mode}`",
        "",
        "## Inferred checks",
    ]
    for name, passed in snapshot.inferred_checks.items():
        lines.append(f"- `{name}`: `{'PASS' if passed else 'FAIL'}`")

    if ablations:
        lines.extend(["", "## Ablations"])
        for item in ablations:
            name = item["name"]
            epochs = item["bc_summary"]["epochs"]
            primitive = item["bc_summary"]["primitive_accuracy"]
            block = item["bc_summary"]["block_accuracy"]
            greedy_mean = item["rollout_summary"]["greedy"]["mean_placed_count"]
            beam_mean = item["rollout_summary"]["beam"]["mean_placed_count"]
            selected = " selected" if item.get("selected") else ""
            lines.append(
                f"- `{name}`{selected}: epochs={epochs}, primitive_acc={primitive:.3f}, "
                f"block_acc={block:.3f}, greedy_mean={greedy_mean:.2f}, beam_mean={beam_mean:.2f}"
            )

    lines.extend(
        [
            "",
            "## Bottlenecks",
            (
                "- Heuristic candidate coverage is still the main recall bottleneck "
                "if it stays far below augmented coverage."
            ),
            (
                "- Block selection accuracy remains the most obvious training weakness "
                "even when primitive accuracy is already high."
            ),
            (
                "- Rollout is still a smoke baseline until at least one strategy "
                "starts completing full cases."
            ),
            "",
            (
                "_Threshold-based checks above are heuristic milestone signals, "
                "not official contest pass/fail criteria._"
            ),
        ]
    )
    return "\n".join(lines)
