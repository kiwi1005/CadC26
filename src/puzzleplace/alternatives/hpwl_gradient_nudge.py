"""Step7R-C HPWL gradient nudge operator.

Perturbs an already-placed block by a small step along the per-block
HPWL pin-centroid gradient. The result is a candidate target box that the
Step7Q-F objective-aware slot finder can legalize.

This module is sidecar-only. It does not mutate layouts in place and does not
touch the contest runtime path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from puzzleplace.data.schema import FloorSetCase
from puzzleplace.ml.step7q_fresh_metric_replay import terminal_centroid

EPS = 1e-9
Box = tuple[float, float, float, float]
Vec = tuple[float, float]
DEFAULT_STEP_LADDER: tuple[float, ...] = (0.25, 0.5, 1.0)

__all__ = [
    "DEFAULT_STEP_LADDER",
    "gradient_vector",
    "nudge_target_box",
    "propose_gradient_variants",
]


def gradient_vector(case: FloorSetCase, block_id: int, current_box: Box) -> Vec:
    """Vector from block center toward the weighted pin centroid."""

    centroid = terminal_centroid(case, block_id)
    if centroid is None:
        return (0.0, 0.0)
    cx = current_box[0] + current_box[2] / 2.0
    cy = current_box[1] + current_box[3] / 2.0
    return (centroid[0] - cx, centroid[1] - cy)


def nudge_target_box(current_box: Box, gradient: Vec, step_factor: float) -> Box | None:
    """Return current_box shifted by step_factor * min(w,h) along gradient."""

    gx, gy = gradient
    magnitude = (gx * gx + gy * gy) ** 0.5
    if magnitude < EPS:
        return None
    w, h = current_box[2], current_box[3]
    step = step_factor * min(w, h)
    dx = gx / magnitude * step
    dy = gy / magnitude * step
    return (current_box[0] + dx, current_box[1] + dy, w, h)


def propose_gradient_variants(
    case: FloorSetCase,
    replay_row: dict[str, Any],
    *,
    step_ladder: Sequence[float] = DEFAULT_STEP_LADDER,
) -> list[dict[str, Any]]:
    """Generate gradient-nudge candidate variants for one Step7Q-F replay row.

    Returns an empty list when the gradient magnitude is below ``EPS`` (block
    already at the pin centroid). The caller can record the zero-gradient
    reason on its own row.
    """

    block_id_value = replay_row.get("block_id")
    if block_id_value is None:
        return []
    block_id = int(block_id_value)
    target_box_field = replay_row.get("target_box")
    if not isinstance(target_box_field, list | tuple) or len(target_box_field) != 4:
        return []
    current_box: Box = (
        float(target_box_field[0]),
        float(target_box_field[1]),
        float(target_box_field[2]),
        float(target_box_field[3]),
    )

    gradient = gradient_vector(case, block_id, current_box)
    magnitude = (gradient[0] ** 2 + gradient[1] ** 2) ** 0.5
    if magnitude < EPS:
        return []

    parent_candidate_id = str(replay_row.get("candidate_id", ""))
    case_id = str(replay_row.get("case_id", ""))

    variants: list[dict[str, Any]] = []
    for step_factor in step_ladder:
        nudged = nudge_target_box(current_box, gradient, float(step_factor))
        if nudged is None:
            continue
        variants.append(
            {
                "schema": "step7r_gradient_variant_v1",
                "parent_candidate_id": parent_candidate_id,
                "case_id": case_id,
                "block_id": block_id,
                "step_factor": float(step_factor),
                "current_box": list(current_box),
                "post_nudge_target_box": list(nudged),
                "gradient_vec": [gradient[0], gradient[1]],
                "gradient_magnitude": magnitude,
            }
        )
    return variants
