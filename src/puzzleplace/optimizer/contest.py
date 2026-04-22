from __future__ import annotations

import math
from pathlib import Path

import torch
from iccad2026_evaluate import FloorplanOptimizer

from puzzleplace.actions import ActionPrimitive, ExecutionState
from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.feedback import load_policy_checkpoint
from puzzleplace.models.policy import DecoderOutput
from puzzleplace.repair import finalize_layout
from puzzleplace.roles import label_case_roles
from puzzleplace.rollout import semantic_rollout

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_PATH = ROOT / "artifacts" / "models" / "agent11_awbc_policy.pt"


def _default_dims(
    area_targets: torch.Tensor,
    target_positions: torch.Tensor | None,
    block_index: int,
) -> tuple[float, float]:
    if target_positions is not None:
        _x, _y, width, height = [float(v) for v in target_positions[block_index].tolist()]
        if width > 0 and height > 0:
            return width, height
    area = float(area_targets[block_index].item())
    side = math.sqrt(max(area, 1e-6))
    return side, side


def contest_case_from_inputs(
    block_count: int,
    area_targets: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: torch.Tensor | None = None,
) -> FloorSetCase:
    return FloorSetCase(
        case_id="contest-case",
        block_count=block_count,
        area_targets=area_targets[:block_count].to(torch.float32),
        b2b_edges=b2b_connectivity.to(torch.float32),
        p2b_edges=p2b_connectivity.to(torch.float32),
        pins_pos=pins_pos.to(torch.float32),
        constraints=constraints[:block_count].to(torch.float32),
        target_positions=(
            target_positions[:block_count].to(torch.float32)
            if target_positions is not None
            else None
        ),
        metrics=None,
    )


class _HeuristicContestPolicy:
    def __call__(
        self,
        case: FloorSetCase,
        *,
        role_evidence=None,
        placements=None,
    ) -> DecoderOutput:
        placements = placements or {}
        role_evidence = role_evidence or label_case_roles(case)
        primitive_logits = torch.full((len(ActionPrimitive),), -4.0, dtype=torch.float32)
        primitive_logits[list(ActionPrimitive).index(ActionPrimitive.PLACE_ABSOLUTE)] = 4.0
        primitive_logits[list(ActionPrimitive).index(ActionPrimitive.PLACE_RELATIVE)] = 1.5
        primitive_logits[list(ActionPrimitive).index(ActionPrimitive.FREEZE)] = 0.5

        block_logits = torch.full((case.block_count,), -20.0, dtype=torch.float32)
        for role in role_evidence:
            if role.block_index in placements:
                continue
            area = float(case.area_targets[role.block_index].item())
            boundary_bonus = (
                2.0
                if bool(
                    case.constraints[
                        role.block_index,
                        ConstraintColumns.BOUNDARY,
                    ].item()
                )
                else 0.0
            )
            fixed_bonus = (
                3.0
                if bool(
                    case.constraints[
                        role.block_index,
                        ConstraintColumns.FIXED,
                    ].item()
                )
                else 0.0
            )
            preplaced_bonus = (
                4.0
                if bool(
                    case.constraints[
                        role.block_index,
                        ConstraintColumns.PREPLACED,
                    ].item()
                )
                else 0.0
            )
            block_logits[role.block_index] = area + boundary_bonus + fixed_bonus + preplaced_bonus

        target_logits = torch.zeros((case.block_count, case.block_count), dtype=torch.float32)
        if len(case.b2b_edges) > 0:
            for src, dst, weight in case.b2b_edges.tolist():
                if src == -1 or dst == -1:
                    continue
                src_idx = int(src)
                dst_idx = int(dst)
                if src_idx < case.block_count and dst_idx < case.block_count:
                    target_logits[src_idx, dst_idx] += float(weight)
                    target_logits[dst_idx, src_idx] += float(weight)

        boundary_logits = torch.zeros((case.block_count, 5), dtype=torch.float32)
        boundary_map = {1: 1, 2: 2, 4: 3, 8: 4}
        for block_index in range(case.block_count):
            boundary_code = int(case.constraints[block_index, ConstraintColumns.BOUNDARY].item())
            boundary_logits[block_index, boundary_map.get(boundary_code, 0)] = 3.0

        geometry = torch.zeros((case.block_count, 4), dtype=torch.float32)
        for block_index in range(case.block_count):
            width, height = _default_dims(case.area_targets, case.target_positions, block_index)
            x = 0.0
            y = 0.0
            if case.target_positions is not None:
                tx, ty, _tw, _th = [float(v) for v in case.target_positions[block_index].tolist()]
                if tx >= 0 and ty >= 0:
                    x = tx
                    y = ty
            geometry[block_index] = torch.tensor([x, y, width, height], dtype=torch.float32)

        return DecoderOutput(
            primitive_logits=primitive_logits,
            block_logits=block_logits,
            target_logits=target_logits,
            boundary_logits=boundary_logits,
            geometry=geometry,
        )


class ContestOptimizer(FloorplanOptimizer):
    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        beam_width: int = 4,
        per_state_candidates: int = 3,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.beam_width = beam_width
        self.per_state_candidates = per_state_candidates
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else CHECKPOINT_PATH
        )
        self._policy = None
        self.last_report: dict[str, float | bool | int] = {}

    def _load_policy(self):
        if self._policy is not None:
            return self._policy
        if self.checkpoint_path.exists():
            self._policy = load_policy_checkpoint(self.checkpoint_path)
        else:
            self._policy = _HeuristicContestPolicy()
        return self._policy

    def _build_initial_state(self, case: FloorSetCase) -> ExecutionState:
        state = ExecutionState()
        if case.target_positions is None:
            return state
        for block_index in range(case.block_count):
            if not bool(case.constraints[block_index, ConstraintColumns.PREPLACED].item()):
                continue
            x, y, width, height = [float(v) for v in case.target_positions[block_index].tolist()]
            if x >= 0 and y >= 0 and width > 0 and height > 0:
                state.placements[block_index] = (x, y, width, height)
                state.frozen_blocks.add(block_index)
        return state

    def _fill_missing_positions(
        self,
        case: FloorSetCase,
        placements: dict[int, tuple[float, float, float, float]],
    ) -> list[tuple[float, float, float, float]]:
        completed = dict(placements)
        x_cursor = max((x + width for x, _y, width, _h in completed.values()), default=0.0)
        for block_index in range(case.block_count):
            if block_index in completed:
                continue
            width, height = _default_dims(case.area_targets, case.target_positions, block_index)
            if case.target_positions is not None:
                x, y, target_w, target_h = [
                    float(v) for v in case.target_positions[block_index].tolist()
                ]
                if (
                    bool(case.constraints[block_index, ConstraintColumns.PREPLACED].item())
                    and x >= 0
                    and y >= 0
                    and target_w > 0
                    and target_h > 0
                ):
                    completed[block_index] = (x, y, target_w, target_h)
                    continue
            completed[block_index] = (x_cursor, 0.0, width, height)
            x_cursor += width + 1.0
        return [completed[idx] for idx in range(case.block_count)]

    def solve_with_report(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor | None = None,
    ) -> tuple[
        list[tuple[float, float, float, float]],
        dict[str, float | bool],
    ]:
        case = contest_case_from_inputs(
            block_count,
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
            target_positions,
        )
        semantic = semantic_rollout(
            case,
            self._load_policy(),
            role_evidence=label_case_roles(case),
        )
        repair = finalize_layout(case, semantic.proposed_positions)
        if repair.report.hard_feasible_after:
            positions = repair.positions
            used_fallback = False
        else:
            positions = self._fill_missing_positions(
                case,
                {idx: box for idx, box in enumerate(repair.positions)},
            )
            used_fallback = True
        self.last_report = {
            "semantic_completed": semantic.semantic_completed,
            "semantic_placed_fraction": semantic.semantic_placed_fraction,
            "hard_feasible_before_repair": repair.report.hard_feasible_before,
            "hard_feasible_after_repair": repair.report.hard_feasible_after,
            "repair_success": repair.report.hard_feasible_after,
            "repair_displacement": repair.report.mean_displacement,
            "fallback_fraction": 1.0 if used_fallback else 0.0,
        }
        return positions, dict(self.last_report)

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor | None = None,
    ) -> list[tuple[float, float, float, float]]:
        positions, _report = self.solve_with_report(
            block_count,
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
            target_positions,
        )
        return positions
