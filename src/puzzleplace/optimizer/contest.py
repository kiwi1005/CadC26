from __future__ import annotations

import math
from pathlib import Path

import torch
from iccad2026_evaluate import FloorplanOptimizer

from puzzleplace.actions import ActionPrimitive, ExecutionState
from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.feedback import load_policy_checkpoint
from puzzleplace.models.policy import DecoderOutput, TypedActionPolicy
from puzzleplace.repair import RepairResult, finalize_layout
from puzzleplace.roles import label_case_roles
from puzzleplace.rollout import semantic_rollout
from puzzleplace.scoring import ObjectiveCandidate, select_objective_candidate

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
        objective_selection_k: int = 1,
        objective_selector: str = "hpwl_bbox_soft_repair_proxy",
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.beam_width = beam_width
        self.per_state_candidates = per_state_candidates
        self.objective_selection_k = max(int(objective_selection_k), 1)
        self.objective_selector = objective_selector
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else CHECKPOINT_PATH
        )
        self._policy = None
        self.last_report: dict[str, float | bool | int | str] = {}

    def _load_policy(self):
        if self._policy is not None:
            return self._policy
        if self.checkpoint_path.exists():
            self._policy = load_policy_checkpoint(self.checkpoint_path)
        else:
            self._policy = _HeuristicContestPolicy()
        return self._policy

    def _candidate_policies(self) -> list[tuple[str, object | None]]:
        policies: list[tuple[str, object | None]] = [("primary", self._load_policy())]
        if self.objective_selection_k <= 1:
            return policies

        policies.append(("heuristic", _HeuristicContestPolicy()))
        seed = 0
        while len(policies) < self.objective_selection_k:
            torch.manual_seed(seed)
            policies.append((f"untrained_seed{seed}", TypedActionPolicy(hidden_dim=32)))
            seed += 1
        return policies[: self.objective_selection_k]

    def _solve_candidate(
        self,
        case: FloorSetCase,
        source_id: str,
        policy: object | None,
    ) -> tuple[ObjectiveCandidate, RepairResult, bool]:
        semantic = semantic_rollout(
            case,
            policy,  # type: ignore[arg-type]
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
        candidate = ObjectiveCandidate(
            source_id=source_id,
            positions=positions,
            repair_report=repair.report,
            semantic_placed_fraction=float(semantic.semantic_placed_fraction),
            semantic_fallback_fraction=(
                1.0 if used_fallback else float(semantic.fallback_fraction)
            ),
            metadata={
                "semantic_completed": bool(semantic.semantic_completed),
                "fallback_used": used_fallback,
            },
        )
        return candidate, repair, used_fallback

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
        dict[str, float | bool | int | str],
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
        solved_candidates: list[ObjectiveCandidate] = []
        solved_repairs: list[RepairResult] = []
        fallback_flags: list[bool] = []
        for source_id, policy in self._candidate_policies():
            candidate, repair, used_fallback = self._solve_candidate(case, source_id, policy)
            solved_candidates.append(candidate)
            solved_repairs.append(repair)
            fallback_flags.append(used_fallback)

        selected_index = 0
        selected_score = 0.0
        if self.objective_selection_k > 1:
            selection = select_objective_candidate(
                case,
                solved_candidates,
                scorer_name=self.objective_selector,
            )
            selected_index = selection.candidate_index
            selected_score = selection.score

        selected_candidate = solved_candidates[selected_index]
        selected_repair = solved_repairs[selected_index]
        used_fallback = fallback_flags[selected_index]
        self.last_report = {
            "semantic_completed": bool(
                selected_candidate.metadata.get("semantic_completed", False)
                if selected_candidate.metadata is not None
                else False
            ),
            "semantic_placed_fraction": selected_candidate.semantic_placed_fraction,
            "hard_feasible_before_repair": selected_repair.report.hard_feasible_before,
            "hard_feasible_after_repair": selected_repair.report.hard_feasible_after,
            "repair_success": selected_repair.report.hard_feasible_after,
            "repair_displacement": selected_repair.report.mean_displacement,
            "fallback_fraction": 1.0 if used_fallback else 0.0,
            "objective_selection_k": len(solved_candidates),
            "objective_selection_used": self.objective_selection_k > 1,
            "objective_selector": self.objective_selector,
            "selected_candidate_source": selected_candidate.source_id,
            "selected_candidate_index": selected_index,
            "selected_objective_score": selected_score,
        }
        return selected_candidate.positions, dict(self.last_report)

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
