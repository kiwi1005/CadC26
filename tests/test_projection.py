from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from hcfp.verify import overlap_pairs


ROOT = Path(__file__).resolve().parents[1]


def _load_projection():
    spec = importlib.util.spec_from_file_location("hcfp_projection_test_mod", ROOT / "src/hcfp/projection.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


projection = _load_projection()


def devices():
    out = ["cpu"]
    if torch.cuda.is_available():
        out.append("cuda")
    return out


def _repeated_signature_boxes(device: str = "cpu") -> torch.Tensor:
    return torch.tensor(
        [
            [2.2216148376, 1.4571588039, 2.8916938305, 3.1972002983],
            [0.1061706543, 3.6712393761, 0.9746763706, 2.8225233555],
            [0.6529381275, 2.1865963936, 2.8806419373, 3.6755690575],
            [1.4217658043, 1.7715048790, 1.9682002068, 3.6074044704],
            [2.0917274952, 1.8759212494, 1.5688374043, 1.9853456020],
        ],
        device=device,
    )


def _component_reset_boxes(device: str = "cpu") -> torch.Tensor:
    return torch.tensor(
        [
            [3.1388082504, 3.9061970711, 2.7061905861, 1.4898124933],
            [3.0557513237, 2.7515659332, 2.5674855709, 1.4099599123],
            [0.7974164486, 2.6873857975, 1.0128545761, 3.2628638744],
            [1.7325341702, 2.1231830120, 2.7367336750, 1.4555559158],
        ],
        device=device,
    )


def _iteration_sensitive_boxes(device: str = "cpu") -> torch.Tensor:
    return torch.tensor(
        [
            [0.87574035, 0.85704213, 2.45705128, 2.15882730],
            [1.20622230, 1.39342475, 1.98927832, 2.96615839],
            [2.85099411, 0.76927847, 2.02672195, 1.51596773],
            [1.99358797, 2.58268523, 2.00041080, 2.23031282],
            [1.06141090, 1.00404620, 2.25653720, 2.39445925],
        ],
        device=device,
    )


class _QualityProblem:
    def __init__(
        self,
        boundary_bits: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> None:
        self.area = torch.ones(2, device=device)
        self.b2b_weight = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], device=device
        )
        self.p2b_edges = torch.empty((0, 3), device=device)
        self.pins = torch.empty((0, 2), device=device)
        self.boundary_bits = (
            torch.zeros((2, 4), dtype=torch.bool, device=device)
            if boundary_bits is None
            else boundary_bits.to(device=device)
        )


def _branch_order(
    branches: torch.Tensor,
    original: torch.Tensor,
    problem: object | None,
) -> torch.Tensor:
    return projection._best_branch_order(
        branches,
        original,
        torch.zeros(2, dtype=torch.bool, device=branches.device),
        torch.zeros(
            (1, branches.shape[1]),
            dtype=torch.bool,
            device=branches.device,
        ),
        1.0e-6,
        False,
        None,
        problem,
    )


@pytest.mark.parametrize("device", devices())
def test_objective_aware_branch_order_prefers_dominating_quality(device: str) -> None:
    original = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [4.0, 0.0, 1.0, 1.0]]],
        device=device,
    )
    branches = torch.tensor(
        [[
            [[0.0, 0.0, 1.0, 1.0], [5.0, 0.0, 1.0, 1.0]],
            [[1.0, 0.0, 1.0, 1.0], [3.0, 0.0, 1.0, 1.0]],
        ]],
        device=device,
    )

    assert _branch_order(branches, original, _QualityProblem(device=device))[0, 0] == 1
    assert _branch_order(branches, original, None)[0, 0] == 0


@pytest.mark.parametrize("device", devices())
def test_objective_aware_branch_order_keeps_hard_feasibility_first(device: str) -> None:
    original = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [4.0, 0.0, 1.0, 1.0]]],
        device=device,
    )
    branches = torch.tensor(
        [[
            [[0.0, 0.0, 1.0, 1.0], [0.5, 0.0, 1.0, 1.0]],
            [[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0]],
        ]],
        device=device,
    )

    assert _branch_order(branches, original, _QualityProblem(device=device))[0, 0] == 1


@pytest.mark.parametrize("device", devices())
def test_objective_aware_branch_order_rejects_boundary_regression(device: str) -> None:
    original = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [4.0, 0.0, 1.0, 1.0]]],
        device=device,
    )
    branches = torch.tensor(
        [[
            [[0.0, 0.0, 1.0, 1.0], [-2.0, 0.0, 1.0, 1.0]],
            [[0.0, 0.0, 1.0, 1.0], [3.0, 0.0, 1.0, 1.0]],
        ]],
        device=device,
    )
    boundary = torch.tensor([[True, False, False, False], [False] * 4])

    assert _branch_order(
        branches,
        original,
        _QualityProblem(boundary, device),
    )[0, 0] == 1


@pytest.mark.parametrize("device", devices())
def test_objective_aware_branch_order_rejects_contact_regression(device: str) -> None:
    original = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [1.0, 0.9, 1.0, 1.0]]],
        device=device,
    )
    branches = torch.tensor(
        [[
            [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]],
            [[0.0, 0.0, 1.0, 1.0], [1.0, 0.9, 1.0, 1.0]],
        ]],
        device=device,
    )
    neutral = torch.full((1, 2, 2), -1, dtype=torch.long, device=device)
    confidence = torch.zeros((1, 2, 2), device=device)
    contact = neutral.clone()
    contact[0, 0, 1] = projection.BDP_LEFT
    contact[0, 1, 0] = projection.BDP_RIGHT
    contact_confidence = confidence.clone()
    contact_confidence[0, 0, 1] = contact_confidence[0, 1, 0] = 1.0
    guidance = projection.ProjectionGuidance(
        neutral,
        confidence,
        contact,
        contact_confidence,
        torch.zeros((1, 2, 2), dtype=torch.bool, device=device),
    )
    problem = _QualityProblem(device=device)
    quality_rank, _, _ = projection._quality_metrics(branches, original, problem)
    order = projection._best_branch_order(
        branches,
        original,
        torch.zeros(2, dtype=torch.bool, device=device),
        torch.zeros((1, 2), dtype=torch.bool, device=device),
        1.0e-6,
        False,
        guidance,
        problem,
    )

    assert quality_rank.tolist() == [[0, 1]]
    assert projection._contact_residual(branches, guidance, 1.0e-6)[0, 0] > 0.0
    assert projection._contact_residual(branches, guidance, 1.0e-6)[0, 1] == 0.0
    assert order[0, 0] == 1


def test_component_beams_branch_each_disconnected_conflict() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 2.0, 2.0],
            [1.0, 0.0, 2.0, 2.0],
            [10.0, 0.0, 2.0, 2.0],
            [11.0, 0.0, 2.0, 2.0],
        ]
    )
    config = projection.ComponentBDPConfig(
        enabled=True,
        beam_width=16,
        max_uncertain_pairs=1,
    )
    pairs, directions = projection.assign_directions(
        boxes,
        component_config=config,
    )
    rows = projection._branch_direction_rows(
        boxes,
        pairs,
        directions[0],
        config,
    )
    active = torch.nonzero(directions[0] >= 0, as_tuple=False).flatten()

    assert active.numel() == 2
    assert len({tuple(row.tolist()) for row in rows[:, active]}) == 16


def test_guided_partial_projection_preserves_original_instead_of_legacy_fallback() -> None:
    boxes = torch.stack((_iteration_sensitive_boxes(), _iteration_sensitive_boxes()))
    direction = torch.full((2, 5, 5), -1, dtype=torch.long)
    confidence = torch.zeros((2, 5, 5))
    direction[0, 0, 1] = projection.BDP_LEFT
    direction[0, 1, 0] = projection.BDP_RIGHT
    confidence[0, 0, 1] = confidence[0, 1, 0] = 1.0
    guidance = projection.ProjectionGuidance(
        direction,
        confidence,
        torch.full_like(direction, -1),
        torch.zeros_like(confidence),
        torch.zeros((2, 5, 2), dtype=torch.bool),
    )

    result = projection.project_disjunctive(
        boxes,
        iterations=1,
        component_config=projection.ComponentBDPConfig(
            enabled=True,
            beam_width=1,
            max_uncertain_pairs=0,
            outer_sweeps=1,
            preserve_feasible=True,
        ),
        guidance=guidance,
    )
    legacy = projection.project_disjunctive(
        boxes[1],
        iterations=1,
    )

    assert not result.ok_mask[0]
    assert torch.equal(result.xywh[0], boxes[0])
    assert float(result.displacement[0]) == 0.0
    assert int(result.initial_pair_count[0]) == int(result.final_pair_count[0])
    assert bool(result.component_proposal_available[0])
    assert not bool(result.component_proposal_hard_ok[0])
    assert result.component_proposal_rollback_reason[0] == "projector_incomplete"
    assert torch.equal(result.component_proposal_xywh[1], boxes[1])
    assert result.component_proposal_rollback_reason[1] == "not_component"
    assert torch.equal(result.xywh[1], legacy.xywh)


@pytest.mark.parametrize("device", devices())
def test_component_commit_rejects_structure_regression_when_conflicts_drop(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boxes = torch.tensor(
        [
            [0.5, 1.6924152374, 0.5860036612, 1.7271790504],
            [1.0622465611, 1.8703191280, 2.2363724709, 1.8167910576],
            [1.0154242516, 1.0809516907, 2.8245306015, 3.3495683670],
        ],
        device=device,
    )
    neutral = torch.full((1, 3, 3), -1, dtype=torch.long, device=device)
    confidence = torch.zeros((1, 3, 3), device=device)
    contact = neutral.clone()
    contact[0, 0, 1] = projection.BDP_LEFT
    contact[0, 1, 0] = projection.BDP_RIGHT
    contact_confidence = confidence.clone()
    contact_confidence[0, 0, 1] = contact_confidence[0, 1, 0] = 1.0
    boundary_lock = torch.zeros((1, 3, 2), dtype=torch.bool, device=device)
    boundary_lock[0, 0, 0] = True
    guidance = projection.ProjectionGuidance(
        neutral,
        confidence,
        contact,
        contact_confidence,
        boundary_lock,
    )
    problem = {
        "boundary_bits": torch.tensor(
            [[True, False, False, False], [False] * 4, [False] * 4],
            device=device,
        )
    }
    config = projection.ComponentBDPConfig(
        enabled=True,
        beam_width=1,
        max_uncertain_pairs=0,
        outer_sweeps=1,
        preserve_feasible=False,
    )

    guarded = projection.project_disjunctive(
        boxes,
        problem=problem,
        iterations=1,
        guidance=guidance,
        component_config=config,
    )

    def allow_structure_regression(branch_boxes, *_args):
        return torch.ones(
            branch_boxes.shape[:2],
            dtype=torch.bool,
            device=branch_boxes.device,
        )

    with monkeypatch.context() as patch:
        patch.setattr(
            projection,
            "_structure_nonregression",
            allow_structure_regression,
        )
        old_commit = projection.project_disjunctive(
            boxes,
            problem=problem,
            iterations=1,
            guidance=guidance,
            component_config=config,
        )

    original_conflicts = torch.triu(
        projection._active_overlap_matrix_exact(boxes, 1.0e-6),
        diagonal=1,
    ).sum()
    old_conflicts = torch.triu(
        projection._active_overlap_matrix_exact(old_commit.xywh, 1.0e-6),
        diagonal=1,
    ).sum()
    original_contact = projection._contact_residual(
        boxes[None, None], guidance, 1.0e-6
    )
    old_contact = projection._contact_residual(
        old_commit.xywh[None, None], guidance, 1.0e-6
    )

    assert not guarded.ok and not old_commit.ok
    assert original_conflicts.item() == 3
    assert old_conflicts.item() == 1
    assert projection._official_boundary_missing(
        boxes[None], problem, 1.0e-6
    ).item() == 0
    assert projection._official_boundary_missing(
        old_commit.xywh[None], problem, 1.0e-6
    ).item() == 1
    assert old_contact.item() > original_contact.item()
    assert not projection._structure_nonregression(
        old_commit.xywh[None, None],
        boxes[None],
        guidance,
        problem,
        1.0e-6,
    ).item()
    assert torch.equal(guarded.xywh, boxes)
    assert guarded.final_pair_count.item() == 3
    assert bool(guarded.component_proposal_available)
    assert not bool(guarded.component_proposal_hard_ok)
    assert guarded.component_proposal_rollback_reason == ("projector_incomplete",)


def test_component_hard_proposal_rejected_by_structure_is_inspectable(monkeypatch: pytest.MonkeyPatch) -> None:
    boxes = _repeated_signature_boxes()
    guidance = projection.ProjectionGuidance(
        torch.full((1, 5, 5), -1, dtype=torch.long),
        torch.ones((1, 5, 5)),
        torch.full((1, 5, 5), -1, dtype=torch.long),
        torch.zeros((1, 5, 5)),
        torch.zeros((1, 5, 2), dtype=torch.bool),
    )

    def reject_structure(branch_boxes, *_args):
        return torch.zeros(
            branch_boxes.shape[:2],
            dtype=torch.bool,
            device=branch_boxes.device,
        )

    with monkeypatch.context() as patch:
        patch.setattr(projection, "_structure_nonregression", reject_structure)
        result = projection.project_disjunctive(
            boxes,
            iterations=1,
            guidance=guidance,
            component_config=projection.ComponentBDPConfig(
                enabled=True,
                beam_width=4,
                outer_sweeps=8,
                preserve_feasible=True,
            ),
        )

    assert torch.equal(result.xywh, boxes)
    assert not result.ok
    assert bool(result.component_proposal_available)
    assert bool(result.component_proposal_hard_ok)
    assert not bool(result.component_proposal_structure_ok)
    assert result.component_proposal_final_pair_count.item() == 0
    assert result.component_proposal_rollback_reason == ("construction_regression",)
    assert result.displacement.item() == 0.0


def test_component_proposal_excludes_unchanged_feasible_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    boxes = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0]]
    )
    guidance = projection.ProjectionGuidance(
        torch.full((1, 2, 2), -1, dtype=torch.long),
        torch.zeros((1, 2, 2)),
        torch.full((1, 2, 2), -1, dtype=torch.long),
        torch.zeros((1, 2, 2)),
        torch.tensor([[[True, False], [False, False]]]),
    )
    calls = 0

    def count_component_core(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("feasible guided rows must bypass component core")

    with monkeypatch.context() as patch:
        patch.setattr(projection, "_project_component_mode", count_component_core)
        result = projection.project_disjunctive(
            boxes,
            guidance=guidance,
            component_config=projection.ComponentBDPConfig(enabled=True),
        )

    assert result.ok
    assert torch.equal(result.xywh, boxes)
    assert calls == 0
    assert not bool(result.component_proposal_available)
    assert bool(result.component_proposal_hard_ok)
    assert bool(result.component_proposal_structure_ok)
    assert result.component_proposal_rollback_reason == ("already_feasible",)


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("batched", (False, True))
def test_component_feasible_fast_path_matches_forced_slow_telemetry(
    device: str,
    batched: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boxes = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [1.0 - 5.0e-7, 0.0, 1.0, 1.0]],
        device=device,
    )
    guidance = projection.ProjectionGuidance(
        torch.full((1, 2, 2), -1, dtype=torch.long, device=device),
        torch.zeros((1, 2, 2), device=device),
        torch.full((1, 2, 2), -1, dtype=torch.long, device=device),
        torch.zeros((1, 2, 2), device=device),
        torch.tensor([[[True, False], [False, False]]], device=device),
    )
    kwargs = {
        "tolerance": 1.0e-6,
        "guidance": guidance,
        "component_config": projection.ComponentBDPConfig(enabled=True),
    }
    inputs = boxes.unsqueeze(0) if batched else boxes

    fast = projection.project_disjunctive(inputs, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(
            projection,
            "_original_ok_mask",
            lambda work, *_args: torch.zeros(
                work.shape[0],
                dtype=torch.bool,
                device=work.device,
            ),
        )
        slow = projection.project_disjunctive(inputs, **kwargs)

    assert float(fast.max_overlap.max()) > 0.0
    for field in projection.ProjectionResult.__dataclass_fields__:
        fast_value = getattr(fast, field)
        slow_value = getattr(slow, field)
        if isinstance(fast_value, torch.Tensor):
            assert torch.equal(fast_value, slow_value), field
        else:
            assert fast_value == slow_value, field


def test_component_mode_sends_only_infeasible_guided_subset_to_core(monkeypatch: pytest.MonkeyPatch) -> None:
    feasible = torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0]])
    infeasible = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.5, 0.0, 1.0, 1.0]])
    boxes = torch.stack((feasible, infeasible))
    direction = torch.full((2, 2, 2), -1, dtype=torch.long)
    confidence = torch.zeros((2, 2, 2))
    direction[:, 0, 1] = projection.BDP_LEFT
    direction[:, 1, 0] = projection.BDP_RIGHT
    confidence[:, 0, 1] = confidence[:, 1, 0] = 1.0
    guidance = projection.ProjectionGuidance(
        direction,
        confidence,
        torch.full_like(direction, -1),
        torch.zeros_like(confidence),
        torch.zeros((2, 2, 2), dtype=torch.bool),
    )
    seen_shapes = []
    real_core = projection._project_component_mode

    def count_component_core(work, *args, **kwargs):
        seen_shapes.append(tuple(work.shape))
        return real_core(work, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(projection, "_project_component_mode", count_component_core)
        result = projection.project_disjunctive(
            boxes,
            iterations=4,
            guidance=guidance,
            component_config=projection.ComponentBDPConfig(
                enabled=True,
                beam_width=1,
                outer_sweeps=2,
            ),
        )

    assert seen_shapes == [(1, 2, 4)]
    assert torch.equal(result.xywh[0], boxes[0])
    assert result.component_proposal_rollback_reason[0] == "already_feasible"
    assert not bool(result.component_proposal_available[0])
    assert result.ok_mask.tolist() == [True, True]
    assert result.component_proposal_rollback_reason[1] in {
        "committed",
        "projector_incomplete",
        "construction_regression",
    }


@pytest.mark.parametrize(
    ("boxes", "direction"),
    [
        ([[0.0, 0.0, 4.0, 10.0], [3.0, 0.0, 4.0, 10.0]], 0),
        ([[3.0, 0.0, 4.0, 10.0], [0.0, 0.0, 4.0, 10.0]], 1),
        ([[0.0, 0.0, 10.0, 4.0], [0.0, 3.0, 10.0, 4.0]], 2),
        ([[0.0, 3.0, 10.0, 4.0], [0.0, 0.0, 10.0, 4.0]], 3),
    ],
)
def test_assigns_minimum_displacement_direction_for_each_side(boxes, direction):
    pairs, directions = projection.assign_directions(torch.tensor(boxes))
    assert pairs.tolist() == [[0, 1]]
    assert directions.item() == direction


@pytest.mark.parametrize("device", devices())
def test_project_resolves_overlap_and_preserves_dimensions(device):
    boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0], [2.0, 0.0, 4.0, 4.0]], device=device)
    result = projection.project_disjunctive(boxes, iterations=8)
    assert result.ok
    assert result.status == "ok"
    assert bool(result.ok_mask)
    assert result.failure_reasons == ("ok",)
    assert result.active_pair_count.item() == 0
    assert result.displacement.item() > 0.0
    assert result.xywh.device.type == torch.device(device).type
    assert torch.allclose(result.xywh[:, 2:4], boxes[:, 2:4])
    assert projection.overlap_matrix(result.xywh).max().item() <= 1.0e-5


@pytest.mark.parametrize("device", devices())
def test_preplaced_centers_are_anchors(device):
    boxes = torch.tensor(
        [
            [0.0, 0.0, 4.0, 4.0],
            [1.0, 0.0, 4.0, 4.0],
            [8.0, 0.0, 2.0, 2.0],
        ],
        device=device,
    )
    result = projection.project_disjunctive(boxes, preplaced_mask=torch.tensor([True, False, False], device=device), iterations=10)
    assert result.ok
    assert torch.equal(result.xywh[0, :2], boxes[0, :2])
    assert torch.allclose(result.xywh[:, 2:4], boxes[:, 2:4])


@pytest.mark.parametrize("device", devices())
def test_batch_projection_is_independent_and_deterministic(device):
    boxes = torch.tensor(
        [
            [[0.0, 0.0, 4.0, 4.0], [2.0, 0.0, 4.0, 4.0]],
            [[0.0, 0.0, 3.0, 3.0], [0.0, 2.0, 3.0, 3.0]],
        ],
        device=device,
    )
    first = projection.project_disjunctive(boxes, iterations=10)
    second = projection.project_disjunctive(boxes, iterations=10)
    assert first.ok
    assert torch.allclose(first.xywh, second.xywh)
    assert torch.allclose(first.xywh[..., 2:4], boxes[..., 2:4])
    assert torch.all(projection.overlap_matrix(first.xywh).amax(dim=(1, 2)) <= 1.0e-5)
    assert first.ok_mask.tolist() == [True, True]
    assert first.active_pair_count.tolist() == second.active_pair_count.tolist()
    assert torch.all(first.displacement >= 0.0)


@pytest.mark.parametrize("device", devices())
def test_overlapping_preplaced_assignment_fails_closed(device):
    boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0], [2.0, 0.0, 4.0, 4.0]], device=device)
    result = projection.project_disjunctive(boxes, preplaced_mask=torch.tensor([True, True], device=device), iterations=8)
    assert not result.ok
    assert result.status == "infeasible"
    assert result.ok_mask.item() is False
    assert result.failure_reasons == ("fixed_pair_overlap",)
    assert result.max_overlap.item() > 0.0
    assert torch.equal(result.xywh, boxes)


@pytest.mark.parametrize("device", devices())
def test_outer_rebuild_catches_newly_created_overlaps(device):
    boxes = torch.tensor(
        [
            [0.6079329252, 3.9580490589, 2.1840376854, 1.6304452419],
            [0.0058529293, 2.8178839684, 3.2314422131, 3.6374118328],
            [2.7010056973, 0.1247475445, 1.9217861891, 2.4937322140],
        ],
        device=device,
    )

    one_pass = projection.project_disjunctive(boxes, iterations=6, outer_iterations=1, beam=1)
    rebuilt = projection.project_disjunctive(boxes, iterations=6, outer_iterations=3, beam=1)

    assert not one_pass.ok
    assert one_pass.failure_reasons == ("residual_overlap",)
    assert rebuilt.ok
    assert rebuilt.failure_reasons == ("ok",)
    assert projection.overlap_matrix(rebuilt.xywh).max().item() <= 1.0e-5


def test_normalized_problem_uses_raw_coordinate_overlap_tolerance() -> None:
    class Problem:
        scale = 100.0
        normalized = True
        preplaced_mask = torch.tensor([False, False])

    boxes = torch.tensor([[0.0, 0.0, 0.1, 0.1], [0.05, 0.0, 0.1, 0.1]])
    result = projection.project_disjunctive(boxes, problem=Problem(), iterations=8)

    assert result.ok
    assert result.max_overlap.item() <= 1.0e-8


def test_overlap_status_matches_exact_extent_tolerance_when_area_is_tiny() -> None:
    tolerance = 1.0e-6
    boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.9995, 0.9995, 1.0, 1.0],
        ]
    )

    assert projection.overlap_matrix(boxes).max().item() <= tolerance
    assert overlap_pairs(boxes, eps=tolerance) == ((0, 1),)

    ok, reasons, _ = projection._verified_status(
        boxes.unsqueeze(0),
        boxes.unsqueeze(0),
        torch.zeros(2, dtype=torch.bool),
        tolerance,
        torch.zeros(1, dtype=torch.bool),
        False,
    )

    assert ok.tolist() == [False]
    assert reasons == ("residual_overlap",)


def test_result_pairs_and_directions_belong_to_the_winning_beam_variant() -> None:
    boxes = _repeated_signature_boxes()

    result = projection.project_disjunctive(
        boxes,
        iterations=1,
        outer_iterations=3,
        beam=4,
    )
    active = [
        (tuple(pair), int(direction))
        for pair, direction in zip(
            result.active_pairs.tolist(),
            result.directions.tolist(),
            strict=True,
        )
        if direction >= 0
    ]

    assert result.ok
    assert result.active_pair_count.item() == len(active) == 2
    assert active == [((1, 2), 3), ((2, 3), 2)]


@pytest.mark.parametrize("device", devices())
def test_repeated_signature_fixture_is_deterministic_in_legacy_mode(device) -> None:
    boxes = _repeated_signature_boxes(device)

    first = projection.project_disjunctive(
        boxes,
        iterations=1,
        outer_iterations=8,
        beam=1,
    )
    second = projection.project_disjunctive(
        boxes,
        iterations=1,
        outer_iterations=8,
        beam=1,
    )

    assert first.ok is False
    assert first.failure_reasons == ("residual_overlap",)
    assert torch.equal(first.xywh, second.xywh)
    assert torch.equal(first.active_pairs, second.active_pairs)
    assert torch.equal(first.directions, second.directions)
    assert torch.equal(first.active_pair_count, second.active_pair_count)
    assert torch.equal(first.max_overlap, second.max_overlap)


@pytest.mark.parametrize("device", devices())
def test_disabled_component_mode_matches_legacy_projection(device) -> None:
    boxes = _repeated_signature_boxes(device)

    legacy = projection.project_disjunctive(
        boxes,
        iterations=1,
        outer_iterations=3,
        beam=4,
    )
    disabled = projection.project_disjunctive(
        boxes,
        iterations=1,
        outer_iterations=3,
        beam=4,
        component_config=projection.ComponentBDPConfig(enabled=False),
    )

    assert disabled.ok == legacy.ok
    assert disabled.status == legacy.status
    assert disabled.failure_reasons == legacy.failure_reasons
    for field in (
        "xywh",
        "max_overlap",
        "ok_mask",
        "displacement",
        "active_pair_count",
        "directions",
        "active_pairs",
        "initial_pair_count",
        "final_pair_count",
        "component_rebuilds",
        "new_pairs_detected",
        "reset_count",
        "beam_states_evaluated",
        "max_component_size",
        "component_proposal_available",
        "component_proposal_xywh",
        "component_proposal_hard_ok",
        "component_proposal_structure_ok",
        "component_proposal_final_pair_count",
        "component_proposal_displacement",
    ):
        assert torch.equal(getattr(disabled, field), getattr(legacy, field))
    assert disabled.component_proposal_rollback_reason == legacy.component_proposal_rollback_reason
    assert disabled.iterations == legacy.iterations


@pytest.mark.parametrize("device", devices())
def test_component_mode_resolves_without_fabricating_reset_telemetry(device) -> None:
    boxes = _repeated_signature_boxes(device)
    config = projection.ComponentBDPConfig(
        enabled=True,
        beam_width=4,
        component_limit=24,
        max_uncertain_pairs=6,
        outer_sweeps=8,
        reset_limit=2,
        preserve_feasible=True,
    )

    first = projection.project_disjunctive(
        boxes,
        iterations=1,
        beam=1,
        component_config=config,
    )
    second = projection.project_disjunctive(
        boxes,
        iterations=1,
        beam=1,
        component_config=config,
    )

    assert first.initial_pair_count.item() == 7
    assert first.component_rebuilds.item() >= 2
    assert first.new_pairs_detected.item() == 0
    assert first.reset_count.item() == 0
    assert first.beam_states_evaluated.item() > 0
    assert first.max_component_size.item() <= config.component_limit
    assert first.final_pair_count.item() == 0
    assert first.ok
    for field in (
        "xywh",
        "max_overlap",
        "ok_mask",
        "displacement",
        "active_pair_count",
        "directions",
        "active_pairs",
        "initial_pair_count",
        "final_pair_count",
        "component_rebuilds",
        "new_pairs_detected",
        "reset_count",
        "beam_states_evaluated",
        "max_component_size",
    ):
        assert torch.equal(getattr(first, field), getattr(second, field))


@pytest.mark.parametrize("device", devices())
def test_component_mode_resets_an_actual_repeated_conflict_signature(device) -> None:
    boxes = _component_reset_boxes(device)
    config = projection.ComponentBDPConfig(
        enabled=True,
        beam_width=4,
        component_limit=24,
        max_uncertain_pairs=6,
        outer_sweeps=8,
        reset_limit=2,
        preserve_feasible=True,
    )

    result = projection.project_disjunctive(
        boxes,
        iterations=1,
        component_config=config,
    )

    assert result.ok
    assert result.initial_pair_count.item() == 3
    assert result.final_pair_count.item() == 0
    assert result.new_pairs_detected.item() == 1
    assert result.reset_count.item() == 1


@pytest.mark.parametrize("device", devices())
def test_component_mode_honors_projection_iterations(device) -> None:
    boxes = _iteration_sensitive_boxes(device)
    config = projection.ComponentBDPConfig(
        enabled=True,
        beam_width=1,
        max_uncertain_pairs=0,
        outer_sweeps=1,
        preserve_feasible=False,
    )

    one = projection.project_disjunctive(
        boxes,
        iterations=1,
        component_config=config,
    )
    four = projection.project_disjunctive(
        boxes,
        iterations=4,
        component_config=config,
    )

    assert not one.ok
    assert four.ok
    assert not torch.equal(one.xywh, four.xywh)


@pytest.mark.parametrize("device", devices())
def test_component_commit_uses_fp64_overlap_predicate(device) -> None:
    boxes = torch.tensor(
        [
            [1.85589051, 0.0, 1.72176063, 1.0],
            [3.57765102, 0.0, 1.0, 1.0],
        ],
        device=device,
    )
    assert not bool(projection._active_overlap_matrix(boxes, 1.0e-8).any())
    assert bool(projection._active_overlap_matrix_exact(boxes, 1.0e-8).any())

    result = projection.project_disjunctive(
        boxes,
        iterations=4,
        tolerance=1.0e-8,
        component_config=projection.ComponentBDPConfig(
            enabled=True,
            beam_width=1,
            outer_sweeps=2,
        ),
    )

    assert result.initial_pair_count.item() == 1
    assert result.final_pair_count.item() == 0
    assert result.ok
    assert result.displacement.item() > 0.0
