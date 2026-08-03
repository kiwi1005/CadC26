from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from hcfp.projection import ComponentBDPConfig, project_disjunctive
from hcfp.projection_guidance import ProjectionGuidance, build_population_guidance


def _case() -> SimpleNamespace:
    return SimpleNamespace(
        n=3,
        boundary_bits=torch.tensor(
            [
                [True, False, False, False],
                [False, False, True, False],
                [False, True, False, True],
            ]
        ),
    )


def _provenance(
    *,
    constraint_records: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    return {
        "topology_seed_orders": (
            {"topology_order_sha256": "vertical"},
            {"topology_order_sha256": "horizontal"},
        ),
        "topology_order_catalog": {
            "vertical": {
                "positive_order": (0, 1, 2),
                "negative_order": (1, 0, 2),
            },
            "horizontal": {
                "positive_order": (0, 1, 2),
                "negative_order": (0, 1, 2),
            },
        },
        "constraint_seed_records": constraint_records,
    }


def test_population_rows_align_constraint_to_recorded_topology() -> None:
    guidance = build_population_guidance(
        _case(),
        _provenance(
            constraint_records=(
                {
                    "kind": "group_contacts",
                    "topology_seed_index": 1,
                    "details": {"moves": ()},
                },
            )
        ),
        residual_count=1,
        constraint_count=1,
        topology_count=2,
    )

    assert guidance.preferred_direction.shape == (4, 3, 3)
    assert torch.equal(guidance.preferred_direction[0], torch.full((3, 3), -1))
    assert torch.equal(guidance.preferred_direction[1], guidance.preferred_direction[3])
    assert torch.equal(guidance.preferred_confidence[1], guidance.preferred_confidence[3])


def test_vertical_topology_maps_above_below_to_bdp_ids() -> None:
    guidance = build_population_guidance(
        _case(),
        _provenance(),
        residual_count=0,
        constraint_count=0,
        topology_count=2,
    )

    assert int(guidance.preferred_direction[0, 0, 1]) == 3
    assert int(guidance.preferred_direction[0, 1, 0]) == 2


def test_contact_moves_store_canonical_inverse_pair() -> None:
    guidance = build_population_guidance(
        _case(),
        _provenance(
            constraint_records=(
                {
                    "kind": "group_contacts",
                    "topology_seed_index": 0,
                    "details": {
                        "moves": (
                            {"anchor": 0, "child": 2, "side": "right"},
                            {"anchor": 1, "child": 2, "side": "above"},
                        )
                    },
                },
            )
        ),
        residual_count=0,
        constraint_count=1,
        topology_count=2,
    )

    assert int(guidance.contact_direction[0, 0, 2]) == 0
    assert int(guidance.contact_direction[0, 2, 0]) == 1
    assert int(guidance.contact_direction[0, 1, 2]) == 2
    assert int(guidance.contact_direction[0, 2, 1]) == 3
    assert float(guidance.contact_confidence[0, 0, 2]) == 1.0


def test_boundary_locks_are_restricted_to_recorded_placed_blocks() -> None:
    guidance = build_population_guidance(
        _case(),
        _provenance(
            constraint_records=(
                {
                    "kind": "combined",
                    "topology_seed_index": 0,
                    "details": {
                        "group": {"moves": ()},
                        "boundary": {"placed": (0, 2)},
                    },
                },
            )
        ),
        residual_count=0,
        constraint_count=1,
        topology_count=2,
    )

    assert torch.equal(
        guidance.boundary_axis_lock[0],
        torch.tensor([[True, False], [False, False], [True, True]]),
    )


def test_guidance_validation_rejects_bad_ranges_and_inverse_mismatches() -> None:
    direction = torch.full((1, 2, 2), -1, dtype=torch.long)
    confidence = torch.zeros((1, 2, 2), dtype=torch.float32)

    bad = direction.clone()
    bad[0, 0, 1] = 4
    with pytest.raises(ValueError, match="BDP ids"):
        ProjectionGuidance(
            bad,
            confidence,
            direction,
            confidence,
            torch.zeros((1, 2, 2), dtype=torch.bool),
        )

    bad = direction.clone()
    bad[0, 0, 1] = 0
    with pytest.raises(ValueError, match="inverse-consistent"):
        ProjectionGuidance(
            bad,
            confidence,
            direction,
            confidence,
            torch.zeros((1, 2, 2), dtype=torch.bool),
        )


def test_malformed_provenance_fails_closed() -> None:
    with pytest.raises(ValueError, match="count mismatch"):
        build_population_guidance(
            _case(),
            _provenance(),
            residual_count=0,
            constraint_count=0,
            topology_count=1,
        )

    bad = _provenance(
        constraint_records=(
            {
                "kind": "group_contacts",
                "topology_seed_index": 0,
                "details": {
                    "moves": ({"anchor": 0, "child": 1, "side": "diagonal"},)
                },
            },
        )
    )
    with pytest.raises(ValueError, match="contact move"):
        build_population_guidance(
            _case(),
            bad,
            residual_count=0,
            constraint_count=1,
            topology_count=2,
        )


def test_guidance_controls_component_direction_and_preserves_legacy_rows() -> None:
    direction = torch.full((2, 2, 2), -1, dtype=torch.long)
    confidence = torch.zeros((2, 2, 2), dtype=torch.float32)
    direction[0, 0, 1] = 3
    direction[0, 1, 0] = 2
    confidence[0, 0, 1] = confidence[0, 1, 0] = 1.0
    locks = torch.zeros((2, 2, 2), dtype=torch.bool)
    guidance = ProjectionGuidance(
        direction,
        confidence,
        torch.full_like(direction, -1),
        torch.zeros_like(confidence),
        locks,
    )
    boxes = torch.tensor(
        [
            [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]],
            [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]],
        ]
    )
    # A large origin makes the FP32-safe component clearance observably larger
    # than the legacy clearance, so this also locks neutral-row routing.
    boxes[..., :2] += 1_000_000.0

    result = project_disjunctive(
        boxes,
        iterations=1,
        component_config=ComponentBDPConfig(
            enabled=True,
            beam_width=1,
            outer_sweeps=2,
        ),
        guidance=guidance,
    )
    legacy = project_disjunctive(boxes[1], iterations=1)

    assert result.ok_mask.tolist() == [True, True]
    assert int(result.directions[0, 0]) == 3
    for field in (
        "xywh",
        "max_overlap",
        "ok_mask",
        "displacement",
        "active_pair_count",
        "directions",
        "initial_pair_count",
        "final_pair_count",
        "component_rebuilds",
        "new_pairs_detected",
        "reset_count",
        "beam_states_evaluated",
        "max_component_size",
    ):
        assert torch.equal(getattr(result, field)[1], getattr(legacy, field))
    assert torch.equal(result.active_pairs, legacy.active_pairs)
    assert result.failure_reasons[1] == legacy.failure_reasons[0]


def test_cyclic_guidance_is_repaired_before_base_projection() -> None:
    direction = torch.full((1, 3, 3), -1, dtype=torch.long)
    confidence = torch.zeros((1, 3, 3), dtype=torch.float32)
    for first, second, forward, reverse in (
        (0, 1, 0, 1),
        (1, 2, 0, 1),
        (0, 2, 1, 0),
    ):
        direction[0, first, second] = forward
        direction[0, second, first] = reverse
        confidence[0, first, second] = 1.0
        confidence[0, second, first] = 1.0
    guidance = ProjectionGuidance(
        direction,
        confidence,
        torch.full_like(direction, -1),
        torch.zeros_like(confidence),
        torch.zeros((1, 3, 2), dtype=torch.bool),
    )
    boxes = torch.tensor(
        [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]]
    )

    result = project_disjunctive(
        boxes,
        iterations=4,
        component_config=ComponentBDPConfig(
            enabled=True,
            beam_width=1,
            max_uncertain_pairs=0,
            outer_sweeps=4,
        ),
        guidance=guidance,
    )

    assert result.ok
    assert projection_is_cycle_free(result.active_pairs, result.directions, 3)


def test_boundary_membership_is_preserved_during_direction_branching() -> None:
    direction = torch.full((1, 2, 2), -1, dtype=torch.long)
    confidence = torch.zeros((1, 2, 2), dtype=torch.float32)
    direction[0, 0, 1] = 1
    direction[0, 1, 0] = 0
    confidence[0, 0, 1] = confidence[0, 1, 0] = 1.0
    locks = torch.zeros((1, 2, 2), dtype=torch.bool)
    locks[0, 0, 0] = True
    guidance = ProjectionGuidance(
        direction,
        confidence,
        torch.full_like(direction, -1),
        torch.zeros_like(confidence),
        locks,
    )
    boxes = torch.tensor(
        [[0.0, 0.0, 2.0, 2.0], [1.0, 0.0, 2.0, 2.0]]
    )

    result = project_disjunctive(
        boxes,
        iterations=4,
        component_config=ComponentBDPConfig(
            enabled=True,
            beam_width=4,
            max_uncertain_pairs=1,
            outer_sweeps=2,
            topology_weight=10.0,
        ),
        guidance=guidance,
    )

    assert result.ok
    assert float(result.xywh[0, 0]) == float(result.xywh[:, 0].amin())


def test_contact_branching_preserves_existing_latch_geometry() -> None:
    direction = torch.full((1, 3, 3), -1, dtype=torch.long)
    confidence = torch.zeros((1, 3, 3), dtype=torch.float32)
    direction[0, 0, 1] = 0
    direction[0, 1, 0] = 1
    confidence[0, 0, 1] = confidence[0, 1, 0] = 1.0
    guidance = ProjectionGuidance(
        torch.full_like(direction, -1),
        torch.zeros_like(confidence),
        direction,
        confidence,
        torch.zeros((1, 3, 2), dtype=torch.bool),
    )
    boxes = torch.tensor(
        [
            [0.0, 0.0, 2.0, 1.0],
            [2.0, 0.9, 2.0, 1.0],
            [2.0, 0.9, 2.0, 1.0],
        ]
    )

    result = project_disjunctive(
        boxes,
        iterations=4,
        component_config=ComponentBDPConfig(
            enabled=True,
            beam_width=4,
            max_uncertain_pairs=1,
            outer_sweeps=2,
        ),
        guidance=guidance,
    )

    assert result.ok
    assert float(result.xywh[0, 0] + result.xywh[0, 2]) == float(
        result.xywh[1, 0]
    )
    overlap_y = min(
        float(result.xywh[0, 1] + result.xywh[0, 3]),
        float(result.xywh[1, 1] + result.xywh[1, 3]),
    ) - max(float(result.xywh[0, 1]), float(result.xywh[1, 1]))
    assert overlap_y > 0.0
    assert float(result.xywh[1, 1] - result.xywh[0, 1]) == pytest.approx(0.9)


def test_satisfied_contact_keeps_semi_rigid_relative_translation() -> None:
    guidance = _right_contact_guidance()
    boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.25, 1.0, 1.0],
            [1.0, 0.25, 1.0, 1.0],
        ]
    )

    result = project_disjunctive(
        boxes,
        iterations=4,
        component_config=ComponentBDPConfig(
            enabled=True,
            beam_width=4,
            max_uncertain_pairs=1,
            outer_sweeps=2,
        ),
        guidance=guidance,
    )

    assert result.ok
    assert torch.allclose(
        result.xywh[1, :2] - result.xywh[0, :2],
        boxes[1, :2] - boxes[0, :2],
    )
    assert not torch.equal(result.xywh[1, :2], boxes[1, :2])


def test_planned_contact_does_not_create_rigid_supernode_before_touching() -> None:
    guidance = _right_contact_guidance()
    boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [5.0, 0.0, 1.0, 1.0],
            [5.0, 0.0, 1.0, 1.0],
        ]
    )

    result = project_disjunctive(
        boxes,
        iterations=4,
        component_config=ComponentBDPConfig(
            enabled=True,
            beam_width=4,
            max_uncertain_pairs=1,
            outer_sweeps=2,
        ),
        guidance=guidance,
    )

    assert result.ok
    assert torch.equal(result.xywh[0, :2], boxes[0, :2])
    assert not torch.equal(result.xywh[1, :2], boxes[1, :2])


def _right_contact_guidance() -> ProjectionGuidance:
    direction = torch.full((1, 3, 3), -1, dtype=torch.long)
    confidence = torch.zeros((1, 3, 3), dtype=torch.float32)
    direction[0, 0, 1] = 0
    direction[0, 1, 0] = 1
    confidence[0, 0, 1] = confidence[0, 1, 0] = 1.0
    return ProjectionGuidance(
        torch.full_like(direction, -1),
        torch.zeros_like(confidence),
        direction,
        confidence,
        torch.zeros((1, 3, 2), dtype=torch.bool),
    )


def projection_is_cycle_free(
    pairs: torch.Tensor,
    directions: torch.Tensor,
    n: int,
) -> bool:
    graphs = [[[] for _ in range(n)] for _ in range(2)]
    for (first, second), direction in zip(
        pairs.tolist(), directions.tolist(), strict=True
    ):
        if direction == 0:
            graphs[0][first].append(second)
        elif direction == 1:
            graphs[0][second].append(first)
        elif direction == 2:
            graphs[1][first].append(second)
        elif direction == 3:
            graphs[1][second].append(first)

    def has_cycle(graph: list[list[int]]) -> bool:
        color = [0] * n

        def visit(node: int) -> bool:
            color[node] = 1
            for other in graph[node]:
                if color[other] == 1 or (color[other] == 0 and visit(other)):
                    return True
            color[node] = 2
            return False

        return any(color[node] == 0 and visit(node) for node in range(n))

    return not any(has_cycle(graph) for graph in graphs)
