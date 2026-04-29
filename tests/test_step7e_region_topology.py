from __future__ import annotations

from puzzleplace.diagnostics.placement_trace import candidate_ordering_trace
from puzzleplace.diagnostics.region_topology import (
    block_region_assignment,
    free_space_fragmentation,
    net_community_clusters,
    pin_density_regions,
    region_occupancy,
)
from puzzleplace.research.step6g_synthetic import make_step6g_synthetic_case
from puzzleplace.research.virtual_frame import PuzzleFrame


def test_region_occupancy_and_pin_density_emit_grid_rows() -> None:
    case = make_step6g_synthetic_case(case_id=0, block_count=12)
    frame = PuzzleFrame(0.0, 0.0, 120.0, 80.0, density=0.8, variant="unit")
    placements = {
        idx: (float(idx % 4) * 20.0, float(idx // 4) * 16.0, 12.0, 10.0)
        for idx in range(12)
    }

    occupancy = region_occupancy(case, placements, frame, rows=2, cols=3)
    pins = pin_density_regions(case, frame, rows=2, cols=3)

    assert len(occupancy["regions"]) == 6
    assert len(pins["regions"]) == 6
    assert "utilization_spread" in occupancy
    assert "pin_density" in pins["regions"][0]


def test_assignment_reports_mismatch_and_cluster_spread() -> None:
    case = make_step6g_synthetic_case(case_id=0, block_count=16)
    frame = PuzzleFrame(0.0, 0.0, 160.0, 100.0, density=0.8, variant="unit")
    placements = {idx: (float(idx) * 8.0, 0.0, 6.0, 6.0) for idx in range(16)}
    clusters = net_community_clusters(case)

    assignment = block_region_assignment(case, placements, frame, clusters)

    assert assignment["assignments"]
    assert assignment["max_cluster_spread_regions"] >= 1
    assert "assignment_entropy" in assignment


def test_fragmentation_and_trace_are_reconstructed() -> None:
    case = make_step6g_synthetic_case(case_id=0, block_count=20)
    frame = PuzzleFrame(0.0, 0.0, 200.0, 120.0, density=0.8, variant="unit")
    placements = {
        idx: (0.0 if idx % 2 == 0 else 160.0, float(idx) * 4.0, 12.0, 8.0)
        for idx in range(20)
    }
    clusters = net_community_clusters(case)
    assignment = block_region_assignment(case, placements, frame, clusters)

    fragmentation = free_space_fragmentation(case, placements, frame)
    trace = candidate_ordering_trace(case, placements, frame, assignment)

    assert fragmentation["empty_cell_count"] > 0
    assert trace["trace_confidence"] == "reconstructed"
    assert len(trace["first_k"]) <= 16
