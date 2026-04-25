# Step6I — Boundary Frame Commitment Sidecar Plan

## Grounding

`AGENT.md` says this project should be treated as constrained deformable-rectangle
puzzle construction, not coordinate regression.  Candidate generation and
legality masks should guarantee the geometry properties that the model/ranker is
allowed to choose from.  Boundary constraints are soft in the official problem,
but the exponential penalty makes them worth satisfying early rather than fixing
only after construction.

Step6H added a virtual puzzle frame and proved that whole rectangles can be kept
inside the construction board.  The remaining visible issue from the Step6H
rerun is that soft boundary violations can still worsen even when protrusion is
zero.  The likely cause is action-space priority: boundary blocks have frame-edge
candidates, but the heuristic can still choose non-boundary sites or place
boundary blocks too late.

## Scope

Step6I is still **research sidecar only**.  It must not change contest runtime,
finalizer behavior, official evaluator semantics, or `target_positions` leakage
boundaries.

## Mechanism

1. **Boundary satisfaction predicate**
   - For a block with boundary code bits, a candidate satisfies the virtual frame
     only if its rectangle touches all required frame edges:
     - left: `x == frame.xmin`
     - right: `x + w == frame.xmax`
     - bottom: `y == frame.ymin`
     - top: `y + h == frame.ymax`

2. **Boundary-frame commitment**
   - When `commit_boundary_to_frame=True`, candidate generation keeps only
     frame-satisfying candidates for a boundary block if at least one such
     candidate exists.
   - If none exists, keep the legal pool and let frame relaxation/fallback handle
     it rather than returning an empty action set prematurely.

3. **Boundary-aware scoring**
   - Add a direct heuristic bonus for `boundary_frame_satisfaction`.
   - Penalize boundary candidates that miss their required frame edge.

4. **Boundary-first construction order**
   - In virtual-frame runs, schedule boundary-coded blocks before regular blocks
     while preserving the existing seed/start variants inside each class.
   - This reserves frame edges before interior packing fragments the board.

5. **Diagnostics**
   - Add sidecar metrics:
     - `boundary_frame_satisfied_edges`
     - `boundary_frame_total_edges`
     - `boundary_frame_satisfaction_rate`
     - `boundary_frame_unsatisfied_blocks`

## Acceptance checks

- Unit tests prove candidate commitment filters boundary blocks to frame-edge
  placements when possible.
- Unit tests prove heuristic scores prefer frame-satisfying boundary candidates.
- Step6I rerun on cases `0..4` completes all layouts and reports:
  - `semantic_completion_fraction == 1.0`
  - `hard_infeasible_after_repair_count == 0`
  - `max_protrusion_distance == 0.0` for selected pre-repair layouts
  - boundary-frame diagnostics are present in JSON and visual metrics.

## Non-goals

- Do not integrate with `ContestOptimizer` runtime.
- Do not tune official scoring lambdas or local evaluator semantics.
- Do not use soft-block `target_positions` at inference time.
