# Step6J — Boundary Hull Ownership Plan

## Grounding from new AGENT.md

Step6J starts from the Step6I boundary-frame commitment observation:

- frame protrusion stayed solved (`frame_violations = 0`, `max_protrusion = 0`);
- soft boundary violations improved;
- bbox area and repair displacement regressed;
- case-4-like layouts still had poor boundary satisfaction.

The new diagnosis is: **boundary was committed to the wrong object**.
Official boundary means a block should own the final compact bbox edge, not that
it must touch the virtual frame.  The virtual frame is only a hard containment
upper bound.

## Scope

This is still Step6 sidecar work only.  Do not change contest runtime, official
evaluator semantics, or finalizer behavior.  Do not use soft-block
`target_positions` in inference candidate generation.

## Implementation slice

### J-A / J-B audit first

Add sidecar attribution for selected multistart layouts:

- boundary commit attribution per boundary block:
  - required boundary code and corner flag;
  - committed to virtual frame edge;
  - satisfies final bbox edge pre/post repair;
  - distance to virtual frame edge;
  - distance to final bbox edge;
  - leave-one-out bbox expansion contribution;
  - repair displacement by block.
- repair attribution summary:
  - pre/post bbox area;
  - pre/post boundary satisfaction rate;
  - pre/post internal and external HPWL proxy;
  - mean repair displacement for all vs boundary blocks.

Artifacts:

- `artifacts/research/step6j_boundary_hull_ownership.json`
- `artifacts/research/step6j_boundary_hull_ownership.md`

### J-C / candidate change

Replace hard frame-edge commitment in the sidecar runner with a conservative
predicted-hull preference:

- Estimate a predicted compact hull inside the virtual frame from total area,
  pin/preplaced anchors, and density.
- Generate hull-edge sliding sites for boundary blocks.
- Prefer hull-satisfying candidates in the heuristic score, but keep normal
  compact candidates for comparison.
- Keep virtual-frame edge sites as fallback, guarded by bbox/protrusion metrics.

### J-D / edge capacity first pass

Record edge capacity diagnostics:

- edge length;
- number of required boundary blocks per edge;
- total projected span demand;
- occupancy fraction.

This first pass is diagnostic; segment optimization can follow only if evidence
shows edge conflict is the bottleneck.

## Gates

Required:

- `semantic_completion_fraction == 1.0`
- `hard_infeasible_after_repair_count == 0`
- selected pre-repair layouts have `frame_violations == 0`
- selected pre-repair layouts have `max_protrusion_distance == 0` up to numeric eps

Boundary/bbox tradeoff:

- boundary satisfaction should not collapse versus Step6I;
- soft violation reduction should stay positive or near Step6I;
- bbox regression should be lower than Step6I, or the attribution file must make
  the regression source explicit;
- repair displacement should be lower than Step6I if hard frame commitment was
  the source of churn.
