# HCFP-5090 greenfield implementation plan

> Detailed staged tasks, dependencies, current status, and exit gates are tracked
> in [`hcfp5090_staged_execution_roadmap.md`](hcfp5090_staged_execution_roadmap.md).
> P0 correctness is closed by the official 100-case replay documented in
> [`hcfp5090_p0_correctness_2026-08-01.md`](hcfp5090_p0_correctness_2026-08-01.md).
> The initial P1 analytic/BDP and telemetry slice remains on HOLD because its
> exact validation QoR is still at the feasible cost cap.

## Goal

Build the new HCFP solver as the only runtime path in this branch. The initial
release must be a safe, inspectable analytic baseline before learned modules or
GPU-specific compilation are introduced.

## Verified deployment constraints

- Official solve accepts seven inputs, with `target_positions` optional.
- Contest execution is one process launch per case under an external timeout;
  startup cost is part of runtime.
- The deployment GPU is an A100 80 GB. The hard path therefore uses FP32
  geometry and does not require FP8 or RTX 5090-only features.
- Network access is unavailable during evaluation.
- The official v10 evaluator is pinned by commit and SHA256 in
  `src/hcfp/reference.py`.

## Milestone 0 — contract and safe incumbent

1. Parse official padded tensors into a validated `FloorplanCase`.
2. Reproduce hard-constraint and metric predicates in an exact-compatible
   verifier.
3. Construct a deterministic shelf fallback that preserves every hard target.
4. Expose the official optimizer and JSON stdin/stdout entrypoints.

Acceptance: every exception or non-finite candidate returns the safe incumbent;
preplaced and fixed-shape tests pass exactly.

## Milestone 1 — analytic HCFP core

1. Create a batched population with diverse deterministic layouts.
2. Apply typed net, pin, overlap, grouping, boundary, compaction, and MIB shape
   channels in FP32.
3. Preserve hard geometry after each fixed-step update.
4. Project the best candidates with BDP v0 and verify before promotion.

Acceptance: CPU/CUDA deterministic smokes pass, overlap decreases on collision
fixtures, and no candidate can replace the fallback without verification.

## Milestone 2 — topology and recovery

1. Add conflict diagnostics and component-local event actions.
2. Add a small direction beam and improve BDP residual handling.
3. Maintain safe, fast-feasible, and exact-feasible incumbent tiers.

Acceptance: projection success and repair displacement improve without new hard
violations or runtime tails.

## Milestone 3 — learned modules

Introduce SCENE, POP-INIT, HiCoDy residuals, and PVR in that order. Each module
stays optional and must beat its analytic counterpart on post-BDP exact metrics.
ETR training begins only after event labels can be generated from exact replay.

## Promotion gates

- Hard feasibility, fixed/preplaced replay, and official scorer parity: 100%.
- Fallback remains reachable on every failure path.
- No new cost=10 case; the 106–120 block subset may not regress.
- Runtime decisions use median and p95 evidence, including process startup.
- Learned or compiled paths remain disabled unless their post-BDP exact result
  is better and portable on the official A100 environment.
