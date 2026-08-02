# HCFP-5090 official selector-label phase

Date: 2026-08-02

## Outcome

Replace the repair ranker's proxy-only supervision with labels that preserve the
FloorSet v10 training baseline and can express the runtime-independent official
quality objective. This phase must not change the contest runtime or promote the
learned lane.

## Current evidence

- The full 100-case oracle audit found no learned oracle wins after BDP.
- Raw-safe reselection removed seven unnecessary analytic replays and improved
  those seven uncapped objectives without changing the normal learned fast path.
- Existing replay labels use absolute HPWL, bounding-box area, soft violation,
  and displacement. They cannot distinguish the same candidate quality relative
  to the case-specific official solution baseline.
- FloorSet-Lite stores `metrics_sol` as
  `[area, pins, nets, b2b_nets, p2b_nets, hard_constraints,
  b2b_weighted_wl, p2b_weighted_wl]`.

## Data contract

1. Preserve two raw-unit scalars on `SolutionLabels`:
   `baseline_area` and `baseline_hpwl`.
2. FloorSet-Lite extraction uses `metrics_sol[0]` and
   `metrics_sol[6] + metrics_sol[7]` after finite/non-negative validation.
3. Fixtures and old shard payloads derive compatible baselines from their
   ground-truth rectangles and case connectivity. Existing artifacts therefore
   remain readable.
4. Geometry labels stay normalized. Baselines stay in official raw units and
   remain invariant under D4 augmentation.
5. No new dependency, checkpoint field, runtime branch, or model parameter is
   introduced in the data-contract subphase.

## Replay target contract

The first bounded experiment showed that the capped runtime-independent cost is
not a usable training target: 241/256 candidates were hard-infeasible, the other
15 were capped at `9.999999`, and 21/32 records were completely tied. The final
target therefore preserves the official ordering without preserving its cap:

```text
hard feasible -> log(1 + alpha * (relu(hpwl_gap) + relu(area_gap)))
                 + beta * soft_violation_relative
hard infeasible -> max(feasible target) + 1 + exact-tail repair residual
```

The feasible branch is a monotone transform of the uncapped v10 objective with
RuntimeFactor fixed at 1. The infeasible branch is strictly worse and uses
post-BDP overlap, conflict components, projection status, and displacement only
to break the official hard-cost tie. Candidate HPWL and area remain in the same
raw units as the stored baseline.

## Stages and gates

### L1 — baseline ingestion

- Thread `metrics_sol` through the direct FloorSet-Lite stream.
- Serialize and deserialize both baselines.
- Prove old payload compatibility and D4 invariance.

Gate: targeted data/Lite tests and full shard round-trip pass.

### L2 — objective-aligned replay

- Add a small pure function for lexicographic official-quality targets.
- Use it in replay generation with exact post-repair telemetry.
- Keep schema compatibility explicit; old replay records must either remain
  readable or fail with a precise version error.

Gate: feasible ordering matches the pinned evaluator formula and every
infeasible target ranks after every feasible target.

### L3 — bounded label experiment

- Generate a small deterministic replay set from training samples only.
- Report feasibility split, target-score spread, and ranking disagreement with
  the old proxy.
- Train only if targets contain useful non-tied signal.

Gate: evidence artifact records input provenance, checkpoint hash, seed, and
label distribution. Validation/test data is forbidden.

### L4 — promotion decision

The selector can enter a runtime experiment only after top-1 regret improves on
held-out training data and the full official validation benchmark remains
100% hard-feasible with no new `cost=10` case. Until then it stays research-only.

## Stop conditions

- Stop and repair the contract if a baseline changes under augmentation or
  normalized/raw units are mixed.
- Do not spend additional long training if the bounded replay has tied labels or
  no ranking disagreement.
- Do not modify the strict legalizer, Pareto gate, submission entrypoint, or
  fallback in this phase.
