# HCFP-5090 Q4 component-aware BDP results

Date: 2026-08-03
Branch: `feat/hcfp5090-qor-first`
Base: `origin/main` at `2ddc494`
Status: **exact-safe QoR checkpoint; runtime promotion gate remains blocked**

## Decision

Q4 establishes an exact-safe implementation checkpoint. Structured candidates
now use cycle-free conflict-component branching, active-contact semi-rigid
motion, exact FP64 commit checks, construction-nonregression gates, and
post-projection HPWL/bbox-aware branch ordering. Guided rows either cross the
hard-feasibility boundary or retain their original geometry; they no longer
fall back to a legacy partial projection that can destroy a raw-repairable
candidate. An uncommitted component proposal is now retained separately from
the primary candidate and can enter the runtime only after raw constraint
replay, exact hard verification, and the existing dominance guard.

The retained-proposal diagnostic closes the earlier feasibility/QoR gap:
the exact per-candidate raw/projected/proposal portfolio reaches 364/512
hard-feasible constraint candidates, above the legacy control's 338/512, and
improves constraint-oracle weighted `J` to `2.276194`. It also changes the
runtime-selected result on held-out case 14 from `J=3.052426` with 45 soft
violations to `J=2.415978` with 27 soft violations. The original candidate
remains recoverable and no proposal can bypass exact verification or Pareto
dominance.

Clean-commit large16 evidence now confirms the QoR result and exact output
stability. Commit `846147d` produces the same sample list and zero placement
hash differences across all 1,552 raw, post-BDP, and exact-portfolio records
relative to `e57058d`. Constraint coverage remains 364/512 and weighted
constraint-oracle `J` remains `2.276194`. Vectorizing the rigid-component
collision predicate lowers solver p50 from 22.913 to 10.070 seconds and p95
from 27.623 to 14.387 seconds without changing a selected placement.

This is still not default-promotion proof. The paired analytic p95 is 0.0441
seconds, leaving a 326.1x learned/analytic p95 ratio. Schema v7 separates solver
runtime, runtime-final selection, offline candidate audit, and full audit wall
time; the runtime failure is in the learned solver rather than evaluator
overhead.

The implementation therefore remains opt-in. Q2 still has a 77.3% group
connectivity result rather than the 95% target. Q3 may now be implemented as a
default-off lane because the clean benchmark preserves the Q2/Q4 oracle gain,
but it cannot be promoted until runtime is reduced further. The analytic
incumbent, exact verifier, fallback, and dominance safety invariant remain
mandatory.

## Implemented causal path

```text
topology / contact / boundary provenance
  -> cycle-free preferred separation directions
  -> conflict components rebuilt after every sweep
  -> bounded uncertain-pair beam per component
  -> fixed-direction projection with dynamic conflict rebuild
  -> active-contact-only semi-rigid supernodes
  -> hard / construction / quality / motion branch tiers
  -> exact FP64 commit gate
  -> exact verifier and dominance safety invariant
```

Key implementation points:

- `ProjectionGuidance` aligns topology, contact, and boundary decisions with
  residual, constraint, and topology candidate rows.
- Contact-tree edges remain planned constraints until their side gap and
  orthogonal overlap satisfy the latch-on geometry. Only active contacts form
  semi-rigid membership.
- H/V cycles are repaired before the base branch enters projection. Every
  uncertain branch is cycle checked.
- Disconnected conflict components build independent bounded beams and are
  merged deterministically.
- Repeated conflict signatures may reset to the second branch only inside the
  same construction-admissibility stratum.
- Branch ordering is lexicographic: fixed-pair safety, hard feasibility,
  boundary/contact nonregression, conflict residual, construction Pareto rank,
  HPWL/bbox Pareto rank, official-local quality, displacement, then branch ID.
- HPWL/bbox are evaluated on complete projected branch geometry. They are not
  approximated as independent pair penalties.
- Component movement commits only after an exact feasibility crossing and no
  boundary/contact regression. Otherwise the original candidate is retained
  byte-for-byte.
- Neutral rows use legacy BDP with legacy clearance. Guided rows do not re-enter
  legacy BDP after a component failure.
- Cycle checks reuse CPU pair values instead of transferring the dense pair
  tensor from CUDA for every branch.

The explicit Per-RMAP-style pre-projection HPWL perturbation remains deferred.
The current implementation ranks already projected branch states; adding
another motion source before exact Q4 evidence exists is not justified.

## Correctness fixes discovered during Q4

1. FP32 projection could report no overlap while FP64 exact replay still found
   micro-overlaps. Component assignment and commit now use the FP64 predicate,
   with FP32-ULP-safe clearance.
2. Denormalization previously performed FP32 arithmetic before conversion to
   FP64. The case and placement now convert first, then denormalize and replay
   raw hard targets.
3. Zero-step dynamics reconstructed boxes through center/aspect state and could
   change geometry. It now returns the supplied boxes exactly.
4. Post-relax constraint provenance could reuse the initial hash after geometry
   changed. Provenance is now attached only when the actual candidate hash
   matches; stale sources fail closed.
5. The mixed component/legacy router selected legacy whenever component status
   was not internally feasible. That bypassed preserve-original semantics and
   caused four official hard regressions. Guided failures now retain their raw
   geometry; only neutral rows use legacy BDP.
6. Audit displacement averaged failed no-commit candidates without showing
   feasibility coverage. Schema v4 reports hard-feasible-conditioned movement,
   new feasibility, regressions, no-commit count, and per-case p50/p95 runtime.
7. Mixed routing applied FP32-safe component clearance to neutral learned rows.
   Neutral rows now retain legacy clearance so component and neutral effects can
   be attributed separately in the clean rerun.
8. A component proposal that did not pass the normalized commit gate could
   still become exact-feasible after raw group/boundary replay. `ProjectionResult`
   now retains only changed, uncommitted proposals. Already-feasible no-ops and
   already-committed geometry are not duplicated in the runtime portfolio.
9. Audit wall time previously included hundreds of official evaluator calls per
   case. Schema v7 records solver core, runtime-final selection, offline
   candidate audit, and full audit wall time separately. The compatibility
   `runtime_seconds` field is the solver total, not the audit wall clock.
10. Group construction performed 3.48 million Python scalar rectangle checks on
    one 119-block case. The collision predicate now translates only the rigid
    component and evaluates every component/outside rectangle pair in FP64.
    Strict edge-touch semantics, move ordering, provenance, and final geometry
    remain unchanged.

## Clean exact-safe large16 benchmark

Authoritative clean artifact:

```text
artifacts/benchmarks/hcfp5090-q4-clean-846147d-vector-overlap-large16.json
SHA-256: 846d6834a5cd395bb8ef591d7f198dcfaccea9a2843933e77acd1815c0a3fd8d
solver commit: 846147d70a23511364d395dce2dc1430376a5d1d
sample-list SHA-256:
e79491726efed65ebb6c55f7e63564e4896cc7d345bca18da725660de7d8fab9
```

The comparator alternates execution order: eight cases run learned first and
eight run analytic first. Analytic raw replay is hard feasible in 15/16 cases;
the existing safe fallback makes the final analytic comparator hard feasible
in 16/16.

| Metric | `e57058d` clean baseline | `846147d` vectorized | Change |
|---|---:|---:|---:|
| solver p50 | 22.9128 s | 10.0702 s | -56.05% |
| solver p95 | 27.6234 s | 14.3866 s | -47.92% |
| solver mean | 21.6225 s | 10.3203 s | -52.27% |
| solver total | 345.9601 s | 165.1246 s | -52.27% |
| constraint exact coverage | 364/512 | 364/512 | identical |
| constraint weighted oracle `J` | 2.276194 | 2.276194 | identical |
| selected placement hashes | 16 | 16 | 0 mismatches |

All 1,552 raw hashes, 1,552 post-BDP hashes, 424 retained-proposal hashes, and
1,552 exact-portfolio hashes match the clean baseline. After excluding timing
and solver-provenance fields, the two schema-v7 reports have zero semantic
differences.

The remaining runtime gap is explicit: analytic p50/p95 are 0.0240/0.0441
seconds and learned/analytic p50/p95 ratios are 419.9x/326.1x. Q4 therefore
passes the exact-safe QoR checkpoint but fails the 1.20x runtime promotion gate.

## Retained-proposal diagnostic

The following artifact is a dirty-worktree diagnostic. It records the solver
commit and dirty fingerprint, but cannot serve as clean promotion evidence:

```text
artifacts/benchmarks/hcfp5090-q4-proposal-portfolio-diagnostic-large16.json
SHA-256: 4d5eb6487da3e165bd452f6caf433bc28875fe3c51c6bd6ab2cf07a9eda39693
sample-list SHA-256:
e79491726efed65ebb6c55f7e63564e4896cc7d345bca18da725660de7d8fab9
```

| Constraint candidate gate | Hard feasible / 512 | Weighted oracle `J` |
|---|---:|---:|
| raw exact replay | 316 | 2.279428 |
| component primary | 322 | 2.279428 |
| legacy control | 338 | 2.281423 |
| exact raw/projected/proposal portfolio | 364 | 2.276194 |

The proposal stage repairs 326/512 rows in the diagnostic before unchanged and
committed duplicates are pruned. The exact portfolio selects proposal geometry
for 186 candidate indices; 48 of those selections are hard feasible, while the
remaining selections are diagnostic least-violation representatives when no
stage for that index is feasible. Case-level runtime selection changes only
when the proposal is exact feasible and dominates the incumbent. Case 14 is the
material win: `J` improves by `0.636448` and soft violations fall by 18. Other
selected-placement differences against the legacy artifact are numerical noise.

The latest schema-v7 smoke is:

```text
artifacts/benchmarks/hcfp5090-q4-schema7-smoke-case2.json
SHA-256: 9fb5201e1b23bfef222c90fd3015ee425bafc974eca2ea557d5e4f199e913a67
```

It runs the analytic comparator first, then the learned lane, on held-out index
1. Learned core plus exact selection is 22.765 seconds; analytic solve plus raw
replay and exact check is 0.243 seconds. Offline candidate audit is another
10.185 seconds and is explicitly excluded from both solver measurements. The
learned runtime is dominated by the 22.004-second solver core, not the
0.761-second exact selection. This makes guided-row/component projection
runtime the next direct engineering target.

## Provisional historical paired audit

Both runs use:

- checkpoint `hcfp5090-q2-constraints-s1000-seed5090.pt`;
- checkpoint SHA-256
  `5c013e14b7b172f40a8be2434a0f185645bdd3d2a7d26125504b72da6c29a4be`;
- the same 16 training-source held-out cases from 16 files;
- 107--120 blocks, held-out seed `5091`, training seed `5090`;
- sample-list SHA-256
  `e79491726efed65ebb6c55f7e63564e4896cc7d345bca18da725660de7d8fab9`;
- 16 topology and 16 constraint seeds per case;
- official-v10 evaluator commit
  `aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa`;
- exact raw-coordinate replay and audit schema v4.

Artifacts:

```text
artifacts/benchmarks/hcfp5090-q4-runtime-legacy-large16.json
SHA-256: 9a61afd3ec64da0bc8f7afe61b8b55660e8afae13fd9324371fccd3291149781

artifacts/benchmarks/hcfp5090-q4-runtime-component-preserve-large16.json
SHA-256: 38347714624542ab8a73bbeb44d08954d835b76f0c2cbe58d2133743ecc4d189
```

These artifacts are retained as historical diagnostics, not promotion proof.
They omit `solver_commit` and `dirty_patch_hash`. The legacy artifact was
written at 22:10, the component artifact at 22:27, and projection code changed
between and after those runs. They also predate the neutral-clearance isolation
fix above. A clean-commit rerun must produce legacy, component, and pure-analytic
comparators with the solver commit embedded in every artifact.

## Paired result

| Metric | Legacy control | Q4 component | Decision |
|---|---:|---:|---|
| total post-BDP hard feasible | 1,004 / 1,552 | 1,052 / 1,552 | confounded by neutral clearance |
| constraint post-BDP hard feasible | 338 / 512 | 322 / 512 | gate miss |
| learned-residual post-BDP hard feasible | 42 / 256 | 106 / 256 | confounded by neutral clearance |
| constraint hard regressions | 4 | 0 | exact-safety evidence |
| constraint newly hard feasible | 26 | 6 | gate miss |
| constraint oracle weighted `J` | 2.281423 | 2.279428 | diagnostic improvement |
| constraint cap-cross cases | 10 / 16 | 10 / 16 | tie |
| constraint weighted movement | 1.221576 | 0.150822 | dominated by 506 no-commits |
| feasible-conditioned constraint movement | 0.018450 | 0.247610 | regression, 13.4x |
| topology weighted movement | 0 | 0 | tie |
| runtime p50 | 23.126 s | 27.747 s | learned/learned 1.200x |
| runtime p95 | 29.163 s | 33.527 s | learned/learned 1.150x; analytic gate unproven |
| runtime total | 374.118 s | 436.035 s | 1.166x |

Constraint-oracle comparison is 2 wins, 14 ties, and 0 losses. Final selected
weighted `J` changes by only about `-6.5e-11`, so selected QoR is effectively a
tie; 13/16 selected placement hashes are identical and the other three differ
only at numerical-noise scale. Boundary and MIB oracle totals do not regress;
grouping improves by one violation. One case also improves bbox/area. HPWL
differences outside those wins are at numerical-noise scale.

The lower constraint feasibility count is a promotion blocker: Q4 intentionally refuses
legacy partial movement for guided failures. This removes 4 regressions and 506
of 512 constraint candidates become no-commit. The total candidate pool still
gains 48 hard-feasible candidates because component-safe clearance also reached
neutral learned-residual legalization in this historical run. That routing
confound is fixed in the current code and the gain must be remeasured.

Hard-feasible-conditioned displacement rises because six previously infeasible
MIB candidates now enter that denominator with nontrivial movement. The direct
Q2 metric was movement over every constraint candidate; on that paired metric,
Q4 falls from `1.221576` to `0.150822`.

The branch metric named construction Pareto rank is a direct-dominator count,
not a complete Pareto-front rank. HPWL and bbox are normalized local proxies;
formal safety continues to come from the exact tail. Runtime also needs a pure
analytic comparator: 4/16 per-case learned/learned ratios exceed 1.20 and the
worst is 1.351x. This is a single-run 16-case diagnostic, not a full promotion
or release result.

## Compatible-MIB probe

The only area-compatible MIB held-out case is
`worker_89/layouts_1232.th:0` (held-out index 14). With beam 8 and eight sweeps,
Q4 legalizes the shared-shape candidate with zero MIB violation. Objective-aware
branch ordering lowers its movement from `51.77` to `39.28` (24.1%), but its
`J` remains about `3.014`, worse than the existing constraint oracle `2.421`.
This proves hard-by-construction MIB can survive exact legalization, but it is
not a promotion-quality candidate yet. A larger MIB-only budget therefore
stays off by default.

## Promotion and next stage

Q4 is an exact-safe QoR checkpoint, not a promoted stage. Default learned
activation remains shadow-only. The feasibility/QoR portfolio target is met in
the clean large16 benchmark, but the learned-versus-analytic runtime target is
not close. The immediate sequence is:

1. keep the clean `846147d` artifact as the Q4 semantic/runtime baseline;
2. preserve the now-vectorized construction path and profile the remaining
   topology construction, exact-tail, and final-selection costs separately;
3. implement geometry-aware dynamics only as an opt-in candidate source, with
   live force gates and exact hash/provenance boundaries;
4. require exact portfolio coverage of at least 338/512, zero selected hard
   regressions, no weighted-`J` regression, and learned p95 at most 1.20 times
   analytic p95;
5. use `scripts/benchmark_hcfp.py` for final official-wrapper wall-time proof.

Q3 collective dynamics may now proceed as a default-off training/runtime lane
because the structure/constraint path has positive exact-safe oracle density.
It cannot be promoted while the runtime gate fails. Exact verification and
Pareto-safe fallback remain non-negotiable.
