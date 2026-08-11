# HCFP-5090 Q7 MIB preservation and bounded group-repair results

Date: 2026-08-11

Branch: `feat/hcfp5090-qor-first`

Evidence parent: `dfde049`

Decision: **QoR checkpoint PASS; Q6-preservation and runtime promotion HOLD**

## Implemented causal fixes

The Q6 MIB regression was caused before BDP: per-block learned aspect ratios were
converted to dimensions independently, while only a small constraint-candidate
subset received group-level MIB construction. BDP translates rectangles and
does not change their dimensions, so it could not restore a shared MIB shape.

Q7 makes compatible MIB groups shared-by-construction in the common exact shape
projection used by learned residual and topology geometry. It preserves hard
fixed/preplaced shapes, skips incompatible area-tolerance intersections, and
keeps every soft member within the official one-percent hard-area tolerance.

Q7 also adds a bounded post-tail grouping repair. It examines at most 12
contact moves and admits a move only when:

- exact hard verification passes;
- boundary, grouping, and MIB counts are componentwise non-regressive;
- normalized total soft violation, bounding area, and HPWL Pareto-dominate the
  current placement.

Collective rollout and active ranker pruning remain disabled.

## Official visible 100 result

The run uses the Q6 checkpoint and search configuration unchanged:

```text
flow_steps=0
collective_steps=0
topology_seeds=16
constraint_seeds=16
flow_seed=7001
execution_seed=7001
```

| Metric | Q6 | Q7 | Delta |
|---|---:|---:|---:|
| Hard feasible | 100/100 | 100/100 | 0 |
| Exact uncapped | 32 | 70 | +38 |
| Weighted capped cost | 9.724486 | 8.989727 | -0.734760 |
| 106--120 weighted cost | 9.814157 | 9.190193 | -0.623964 |
| 116--120 exact uncapped | 0/5 | 3/5 | +3 |
| Boundary violations | 1,506 | 1,601 | +95 |
| Grouping violations | 1,247 | 815 | -432 |
| MIB violations | 243 | 0 | -243 |
| Runtime p50 | 2.54595 s | 2.59693 s | +0.05097 s |
| Runtime p95 | 6.55456 s | 6.60742 s | +0.05286 s |

Against the analytic incumbent, Q7 records 70 improvements, 30 ties, and zero
regressions. The original near-cap cohort crosses on 11/12 cases; case 14 is the
only remaining member. Case 98, which was just outside that cohort, also crosses.
The required large-case targets therefore both pass:

| Case | Q6 cost | Q7 cost | Q7 MIB | Result |
|---:|---:|---:|---:|---|
| 96 | 9.999999 | 9.178030 | 0 | exact uncapped |
| 98 | 9.999999 | 9.771748 | 0 | exact uncapped |

## Remaining selection defect

Q7 is not yet a drop-in replacement for Q6 on every already-improved case.
Comparing Q7 directly with the preserved Q6 learned output gives 65 improvements,
29 ties, and six regressions. Five are small; case 63 regresses from `7.427046`
to the cap. This is not a hard-feasibility failure and Q7 still ties the analytic
incumbent there. It means MIB-preserving geometry replaced, rather than augmented,
the prior candidate family.

The next selection slice should retain a bounded legacy-shape challenger for
the affected observable case signature and let exact post-tail scoring choose
between legacy and MIB-preserving candidates. It must not use case IDs or stored
visible solutions. Until that portfolio preserves Q6 per-case QoR, Q7 remains a
research checkpoint rather than the default submission lane.

## Verification

```text
targeted geometry/learned/raw-repair tests: PASS
full pytest: 567 collected, PASS
Ruff src/tests/scripts/submission: PASS
compileall src/scripts/submission: PASS
git diff --check: PASS
official visible 100: 100/100 hard feasible
```

Evidence artifact:

```text
artifacts/benchmarks/hcfp5090-q7-mib-group-greedy-official100-seed7001.json
SHA-256 6abb7264b2849e555b55a3c02a6c0ea62f005c40c585972ab8ff080fc3929d7a
```

Static type checking was not run because the active Python environment does
not contain `mypy`.
