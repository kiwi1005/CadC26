# HCFP-5090 current-case diagnosis and batch grill

Date: 2026-08-11

Branch: `feat/hcfp5090-qor-first`

Status: all G1--G8 recommendations accepted; no solver change

Evidence snapshot: Q6 official visible 100, canonical seed 7001

## Scope and evidence boundary

This report diagnoses the latest preserved 100-case official-visible result, not a new run from the current `HEAD`. The benchmark was produced from clean solver commit `6b0f427` and later recorded by the Q6 release documentation. The three Q6 seeds produced identical final geometry, so seed 7001 is used as the canonical per-case source.

Primary evidence:

- `artifacts/benchmarks/hcfp5090-q6-shadow-official100-seed7001.json`;
- `artifacts/visualizations/hcfp5090-q6-official100-png/case_000.png` through `case_099.png`;
- `artifacts/reports/hcfp5090-q6-case-diagnosis-official100-seed7001.json`;
- `artifacts/reports/hcfp5090-q6-case-diagnosis-official100-seed7001.csv`;
- `artifacts/reports/hcfp5090-q6-cap-attribution-official100-seed7001.json`;
- `docs/research/hcfp5090_q6_release_results_2026-08-08.md`.

The reconstructed boundary, grouping, and MIB counts reproduce `violations_relative` for all 100 cases in both analytic and Q5-shadow lanes. This makes the per-constraint totals below exact for this evidence snapshot. Raw, projected, and post-repair stages are not present in the Q6 benchmark rows, so this report cannot attribute a particular final violation to one tail substage.

## Executive diagnosis

The current result is not blocked by hard feasibility. It is blocked by incomplete soft-constraint construction or preservation, weak quality on a small set of structurally bad cases, and runtime.

| Signal | Current result | Meaning |
|---|---:|---|
| Hard feasible | 100/100 | Exact legalization and fallback are working |
| Exact uncapped | 32/100 | 68 cases remain above the official cap |
| Near cap, margin greater than -0.15 | 12 cases | Several can cross with one to four soft fixes |
| Far from cap, margin at most -0.50 | 20 cases | Local repair alone is unlikely to be enough |
| Improved / tied / regressed capped cost | 32 / 68 / 0 | Pareto safety is working |
| Geometry changed from analytic | 88 cases | The learned candidate lane is active on most cases |
| Cap-saturated improvements | 56 cases | Useful factor improvements are hidden by the cap |
| Unchanged from analytic | 12 cases | No accepted challenger displaced the incumbent |
| Known candidate coverage failures | cases 65 and 89 | No exact-eligible learned initial candidate |

One terminology trap matters here. `src/hcfp/benchmark.py` labels every cost at least `9.99` as `capped_feasible`; exact cap status instead uses `cap_margin > 0`. Case 84 has cost `9.997674` and exact cap margin `+0.000233`: it is exact uncapped but still receives the stricter benchmark label. This report uses exact cap margin, yielding 32 uncapped and 68 capped cases rather than the benchmark's 31/69 split.

## What the active Q6 path actually measures

The Q6 command used `flow_steps=0` and `collective_steps=0`. The checkpoint advertises `ranker=false`, so active ranker pruning is also disabled. The current PNGs therefore diagnose:

```text
structured topology and constraint candidates
  -> exact/projected tail
  -> incumbent and Pareto guard
```

They do not diagnose the disabled collective rollout, and a ranker mistake is not the cause of the final placement. Q3 and Q5 remain separate promotion blockers, but neither should be used to explain a Q6 final-case failure without candidate-level evidence.

## Problem 1: boundary and grouping improve, but remain unfinished

| Soft constraint | Analytic total | Q5-shadow total | Delta | Result |
|---|---:|---:|---:|---|
| Boundary | 2,136 | 1,506 | -630 | Material improvement, still largest residual |
| Grouping | 1,668 | 1,247 | -421 | Material improvement, still severe on large cases |
| MIB | 55 | 243 | +188 | Systematic regression |
| Total | 3,859 | 2,996 | -863 | Net improvement despite MIB regression |

Among the 68 capped cases, 37 are boundary-led and 30 are grouping-led by the largest soft contribution. The remaining case is quality-required under the counterfactual cap-blocker definition. No capped case is purely hard-blocked.

The 106--120-block subset retains 318 boundary, 302 grouping, and 36 MIB violations. In the 116--120 bucket, all five cases remain capped and retain 93 boundary, 119 grouping, and 13 MIB violations.

Interpretation: Q2 construction is useful, but its 77.3% group-connectivity gate and unfinished MIB carry-through are visible in final official cases. Boundary and contact decisions are not being completed or preserved sufficiently as case size grows.

## Problem 2: MIB is a tail-preservation regression, not a search-volume problem

The Q5-shadow lane reduces boundary and grouping violations but increases MIB violations from 55 to 243. The largest individual regressions include:

| Case | Blocks | MIB delta | Final MIB | Cap margin | Required soft fixes |
|---:|---:|---:|---:|---:|---:|
| 44 | 65 | +6 | 6 | -0.1405 | 4 |
| 50 | 71 | +6 | 6 | -0.1897 | 5 |
| 20 | 41 | +5 | 5 | +0.2716 | 0 |
| 35 | 56 | +5 | 6 | -0.2498 | 6 |
| 46 | 67 | +5 | 6 | -0.3952 | 11 |
| 85 | 106 | +5 | 6 | +0.1262 | 0 |

Case 44 is the clearest low-cost target: the learned geometry lowers boundary and grouping counts but introduces six MIB violations; eliminating four soft violations would cross the cap. Adding more random topology samples does not directly address this failure. The first question is where shared MIB shape is lost between construction and the final exact result.

## Problem 3: near-cap cases should not share a strategy with structural failures

Twelve cases are within 0.15 log-cost of the cap:

| Case | Blocks | Cap margin | Soft fixes needed | Boundary / group / MIB | Quality gap |
|---:|---:|---:|---:|---:|---:|
| 33 | 54 | -0.0200 | 1 | 15 / 11 / 3 | 2.4345 |
| 38 | 59 | -0.0267 | 1 | 21 / 8 / 0 | 3.4974 |
| 5 | 26 | -0.0321 | 1 | 9 / 4 / 2 | 3.6041 |
| 96 | 117 | -0.0368 | 2 | 5 / 29 / 3 | 4.6466 |
| 39 | 60 | -0.0447 | 1 | 18 / 2 / 3 | 4.2332 |
| 2 | 23 | -0.0581 | 1 | 14 / 3 / 0 | 4.0171 |
| 77 | 98 | -0.0675 | 2 | 14 / 19 / 5 | 3.1002 |
| 66 | 87 | -0.0975 | 3 | 13 / 11 / 3 | 5.4876 |
| 47 | 68 | -0.1042 | 3 | 4 / 14 / 3 | 6.7285 |
| 0 | 21 | -0.1311 | 2 | 9 / 4 / 0 | 5.3622 |
| 44 | 65 | -0.1405 | 4 | 17 / 8 / 6 | 3.9801 |
| 14 | 35 | -0.1481 | 3 | 13 / 3 / 2 | 4.4115 |

Recommended interpretation: these are targeted repair cases. Case 96 is especially valuable because it has 117 blocks and only needs two soft fixes. Cases 33, 38, 5, 39, and 2 each need one exact soft fix. They should not wait for a new global generator.

## Problem 4: cases 65 and 89 are coverage failures with shelf-collapse geometry

Cases 65 and 89 have no exact-eligible learned initial candidates in all three Q6 seeds. Their final geometry is unchanged from the analytic incumbent.

| Case | Blocks | Cap margin | Quality gap | B / G / M | BBox aspect | Utilization |
|---:|---:|---:|---:|---:|---:|---:|
| 65 | 86 | -1.7362 | 17.1846 | 24 / 24 / 0 | 6.37 | 10.6% |
| 89 | 110 | -1.8955 | 19.7570 | 26 / 22 / 0 | 6.40 | 8.5% |

Both PNGs show a long shelf-like envelope rather than a compact topology. Case 89 is classified as quality-required: removing every soft contribution would still not cross the cap, although it also has 48 soft violations. Calling it quality-only would therefore be incorrect.

These cases need candidate coverage and topology adaptation, not ranker tuning. Until at least one exact-eligible non-shelf candidate exists, the selector has nothing useful to choose.

## Problem 5: the highest-weight large cases split into quick wins and structural work

| Case | Blocks | Cap margin | Soft fixes needed | B / G / M | Geometry changed? | Diagnosis |
|---:|---:|---:|---:|---:|---|---|
| 95 | 116 | -0.9091 | 31 | 28 / 23 / 5 | yes | Structural reseed and constraint co-design |
| 96 | 117 | -0.0368 | 2 | 5 / 29 / 3 | yes | Immediate grouping/contact repair target |
| 97 | 118 | -0.6062 | 19 | 21 / 24 / 2 | yes | Structural reseed and tail preservation |
| 98 | 119 | -0.1692 | 5 | 6 / 18 / 3 | yes | Second near-cap large target |
| 99 | 120 | -0.6575 | 23 | 33 / 25 / 0 | no | Candidate acceptance/coverage investigation |

Case 99 looks orderly in the PNG, but many boundary-marked blocks remain in the interior and grouped blocks remain disconnected. Because its geometry is identical to analytic, this is not evidence that Q2 made a bad final move; it is evidence that no challenger passed the incumbent gate.

## Problem 6: cap saturation hides 56 meaningful geometry changes

Eighty-eight cases differ geometrically from analytic, but only 32 cross into a lower capped cost. Fifty-six changed cases remain tied at `9.999999`. Across those 56 cases, the average deltas versus analytic are:

- HPWL gap: `-2.8267`;
- area gap: `-0.8640`;
- boundary violations: `-7.6250`;
- grouping violations: `-1.9821`;
- MIB violations: `+2.4821`;
- total soft violations: `-7.1250`.

These are not no-op cases. Capped cost is simply too coarse for training, ablation, and debugging before cap crossing. Exact uncapped `J`, cap margin, and raw violation counts should remain the diagnostic targets, while official capped cost remains the release target.

## Runtime remains a separate release blocker

Q6 learned runtime is about `2.55 s` p50 and `6.55 s` p95 on RTX 5090, versus `0.236 s` and `0.508 s` for analytic. Local official scoring fixes runtime factor to one, so the current QoR gain does not prove submission gain under cross-submission runtime normalization. This does not change the case diagnosis above, but it prevents default promotion.

## Recommended causal order

```text
preserve compatible MIB shape through the exact tail
  -> targeted one-to-five-fix repair for near-cap cases
  -> restore exact-eligible topology coverage for 65 and 89
  -> structural reseeding for far-cap large cases
  -> only then revisit ranker pruning or collective rollout
  -> profile and budget the promoted path
```

This order attacks observed final-case failures. It does not assume that a larger model, more random samples, or more flow steps will solve them.

## Batch grill: decisions for the next implementation slice

Resolution on 2026-08-11: the user accepted recommendation A for G1--G8.

| Decision | Accepted direction |
|---|---|
| G1 | Develop against exact uncapped `J` and cap margin; promote against official capped cost |
| G2 | Repair MIB preservation before expanding candidate volume |
| G3 | Use cohort-specific case-signature budgets |
| G4 | Target cases 96 and 98 before the other 116--120 cases |
| G5 | Keep collective rollout and active ranker pruning disabled |
| G6 | Permit bounded, exact-verified post-tail soft repair |
| G7 | Cross at least 8/12 near-cap cases including 96 and 98, eliminate aggregate MIB regression, and preserve zero hard/Pareto regressions |
| G8 | Allow slow shadow experiments but block default promotion until runtime gates pass |

The original questions and rationale are retained below as the decision record.

### G1 — Canonical optimization signal

- **A — Use exact uncapped `J` and cap margin for development; keep official capped cost for promotion.**
- B — Optimize official capped cost directly.

Recommendation: A. Fifty-six cap-saturated cases prove that capped cost discards useful gradients and ordering information.

### G2 — First QoR intervention

- **A — Make MIB preservation non-regressive through projection and exact repair before expanding search.**
- B — Add more topology candidates first.
- C — Reactivate collective dynamics first.

Recommendation: A. The measured `+188` MIB regression is a direct causal leak; more candidates can reproduce the same leak.

### G3 — Case-budget policy

- **A — Use cohort-specific budgets: targeted repair for near-cap cases, special coverage recovery for 65/89, structural reseeding for far-cap large cases.**
- B — Keep one uniform candidate budget for all cases.

Recommendation: A. The required action differs fundamentally between case 33 needing one soft fix and case 89 needing a replacement topology.

### G4 — First large-case promotion target

- **A — Uncap cases 96 and 98 before attempting cases 95, 97, and 99.**
- B — Optimize all five 116--120 cases together.
- C — Start with case 99 because it has the largest score weight.

Recommendation: A. Cases 96 and 98 need only two and five soft fixes; they are the highest-probability weighted gains. Case 99 needs 23 fixes and has no accepted geometry change.

### G5 — Collective dynamics and ranker status

- **A — Keep collective rollout and active ranker pruning disabled while candidate/tail defects are repaired.**
- B — Enable them in the next benchmark to increase diversity.

Recommendation: A. Collective transformed rows failed feasibility in Q3, and the broad ranker remains below promotion gates. Neither controls the current Q6 output.

### G6 — Near-cap repair authority

- **A — Permit a bounded post-tail soft-repair pass only when exact hard feasibility and Pareto non-regression are reverified.**
- B — Require all soft constraints to be solved before the exact tail.

Recommendation: A. A bounded exact-verified pass is the shortest route for the one-to-five-fix cohort, provided it cannot bypass the incumbent guard.

### G7 — Next sprint's measurable gate

- **A — Require at least 8 of the 12 near-cap cases to cross, including 96 and 98; MIB delta versus analytic must be at most zero; hard regressions and Pareto regressions remain zero.**
- B — Gate only on weighted average cost.
- C — Gate only on more exact-eligible candidates.

Recommendation: A. It is specific enough to falsify the repair strategy and protects the observed 32 improvements.

### G8 — Runtime policy during QoR repair

- **A — Allow slow shadow experiments, but prohibit default promotion until p95 and cold-start gates pass.**
- B — Reject any QoR experiment that exceeds current analytic runtime.
- C — Ignore runtime until submission freeze.

Recommendation: A. It preserves research throughput without confusing local `RuntimeFactor=1` evidence with submission readiness.

## ADR candidates after the grill

The accepted decisions are recorded in:

1. `docs/adr/0001-use-exact-cap-margin-for-development.md`;
2. `docs/adr/0002-preserve-constraints-before-expanding-search.md`;
3. `docs/adr/0003-route-a-solver-portfolio-by-case-signature.md`.

The domain terms resolved during this diagnosis are recorded in the repository root `CONTEXT.md`.
