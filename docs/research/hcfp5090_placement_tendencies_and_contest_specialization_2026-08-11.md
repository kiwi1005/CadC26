# HCFP-5090 placement tendencies and contest-aware specialization

Date: 2026-08-11

Evidence: Q6 official-visible 100, canonical seed 7001

Decision: specialize by transferable case signature, not visible case identity

## Evidence boundary

This report analyzes the latest preserved Q6 official-visible benchmark rather than rerunning the current `HEAD`. Its generated feature report is:

```text
artifacts/reports/hcfp5090-q6-placement-tendencies-official100-seed7001.json
```

The official contest README says that the hidden 100 has the same format and 21--120 block range as validation and is used for final ranking. It does not promise that a hidden case with a given index or block count is the same topology as its visible counterpart. The visible set happens to satisfy `test_id = block_count - 21` for all 100 cases; consequently, a parameter table keyed by exact block count would have only one visible observation per entry and would behave like case memorization.

## Tendency 1: the learned lane is an aggressive shape-mode switch

| Cohort | Cases | Q5 movable-square fraction | Analytic movable-square fraction | Q5 aspect entropy |
|---|---:|---:|---:|---:|
| Improved and exact uncapped | 32 | 0.16% | 99.80% | 3.52 |
| Geometry changed but cap-saturated | 56 | 0.05% | 99.85% | 3.79 |
| Unchanged analytic incumbent | 12 | 100% | 100% | 0.00 |

The accepted learned placements do not make small local shape changes. They move almost every movable block away from the analytic square shape and create a high-diversity aspect-ratio population. This helps HPWL and area, but it also explains why MIB is fragile: per-block shape diversity conflicts with group-shared dimensions.

Recommended specialization:

- retain an explicit square-preserving structured seed family;
- add a moderate, contact-oriented aspect family rather than only analytic squares and high-entropy per-block shapes;
- broadcast one shape variable through compatible MIB groups in every family;
- condition aggressive aspect diversity on terminal density and group/MIB membership.

## Tendency 2: grouping completion is the strongest success signature

| Final group satisfaction | Cases | Exact uncapped | Uncap rate |
|---|---:|---:|---:|
| 0% | 56 | 7 | 12.5% |
| greater than 0%, at most 25% | 8 | 2 | 25.0% |
| greater than 25%, at most 50% | 7 | 2 | 28.6% |
| greater than 50%, at most 75% | 13 | 8 | 61.5% |
| greater than 75% | 16 | 13 | 81.3% |

Mean group satisfaction is 57.8% for the 32 improved cases, 14.6% for the 56 cap-saturated changed cases, and 0% for the 12 unchanged cases. The 116--120 bucket also has 0% mean group satisfaction.

Boundary satisfaction does not separate success nearly as strongly: improved and cap-saturated changed cases average 39.3% and 40.4%. Boundary remains a large absolute penalty, but group contact completion is the clearer routing and promotion signal.

Recommended specialization:

- use group-contact completion as an early-stop and candidate-pruning signal;
- allocate strict contact-tree replay to large cases whose analytic or first-pass group satisfaction is low;
- preserve latched group components as rigid or semi-rigid units through later repair;
- treat cases 96 and 98 as the first large group-repair probes.

## Tendency 3: near-square envelopes win; elongated envelopes fail

| BBox extreme aspect ratio | Cases | Exact uncapped | Uncap rate |
|---|---:|---:|---:|
| at most 1.50 | 72 | 29 | 40.3% |
| greater than 1.50, at most 2.00 | 24 | 3 | 12.5% |
| greater than 2.00 | 4 | 0 | 0% |

Cases 65 and 89 have aspect ratios above 6.3 and utilization below 10.6%; both are exact-candidate coverage failures. The tendency supports multiple outline hypotheses, but not a universal square outline: pins and preplaced anchors can require directional envelopes.

Recommended specialization:

- detect shelf collapse from incumbent geometry using `aspect > 5` and `utilization < 0.12` as an initial visible-calibrated signature;
- for that signature, bypass shelf-centered residual candidates and force alternate sequence-pair orientations;
- otherwise prefer outline hypotheses near aspect 1.0--1.5, with anchor-span-aware elongated alternatives retained as a portfolio member.

The numeric thresholds are calibration candidates, not new hard invariants. They must be tested on internal unseen training cases before runtime promotion.

## Tendency 4: terminal-dense cases are much harder

| Pin-to-block edges per block | Cases | Exact uncapped | Uncap rate |
|---|---:|---:|---:|
| at most 2 | 16 | 10 | 62.5% |
| greater than 2, at most 10 | 53 | 15 | 28.3% |
| greater than 10, at most 20 | 19 | 5 | 26.3% |
| greater than 20 | 12 | 2 | 16.7% |

The terminal-dense cases above 20 edges per block include 82, 86, 89, 94, 95, 97, and 99 among the high-weight failures. Their mean cap margin is `-0.526`, compared with `-0.009` for the lowest-density band.

Recommended specialization:

- predict or construct the envelope from pin/preplaced anchor span before compaction;
- reduce independent per-block aspect exploration in terminal-dense cases;
- allocate topology diversity to cluster-to-anchor ordering rather than extra random geometry noise;
- report terminal-attraction work separately because it can dominate large-case runtime.

## Tendency 5: performance is bucketed, but exact block-count tuning would memorize visible cases

| Blocks | Cases | Exact uncapped | Uncap rate | Mean cap margin |
|---|---:|---:|---:|---:|
| 21--64 | 44 | 19 | 43.2% | -0.092 |
| 65--95 | 31 | 5 | 16.1% | -0.328 |
| 96--105 | 10 | 5 | 50.0% | -0.042 |
| 106--115 | 10 | 3 | 30.0% | -0.443 |
| 116--120 | 5 | 0 | 0% | -0.476 |

The official score assigns approximately 34% weight to 116--120, 22% to 111--115, and 15% to 106--110. A large-case portfolio is therefore justified, but it must use coarse buckets plus topology, constraint, terminal, and incumbent features. A 100-entry per-block-count table is a visible-case lookup in disguise.

## Appropriate contest overfitting

The recommended policy is asymmetric:

| Specialization level | Decision | Reason |
|---|---|---|
| Official scorer and weight formula | Strongly specialize | This is the true objective |
| FloorSet constraint and generator statistics | Strongly specialize | Hidden uses the same format and range |
| Case-signature thresholds and solver portfolio | Specialize with internal checks | Transfers to unseen instances |
| Visible100 threshold calibration | Allow | Useful contest feedback, but validate direction on internal data |
| Exact block-count or case-ID lookup | Reject | One visible case per block count; hidden identity differs |
| Stored visible geometry or input-hash lookup | Reject | Does not target hidden ranking and destroys generalization |
| Hard verifier or Pareto safety weakening | Reject | A single hard failure is disproportionately costly |

The exact verifier and Pareto guard make aggressive portfolio experiments safer for QoR: a bad challenger can be rejected. They do not make over-specialization free, because unnecessary candidates still increase runtime and can worsen the official runtime factor.

## Proposed signature router

```text
MIB present and compatible
  -> group-shared shape in every candidate family

initial required_soft_fixes_to_uncap <= 5
  -> bounded targeted soft repair

incumbent aspect > 5 and utilization < 0.12
  -> shelf-collapse recovery portfolio

p2b_edges / N > 20
  -> terminal-dense topology and anchor-envelope portfolio

N >= 106 and group satisfaction <= 5%
  -> strict contact-tree replay with large-case budget

otherwise
  -> standard structured topology and constraint portfolio
```

The router must use measurements available for the unseen case, preferably from the analytic incumbent and exact pre-tail telemetry. It must not use a visible test identifier.

## Experiment order after accepted G1--G8

1. Preserve MIB shared shape and prove aggregate MIB delta versus analytic is at most zero.
2. Add a low-fix repair portfolio and cross at least 8/12 near-cap cases, including 96 and 98.
3. Add square-preserving, moderate-aspect, and aggressive-aspect candidate families; measure oracle and selected contribution by signature.
4. Recover exact-eligible non-shelf candidates for 65 and 89.
5. Add terminal-dense and large-group-collapse routing.
6. Calibrate thresholds on visible100, then require the direction of gain to hold on a sample-ID-disjoint internal split.
7. Promote only after hard feasibility, Pareto, p95, cold-start, and wrapper gates pass.

This is deliberate contest specialization: exploit the scorer and generator distribution aggressively, while keeping the part that must generalize—the signature-to-policy mapping—independent of visible case identity.
