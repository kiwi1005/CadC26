# HCFP-5090 exact-overlap runtime vectorization phase

Date: 2026-08-02

## Outcome

Remove the dominant CPU verifier bottleneck without changing candidate
generation, projection, exact feasibility, incumbent selection, Pareto guard,
or official-output geometry.

## Fresh profiler evidence

The clean Pareto-guard validation establishes the current runtime gate:

- analytic p50/p95: `0.604 / 1.384 s`;
- learned p50/p95: `1.568 / 3.630 s`;
- learned/analytic p50 and p95 ratios: `2.595x / 2.623x`.

A synchronized stage timer on the current branch and checkpoint shows that
checkpoint/model setup is not material:

| Case | Blocks | Checkpoint load | Learned population | Analytic tail | Learned tail |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 21 | 0.007 s | 0.009 s | 0.260 s | 0.218 s |
| 50 | 71 | 0.006 s | 0.035 s | 1.000 s | 0.604 s |
| 99 | 120 | 0.006 s | 0.085 s | 2.431 s | 1.172 s |

The case-99 tail breakdown attributes only about `0.022 s` to relaxation and
`0.13 s` to projection. The CPU incumbent scan plus telemetry consumes about
`2.03 s` in the analytic tail and `0.87 s` in the learned tail. Both paths
repeatedly call the exact verifier. Luna MAX isolated `overlap_pairs()` as the
dominant verifier cost: it currently executes an `O(N^2)` Python loop over
CPU FP64 rectangles.

## Minimal change

Replace only `hcfp.verify.overlap_pairs()` with a CPU FP64 tensor expression:

1. compute all pairwise x/y intersection lengths with broadcast min/max;
2. retain the exact strict predicate `overlap_x > eps and overlap_y > eps`;
3. select only the strict upper triangle;
4. return pairs in the same row-major `(i, j)` order and tuple type.

No tolerance, dtype, preplaced exemption, verifier contract, projection
configuration, candidate count, or selection key may change.

## Verification gates

- Unit boundary cases: edge touch, exactly epsilon, and one ULP below/above
  the threshold.
- Differential test: compare vectorized output with the former scalar
  definition over deterministic random FP64 rectangles and multiple epsilon
  values.
- Ordering test: multiple overlaps retain scalar row-major pair order.
- Existing preplaced and official-parity tests remain green.
- Full repository tests, Ruff, compileall, and `git diff --check` pass.
- Determinism replay on cases 88/97 keeps selected placements, metrics,
  candidate tensors, incumbent snapshots, and feasibility flags unchanged.
- Material runtime A/B demonstrates a reduction before running validation 100.
- Official validation 100 retains 100/100 feasibility, the same 11 Pareto-safe
  learned improvements, zero regressions, and no new `cost=10` case.

## Stop conditions

- Reject the change if any differential pair set or pair ordering differs.
- Reject it if selected raw placements or official metrics differ on 88/97.
- Do not combine analytic and learned projection batches.
- Do not reduce beam, iterations, population, or exact candidate coverage.
- Do not add checkpoint caching, custom CUDA, `torch.compile`, or dependencies
  in this phase.
