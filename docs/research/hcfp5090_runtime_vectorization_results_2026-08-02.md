# HCFP-5090 exact-overlap runtime vectorization results

Date: 2026-08-02

## Decision

**ACCEPT the verifier vectorization. Keep the learned sidecar default-off.**

The change removes the dominant runtime bottleneck with byte-identical
candidate geometry and exact official-quality parity. The learned lane now
passes the p95 runtime gate relative to the analytic baseline, but its median
is still `1.71x` analytic because it always pays model and second-tail overhead
on the many cases where the Pareto guard ultimately retains analytic output.

## Change

Commit `23ba270df2240c3894c64de27b1c6dd58c5f00e5` replaces the CPU FP64 Python
pair loop in `hcfp.verify.overlap_pairs()` with one broadcast intersection
test and a strict upper-triangle extraction.

The following semantics are unchanged:

- exact CPU FP64 input conversion;
- strict `overlap_x > eps and overlap_y > eps` predicate;
- edge-touch and epsilon-boundary legality;
- row-major `(i, j)` result order;
- hard verifier, incumbent, projection, Pareto, and fallback contracts.

## Differential and regression evidence

- deterministic random FP64 differential tests against the former scalar
  definition over multiple block counts and epsilon values;
- exactly-epsilon and one-ULP-below/above threshold tests;
- multiple-pair ordering test;
- full repository test suite passed;
- Ruff, compileall, and `git diff --check` passed.

The before/after case 88 and 97 audit matched exactly for every protected
field:

- selected tensor hash;
- raw and projected candidate tensor hashes;
- hard-feasible and projection-ok flags;
- incumbent snapshot;
- raw official placements.

Artifacts:

```text
artifacts/benchmarks/hcfp5090-overlap-vectorization-baseline.json
artifacts/benchmarks/hcfp5090-overlap-vectorization-after.json
artifacts/benchmarks/hcfp5090-overlap-vectorization-material10.json
```

## Clean-provenance validation 100

```text
artifacts/benchmarks/hcfp5090-overlap-vectorization-validation100.json
SHA256 6196a45f4e1f9938ebb0981212a56910d86628fe219fa046b6c37b11cf3f804f
git commit 23ba270df2240c3894c64de27b1c6dd58c5f00e5
checkpoint state 998026a212d5362fa5d113d07fa929bdf953d1622ae2b33f88119f0a4fc9a3af
```

Results:

- analytic hard feasibility: `100/100`;
- learned hard feasibility: `100/100`;
- analytic and learned official capped weighted cost: `9.999999`;
- every analytic and learned position/quality field matches the prior clean
  Pareto-guard report;
- all 11 learned improvements remain present: cases
  `4, 40, 41, 49, 54, 57, 68, 77, 89, 91, 93`;
- no new `cost=10` case and no quality regression relative to that report.

## Runtime result

| Lane | Metric | Before | After | Speedup |
| --- | --- | ---: | ---: | ---: |
| analytic | p50 | 0.604 s | 0.215 s | 2.81x |
| analytic | p95 | 1.384 s | 0.475 s | 2.91x |
| learned | p50 | 1.568 s | 0.367 s | 4.27x |
| learned | p95 | 3.630 s | 0.505 s | 7.20x |
| learned | max | 4.377 s | 0.697 s | 6.28x |

The new learned/analytic ratios are:

- p50: `1.707x`;
- p95: `1.062x`.

Thus the learned p95 is within the `analytic x 1.10` gate, but the median gate
is not yet met.

## Visualization parity

The regenerated HTML files are byte-identical to the previous clean report:

```text
case_0.html  07459bfdc210b351d02312b0fade6036374e9a9887d6da09e4e93337f80f2f6d
case_50.html 665f23c60d6008e9c03a3460680519e1e9ed3f0dff507e161944123fc00ae6f5
case_88.html fa484e893c18cba0adc90cbfbd22065366cb457e7a6f434f9bc9f17e41efab29
case_97.html 5655bbff5a9478ca5f10553386b55bd0d393ffa5206523a8c931968f2546c481
case_99.html 7d9c6f988b229217c54845dc8b62642e67ccd299c8d178dbb17ff9941fdd3c6a
```

## Next phase

Profile a conservative learned-activation policy. It must skip the learned
tail on clear analytic-retention cases while preserving all 11 current gains,
100/100 feasibility, and the large-case subset. Do not reduce beam,
projection iterations, population, or exact verification coverage merely to
hit the median gate.
