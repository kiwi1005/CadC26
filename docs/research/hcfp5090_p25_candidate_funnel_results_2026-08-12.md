# HCFP-5090 P2.5 Candidate Funnel Results

## Hypothesis

The remaining large-case failures are not all generation failures. A stage-by-stage candidate audit can separate generation, repair-preservation, and selection gaps before adding another topology model.

## Minimal change

- Added `scripts/audit_hcfp_candidate_funnel.py` to record exact raw, post-BDP, post-repair, and selected metrics with candidate provenance.
- Added an experiment-only selector, enabled by `HCFP_CANDIDATE_FUNNEL_PROXY=1`, that compares provenance-bearing candidates against the incumbent after exact constraint/group repair.
- The relative proxy permits a bounded HPWL/area trade for a sufficiently large soft-constraint reduction:

  ```text
  2 * delta_soft
  + 0.5 * log(candidate_area / incumbent_area)
  + 0.5 * log(candidate_hpwl / incumbent_hpwl)
  ```

Hard-infeasible candidates are never promoted and the incumbent remains available.

## Diagnostic cases

| Case | Best available finding | Failure class before selector experiment |
| --- | --- | --- |
| 88 | Post-repair treemap cost 8.624424 existed; selected result remained capped | Selection gap |
| 89 | Best raw/post-repair cost 11.036214 | Generation gap |
| 93 | Post-repair constraint cost 9.414884 existed; selected result remained capped | Selection gap |
| 98 | Treemap cost 9.797765 selected and preserved | Near-cap canary |

## Large15 experiment

Compared with the frozen P2 treemap portfolio:

| Metric | P2 | P2.5 | Delta |
| --- | ---: | ---: | ---: |
| Weighted cost | 7.492749 | 7.332108 | -0.160641 |
| Below cap | 12/15 | 14/15 | +2 |
| Hard feasible | 15/15 | 15/15 | unchanged |
| Improved / tied / regressed | - | 5 / 10 / 0 | zero regressions |

Changed cases:

```text
86: 5.935267 -> 5.488714
88: 9.999999 -> 8.624424
92: 8.588517 -> 8.291613
93: 9.999999 -> 9.414884
99: 5.714720 -> 5.502958
```

Case 98 remained unchanged at 9.797765.

## Decision

**KEEP** the candidate-funnel proxy as an experiment path. P2.5 confirms that cases 88 and 93 were selector failures, while case 89 is the remaining visible pure generation gap. Keep the proxy default-off until broader held-out calibration is done in P5.

Artifacts:

- `artifacts/benchmarks/hcfp5090-p25-candidate-funnel-cases88-89-93-98.json`
- `artifacts/benchmarks/hcfp5090-p25-weighted-proxy-large15.json`

