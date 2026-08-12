# HCFP-5090 P6 family contribution and full100 result

## Hypothesis

Four additive B*-Tree seeds provide unique QoR wins beyond the frozen P2.5 portfolio without sacrificing hard feasibility, and their gain is not limited to the visible large15 subset.

## Experiment

The exact scorer compared the same checkpoint and portfolio with `btree_seeds=0/2/3/4`. All configurations retained the analytic incumbent, 16 topology seeds, 16 constraint seeds, one treemap seed and the candidate-funnel proxy.

### Seed-count ablation on large15

| B*-Tree seeds | Weighted cost | Runtime p50 | Runtime p95 | Unique improvements vs 0 seeds |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 7.332108 | 7.272 s | 10.492 s | - |
| 2 | 7.324685 | 7.934 s | 11.602 s | 1 |
| 3 | 7.128942 | 8.054 s | 11.555 s | 2 |
| 4 | **7.095689** | 8.281 s | 12.184 s | **3** |

The fourth seed adds the case 92 win; three seeds already add the case 85 and 98 wins. No configuration regressed a case because the family is additive and selector-gated.

### Full100 marginal contribution

| Portfolio | Weighted cost | Capped feasible | Hard feasible | Runtime p50 | Runtime p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| P2.5, no B*-Tree | 7.142119 | 3 | 100/100 | 3.102 s | 9.169 s |
| P2.5 + 4 B*-Tree | **6.956479** | **2** | **100/100** | 3.611 s | 10.483 s |

B*-Tree produced 23 improvements, 77 ties and zero regressions. Unique wins occurred across all block-count ranges:

| Block count | Cases | B*-Tree wins | Mean cost before | Mean cost after |
| --- | ---: | ---: | ---: | ---: |
| 21-50 | 30 | 8 | 6.205707 | 5.679088 |
| 51-80 | 30 | 9 | 6.300710 | 5.950090 |
| 81-105 | 25 | 3 | 6.730991 | 6.671466 |
| 106-120 | 15 | 3 | 7.354227 | 7.168886 |

Artifacts:

- `artifacts/benchmarks/hcfp5090-p6-btree0-full100.json`
- `artifacts/benchmarks/hcfp5090-p6-btree2-large15.json`
- `artifacts/benchmarks/hcfp5090-p6-btree3-large15.json`
- `artifacts/benchmarks/hcfp5090-p34-btree-runtime-large15.json`
- `artifacts/benchmarks/hcfp5090-p6-btree4-full100.json`

## Decision

**KEEP four B*-Tree seeds.** The approximately 0.19 weighted full100 gain and zero-regression behavior justify the measured runtime increase. The ranker stays shadow-only; mask/TTO remains deferred because the remaining capped cases are not shown to be low-dimensional near misses.

Two fresh-process case 98 runs returned bit-identical placements and cost `8.017462`; measured cold-start runtimes were `12.266 s` and `12.346 s`. This verifies deterministic local RTX 5090 startup behavior. A100 profiling and frozen submission packaging remain environment/release tasks rather than claims of this experiment.
