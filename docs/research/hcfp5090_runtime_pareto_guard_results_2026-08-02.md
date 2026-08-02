# HCFP-5090 runtime Pareto guard results — 2026-08-02

## Decision

The split-tail analytic guard passes the quality and hard-feasibility gates but
the learned lane remains **HOLD/default-off**. It improves the deterministic
weighted uncapped objective by 7.50% overall and 6.25% on 106--120 block cases
with no per-case regression, but every visible case still saturates the
official `9.999999` cap and learned runtime p95 is 2.62x the analytic lane.

## Implemented contract

- Run the standalone analytic tail unchanged.
- Run ranker-pruned learned candidates in a separate exact-tail batch.
- Merge both batches back into the stable source layout:

```text
fallback
analytic initial
learned initial
analytic relaxed
learned relaxed
```

- Preserve standalone analytic exact and fast source indices in the merged
  snapshot.
- After raw hard-target replay, replace the selected placement only with a
  raw-feasible analytic candidate that is no worse in soft violation, bounding
  box area, and HPWL, and materially better in at least one dimension.
- If the standalone analytic result is the merged winner, return its original
  safe incumbent rather than a projected copy of fallback index zero.

No official evaluator, baseline lookup, dependency, submission entrypoint, or
normalized hard tolerance was added to the runtime.

## Clean-provenance validation 100

Artifact:

```text
artifacts/benchmarks/hcfp5090-pareto-guard-final-validation100.json
SHA256 69fbb79df7a774c69c4caa9728d27f367aba518d6972d12335731a356540f4f7
implementation commit c5c1925c3b1883ab0a2621421214dc44a757bcf6
checkpoint state 998026a212d5362fa5d113d07fa929bdf953d1622ae2b33f88119f0a4fc9a3af
evaluator commit aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa
evaluator SHA256 64db37865b42baf11add62bdbf035690dca086cd4be7b5b4e58db756f20d8498
flow seed 0
execution seed 0
```

| Metric | Analytic | Learned + split-tail guard |
| --- | ---: | ---: |
| Hard-feasible | 100/100 | 100/100 |
| 106--120 hard-feasible | 15/15 | 15/15 |
| Official weighted capped cost | 9.999999 | 9.999999 |
| Weighted uncapped objective | 34.777064 | 32.167838 |
| 106--120 weighted uncapped objective | 35.234287 | 33.031899 |
| Runtime p50 | 0.604138 s | 1.567554 s |
| Runtime p95 | 1.384374 s | 3.630491 s |
| Runtime max | 1.886557 s | 4.377393 s |

Eleven cases improve, 89 tie the analytic lane, and none regress:

```text
4, 40, 41, 49, 54, 57, 68, 77, 89, 91, 93
```

The improved large cases are 89, 91, and 93 with 110, 112, and 114 blocks.
Cases 88 and 97 now exactly match their analytic placements and metrics.

## Determinism and visualization

An independent two-case process exactly reproduced positions, feasibility,
HPWL gap, area gap, soft violation, and capped cost for cases 88 and 97:

```text
artifacts/benchmarks/hcfp5090-pareto-guard-final-determinism-88-97.json
SHA256 970337039b454e7815691796eabb8fd16429c22c115b6ac60ae5a55302c659d1
```

Visualization hashes:

```text
case_0.html  07459bfdc210b351d02312b0fade6036374e9a9887d6da09e4e93337f80f2f6d
case_50.html 665f23c60d6008e9c03a3460680519e1e9ed3f0dff507e161944123fc00ae6f5
case_88.html fa484e893c18cba0adc90cbfbd22065366cb457e7a6f434f9bc9f17e41efab29
case_97.html 5655bbff5a9478ca5f10553386b55bd0d393ffa5206523a8c931968f2546c481
case_99.html 7d9c6f988b229217c54845dc8b62642e67ccd299c8d178dbb17ff9941fdd3c6a
```

## Verification

```text
pytest: 157 passed
Ruff: passed
compileall src/hcfp submission: passed
git diff --check: passed
```

Luna MAX review caught and closed source-index parsing, unreachable fast-tier,
fallback-index, replay-metric, and full-pool GPU-to-CPU synchronization defects
before the implementation commit.

## Remaining gates

1. Runtime is still the blocking promotion gate: learned p50/p95 are 2.59x and
   2.62x analytic.
2. The official cap hides all uncapped QoR gains on visible validation, so the
   learned lane cannot be promoted on weighted capped score evidence.
3. Case 95 still exposes a small selector opportunity: the exact candidate
   oracle is `22.211647` while the incumbent is `22.626706`. Evidence:
   `artifacts/benchmarks/hcfp5090-final-attribution-95.json`, SHA256
   `7d5e5b6798dfadb3ebeb99526941c717f1df2df6e7fe2245d51e4aabb5eac564`.
4. Keep the contest default on the analytic lane. The next phase is profiler-led
   removal of duplicate tail overhead, followed by the same material and
   full-100 gates.
