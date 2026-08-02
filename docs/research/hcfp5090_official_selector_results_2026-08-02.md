# HCFP-5090 official selector results — 2026-08-02

## Decision

The official-baseline data path and objective-aligned ranker training loop are
implemented and verified. The new ranker improves held-out replay regret, but
the learned lane remains **HOLD/default-off** because deterministic full replay
regresses the weighted uncapped objective, runtime p95 is still more than twice
the analytic lane, and cases 88 and 97 have material QoR regressions.

## Data and replay contract

- `SolutionLabels` preserves raw-unit `baseline_area` and `baseline_hpwl`.
- FloorSet-Lite reads `metrics_sol[0]` and `metrics_sol[6] + metrics_sol[7]`.
- Legacy shard payloads derive compatible values from ground-truth geometry;
  new shards preserve official stored metrics exactly.
- D4 augmentation keeps both raw baselines invariant.
- Replay schema v2 marks `target_kind=official_v10_lexicographic_v1`.
- Legacy proxy replay remains readable but ranker training rejects it.

The first direct capped-cost trial produced only two FP32 values:

```text
hard infeasible: 241 / 256 -> 10.0
capped feasible: 15 / 256 -> 9.999999
all-tied records: 21 / 32
```

The retained lexicographic label is rank-equivalent to the uncapped official
objective for feasible candidates. Official-tied infeasible candidates are
ordered by post-BDP repair residual. The final train32 replay has 256 unique
targets and 32/32 records contain ranking signal.

## Deterministic selector A/B

Ranker-only checkpoint updates previously changed the flow noise because the
entire checkpoint hash was used as seed. `LearnedConfig.seed` now permits an
explicit A/B seed; replay and benchmark tools use `0`, so ranker checkpoints
share the same candidate pool. An ordinary contest learned invocation without
`HCFP_FLOW_SEED` retains the prior checkpoint-hash behavior. The replay CLI
records both dataset seed and flow seed.

Artifacts:

```text
train32 replay
  artifacts/replay/hcfp5090-official-v10-r32.jsonl
  SHA256 6349a5a7efbe219c28c53a28fd14772dd334f967d4ac5e89a81303500b2036a1
heldout16 replay
  artifacts/replay/hcfp5090-official-v10-heldout-r16.jsonl
  SHA256 c03790ac51020921a3d7cda2d063ef0395da5f00b55551a0c66f486ac51498b9
ranker checkpoint
  artifacts/checkpoints/hcfp5090-ranked-official-r32-s2000-v1.pt
  SHA256 7d00e83e736d0e1ed150dd1808a7464beb885137d6a9c09c3c7aeb2668079b21
  state hash 998026a212d5362fa5d113d07fa929bdf953d1622ae2b33f88119f0a4fc9a3af
regret report
  artifacts/benchmarks/hcfp5090-ranker-regret-official-v1.json
  SHA256 9474b0dfc6a974ea21f48566f6a46e250205b44192fb51c85eb810f978f76c12
```

| Split | Ranker | Top-1 exact | Mean regret | p95 regret |
| --- | --- | ---: | ---: | ---: |
| train32 | legacy proxy | 12/32 | 0.408253 | 1.760118 |
| train32 | official s2000 | 17/32 | 0.328732 | 1.729140 |
| heldout16 | legacy proxy | 5/16 | 0.266364 | 0.618441 |
| heldout16 | official s2000 | 8/16 | 0.090488 | 0.408290 |

Held-out mean regret improves by 66.0%. This is a training-only split drawn
from FloorSet-Lite with seed `20260802`; official validation was not used to fit
the checkpoint or select training steps.

## Official validation 100

Full report:

```text
artifacts/benchmarks/hcfp5090-ranked-official-top4-validation100.json
SHA256 0123eb81fc0cead101d01e57a820b3ea451b0dccc369498d7c5a79b7d4472c9c
```

| Metric | Analytic | Learned + official ranker top-4 |
| --- | ---: | ---: |
| Hard-feasible | 100/100 | 100/100 |
| 106–120 hard-feasible | 15/15 | 15/15 |
| Official weighted cost | 9.999999 | 9.999999 |
| Weighted uncapped objective | 34.777064 | 37.063082 |
| 106–120 weighted uncapped objective | 35.234287 | 40.870537 |
| Runtime p50 | 0.590918 s | 1.379398 s |
| Runtime p95 | 1.375935 s | 3.105029 s |
| Runtime max | 1.884994 s | 3.786288 s |

The learned lane preserves hard feasibility but regresses the weighted uncapped
objective by 6.57% overall and 15.99% on the 106–120 bucket. Runtime p95 is
2.26x analytic. Eight material cases improve, but cases 88 and 97 regress; the
large case 97 dominates the weighted result. Case-level Pareto/incumbent
selection must be repaired before default use.

The final runner enables CUDA deterministic algorithms with execution seed
`0`. Two independent 48/51/88 processes produced identical HPWL, area, soft,
and cost tuples; isolated case 97 also matches the full-run tuple.

Visualizations:

```text
case_0.html  07459bfdc210b351d02312b0fade6036374e9a9887d6da09e4e93337f80f2f6d
case_50.html 665f23c60d6008e9c03a3460680519e1e9ed3f0dff507e161944123fc00ae6f5
case_99.html 7d9c6f988b229217c54845dc8b62642e67ccd299c8d178dbb17ff9941fdd3c6a
```

## Next gate

1. Preserve a standalone analytic incumbent in the learned comparison lane.
2. Add a baseline-free Pareto guard so learned output cannot worsen HPWL, area,
   and soft violation together.
3. Profile candidate-count/tail steps on the 10 materially changed cases.
4. Re-run official 100 only after cases 88 and 97 are non-regressing.

These gates are complete. The split-tail implementation, clean-provenance
validation 100, runtime measurements, determinism evidence, and continued HOLD
decision are recorded in
[`hcfp5090_runtime_pareto_guard_results_2026-08-02.md`](hcfp5090_runtime_pareto_guard_results_2026-08-02.md).
