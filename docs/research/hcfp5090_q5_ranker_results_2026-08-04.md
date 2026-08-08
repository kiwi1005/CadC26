# HCFP-5090 Q5 repair-aware ranker results

Date: 2026-08-04  
Branch: `feat/hcfp5090-qor-first`  
Implementation commit: `2cb1ddb`  
Decision: **historical dev16 pilot PASS on 2/3 seeds; broad promotion HOLD**

## 2026-08-08 broad-validation update

The 16-case result below remains a valid historical pilot, but it is no longer
sufficient promotion evidence. A checksum-validated, sample-ID-disjoint
512-case development replay now shows that the ranker does not generalize at
the required rate.

| Checkpoint | Initial top-1 | Initial top-4 | False promotion | Decision |
|---|---:|---:|---:|---|
| v4 seed 5104 | 44.1% | 70.7% | 4 | hold |
| v4 seed 5105 | 43.4% | 69.9% | 2 | hold |
| v5 seed 5104 | 49.2% | 72.1% | 2 | best v5, hold |
| v5 seed 5105 | 47.3% | 71.9% | 2 | hold |
| v5 seed 5106 | 48.6% | 71.9% | 1 | hold |

The v5 replay scale improves top-1 by about five points and top-4 by one to two
points, but remains far below the 75% / 93.75% gates. The exact-eligible
post-tail shadow remains output-neutral and safe; pre-tail pruning remains
disabled because the unfiltered false-promotion evidence is real.

The 5,000-record training manifest is
`artifacts/replay/hcfp5090-q5-dagger5000-manifest.json`. The broad development
manifest is `artifacts/replay/hcfp5090-q5-dev512-manifest.json`; it contains 512
unique samples, 1,024 paired records, eight checksum-valid shards, and zero
sample overlap with the training and earlier replay sets. Full evidence is in
[`hcfp5090_q6_release_results_2026-08-08.md`](hcfp5090_q6_release_results_2026-08-08.md).

## Outcome

Q5 now has an end-to-end, fail-closed training and diagnosis path:

```text
paired learned candidates
  -> pre-tail repair-aware features
  -> post-repair exact list targets
  -> candidate-list training
  -> held-out top-1/top-4/false-promotion gates
  -> output-neutral runtime shadow
  -> selected-versus-oracle visual replay
```

The important result is not a larger model. It is that the ranker now observes
the same initial raw/post-BDP slice that runtime can reproduce, while labels
come from the exact post-repair order. Two independent initialization seeds
meet all three initial-stage gates on the disjoint 16-sample pilot:

- exact top-1 at least 12/16;
- top-4 oracle recall at least 15/16;
- false promotion exactly zero.

The third seed misses all three thresholds. All three post-relax evaluations
miss the top-1 and top-4 gates. The ranker therefore remains shadow-only and
cannot alter candidate selection, exact source, fallback, or the Pareto guard.

This is checkpoint-level evidence, not a release result. The replay target is
still below 5,000 records, the dev set contains only 16 sample-ID-disjoint
cases, and 100-case three-seed validation plus A100 cold-start profiling have
not run.

## Implemented contract

### Repair-aware feature v4

`repair_aware_ranker_features_v4_device_parity` contains 26 candidate-local
values derived only from information available before exact repair:

- the existing eight raw candidate metrics;
- the same eight metrics after BDP;
- mean and maximum center movement through BDP;
- explicit `learned`/`constraint`/`topology` source one-hot values;
- boundary, grouping-connectivity, and MIB-shape pre-tail proxies;
- initial/post-relax stage;
- projection success reconstructed from geometry.

Post-repair feasibility, cap margin, target rank, teacher displacement, and
post-repair geometry are not read by feature construction. A regression test
mutates post-repair targets and proves the features remain byte-identical.
CPU and CUDA grouping-proxy paths also match exactly on the device-parity test.

The feature dimension, version, normalization, and scene-embedding policy are
checkpointed. Runtime and evaluation reject incompatible contracts rather than
silently padding or reinterpreting features.

### Training path

The promoted pilot recipe uses 256 initial-stage candidate lists:

```text
128 initial lists from paired train128
+
128 initial-only lists from a second disjoint replay
=
256 listwise records
```

The two training replays and the 16-sample dev replay have pairwise-disjoint
sample IDs. Features use training-only global z-score normalization. Constant
columns retain identity scale instead of producing a near-zero divisor.

The ranker does not use the pooled scene embedding in this pilot. That removes
an encoder forward pass from every ranker step and makes the learned function
depend only on candidate evidence. After the 26-D features are prepared, the
training script releases raw, post-BDP, and post-repair geometry tensors from
the in-memory record representation. This reduces training memory without
changing targets.

The loss is:

```text
ListMLE
+ 0.25 * feasibility ordering
+ 0.05 * standardized pointwise J regularization
```

Continuation is allowed only when feature version, dimension, normalization,
and scene-embedding policy exactly match. Changing any part of a trained ranker
contract fails closed.

### Runtime shadow

The runtime shadow activates only when checkpoint metadata says the ranker was
trained and the 26-D feature contract matches. It ranks only candidates that
are both hard-feasible and projection-valid. Its evidence is written under
`ranker_shadow_*` fields in the incumbent snapshot.

The shadow uses the exact merged learned **initial** slice represented by replay:

```text
raw_candidates[1 + analytic_count : 1 + analytic_count + learned_count]
projected_candidates[same slice]
```

It does not use the later post-relax half, does not change `selected`, and does
not change `exact_source`. Incompatible checkpoints, non-finite features, and
runtime exceptions are recorded as a skipped/failure reason while preserving
the incumbent.

A real FloorSet CUDA smoke on `worker_11/layouts_9744.th:17` produced identical
selected geometry before and after the Q5 checkpoint:

```text
selected SHA-256: a659315fac411c3a972effd7dfb468da7c3d5266cdcfd90afc7d9810289fda3a
exact source:      candidate_33
shadow stage:      initial
eligible rows:     17
shadow top-4:      candidate_35, candidate_36, candidate_33, candidate_37
shadow failure:    null
```

This proves output neutrality for the smoke case; it does not prove that active
ranker selection is safe.

## Replay evidence

| Replay | Samples | Records used here | SHA-256 |
|---|---:|---:|---|
| `hcfp5090-q5-paired-train128-disjoint.jsonl` | 128 | 128 initial | `15ecea58a3c2e52de0f41661e164d95a132431d1abebadd50178276ac6951752` |
| `hcfp5090-q5-initial-train128b-disjoint.jsonl` | 128 | 128 initial | `d9d3bd72a35ff5d308ea4c1db7f11c38cb86eeb010d8910363ef66000a0c38c9` |
| `hcfp5090-q5-paired-dev16.jsonl` | 16 | 16 initial + 16 post-relax | `599678f9f708284880aac11731ebca330059445547f587273c8884c2b51d73a1` |

The dev replay is schema v3, uses checkpoint state
`b3a92c183f3a4f840955f62b8f61cd4e0573ae07b6e2697201280f52deff17d0`,
and contains 1,216 distinct post-repair candidate targets across its 32 lists.
Sample-ID exclusion is verified, but this small pilot is not a replacement for
the broader internal validation-like split required by Q6.

## Held-out result

All rows below use the same 16-sample dev replay and the exact stage-specific
promotion policy.

| Seed | Stage | Top-1 | Top-4 recall | False promotion | Weighted rank regret | Weighted score regret | Gate |
|---:|---|---:|---:|---:|---:|---:|---|
| 5102 | initial | 11/16 | 14/16 | 1 | 0.5000 | 0.004542 | fail |
| 5104 | initial | 12/16 | 15/16 | 0 | 0.4375 | 0.004542 | pass |
| 5105 | initial | 12/16 | 15/16 | 0 | 0.4375 | 0.004542 | pass |
| 5102 | post-relax | 6/16 | 9/16 | 0 | 2.8125 | 0.091828 | fail |
| 5104 | post-relax | 6/16 | 10/16 | 0 | 2.0625 | 0.056677 | fail |
| 5105 | post-relax | 3/16 | 6/16 | 0 | 6.3750 | 0.157416 | fail |

The previous 1,000-step listwise baseline achieved only 5/16 initial top-1,
13/16 top-4, two false promotions, weighted rank regret 7.0625, and weighted
score regret 0.045904. Relative to that baseline, seed 5104 reduces initial
weighted rank regret by 93.8% and weighted score regret by 90.1%.

The matching initial score regret for seeds 5102/5104/5105 is not sufficient to
declare seed robustness: seed 5102 still promotes one infeasible/worse-tier row
and misses the discrete gate. The correct conclusion is 2/3 checkpoint passes,
not unanimous promotion.

## Checkpoints and reports

| Seed | Checkpoint SHA-256 | State hash | Training loss window |
|---:|---|---|---:|
| 5102 | `71a3eb43e7790a6d91d594afb1e23bdcd5bbcc229326c59be3f7255a536949f7` | `0a752df0b2b79bfeae57c19ce512f8e020784070d8621cc1b8b9cef32965bcae` | 2.354789 -> 1.923401 |
| 5104 | `ac1986a02c69c3068b696b132458d871777076367be76c37c20968f33771d78b` | `47fa80a14efad2403b5ed4afb63fc34035e7dd17da1c3a41f2b12d458ece4c49` | 2.354545 -> 1.920714 |
| 5105 | `12fedb3f47c4e0321b4b5e04eea526c4ef54a267884fd327d2a6e143f768d1cc` | `af396a60850003c8c263b405b9bc5f9f7d86c518f10e5109da17539e2775f850` | 2.334945 -> 1.921586 |

Evaluation report SHA-256 values:

- seed 5102: `d6bd5803eea94ad6912ac7e1cbe6082663fc4c34ab9151b8da3bf70d7ccec32a`;
- seed 5104: `19413dcc14b98fd198df2d408cb8d1ec900f689067ad6d118567e0c50fd2bf69`;
- seed 5105: `80c34326d3daee1f7da38c72e530670798bb0996a6a891f9c4ea77513a4e4dba`.

## Visual diagnosis

The visualizer validates replay/evaluation row identity before rendering and
refuses to overwrite an existing bundle unless `--force` is explicit. For each
case it draws selected and oracle geometry at raw, post-BDP, and post-repair
stages, for six SVG panels total.

Seed 5104's bundle is:

```text
artifacts/visualizations/hcfp5090-q5-ranker-v4-dev16-seed5104/index.html
index SHA-256:    8a93dcfc5b360c7311b9db8749b45f775a2f31de5499b25c683611ff7c5e1729
manifest SHA-256: d5de3c0de113586fa0fbb71837be993cc197251ddd346ef18a68ace8c1e80260
```

There are four top-1 misses. Three retain the oracle inside top-4. Only
`worker_97/layouts_8400.th:85` misses top-4, with rank regret 2 and score regret
0.008630. Luna-MAX miss analysis tested deterministic boundary/group/MIB proxy
guards, but they reduced held-out recall. No unsupported heuristic guard was
added; the miss remains a training/data problem.

## Verification

Fresh verification for the implementation commit:

```text
full repository pytest: PASS
Q5 targeted pytest:     146 passed
Ruff:                    PASS
compileall:              PASS
git diff --check:        PASS
CPU/CUDA feature parity: PASS
real FloorSet CUDA smoke: output-neutral
```

Luna-MAX independently reviewed target leakage, false-promotion gating,
continuation compatibility, runtime slice alignment, output neutrality, and
visual row identity. The review found no promotion-bypassing path.

## Promotion decision and next work

Q5 is promoted only to an **initial-stage shadow checkpoint**. It is not enabled
for runtime selection. The next gates remain:

1. expand training-source replay toward 5,000 records with the required
   hard-negative/near-cap/large-case/positive composition;
2. build a broader validation-like internal split rather than tuning on the
   current 16 cases;
3. repair or explicitly exclude the post-relax stage until its gate passes;
4. run 100 cases with at least three deterministic seeds and prove zero Pareto
   regressions under an active-selection experiment;
5. profile checkpoint cold load, p50/p95, memory, and fallback on A100;
6. only then consider changing `capabilities.ranker` or pruning exact-tail rows.

Activation remains shadow-only. Q3 post-relax remains a hard-negative source,
not a promoted candidate stage.
