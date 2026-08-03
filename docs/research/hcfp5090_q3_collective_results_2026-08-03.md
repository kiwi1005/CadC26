# HCFP-5090 Q3 collective dynamics result

Date: 2026-08-03  
Decision: **HOLD — implemented and trainable, but not eligible for runtime promotion**

## Outcome

The geometry-aware collective controller is connected to the active typed-force
path and can be trained independently, but its current two-step rollout destroys
the hard-feasible structure of every post-relax topology candidate in the clean
large-case audit. The unchanged initial half of the learned population preserves
the Q2/Q4 oracle result, so the Pareto-safe final answer does not regress; this is
fallback protection, not evidence that collective relaxation helps QoR.

The next use of Q3 is therefore data generation: its failed post-relax states are
high-value Q5 DAgger negatives paired with exact BDP/raw-repair teacher states.
`collective_steps` remains default-off.

## Evidence boundary

Collective artifact:

- `artifacts/benchmarks/hcfp5090-q3-warmema-b3a92c1-collective2-large16.json`
- SHA-256: `895d589fcfb4f1e9192b2551f601045e5596a1e9e971025b8e032e4c4ec0e302`
- clean solver commit: `3c126caa94f7d8a30f0bfcf76da370ba77868cd7`
- checkpoint state: `b3a92c183f3a4f840955f62b8f61cd4e0573ae07b6e2697201280f52deff17d0`
- checkpoint file SHA-256: `729ea4e15ed451603b80c34f865ebc83917919a30af290d3bc823bdc6ee26a42`

Control artifact:

- `artifacts/benchmarks/hcfp5090-q4-clean-846147d-vector-overlap-large16.json`
- SHA-256: `846d6834a5cd395bb8ef591d7f198dcfaccea9a2843933e77acd1815c0a3fd8d`
- clean solver commit: `846147d70a23511364d395dce2dc1430376a5d1d`
- Q2 parent checkpoint state: `5f6f23da42410e92d48b2b665c3bd0ec75577a334f26e0a6e28e39369c9da477`

Both artifacts use the same 16 held-out FloorSet-Lite training samples, one
sample per source file, spanning 106--120 blocks. Their held-out sample-list
hash is
`e79491726efed65ebb6c55f7e63564e4896cc7d345bca18da725660de7d8fab9`,
with zero overlap against the recorded Q2/Q3 training lineage. The Q3 checkpoint
is initialized from the control checkpoint and trains only `collective.*`
parameters; therefore the initial topology/constraint seeds remain the relevant
control surface. The comparison is not represented as a same-commit runtime A/B.

## Candidate survival

Both runs evaluate the same portfolio size:

| Candidate type | Rows over 16 cases |
|---|---:|
| fallback | 16 |
| analytic | 256 |
| learned residual | 256 |
| constraint | 512 |
| topology | 512 |

Hard-feasible counts are:

| Stage/type | Q3 collective-2 | Q2/Q4 control |
|---|---:|---:|
| raw analytic | 46 | 46 |
| raw residual | 0 | 0 |
| raw constraint | 158 | 316 |
| raw topology | 256 | 512 |
| post-BDP analytic | 96 | 96 |
| post-BDP residual | 21 | 42 |
| post-BDP constraint | 161 | 322 |
| post-BDP topology | 256 | 512 |
| exact residual | 21 | 42 |
| exact constraint | 182 | 364 |
| exact topology | 256 | 512 |

The exact factor-of-two loss is structural: the merged population contains an
initial and a post-relax copy of each learned candidate. The initial copy
survives; the collective-transformed copy does not. Increasing rollout steps or
candidate count would multiply this failure mode rather than fix it.

## QoR and runtime

The best constraint candidate is unchanged within audit precision:

| Metric | Q3 collective-2 | Q2/Q4 control |
|---|---:|---:|
| raw weighted constraint oracle `J` | 2.2794279785050833 | 2.2794279785050833 |
| exact weighted constraint oracle `J` | 2.276194324785272 | 2.276194324741771 |
| exact constraint-over-topology weighted `J` gain | 0.5811977553 | same construction lane |
| selected-over-analytic weighted `J` gain | 1.1551105577 | 1.1551105576 |

All 16 selected placements remain hard feasible and beat the analytic comparator,
but they are selected from the unchanged initial/guarded alternatives. No Q3
post-relax candidate supplies the measured oracle.

Measured solver runtime is 8.564 seconds p50 and 11.253 seconds p95, versus
10.070/14.387 seconds in the older control artifact and 0.0236/0.0456 seconds
for the same-run analytic comparator. Cross-commit timing is diagnostic only;
even the Q3 measurement remains about 247x analytic at p95 and fails the runtime
gate decisively.

## Determinism finding

The Q5 two-case replay smoke reproduced byte-identical `initial` geometry hashes
for a fixed sample/seed, but the CUDA collective `post_relax` hashes changed
between identical invocations. Replay rows remain self-consistent because every
realized geometry is hash-bound to its provenance and exact target, but Q3 does
not currently satisfy a bitwise rollout-reproducibility gate.

## Decision and next repair

Q3 remains useful only behind an explicit experiment flag until all of the
following hold:

1. post-relax topology/constraint feasibility is no worse than initial;
2. exact repair displacement decreases rather than increases;
3. same-seed CUDA rollout is reproducible or its bounded nondeterminism is
   explicitly calibrated;
4. large-case p95 approaches the analytic budget;
5. a clean held-out run attributes an oracle improvement to post-relax rows.

Q5 now records paired `initial` and `post_relax` lists, post-BDP geometry, exact
constraint repair, uncapped `J`, cap margin, violation counts, and center-based
teacher displacement. That dataset is the correct next input for repairing the
collective policy and training the post-repair listwise ranker.
