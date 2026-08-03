# HCFP-5090 Q1 structured-topology pilot

Date: 2026-08-03
Branch: `feat/hcfp5090-qor-first`
Status: **PASS — internal held-out promotion gate passed; Q2 unblocked**

## Implemented causal path

Q1 now has an opt-in path from learned structure to actual geometry:

```text
same-group / same-MIB typed scene messages
  -> dual Sinkhorn permutations
  -> deterministic hard sequence pair
  -> preplaced adaptation
  -> cycle-free H/V DAGs
  -> exact anchored longest-path packing
  -> topology candidates with provenance
  -> existing BDP and exact verifier
```

The default model and runtime remain unchanged. A topology checkpoint must set
`topology_enabled`, and runtime must request a positive `topology_seeds` count.
An explicit topology audit fails closed when no topology candidate is accepted;
the normal contest lane still falls back to the unchanged analytic incumbent.

## Exactness fixes found during integration

Two integration defects were found by direct official-evaluator probes:

1. Packing and then snapping preplaced blocks can invalidate decoded relations.
   Anchors are now part of the longest-path constraints and are copied exactly.
2. A zero-gap SP contact can become a tiny overlap after the required FP32
   `xywh -> state -> xywh` round trip. A normalized `1e-5` spacing is applied to
   edges involving movable blocks; direct anchor-to-anchor edges retain zero
   spacing so official target coordinates remain exact.

After this correction, all 64 topology candidates in the four-case 30-step
pilot were raw and post-BDP hard feasible.

## Training smoke

The first CUDA BF16 smoke used 64 training samples and 30 structure steps:

```text
first structure loss: 2.0480759
last structure loss:  1.2044708
checkpoint: artifacts/checkpoints/hcfp5090-q1-topology-smoke.pt
```

A superseded diagnostic configured a 2,048-sample source stream and ran 1,000
steps:

```text
first structure loss: 2.0480759
last structure loss:  0.2443405
checkpoint: artifacts/checkpoints/hcfp5090-q1-topology-s1000.pt
sha256: 7e496520d2a03eb772cfaff7b426cd62672d0c0d00aecaf60bb4c7500c9fc31f
```

Its sidecar did not record the sample IDs actually pulled by the 1,000
training steps, so it is retained only as diagnostic evidence. The canonical
promotion checkpoint was retrained with the same seed and architecture after
adding schema-v2 training-stream provenance:

```text
first structure loss: 2.0480759
last structure loss:  0.2443917
checkpoint: artifacts/checkpoints/hcfp5090-q1-topology-s1000-seed5090.pt
checkpoint file sha256: 6f2ffaf929735d805fcaef02ed148573e3e172f23d50d4f867bdbc30440a55db
checkpoint state hash: 48ba552c518f6c7e93a56c86e3e5a210234fb9a8438abe30ee3d4b88a8911f3a
training report sha256: 8d5c9b945647e9b08e8be5b9c42020390647dcef33f3915c82ae52d3f97645d3
```

The report records the exact ordered consumed stream, not the configured
source limit. The held-out audit reconstructs that stream from root, sampling
mode, seed, and limit; it rejects any checkpoint, config, count, or hash
mismatch before selecting held-out cases.

The official validation cases were used only for evaluation, never training.

## Four-case oracle evidence

Artifact:
`artifacts/benchmarks/hcfp5090-q1-pilot-topology-4case.json`

Cases `0, 50, 95, 99` used population 8, eight requested topology candidates,
no flow/dynamics, four projection iterations, and direction beam 1.

For the two 106--120-block cases in the 30-step pilot:

| Stage | Weighted topology oracle gain over analytic |
|---|---:|
| raw | `+59.6776` |
| post-BDP | `-5.8426` |

This is not evidence that BDP damaged topology candidates: all 32 large-case
topology records were feasible and byte-identical before/after BDP. The reversal
occurred because BDP changed analytic feasibility from 4/32 to 14/32 and
improved those candidates from a weak raw shelf into a stronger
post-projection solution. The correct gate is therefore post-BDP, not raw
geometry.

| Case | Raw analytic | Raw topology | Post analytic | Post topology |
|---|---:|---:|---:|---:|
| 95 / N=116 | `90.88` | `28.30` | `22.63` | `28.30` |
| 99 / N=120 | `82.86` | `25.27` | `19.30` | `25.27` |

The remaining topology gap is primarily structural soft QoR. Case 95 has
63/66 topology soft violations versus 58/66 for analytic; case 99 has 66/67
versus 58/67. Boundary improved only modestly while MIB improved strongly, so
the next topology stage must construct boundary/group contacts rather than run
more overlap projection.

## 1,000-step failure attribution

The fail-closed four-case audit stopped on case 95 with:

```text
fixed coordinates contradict sequence constraints
```

The decoded relations among preplaced pairs were compatible, but movable blocks
created transitive paths between anchors that required coordinates beyond the
fixed targets. Case 95 examples included horizontal paths `46/73 -> 107` and
vertical paths `46/73 -> 16`. The same checkpoint produced no accepted topology
candidate on either sampled 106--120 case when run in diagnostic mode.

The implemented repair preserves learned anchor-anchor and movable-movable
orders, then builds deterministic anchor-prefix/suffix variants so movable
blocks cannot sit between anchors in either sequence. This is a structural
preplaced adaptation, not an exact-target relaxation. Every candidate records
its actual order, H/V edges, aspect source, and order/edge hashes.

After this repair, the 1,000-step checkpoint passes fail-closed audit on all
four pilot cases. All 64 large-case topology records at oracle@16 are hard
feasible. The 106--120 post-BDP weighted gap improved from `-5.8426` to
`-0.2888`, with one topology win and one analytic win.

An oracle@32 diagnostic reached a positive `+0.0940` weighted large-case gain.
Its case-99 winner used aspect source 5, while the first 16 candidates did not
cover that source. This isolates the remaining pilot gap to bounded candidate
allocation: Q1 must choose 16 candidates from the small exact-safe pool using
training-independent candidate-local metrics, not simply emit the first 16.

The final deterministic 16-slot selector keeps the best exact-safe candidate
from each distinct topology order, then fills the remaining budget by raw soft
violation and safe-shelf-normalized HPWL/bbox quality. On the four visible
development cases this changed the large-case post-BDP gain to `+0.0940` and
the all-four weighted gain to `+0.1423`. These visible cases remain development
diagnostics, not promotion evidence.

## Held-out failure and adaptation repair

The first training-only audit correctly failed closed on
`worker_40/layouts_2464.th:53`. Two preplaced conflicts were low-confidence,
but the old repair froze every other geometrically valid anchor relation even
when that relation was ambiguous and low-confidence. Three such decisions
formed a plus-order cycle after the exact singleton relation was inserted.

The repair now:

1. inserts geometrically forced singleton anchor relations first;
2. keeps high-confidence predicted relations hard;
3. exposes low-confidence ambiguous anchor relations as choices;
4. runs a deterministic search bounded to 4,096 states over those choices;
5. rejects the seed if exact anchors and protected relations still cannot form
   two acyclic orders.

The failing case then produced all 16 requested topology seeds. Exact official
raw-coordinate replay remains downstream-authoritative; the normalized
candidate path remains FP32 and within the pinned hard tolerance.

## Training-only internal held-out gate

The initial 16-sample audit was discarded as promotion evidence because all
samples came from one source file and all had 108 blocks. The audit now defaults
to at most one layout per source file.

Canonical artifact:
`artifacts/benchmarks/hcfp5090-q1-s1000-seed5090-topology16-internal-heldout-large16-diverse.json`

Artifact sha256:
`d012fdb1c0838fa4d9eff1dd664c383d45f4c3449215b8d1ea9b42ac0d8dc329`

Split evidence:

```text
training source limit: 2,048, seed 5090
actual consumed training IDs: 1,000 ordered / 1,000 unique
ordered training ID sha256: 09ebf0335ce10503724a5f81c8a6a4ca89a69373c5eaadbc3c2c6346a49f0da2
unique training ID sha256: 7b697f10f1bfe9c5ca8d6565e3aadd694bc7a9ffc25ef3d033d3116c317ba7b0
held-out seed: 5091
held-out samples: 16
distinct source files: 16
block range observed: 107--120
held-out ID sha256: e79491726efed65ebb6c55f7e63564e4896cc7d345bca18da725660de7d8fab9
training/held-out overlap: 0
```

Results:

| Metric | Result |
|---|---:|
| raw weighted topology gain over analytic | `+51.7198` |
| post-BDP weighted topology gain over analytic | `+24.3934` |
| post-BDP topology wins | `16 / 16` |
| selected weighted gain over analytic | `+23.0997` |
| selected wins over analytic | `16 / 16` |
| raw topology-stage hard feasibility | `512 / 512` |
| post-BDP topology-stage hard feasibility | `512 / 512` |
| weighted analytic oracle objective | `42.8548` |
| weighted topology oracle objective | `18.3462` |

The count is 512 because each of 16 structural seeds is audited at both the
initial and post-relax candidate positions for 16 cases. The two stages are
kept separate in provenance and do not increase oracle diversity.

The current exact selector chooses a topology candidate and beats the analytic
oracle on every held-out case, but misses the overall candidate oracle on
12/16 cases. Q1 therefore proves useful structured candidate density and safe
runtime causality; it does not close post-repair ranking regret. That remains a
Q5 listwise-ranker target rather than a reason to widen Q1 sampling.

Topology provenance now stores full positive/negative orders and H/V edges once
per order hash. Candidate, pool, and stage records reference that catalog by
hash. The multi-aspect fixture shrank from 39,056 to 23,111 serialized bytes
without losing reconstruction checks.

## Promotion decision

Q1 passes its internal promotion gate. All five evidence requirements are now
met: deterministic anchor-safe variants, zero-cycle/exact-anchor construction,
bounded 16-slot selection, disjoint training-only held-out evaluation, and
positive weighted post-BDP large-case gain. Q2 constraint construction may now
start; Q1 remains opt-in until later full-case and runtime gates pass.

This is an internal development promotion gate, not release proof. The artifact
pins its checkpoint, training report, commands, source hashes, and exact sample
streams; a release claim still requires rerunning the command from the
committed integration tree and completing the later full-case/runtime gates.

## Verification snapshot

Current integration snapshot:

```text
targeted topology/held-out/attribution tests: passed
diverse 16-case CUDA held-out audit: passed
full pytest: passed after provenance integration
Ruff: passed
compileall: passed
git diff --check: passed
CUDA BF16 30-step smoke: passed
CUDA BF16 1,000-step diagnostic: passed
explicit topology audit: fail-closed on incomplete seeds; repaired case passes
```
