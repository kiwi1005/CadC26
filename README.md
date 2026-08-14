# HCFP-5090

Greenfield heterogeneous collective floorplanner for ICCAD 2026 FloorSet cases
(`21 <= N <= 120`). This branch contains only the new HCFP path:

```text
official case tensors
  -> deterministic safe incumbent
  -> FP32 batched collective dynamics
  -> batched disjunctive projection (BDP)
  -> exact-compatible verification
  -> incumbent-preserving submission output
```

The default contest lane is deliberately analytic and dependency-light. The
repository also contains trainable SCENE/POP-INIT/rectified-flow/controller/
ranker components, auditable tar shards, versioned checkpoints, and an opt-in
learned initializer. Learned weights remain default-off until exact post-BDP
promotion evidence exists.

## Quick start

```bash
PYTHONPATH=src python -m hcfp demo --device auto --json
PYTHONPATH=src python -m pytest
python scripts/audit_floorset_v10.py --fetch-cache --json
```

Run an exact fallback-versus-analytic effect benchmark and emit comparison
visualizations:

```bash
PYTHONPATH=src python scripts/benchmark_hcfp.py \
  --optimizer fallback=scripts/audit_fallback_optimizer.py \
  --optimizer analytic=submission/optimizer.py \
  --baseline fallback \
  --data-path artifacts/floorset-v10 \
  --cases 0,50,99 \
  --device cuda \
  --output artifacts/benchmarks/hcfp-current.json \
  --visualize-dir artifacts/benchmarks/hcfp-current \
  --visualize-cases 0,50,99
```

Render any placement/candidate JSON directly as dependency-free SVG or HTML:

```bash
PYTHONPATH=src python scripts/visualize_hcfp.py placement.json -o floorplan.svg
```

Build a checksummed training shard, run a bounded training smoke, and profile
the real analytic fast path:

```bash
PYTHONPATH=src python scripts/build_hcfp_shards.py fixtures.json \
  -o artifacts/shards/train-000000.tar \
  --source FloorSet-train --source-version v10 --split train \
  --denylist official-validation-ids.txt
PYTHONPATH=src python scripts/train_hcfp.py artifacts/shards/train-000000.tar \
  -o artifacts/checkpoints/hcfp-smoke.pt --steps 10 --device cuda
PYTHONPATH=src python scripts/profile_hcfp.py --blocks 120 --candidates 32 \
  --device cuda --output artifacts/reports/profile-n120-k32.json
```

For the official 1.008M-case corpus, stream the extracted FloorSet-Lite files
directly and close the exact-tail replay/ranker loop without copying the
dataset:

```bash
PYTHONPATH=src python scripts/train_hcfp.py \
  --floorset-lite-root artifacts/floorset-v10 --sampling score-aware \
  -o artifacts/checkpoints/hcfp-flow.pt --stage all --steps 3000 \
  --population 8 --amp bf16 --ema-decay 0.999 --device cuda
PYTHONPATH=src python scripts/generate_hcfp_replay.py \
  --floorset-lite-root artifacts/floorset-v10 \
  --checkpoint artifacts/checkpoints/hcfp-flow.pt \
  --output artifacts/replay/hcfp-exact-tail.jsonl --limit 32 --device cuda
PYTHONPATH=src python scripts/train_hcfp_ranker.py \
  artifacts/replay/hcfp-exact-tail.jsonl \
  --checkpoint artifacts/checkpoints/hcfp-flow.pt \
  --output artifacts/checkpoints/hcfp-ranked.pt --steps 500 --device cuda
PYTHONPATH=src python scripts/eval_hcfp_ranker.py \
  --replay heldout=artifacts/replay/hcfp-exact-tail-heldout.jsonl \
  --checkpoint ranked=artifacts/checkpoints/hcfp-ranked.pt \
  --output artifacts/benchmarks/hcfp-ranker-regret.json --device cuda
PYTHONPATH=src python scripts/audit_hcfp_oracle.py \
  --data-path artifacts/floorset-v10 \
  --checkpoint artifacts/checkpoints/hcfp-ranked.pt \
  --output artifacts/benchmarks/hcfp-oracle-attribution.json \
  --cases all --device cuda --population 8 --flow-steps 6
```

Build a disjoint learned-tail activation replay and run the hash-bound shadow
training/evaluation loop:

```bash
PYTHONPATH=src python scripts/generate_hcfp_activation_replay.py \
  --floorset-lite-root artifacts/floorset-v10 \
  --checkpoint artifacts/checkpoints/hcfp-ranked.pt \
  --output-prefix artifacts/replay/hcfp-activation \
  --train-count 1024 --calibration-count 512 --heldout-count 512 \
  --layouts-per-file 16 --seed 20260806 --device cuda
PYTHONPATH=src python scripts/train_hcfp_activation.py \
  --train-replay artifacts/replay/hcfp-activation.train.jsonl \
  --calibration-replay artifacts/replay/hcfp-activation.calibration.jsonl \
  --output artifacts/checkpoints/hcfp-activation.json --device cuda
PYTHONPATH=src python scripts/eval_hcfp_activation.py \
  --policy artifacts/checkpoints/hcfp-activation.json \
  --training-report artifacts/checkpoints/hcfp-activation.json.training.json \
  --replay heldout=artifacts/replay/hcfp-activation.heldout.jsonl \
  --output artifacts/benchmarks/hcfp-activation-heldout.json
```

The trainer requires at least 32 train and 16 calibration positives by default.
The current shadow policy failed disjoint held-out recall/skip gates, so no
activation policy is connected to the contest runtime.

The oracle audit keeps all learned candidates by default, applies the same
analytic/BDP tail as runtime, and attributes raw and post-BDP official quality
to fallback, analytic, and learned sources. `--tail-topk` is only for a
secondary ranker-pruning comparison, not the primary oracle@K measurement.

Set `HCFP_CHECKPOINT=/absolute/path/model.pt` to opt into the learned
initializer. Loading is schema/hash/normalization checked; any missing,
damaged, or incompatible checkpoint falls back to the verified analytic lane.
For the score-dominant 106--120 block bucket, runtime enables 16 topology and
16 constraint seeds by default. `HCFP_LARGE_CHECKPOINT` may point that bucket
at a large-case fine-tuned checkpoint while smaller cases keep
`HCFP_CHECKPOINT`.
For effect attribution, benchmark `scripts/audit_learned_optimizer.py` with
`--checkpoint learned=/absolute/path/model.pt`; this strict adapter requires a
valid checkpoint while the per-case raw gate still retains the verified
analytic/safe incumbent when a learned winner is not officially legal.
If a normalized winner fails only after raw denormalization, the lane first
reuses a verified non-fallback projected candidate; fallback-only pools retain
the analytic replay to avoid trading quality for runtime.

The official submission surface is `submission/optimizer.py`; the standalone
JSON stdin/stdout executable is `submission/binary_main.py`. Geometry remains
FP32 even when later learned layers use reduced precision. Preplaced geometry
and fixed-shape dimensions are copied exactly from `target_positions`.

The canonical design is [`HCFP5090_完整技術報告.md`](HCFP5090_完整技術報告.md).
The implementation order and acceptance gates live in
[`docs/research/hcfp5090_greenfield_plan.md`](docs/research/hcfp5090_greenfield_plan.md).
The first complete-framework effect result is recorded in
[`docs/research/hcfp5090_framework_effect_2026-08-01.md`](docs/research/hcfp5090_framework_effect_2026-08-01.md).
The official-data training, exact replay, learned sidecar, 100-case benchmark,
and current HOLD decision are recorded in
[`docs/research/hcfp5090_training_closed_loop_2026-08-01.md`](docs/research/hcfp5090_training_closed_loop_2026-08-01.md).
The raw-safe reselection implementation and 100-case evidence are recorded in
[`docs/research/hcfp5090_raw_reselection_2026-08-02.md`](docs/research/hcfp5090_raw_reselection_2026-08-02.md).
The official-baseline replay, selector training, deterministic seed contract,
and latest 100-case HOLD decision are recorded in
[`docs/research/hcfp5090_official_selector_results_2026-08-02.md`](docs/research/hcfp5090_official_selector_results_2026-08-02.md).
The split-tail Pareto guard, clean-provenance validation 100, and current
runtime-limited HOLD decision are recorded in
[`docs/research/hcfp5090_runtime_pareto_guard_results_2026-08-02.md`](docs/research/hcfp5090_runtime_pareto_guard_results_2026-08-02.md).
The exact-overlap vectorization, byte-identical validation 100 replay, and
remaining median-runtime gate are recorded in
[`docs/research/hcfp5090_runtime_vectorization_results_2026-08-02.md`](docs/research/hcfp5090_runtime_vectorization_results_2026-08-02.md).
The training-only activation replay, shadow policy result, held-out failure,
and explicit no-promotion decision are recorded in
[`docs/research/hcfp5090_activation_shadow_results_2026-08-02.md`](docs/research/hcfp5090_activation_shadow_results_2026-08-02.md).
The current large-case QoR-first checkpoint, exact 15-case result, runtime
policy, and per-case analytic/learned PNG comparisons are recorded in
[`docs/research/hcfp5090_qor_first_large_structure_2026-08-12.md`](docs/research/hcfp5090_qor_first_large_structure_2026-08-12.md).
The approved next-stage task DAG for latent-outline recovery, exact-area
treemaps, `tree_sol`-supervised B*-Trees, mask/TTO refinement, near-cap replay,
and submission freeze is tracked in
[`docs/research/hcfp5090_latent_outline_exact_packing_plan_2026-08-12.md`](docs/research/hcfp5090_latent_outline_exact_packing_plan_2026-08-12.md).
The completed P0 bbox/cap evidence and the P1 training-only outline-recovery
audit are recorded in
[`docs/research/hcfp5090_p0_p1_outline_results_2026-08-12.md`](docs/research/hcfp5090_p0_p1_outline_results_2026-08-12.md).
The P7 axis-dual B*-Tree, sparse-island rescue, failed challenger ablations,
baseline-head calibration, full100 QoR result and per-case visual audit are
recorded in
[`docs/research/hcfp5090_p7_frontier_completion_2026-08-13.md`](docs/research/hcfp5090_p7_frontier_completion_2026-08-13.md).
The P8 dense contact patch, boundary witness, obstacle-region and
connectivity-aware B*-Tree experiments, including the guarded full100 result,
are recorded in
[`docs/research/hcfp5090_p8_constraint_topology_results_2026-08-13.md`](docs/research/hcfp5090_p8_constraint_topology_results_2026-08-13.md).
The follow-on Case70 BFOD/P10 sidecar, learned contact-ranker experiment, and
plan-versus-implementation status are recorded in
[`docs/research/hcfp5090_bfod_v1_p10_progress_2026-08-14.md`](docs/research/hcfp5090_bfod_v1_p10_progress_2026-08-14.md).

Local development targets the RTX 5090. The contest-safe path targets the
official A100 environment and does not depend on FP8, persistent workers,
`torch.compile`, CUDA Graph capture, Shapely, SciPy, or network access.
