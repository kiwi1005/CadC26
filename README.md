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

Set `HCFP_CHECKPOINT=/absolute/path/model.pt` to opt into the learned
initializer. Loading is schema/hash/normalization checked; any missing,
damaged, or incompatible checkpoint falls back to the verified analytic lane.
For effect attribution, benchmark `scripts/audit_learned_optimizer.py` with
`--checkpoint learned=/absolute/path/model.pt`; this strict adapter refuses to
silently label an analytic fallback as learned.

The official submission surface is `submission/optimizer.py`; the standalone
JSON stdin/stdout executable is `submission/binary_main.py`. Geometry remains
FP32 even when later learned layers use reduced precision. Preplaced geometry
and fixed-shape dimensions are copied exactly from `target_positions`.

The canonical design is [`HCFP5090_完整技術報告.md`](HCFP5090_完整技術報告.md).
The implementation order and acceptance gates live in
[`docs/research/hcfp5090_greenfield_plan.md`](docs/research/hcfp5090_greenfield_plan.md).
The first complete-framework effect result is recorded in
[`docs/research/hcfp5090_framework_effect_2026-08-01.md`](docs/research/hcfp5090_framework_effect_2026-08-01.md).

Local development targets the RTX 5090. The contest-safe path targets the
official A100 environment and does not depend on FP8, persistent workers,
`torch.compile`, CUDA Graph capture, Shapely, SciPy, or network access.
