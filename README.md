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

The first milestone is deliberately analytic and dependency-light. SCENE,
POP-INIT, learned HiCoDy residuals, ETR, and PVR remain later promotion-gated
modules; they are not required for the safe contest baseline.

## Quick start

```bash
PYTHONPATH=src python -m hcfp demo --device auto --json
PYTHONPATH=src python -m pytest
python scripts/audit_floorset_v10.py --fetch-cache --json
```

The official submission surface is `submission/optimizer.py`; the standalone
JSON stdin/stdout executable is `submission/binary_main.py`. Geometry remains
FP32 even when later learned layers use reduced precision. Preplaced geometry
and fixed-shape dimensions are copied exactly from `target_positions`.

The canonical design is [`HCFP5090_完整技術報告.md`](HCFP5090_完整技術報告.md).
The implementation order and acceptance gates live in
[`docs/research/hcfp5090_greenfield_plan.md`](docs/research/hcfp5090_greenfield_plan.md).

Local development targets the RTX 5090. The contest-safe path targets the
official A100 environment and does not depend on FP8, persistent workers,
`torch.compile`, CUDA Graph capture, Shapely, SciPy, or network access.
