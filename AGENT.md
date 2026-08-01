# AGENT.md — HCFP-5090

This repository is a greenfield ICCAD 2026 FloorSet solver. Keep all work inside
`/home/hwchen/PROJ/CadC26` and keep the runtime implementation under `src/hcfp/`.

## Canonical sources

- Architecture: `HCFP5090_完整技術報告.md`
- Packaged design material: `HCFP5090_中文完整方案_2026-08-01/`
- Implementation plan: `docs/research/hcfp5090_greenfield_plan.md`
- Pinned official reference: `src/hcfp/reference.py`
- Contest entrypoint: `submission/optimizer.py`

## Runtime invariants

- Support the official seven-argument solve contract, including optional
  `target_positions`.
- Preserve preplaced `(x, y, w, h)` and fixed-shape `(w, h)` exactly.
- Keep coordinates, dimensions, overlap predicates, and force accumulation in
  FP32 or higher precision; never require FP8 for contest execution.
- Start from a deterministic safe fallback and replace the incumbent only with
  finite, verified candidates.
- Keep GPU inner loops free of Python block/candidate loops and geometry
  packages such as Shapely.
- Treat grouping, MIB, and boundary as soft constraints; treat overlap, area,
  fixed shape, and preplaced geometry as hard constraints.
- Do not add dependencies without a demonstrated need.

## Verification

Run targeted tests first, then the full suite:

```bash
PYTHONPATH=src python -m pytest tests/test_<area>.py -q
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m compileall -q src/hcfp submission
```

When CUDA is available, run the device parity smoke. Before contest packaging,
audit the pinned official FloorSet checkout and run the JSON submission smoke
from both the repository root and `submission/` working directory.
