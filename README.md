# FloorSet Puzzle Research Workspace

CadC26 is a local FloorSet-Lite / ICCAD 2026 floorplanning research workspace.
It contains a runnable Python package, contest smoke entrypoints, tests, and
sidecar research experiments. The upstream FloorSet checkout lives under
`external/FloorSet/`, stays untracked, and must not be modified in place.

## Current project direction

The active architecture is Step7 **DGLPR**:

```text
Diagnose -> Generate Alternatives -> Legalize -> Pareto Select -> Refine / Iterate
```

As of 2026-05-09:

- Step6 is frozen historical evidence.
- Step7P/Q/R local-operator expansion produced 0 strict meaningful winners.
- Step7S certified local smooth-face stationarity on the 8 representative cases.
- Step7T active-soft repair is the current positive branch: 3 strict winners on
  3/8 representative cases, `phase4_gate_open=true`, exact visual sanity passed.
- Step7U bounded blocker MILP is currently an obstruction certificate, not a
  generator.

Read `docs/cadc26_research_handoff.md` for the condensed Step4/6/7 history,
current bottleneck, and cleanup retention policy. Old per-step diary docs and
Oracle prompt docs were merged there and deleted.

Do not integrate runtime/finalizer changes from Step7 sidecars until a fresh
Phase4 review validates the exact active-soft winners.

## Environment

### 1. Clone the official FloorSet repo locally only

```bash
git clone https://github.com/IntelLabs/FloorSet.git external/FloorSet
```

`external/FloorSet/` is ignored by git on purpose so upstream source and
downloaded datasets do not get vendored into this repository.

### 2. Use the repo virtual environment

This checkout normally has `.venv` pointing at the shared `cadc-baseline`
environment. Prefer it instead of creating a second environment:

```bash
source .venv/bin/activate
python -m pip install -e .[dev]
```

For commands in automation, use:

```bash
PYTHONPATH=src .venv/bin/python <script-or-module>
```

## Smoke download / loader check

```bash
python scripts/download_smoke.py
```

The script imports official FloorSet dataloaders, auto-approves the dataset
prompt for non-interactive runs, and prints validation/training batch structure.
Expected local-only data roots:

- validation: `external/FloorSet/LiteTensorDataTest/`
- training: `external/FloorSet/floorset_lite/`

## Main entrypoints

Contest/runtime:

- `contest_optimizer.py`
- `src/puzzleplace/optimizer/contest.py`
- `python scripts/evaluate_contest_optimizer.py`
- `python scripts/run_smoke_regression_matrix.py`

Research sidecars:

- Step7 scripts live under `scripts/step7*.py`.
- Step7 modules live under `src/puzzleplace/{diagnostics,alternatives,experiments,ml,repack,search}/`.
- Generated evidence lives under ignored `artifacts/research/`.
- Closed experiment conclusions should be merged into
  `docs/cadc26_research_handoff.md` instead of accumulating one doc per step.

## Verification shortcuts

Use the repository environment and `PYTHONPATH=src`:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/<focused_test>.py
PYTHONPATH=src .venv/bin/python -m ruff check <touched_python_files>
PYTHONPATH=src .venv/bin/python -m mypy <typed_modules_or_scripts>
```

For broad checks after larger code changes:

```bash
PYTHONPATH=src .venv/bin/python -m pytest
PYTHONPATH=src .venv/bin/python -m ruff check src scripts tests
```

## Red lines

- Do not edit `external/FloorSet/` in place.
- Do not promote sidecar behavior into contest runtime/finalizer paths without
  explicit scope widening and fresh verification.
- Do not lower `MEANINGFUL_COST_EPS=1e-7` to rescue local micro improvements.
- Do not reopen Step7P/Q/R-style local operators unless FloorSet/data/threshold
  changes and Step7S is rerun first.
- Do not report only averages; preserve per-case and per-profile evidence.
