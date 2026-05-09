# HELP

Updated: 2026-05-09 Asia/Taipei.

## Current status

This checkout is an active git-backed CadC26 / FloorSet-Lite research repo on
`experiment/new-method`. Step7 is the active architecture; Step6 is frozen
historical evidence.

Current bottleneck:

```text
Step7S closed smooth local-search faces; Step7T active-soft repair is the current
positive path with 3 strict winners / 8 representative cases; Step7U is blocker
evidence unless widened to larger MILP/CP-SAT.
```

Primary handoff doc:

- `docs/cadc26_research_handoff.md`

Old per-step docs and Oracle prompt docs were merged into that file and deleted.
Generated artifact Markdown/log files are disposable after the key facts are in
the handoff; JSON/JSONL evidence remains under `artifacts/research/` when exact
replay data may still be needed.

## Useful commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/<focused_test>.py
PYTHONPATH=src .venv/bin/python -m ruff check <touched_python_files>
PYTHONPATH=src .venv/bin/python -m mypy <typed_modules_or_scripts>
```

## Documentation map

- `README.md` — external onboarding, environment, and current status.
- `AGENT.md` — AI/operator routing, architecture, and red lines.
- `docs/cadc26_research_handoff.md` — condensed Step4/6/7 research history and
  cleanup retention policy.
- `docs/05_semantic_rollout.md`, `docs/06_repair_finalizer.md`,
  `docs/07_strict_evaluation.md`, `docs/08_known_failure_modes.md`,
  `docs/research-experiment-manual.md` — evergreen conceptual/manual docs.

## Red lines

- Do not edit `external/FloorSet/` in place.
- Do not treat Step6 sidecars as active architecture unless explicitly reopened.
- Do not promote a Step7 sidecar into contest runtime without explicit scope
  widening and fresh verification.
- Do not lower `MEANINGFUL_COST_EPS` to count micro improvements as strict wins.
- Do not omit `PYTHONPATH=src` unless the package is installed in the active env.
