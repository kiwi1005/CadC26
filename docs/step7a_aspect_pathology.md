# Step7A: Aspect Pathology and Role-Aware Shape Diagnosis

## Cleanup / implementation plan

1. Keep Step6 sidecars frozen and traceable; do not move them during Step7A.
2. Add a new diagnostics module at `src/puzzleplace/diagnostics/aspect.py`.
3. Reuse Step6P representative alternatives as the input layout set.
4. Compare original vs selected representative layouts for aspect distortion.
5. Emit per-case, per-role, candidate-family, correlation, and visualization artifacts.

## Acceptance checks

- Sidecar only; no runtime integration.
- Report multiple thresholds: `abs(log(w/h)) > 1.5`, `> 2.0`, `> 3.0`.
- Include large/XL coverage status explicitly.
- Include representative failure PNGs.
- Run ruff, mypy, targeted pytest, and artifact sanity checks.
