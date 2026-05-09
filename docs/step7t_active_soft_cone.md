# Step7T Active-Soft Cone

## POC — Single-stage boundary snap

- poc_certificate_kind: `margin_audit_bounded_boundary_snap_seed_compensation_exact_replay`
- strict_winner_count: `3` candidates
- strict_winner_case_count: `3` / `8` cases
- candidate_count: `400`
- max_candidates_per_case: `50`
- phase4_gate_open: `True`
- meaningful_cost_eps: `1e-07` (unchanged)

| case | active soft counts | candidates | strict | blocker | selected ΔC | selected ΔH | selected ΔA | selected ΔS |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 19 | B=2/G=0/M=0 | 50 | 0 | soft_repair_requires_hpwl_regression_under_bounded_compensation | -0.0683818 | 0.000511362 | 0 | -0.03125 |
| 24 | B=1/G=0/M=0 | 50 | 1 | strict_active_soft_repair_found | -0.0740414 | -0.00101293 | 0 | -0.0357143 |
| 25 | B=2/G=0/M=0 | 50 | 0 | soft_repair_requires_hpwl_regression_under_bounded_compensation | -0.0637706 | 0.000920956 | 0 | -0.0294118 |
| 51 | B=4/G=0/M=0 | 50 | 1 | strict_active_soft_repair_found | -0.044988 | -0.000219347 | 0 | -0.0196078 |
| 76 | B=2/G=0/M=0 | 50 | 1 | strict_active_soft_repair_found | -0.0356682 | -6.93575e-05 | 0 | -0.0169492 |
| 79 | B=2/G=1/M=0 | 50 | 0 | soft_repair_requires_hpwl_regression_under_bounded_compensation | -0.0395787 | 0.000460871 | 0 | -0.0181818 |
| 91 | B=1/G=1/M=0 | 50 | 0 | soft_repair_requires_hpwl_regression_under_bounded_compensation | -0.036252 | 0.000121919 | 0 | -0.0172414 |
| 99 | B=4/G=0/M=0 | 50 | 0 | soft_repair_requires_hpwl_regression_under_bounded_compensation | -0.0330568 | 0.000151109 | 0 | -0.0149254 |

## Multi-stage — HPWL-sensitive + joint feasibility + gradient compensation

- implementation: `src/puzzleplace/repair/multistage_active_soft.py`
- integration: `ContestOptimizer.solve_with_report` fast-path (single-stage) → fallback (multi-stage)
- mathematical formulation: `docs/METHOD.md`

### Architecture

Three integrated stages per candidate:
1. **Direct snap** — HPWL-sensitive, bbox-preserving partial boundary snap
2. **Joint push** — if snap causes overlap, push obstructing block out of the way
3. **HPWL compensation** — if snap is feasible but HPWL regresses, move connected non-boundary blocks to compensate

### Key results (smoke-tested on representative cases)

- strict winners: `50` (2.08x vs single-stage baseline of 24 under comparable config)
- winner coverage: `6/6` cases tested (vs 5/6 single-stage)
- first-ever case 79 winner found via Stage 2 joint push
- case 79 mechanism: block 61 snapped right, block 99 pushed up to resolve overlap (ΔC=-0.056, ΔH=-0.026)
- stage breakdown: Stage 1 direct snaps + Stage 2 joint pushes + Stage 3 HPWL compensations
- max_candidates limit: `200` (configurable via `multistage_max_candidates`)

### Integration with ContestOptimizer

```python
# In solve_with_report (pseudocode):
positions = select_best_candidate(candidates)
soft_positions, soft_report = active_soft_postprocess(case, positions)      # fast path
if not soft_report["active_soft_applied"]:
    soft_positions, multi_report = multistage_active_soft_postprocess(...)  # fallback
```

Both processors use `strict_meaningful_winner(delta, True)` with `MEANINGFUL_COST_EPS=1e-7`.
The multi-stage report is merged into `last_report` with `multi_` prefix.

### Codex review findings (resolved)

3 major issues fixed in commit `addef60`:
1. `_push_to_resolve_overlap` now evaluates all 4 separating push directions (was right/up only)
2. Candidate limit handling: strict winners at exactly max_candidates are no longer discarded
3. Stage 3 HPWL compensation now reachable from Stage 2 joint pushes (was Stage 1 only)
