# Step7T Active-Soft Cone POC

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
