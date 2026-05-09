---
status: closed
opened: 2026-05-09
closed: 2026-05-09
predecessor: docs/step7r_chain_move_operator.md
predecessor_close_decision: artifacts/research/step7r_close_decision.json
oracle_response: artifacts/research/oracle_step7s_critical_cone_response.md
terminal_certificate: artifacts/research/step7s_critical_cone_certificate.md
terminal_summary: artifacts/research/step7s_critical_cone_summary.json
terminal_result: local_kkt_stationarity_certified_with_hpwl_hinge_cap
---

# Step7S: Critical-Cone Active-Set Descent Certificate

Step7S is the post-Step7R lane, opened on the basis of a mathematical
derivation rather than another empirical operator family. The Oracle
analysis showed that:

1. Step7Q-F's 27 AVNR rows collapse to 2 unique geometric outcomes after
   the slot-finder projection;
2. The best observed official decrease is `-1.477e-8`, so reaching the
   strict gate `MEANINGFUL_COST_EPS = 1e-7` requires aggregating roughly
   **7-8 effective aligned single-block descent units**;
3. This cannot come from a single-block or k=2 local operator — only from
   coordinated multi-block descent on the active-contact closure;
4. The same primal/dual program emits a KKT certificate of local
   Pareto-stationarity if no such direction exists.

## Decision boundary

Step7S **is**:

- a sidecar that emits a primal/dual certificate per representative case;
- evaluated under the unchanged Phase4 strict gate;
- bounded to the active-contact closure of one seed block per run.

Step7S **is not**:

- a contest runtime / finalizer integration;
- a direct-coordinate RL solver;
- a relaxation of `MEANINGFUL_COST_EPS`;
- a reopening of Step7P/Q/R chain or gradient operators;
- a global re-placement.

## Phase plan

### Phase E (verifiable_immediately) — assumption checks

Run before opening any new modeling lane. All four must pass.

| script | output |
|--------|--------|
| `scripts/verify_official_hinge_scalarization.py` | `artifacts/research/step7s_hinge_scalarization_check.json` |
| `scripts/verify_step7q_avnr_dedup.py` | `artifacts/research/step7s_avnr_unique_candidate_check.json` |
| `scripts/verify_step7q_avnr_hinge_caps.py` | `artifacts/research/step7s_avnr_hinge_cap_check.json` |
| `scripts/verify_official_coordinate_precision.py` | `artifacts/research/step7s_coordinate_precision_check.json` |

Gate: every check must report `verdict = pass`. If any fails, fix the model
assumption before continuing — do not patch the CCQP to compensate.

### Phase F (verifiable_with_small_extension) — instrumentation

| script | output |
|--------|--------|
| `scripts/step7s_contact_graph.py` | `artifacts/research/step7s_contact_graph.json` |
| `scripts/step7s_slot_jacobian_probe.py` | `artifacts/research/step7s_slot_jacobian_probe.json` |
| `scripts/step7s_gradient_check.py` | `artifacts/research/step7s_gradient_check.json` |

Gate: contact graph stable (active contacts agree under ±1e-6 perturbation),
slot finder image rank ≤ 4 on a 6-direction probe, analytic gradients agree
with finite differences within 1e-6 relative.

### Phase S1 (smallest CCQP) — case 24, seed block 32

`scripts/step7s_critical_cone_certificate.py --case-id 24 --seed-block 32`

Output: `artifacts/research/step7s_case24_block32_cone_certificate.json` with:

- `closure_size`, `closure_block_ids`
- `sigma_pred` (predicted official descent strength on the cone)
- `rho_pred` (CCQP slack)
- `delta_predicted` (CCQP linearized deltas)
- `delta_exact` (after exact line search and replay)
- `dual_multipliers` (`λ_H, λ_A, λ_S, μ, ν`) when `rho_pred = 0`
- `result`: one of `strict_winner` | `avnr_only` | `infeasible_cone` | `kkt_stationary`

Phase S2 gate: open if any of the 8 representative cases produces
`strict_winner` AND total strict count ≥ 3 across the 8 cases.

### Phase S2 (full 8-case CCQP) — only after S1

Run CCQP on every representative case using its largest-AVNR seed block.
Aggregate primal/dual certificates into
`artifacts/research/step7s_critical_cone_summary.json`. Two terminal results:

- `strict_winner_set`: ≥ 3 exact strict winners → Phase4 gate review reopens;
- `local_kkt_stationarity_certified`: KKT multipliers proving Pareto-stationarity
  on the current active faces at threshold 1e-7. Project gains a hard
  certificate that local single-axis Step7P/Q/R operators were exhausted.

## Forbidden in Step7S

- modifying `contest_optimizer.py`, `external/FloorSet/`, or finalizer paths;
- relaxing `MEANINGFUL_COST_EPS` from `1e-7`;
- passing CCQP displacements back through the Step7Q-F slot finder
  (Oracle predicted this collapses to the same fixed point);
- claiming `phase4_gate_open = true` without ≥ 3 exact strict winners;
- adding numerical-noise epsilon inflation to make small AVNR moves count
  as strict.

## Tooling

- QP solver: prefer `scipy.optimize.linprog` for the linearized version
  (drop `‖δ‖²` regularizer and treat as LP with `max ρ`), or
  `scipy.optimize.minimize(method='SLSQP')` for the full QP. If neither is
  numerically robust at this scale, switch to `cvxpy` (already an indirect
  dependency through other tooling? — verify before adding).
- Parallelism: Phase S2 runs 8 cases in parallel via
  `concurrent.futures.ProcessPoolExecutor(max_workers=min(8, 48))`. Phase S1
  is single-case so serial.
- Gradients: use the existing FloorSet evaluator (`evaluate_positions`) for
  ground-truth metric values; analytic HPWL gradient is per-pin
  `∂HPWL/∂center_block = sign(center_block - pin_centroid)` summed by net
  weight (Manhattan distance subgradient). Boundary-of-bbox blocks have a
  rank-1 effect on bbox gap.

## Orchestration note

Same split as Step7R: main agent (this Claude session) plans, verifies, and
runs CCQP; if any sub-task is large enough, it is delegated to Codex with
explicit parallelism instructions. Main agent never relaxes the strict
gate. The Step7S terminal artifact is a primal/dual certificate, not an
operator replay summary.
