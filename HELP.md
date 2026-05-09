# HELP

## 2026-04-29 current status

This checkout is now an active git-backed Python research repo, not the early
docs-only workspace.

Evidence:

- Branch: `experiment/new-method`.
- Current HEAD: `ee63724 Add Step7G spatial locality routing and sync AGENT.md through Step7G`.
- Package root: `src/puzzleplace/`.
- Tests: `tests/`.
- Scripts: `scripts/`.
- Config: `pyproject.toml`.
- Local environment: `.venv` points to `/home/hwchen/.local/share/mamba/envs/cadc-baseline`.
- Upstream FloorSet remains local-only under `external/FloorSet/`.

## Current architecture

The active project framing is Step7:

```text
Diagnose
-> Generate Alternatives
-> Legalize
-> Pareto Select
-> Refine / Iterate
```

Short name: **DGLPR loop**.

Do not restart from the older "coordinate regression" or boundary-first Step6
gate-stack framing.  Step6G-P artifacts are preserved as historical sidecars
and evidence, but new work should live in the Step7 module layout.

## Step7 evidence chain

- Step7A: aspect pathology / role-aware shape diagnosis.
- Step7B: role-aware shape policy smoke.
- Step7D: representative large/XL coverage replay.
- Step7E: region/topology failure diagnosis.
- Step7F: bounded repair radius study.
- Step7G: spatial locality map and move routing.
- Step7H: route-aware candidate diversification.
- Step7C-thin: route-aware layout edit loop.

Latest Step7G result:

- Decision: `promote_locality_routing_to_step7c`.
- Coverage: 8 Step7F candidate cases plus 9 Step7E locality-map cases.
- Calibration: 8/8 candidates correctly classified as global against Step7F
  weak labels.
- Routing: all 8 route to `global_route_not_local_selector`.
- Routing quality: invalid local-repair attempt rate drops from `1.0` to `0.0`
  while preserving 3 safe-improvement / non-empty-Pareto cases in the non-local
  reporting path.
- Visualization sanity audit: OK with `trace_confidence = reconstructed`;
  arrow endpoint, raw-distance, block id matching, and coordinate frame checks
  passed, but the exact construction trace is not present.



Latest Step7H result:

- Decision: `promote_route_aware_iteration_to_step7c`.
- Candidate diversity: 72 candidates; local 32, regional 24, macro 8, global 8.
- Non-global candidate rate: 0.8889.
- Invalid local attempt rate: 0.0.
- Useful regional/macro candidate count: 32.
- Synthetic probes: 4 / 4 pass with no under/over-predicted globality.



Latest Step7C-thin result:

- Decision: `promote_to_step7c_iterative_loop`.
- Actual edit candidates: 40 from 72 Step7H descriptors.
- Actual routes: local 16, regional 8, macro 8, global 8.
- Actual non-global rate: 0.8.
- Invalid local attempt rate: 0.0.
- Route stability descriptor-to-edit: 1.0.
- Actual Pareto front non-empty: 8 / 8 cases; original included in all cases.
- Caveat: edits are deterministic synthetic rectangle edits, not official real-case placement edits yet.

Primary artifacts:

- `docs/step7g_spatial_locality_routing.md`
- `artifacts/research/step7g_decision.md`
- `artifacts/research/step7g_calibration_report.json`
- `artifacts/research/step7g_visualization_audit.json`
- `artifacts/research/step7g_visualizations/`

## Next safe task

If the user does not specify another branch, the next safe task is:

```text
Step7C proper: route-aware real-case iteration loop
```

Suggested scope:

- sidecar only
- no contest runtime integration
- no RL / full NSGA-II
- no hard locality rejection gate
- preserve original layout as a candidate
- use Step7G locality routing before choosing bounded local repair vs global
  route fallback
- report per-case outcomes, not only averages

The first target should convert Step7H proxy candidate lanes into actual sidecar layout edits and verify legality/quality after route-aware repair selection.

## Commands that are usually useful

Use the repo environment explicitly:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_step7g_spatial_locality.py
PYTHONPATH=src .venv/bin/python -m mypy src/puzzleplace/diagnostics/spatial_locality.py src/puzzleplace/alternatives/locality_routing.py
PYTHONPATH=src .venv/bin/python scripts/step7g_run_spatial_locality_routing.py
```

For new Step7 sidecars, follow the same pattern:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_step7*_*.py
PYTHONPATH=src .venv/bin/python -m mypy <new typed modules>
PYTHONPATH=src .venv/bin/python -m ruff check <touched python files>
```

## 2026-05-08 Step7P-CCR handoff

The current active research lane is Step7P-CCR causal closure repack, not the
older Step7C proper recommendation above.  Step7P was opened because Step7L/M/N/O
and the Oracle stagnation diagnosis agreed that the project was reranking or
micro-perturbing a weak candidate universe instead of creating a causal
move/repack operator.

Completed gates:

- Phase0 stagnation lock: `decision=start_causal_closure_repack`; Step7O Phase3,
  Step7N reservoir reopen, Step7M micro widening, and GNN/RL are forbidden.
- Phase1 causal atlas: 325 subproblems, 8 represented cases, 6 intent families,
  `phase2_gate_open=true`.
- Phase2 synthetic repacker: 5 fixtures pass no-overlap, area/fixed/preplaced,
  MIB, boundary, and HPWL-only soft-regression rejection gates;
  `phase3_gate_open=true`.

Current blocker:

- Phase3 request deck was repaired: 120 unique requests, 8 represented cases,
  case `51` restored, `forbidden_request_term_count=0`, and
  `phase3_replay_gate_open=true`.
- Phase3 bounded replay now stops at `decision=stop_phase3_replay_gate`.
- Replay evidence: `strict_meaningful_winner_count=0`,
  `hard_feasible_nonnoop_count=79`, `overlap_after_splice_count=41`,
  `actual_all_vector_nonregressing_count=18`, `soft_regression_rate=0.525`,
  `bbox_regression_rate=0.15833333333333333`, `phase4_gate_open=false`.
- Phase4 artifact is intentionally blocked by the replay gate:
  `decision=blocked_by_phase3_replay_gate`.

Three-branch redesign result:

- Artifact: `artifacts/research/step7p_operator_branch_summary.json`.
- Branch A (`overlap_zero_hard_only`) reduces overlap to 0 but worsens
  soft/bbox regression.
- Branch B (`vector_guarded_narrow`) removes overlap/soft/bbox regression but is
  too narrow: 30 requests and 3 cases.
- Branch C (`balanced_failure_budget`) is the best partial method: 96 requests,
  8 cases, overlap 41 -> 36, soft regression 0.525 -> 0.2916666666666667,
  bbox regression 0.15833333333333333 -> 0.0.

RL/GNN readiness:

- Artifact: `artifacts/research/step7p_gnn_rl_readiness_summary.json`.
- `decision=open_constrained_operator_learning_phase`.
- `gnn_rl_gate_open=true`.
- `allowed_phase=step7q_constrained_operator_learning`.

Next safe task:

```text
start Step7Q constrained GNN/RL operator learning from branch_c_balanced_failure_budget
-> learn overlap/soft/bbox/strict-winner operator policy sidecar only
-> continue Phase4 only if phase4_gate_open=true
```

Do not run real Phase4 ablation, contest runtime integration, finalizer changes,
or direct coordinate RL from the current replay rows. Branch C reduces the
requested failure metrics but still has `strict_meaningful_winner_count=0` and
`phase4_gate_open=false`.

## Documentation map

- `README.md` — external onboarding, environment, current scripts.
- `AGENT.md` — AI/operator architecture memory and current Step7 routing.
- `docs/step6_legacy_inventory.md` — frozen Step6 sidecar map.
- `docs/step7*.md` — Step7 study plans/results.
- `artifacts/research/step7*_decision.md` — generated decisions and evidence.

## Red lines

- Do not edit `external/FloorSet/` in place.
- Do not treat Step6 sidecars as the active architecture unless the user asks to
  return to that branch.
- Do not promote a Step7 sidecar into contest runtime without explicit scope
  widening and fresh verification.
- Do not collapse large/XL evidence into a mean score; keep per-case and
  per-profile breakdowns.
- Do not omit `PYTHONPATH=src` unless the package has been installed in the
  active environment.


## 2026-05-08 Step7Q-A data mart

Step7Q-A non-leaky operator-learning data mart passed: `artifacts/research/step7q_operator_data_mart_summary.json` reports `decision=promote_to_constrained_risk_ranking`, `example_count=325`, `feature_label_leakage_count=0`, `strict_meaningful_positive_count=0`, and `allowed_next_phase=step7q_constrained_risk_ranking`. Next work is constrained risk/ranking, not direct coordinate RL or Phase4.

## 2026-05-08 Step7Q-B status

No user-blocking question. Step7Q-B found and fixed a mask issue: Phase3 request rows include a `forbidden` safety policy list, which must not be interpreted as actual forbidden operator usage. After rebuilding Step7Q-A and running the new policy smoke, `artifacts/research/step7q_operator_policy_summary.json` reports a passing risk profile (`96` rows, `8` cases, overlap `23`, soft regression `0.052083333333333336`, bbox `0.0`) but zero strict meaningful winners, so Phase4 remains closed. Next bottleneck: parameterize/synthesize new operator variants that can produce strict winners while staying inside the risk filter.

## 2026-05-08 Step7Q-C status

No user-blocking question. Step7Q-C generated `artifacts/research/step7q_parameter_expansion_deck.jsonl` and summary `artifacts/research/step7q_parameter_expansion_summary.json`. The deck is balanced and finite-action only, with no forbidden action terms or direct coordinate fields. It is ready for a fresh metric replay executor, but has zero strict winner evidence because it has not executed geometry. Phase4 remains closed.

## 2026-05-08 Step7Q-D status

No user-blocking question. Step7Q-D ran fresh metric replay for the parameter expansion deck and found the bottleneck: 87/96 actions overlap after splice; only 9 produced fresh metrics and none became strict meaningful winners. Phase4 remains closed. Next work should be an obstacle-aware slot/target executor for finite Step7Q actions, not another risk-ranking pass.

## 2026-05-08 Step7Q-E status

No user-blocking question. Step7Q-E solved the overlap problem with slot-aware replay (`overlap_after_splice_count=0`, `fresh_hard_feasible_nonnoop_count=96`) but exposed the new bottleneck: slot selection destroys objectives (`soft_regression_rate=0.9270833333333334`, `bbox_regression_rate=0.90625`, `hpwl_regression_rate=0.90625`) and still has zero strict winners. Next step: objective-aware slot scoring/selection, not legality-only search.

## 2026-05-08 Step7Q-F status

No user-blocking question. Objective-aware slot scoring improved but did not solve the bottleneck: all-vector nonregressing rose from 7 to 27 and regression rates dropped, but strict winners remain 0 and risk gate is still closed. Next step should improve the slot candidate generator/pool, not only the scorer.
