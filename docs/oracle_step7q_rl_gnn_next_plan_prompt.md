# Oracle review request: CadC26 Step7Q constrained RL/GNN operator-learning plan

You are reviewing a local ICCAD 2026 FloorSet-Lite floorplanning research workspace (`/home/hwchen/PROJ/CadC26`). Please act as an external architecture/research critic and planner.

## Project briefing

The repo is a Python sidecar research workspace for FloorSet-Lite floorplanning. It contains Step7 sidecar experiments around diagnosis, candidate generation, replay, vector gating, and constrained selection. The upstream FloorSet checkout is local-only under `external/FloorSet/` and must not be modified. Contest runtime/finalizer integration is explicitly out of scope for this review.

Current active architecture:

```text
Diagnose -> Generate Alternatives -> Legalize/Repack -> Full-case/vector replay -> constrained gate -> next phase
```

Relevant environment/verification commands:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_step7p_*.py tests/test_step7q_*.py -q
PYTHONPATH=src .venv/bin/python -m ruff check <touched files>
PYTHONPATH=src .venv/bin/python -m mypy <typed modules>
```

## Current situation summary

We were stuck in Step7P causal closure repack. The original issue was that many previous Step7 lines only ranked or micro-perturbed a weak candidate universe. Step7P tried to create a causal move/repack operator.

Step7P evidence so far:

1. Phase0 stagnation lock passed.
   - Existing Step7L/M/N/O candidate universe has no broad new winner signal.
   - Step7O Phase3, Step7N reservoir reopen, Step7M micro widening, and GNN/RL were kept closed.

2. Phase1 causal atlas passed.
   - `subproblem_count=325`
   - `represented_case_count=8`
   - `largest_case_share=0.32`
   - diverse failure/intent labels.

3. Phase2 synthetic causal repacker passed.
   - synthetic fixtures prove no-overlap, area/fixed/preplaced/MIB/boundary guards, and reject HPWL-only soft regression.

4. Phase3 request gate was repaired and now passes.
   - `request_count=120`
   - `represented_case_count=8`
   - `case51` restored via exact non-forbidden closure fallback.
   - forbidden request term count = 0.

5. Phase3 bounded replay still fails.
   - `hard_feasible_nonnoop_count=79`
   - `overlap_after_splice_count=41`
   - `actual_all_vector_nonregressing_count=18`
   - `soft_regression_rate=0.525`
   - `bbox_regression_rate=0.15833333333333333`
   - `strict_meaningful_winner_count=0`
   - `phase4_gate_open=false`

6. Replay blocker diagnosis showed selector/request ordering alone cannot solve it.
   - non-forbidden exact hard source rows = 107
   - all-vector-nonregressing source rows = 30
   - strict meaningful source rows = 0

7. We then tried three operator-source branches:
   - Branch A: hard-only / overlap-zero. Reduces overlap to 0 but worsens soft/bbox regression.
   - Branch B: all-vector guarded. Good vectors but too narrow: 30 requests / 3 cases.
   - Branch C: balanced failure budget. Best partial method:
     - `request_count=96`
     - `represented_case_count=8`
     - `hard_feasible_nonnoop_count=60`
     - `largest_case_share=0.25`
     - `overlap_after_splice_count=36` vs baseline 41
     - `soft_regression_rate=0.2916666666666667` vs baseline 0.525
     - `bbox_regression_rate=0.0` vs baseline 0.15833333333333333
     - `strict_meaningful_winner_count=0`
     - `phase4_gate_open=false`

8. Based on this plateau, we opened a constrained RL/GNN readiness gate:
   - artifact: `artifacts/research/step7p_gnn_rl_readiness_summary.json`
   - `decision=open_constrained_operator_learning_phase`
   - `gnn_rl_gate_open=true`
   - `allowed_phase=step7q_constrained_operator_learning`

Important: this does NOT mean direct coordinate RL, contest runtime integration, finalizer changes, or Phase4. It means a sidecar learning phase for causal operator policy/scoring only.

## Constraints and red lines

- Do not modify or rely on changing `external/FloorSet/`.
- Do not integrate into `contest_optimizer.py` or runtime solver.
- Do not change finalizer semantics.
- Do not use RL as a direct coordinate generator.
- Do not call Phase4 successful unless `phase4_gate_open=true` from vector replay.
- Avoid scalar penalty soup; keep HPWL/BBox/Soft/official-like deltas separate.
- Do not claim success from micro deltas or case024/case025-only improvements.
- Preserve per-case evidence; do not summarize only by average.

## What I need from you

Please review the attached files and answer the following:

1. **Gate sanity check**: Was it reasonable to open only a constrained Step7Q GNN/RL operator-learning phase from the current Step7P evidence? If not, which criterion is too weak or missing?
2. **Root-cause diagnosis**: Given Branch C reduces overlap/soft/bbox but still has 0 strict meaningful winners, what is the most likely missing causal mechanism?
3. **Recommended Step7Q architecture**: Propose a concrete RL or GNN system architecture that is appropriate for this repo and these artifacts. It should be sidecar-only and should learn causal operator/source selection or local operator parameters, not direct final coordinates.
4. **Training data and labels**: Specify exactly what examples, graph nodes/edges/features, action space, labels/rewards, masks, and negative samples should be built from current artifacts.
5. **Experiment plan**: Give a phased, executable plan with gates, expected artifacts, and tests. Include kill gates and promotion gates.
6. **Minimal implementation slice**: Recommend the first 1-2 code files/scripts/tests to implement next, with exact artifact names and acceptance criteria.
7. **Risk review**: Identify what would cause another “原地踏步” loop and how to prevent it.

Desired output format:

- A concise verdict first.
- Then a concrete plan broken into phases.
- Include specific artifact names and metrics.
- Be explicit about whether to use GNN, RL, imitation/ranking, offline RL, or a hybrid.
- Be critical: if our `gnn_rl_gate_open=true` is premature, say so and give the corrected gate.

