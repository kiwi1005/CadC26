# Oracle review request: Step7L is complete but negative; plan the next architecture

You are reviewing `/home/hwchen/PROJ/CadC26`, a Python research repo for ICCAD/FloorSet-style macro placement experiments. Please act as an external research architect, not as an implementation worker. The user asked for a follow-up plan after Step7L completed.

## Project constraints

- This repo uses sidecar research pipelines. Do **not** propose contest runtime or finalizer mutations unless explicitly justified as a later, gated integration step.
- Use `PYTHONPATH=src .venv/bin/python` for local commands.
- `external/FloorSet/` is a read-only official source/evaluator/data checkout.
- Validation/test labels must not train request generators. They can be used only during replay/evaluation.
- Prior evidence gates matter: fresh replay labels, overlap-after-splice, unique signature counts, archive unique totals, HPWL/bbox/soft regressions, and case025-excluded winners. Raw win counts alone are misleading.
- Current high-level diagnosis before Step7L: boundary/legalization is no longer the blocker; target direction / topology-demand / objective alignment is the blocker.

## What just happened: Step7L completion

Step7L was built as a deterministic learning-guided topology/terminal target-prior lane. It is now complete for its deterministic-prior scope.

### Phase 0 data-boundary audit

- Decision: `promote_to_step7l_heatmap_baseline`
- Validation leakage count: `0`
- It separated:
  - FloorSet training `fp_sol` labels for supervised layout/target priors.
  - Visible validation inputs for request generation only.
  - Step7 sidecar replay labels for quality/ranker diagnostics only.

### Phase 1 deterministic heatmap baseline

- Decision: `promote_heatmap_to_candidate_request_dry_run`
- Generated `15136` heatmap examples from `256` training samples.
- Key metric: topology recall@5 = `0.152154`.
- Topology beat center baseline but did not prove objective usefulness.

### Phase 2 request deck

- Decision: `promote_to_repacker_interface_design_not_replay`
- Artifact: `artifacts/research/step7l_phase2_candidate_requests.jsonl`
- Cases: `19,24,25,51,76,79,91,99`
- Request count: `296`
- Unique request signatures: `296`
- Request families:
  - `original_anchor`: 8
  - `topology`: 96
  - `terminal`: 96
  - `union_diversified`: 96
- `uses_validation_target_labels: false`

### Phase 2 replay bridge

A sidecar single-block obstacle-aware target-window replay bridge consumed the request deck. It used validation labels only to reconstruct baseline geometry and evaluate the replay; it did not change runtime/finalizer code.

- Decision: `complete_step7l_deterministic_prior_and_defer_gnn_rl`
- Generated candidate count: `288`
- Fresh hard-feasible non-noop: `256`
- Full-case overlap after splice: `0`
- Fresh official-like improving: `0`
- Fresh quality-gate pass: `0`
- Metric regressions: `256`
- HPWL regressions: `256`
- BBox regressions: `256`
- Soft regressions: `256`
- case025-outside winners: `0`
- Phase 3 GNN gate open: `false`
- Phase 4 offline-RL gate open: `false`

Interpretation from the local run:

> Step7L proved the heatmap requests are interface-complete and often hard-feasible after a sidecar single-block obstacle-aware bridge, but they produce zero fresh official-like improving / gate-pass candidates and regress HPWL/bbox/soft metrics. Therefore Step7L is complete for the deterministic-prior scope; GNN/RL gates stay closed until a cost/soft/HPWL-constrained target prior or larger replay corpus exists. The next bottleneck is objective-aligned target direction, not legality or replay plumbing.

## Prior relevant evidence

Earlier Step7S work showed:

- Boundary snap and soft-boundary signals can preserve known wins but collapse to case025-heavy / single-signature evidence.
- Larger boundary/soft-boundary corpora preserve legality but do not transfer; non-case winners remain absent.
- Eligibility gating can avoid regressions but collapses diversity.
- Oracle's earlier review recommended an opportunity atlas, HPWL-corridor constrained soft-boundary generator, and optionally training-guided source/target selector.

## Exact question

Given the negative but informative Step7L completion above, propose the next research/engineering plan.

Please do **not** simply say “train a bigger GNN” or “do RL now.” The local evidence gate closed those branches. Instead, design a concrete next system that directly targets objective-aligned target direction.

## Desired output

Return a detailed but actionable plan with these sections:

1. **Diagnosis update**
   - What did Step7L actually prove?
   - What did it falsify?
   - Which earlier hypotheses should be killed or demoted?

2. **Recommended next architecture**
   - Should the next step be Step7M/Step7L2/Step7S-X style? Please name it.
   - Define the system components and data flow.
   - Explain how it constrains HPWL, bbox, and soft violations **before** replay, without creating scalar penalty soup.
   - Explain how it should use topology/terminal heatmaps, if at all, after Step7L's failure.

3. **Experiment plan**
   - 3 to 6 ordered phases/tasks.
   - For each phase: goal, files/modules to add/edit, artifacts to write, smoke commands, metrics, promotion/kill gates.
   - Keep it sidecar-first and CPU-friendly.
   - Include a small first experiment that can run on the 8 focus cases.

4. **Evaluation contract**
   - Exact metrics to report, including case025-excluded accounting.
   - How to avoid raw-win inflation and leakage.
   - What counts as a real promotion signal.

5. **Possible model/RL/GNN usage later**
   - Under what concrete data conditions should GNN or offline RL reopen?
   - What minimal model would be justified first, if any?
   - What should remain deterministic?

6. **Implementation sketch**
   - Suggested file names in this repo.
   - Key dataclasses / schemas.
   - Pseudocode for the first objective-aligned target generator.

7. **Risks and kill criteria**
   - List likely failure modes and how to detect them quickly.

Please ground the plan in floorplanning / placement literature ideas where useful: analytical placement separation of global placement vs legalization, HPWL/density/legalization tradeoffs, graph placement caution, mask/action priors, and offline RL data requirements. Keep citations or paper names concise; I mostly need a practical repo plan.
