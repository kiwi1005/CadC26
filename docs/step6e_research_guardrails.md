# Step6E Research Guardrails After E10

Date: 2026-04-25

New constraint from the research loop:

- Do not add hand-built/manual feature or delta adjustments.
- Do not specialize methods to individual cases or hard-case IDs.
- Do not tune fusion/loss weights as the main path.
- Future experiments must be case-agnostic learned architecture/objective changes and must validate with LOCO before widening.
- Keep Step6 runners in serial smoke mode by default; use `--workers 48` only for independent case/seed or LOCO split jobs after the relevant smoke gate.

Implementation guardrail applied:

- `scripts/run_step6e_majority_advantage_ranker.py` no longer exposes `action_delta_features` in `--ranker-input` choices.
- If an old invocation tries to use `action_delta_features`, the script fails closed with a clear error.
- Default `--case-ids` for that runner is now a neutral 5-case slice `[0, 1, 2, 3, 4]`, not the previous hard-case-only slice.

Historical artifacts from E10 remain preserved as negative evidence, but that route is deprecated and should not be extended.

Next valid direction:

- Learned relation-delta architecture with no hand-authored delta scalars and no per-case branching.
- Acceptance gate should compare against E6 (`LOCO rank 3.4583 / top1 0.2500`) on a neutral slice before any wider run.
