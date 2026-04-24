# Step6F Matched Neutral5 Typed-Graph Baseline Status

Date: 2026-04-25 01:22 Asia/Taipei  
Worker: `worker-2`  
Task: matched neutral5 typed-graph baseline refresh / 48-core usage policy evidence

## Requested baseline

Refresh the matched neutral5 typed-graph majority-advantage baseline under the E13 protocol:

```bash
PYTHONPATH=src /home/hwchen/PROJ/CadC26/.venv/bin/python \
  scripts/run_step6e_majority_advantage_ranker.py \
  --case-ids 0 1 2 3 4 \
  --policy-seeds 0 1 \
  --encoder-kind typed_constraint_graph \
  --ranker-input candidate_features \
  --objective-kind majority_pairwise \
  --workers 1 \
  --output artifacts/research/step6e_typed_graph_majority_advantage_neutral5_loco.json
```

`--workers 1` is the required smoke/default mode after the worker-policy audit. A later `--workers 48` rerun would be valid only after this smoke/import gate succeeds, because the independent work would be the 10 case/seed collection jobs plus 5 LOCO split jobs.

## Worker-time blocker

The worker task could not start in its worktree because the Step6E runner imported `canonical_action_key` from `puzzleplace.actions`, but that worktree did not export the symbol yet.

Attempted command:

```bash
PYTHONPATH=src /home/hwchen/PROJ/CadC26/.venv/bin/python \
  scripts/run_step6e_majority_advantage_ranker.py --help
```

Observed worker-time failure:

```text
ImportError: cannot import name 'canonical_action_key' from 'puzzleplace.actions' (.../src/puzzleplace/actions/__init__.py)
```

Leader follow-up fixed this import/export seam at commit `a23fa5d`; `tests/test_hierarchical_policy.py` now collects and passes in the targeted suite. This artifact still records that the matched neutral5 baseline itself was not rerun during task 6.

## Existing evidence only, not a fresh refresh

`artifacts/research/step6e_experiment_summary.{json,md}` records E13 neutral5 metrics:

| baseline | cases | seeds | pools | micro rank | micro top1 | LOCO rank | LOCO top1 | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E13 typed-graph majority advantage neutral5 | 5 | 2 | 40 | `1.9750` | `0.4500` | `4.0500` | `0.1500` | false |

However, the referenced primary artifact path `artifacts/research/step6e_typed_graph_majority_advantage_neutral5_loco.json` is not present in this worktree. Therefore this lane cannot claim a refreshed matched baseline artifact yet.

## Baseline labeling rule

- E13 neutral5 remains the primary comparable baseline recorded by summary evidence.
- E6 (`LOCO rank 3.4583 / top1 0.2500`) must remain historical hard-slice evidence until rerun under the same neutral5 protocol.

## Next unblock action

Rerun the smoke command above with `--workers 1` on the leader checkpoint. Only after it produces the matched neutral5 artifact should we launch the wider independent collection/LOCO jobs with `--workers 48`.
