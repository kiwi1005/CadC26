# Step6D Experiment Summary

- status: `complete_with_gate_blocker`

## Aggregate Results

| run | mode | cases | pools | ranker | target | pair aux | mean rank | regret | top1 | gate | zero-top1 |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `lane13_loco5` | `leave_one_case_out` | 5 | 120 | `scalar` | `oracle_ce` | `None` | 5.6833 | 0.9444 | 0.1667 | `False` | `[2]` |
| `lane13_loco10` | `leave_one_case_out` | 10 | 240 | `scalar` | `oracle_ce` | `None` | 6.2542 | 1.1657 | 0.1042 | `False` | `[0, 3, 5]` |
| `target_soft_loco5` | `leave_one_case_out` | 5 | 120 | `scalar` | `soft_quality` | `None` | 6.4750 | 1.0369 | 0.1667 | `False` | `[2]` |
| `rel_action_q_micro5` | `micro_overfit` | 5 | 120 | `relational_action_q` | `soft_quality` | `0.0` | 2.9917 | 0.2100 | 0.3000 | `None` | `[]` |
| `rel_action_q_loco5` | `leave_one_case_out` | 5 | 120 | `relational_action_q` | `soft_quality` | `0.0` | 4.8083 | 0.4905 | 0.2167 | `False` | `[]` |
| `rel_action_q_loco10` | `leave_one_case_out` | 10 | 240 | `relational_action_q` | `soft_quality` | `0.0` | 5.4250 | 0.8632 | 0.1458 | `False` | `[1, 4, 6]` |

## Key Comparisons

- `rel_action_q_vs_lane13_5case`: rank delta `-0.8750`, regret delta `-0.4539`, top1 delta `0.0500`
- `rel_action_q_vs_lane13_10case`: rank delta `-0.8292`, regret delta `-0.3025`, top1 delta `0.0417`
- `target_soft_vs_lane13_5case`: rank delta `0.7917`, regret delta `0.0925`, top1 delta `0.0000`

## Interpretation

- Target-only soft_quality on scalar ranker worsened 5-case rank/regret, so target brittleness alone is not the main blocker.
- Relational action-Q with soft_quality and no pairwise auxiliary improved both 5-case and 10-case over Lane 13 on all three aggregate metrics, but still failed the LOCO gate.
- Pairwise auxiliary loss hurt same-case micro-overfit in the first implementation, so it should remain disabled until pairwise target calibration is diagnosed.
- Remaining blocker is transfer/top1 reliability: 10-case still has zero-top1 held-out cases 1, 4, and 6.

## Next Gate

- Do not tune fusion weights.
- Keep relational action-Q as the current best direction, but diagnose zero-top1 cases before widening architecture.
- Next experiment: rollout-return label diagnostic for validation `1`, `4`, and `6`, plus pairwise auxiliary calibration before re-enabling pair loss.
