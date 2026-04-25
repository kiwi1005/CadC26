# Step6P: Pareto Alternative Selector

依照新版 `AGENT.md`，Step6P 先做 sidecar Pareto ranking，不接 runtime、不上 full NSGA-II。

## Plan

1. **Input**：沿用 Step6M move alternatives (`step6m_move_library_eval.json`) 與 Step6O guarded baseline，避免重跑昂貴 search。
2. **Applicability filter**：先移除 no-effect 與角色不適用 moves：MIB move 只給 MIB target、group move 只給 grouping target、soft shape 只給 non-fixed/non-preplaced target、boundary/local moves 只給 attribution-related target。
3. **Original-inclusive candidate set**：每個 case 加入 `original`，其 objective deltas 與 disruption 都是 0。
4. **Hard legality filter**：hard infeasible、frame protrusion、fixed/preplaced/area invalid 不進 Pareto set。
5. **Objectives**：minimize boundary violation delta (`-boundary_delta`)、normalized HPWL delta、normalized bbox delta、disruption cost。
6. **Output**：保留完整 Pareto front，另輸出 `min_disruption`、`closest_to_ideal`、`best_boundary`、`best_hpwl` 代表點；代表點只用來比較，不寫死 runtime rule。

## Gate

- `original` 可以自然勝出。
- simple compaction 必須在 Pareto 上合理才保留。
- boundary edge reassign 等低風險 alternatives 不被 lexicographic selector 隱藏。
- no-effect move ratio 必須在 applicability filter 後下降。
- 產出 `step6p_*` artifacts，且不修改 contest runtime。
