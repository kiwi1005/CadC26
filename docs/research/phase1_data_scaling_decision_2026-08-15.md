# Phase 1 資料 scaling 決策 — pool 100K / 10K steps

日期：2026-08-15
方案：`model_first_e2e_floorplanning_plan_2026-08-15.md` §7 in-flight 項
Checkpoint：`hcfp5090-q2-structure-large-s10000-pool100k-seed6501.pt`
（parent 同 incumbent `hcfp5090-q2-constraints-s1000-seed5090.pt`；structure stage；
lr 3e-5；pool 100K、10K steps、10K unique samples；seed 6501）

## Hypothesis

Direct-generation 大型 structure lane 的 incumbent 只消費了 3K samples
（1M 的 0.3%）。把 sample pool 拉到 100K、steps 拉到 10K，large15 exact
應優於 incumbent 的 `8.822391 / 8-of-15 uncapped`。

## Experiment

凍結的 large15 exact 命令（與 incumbent 評測完全同款：topology 16 /
constraint 16 seeds、flow off、collective off、execution seed 6501）。

## Result

| | pool 3K / 3K steps（incumbent） | **pool 100K / 10K steps** |
| --- | ---: | ---: |
| Weighted capped cost | 8.822391 | **8.720192** |
| Uncapped cases | 8 / 15 | **10 / 15** |
| Hard feasible | 15/15 | 15/15 |

Per-case：7 better、5 worse、3 same。Uncapped 集合改變：
`+{86, 88, 93, 98}`、`−{92, 99}`（87/92/96/97/99 退步，86/88/93/98 穿 cap）。

Per-case churn 顯著：10× 資料不是均勻改善，而是移動每個 case 的 seed
抽樣命運。研究判準（weighted + uncapped count）改善；production
promotion 仍需 Pareto guard，不在本實驗範圍。

## Decision

**KEEP（作為 G1 起點 checkpoint）。** 資料紅利存在且方向正確：
0.3% → 1% 的 1M 消費量即得 −0.102 weighted、+2 uncapped。曲線尚未
飽和的證據不足（單點），但 G1 gate 用這個 checkpoint 作為
direct-generation 基礎已是更強的起點。

## Next experiment

依主線方案 §7：實作 G1 Direct Generation Gate（anchor-only masked state
→ StructureHeads → K=4 one-shot decode → exact score，106–120 主桶）。
訓練吞吐已修復（125→25 ms/step，4.8×；commit `597a018`），本輪訓練
是用舊 code 跑的（1650 s），之後同規模訓練約 ~550 s。

Observed but not blocking current experiment: case 97 退步 +2.376 為
最大單案 regression；若 G1 之後做 per-case 歸因，97 是首選對象。
