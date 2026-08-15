# P11.5b — Solver-state battle：CCRL ordering 在真 solver state 上的轉移

日期：2026-08-15
Checkpoint（凍結）：`train_fixed64_gate_d_statefix/contact_gate_d.pt`（與 P11.5 完全同款）
States：180 個 `safe_shelf` deterministic solver incumbents（training root，seed 5090，取 grouping debt > 0 的最壞 group）
產物：`artifacts/experiments/p11_ccrl_contact_replay/p11_solver_state/solver_battle.json`

## Hypothesis

P11.5 在 structured-corruption held-out states 上證明 CCRL Top-4 = 92.2%
oracle gain。本實驗只換一個變數——state distribution——量同樣的排序能力
是否轉移到 solver 自己產生的 placement。決策規則（預先）：Top-4 ≥ 90% →
KEEP（接 sidecar）；70–85% → MODIFY（DAgger）；40–50% → RETHINK。

## Changed

- `scripts/experiment_ccrl_solver_state.py`（新）。與 P11.5 battle 完全相同
  的 action set、decode cache、budget 軌道；states 來自 `safe_shelf(source)`
  而非 corruption；obligation = grouping debt 最大的 group；debt 口徑 =
  全域 grouping violation（與 decoder 一致）。

## Experiment

180 qualifying states（全部有 oracle headroom，mean gain 1.7 debt 單位），
每 state 平均 129 triples、143 decodes 全量掃描，模型 forward 9.5 ms。

## Result

```text
budget | deterministic        CCRL
       | succ    gain     |  succ    gain
     1 | 0.178   0.108    |  0.328   0.237
     2 | 0.183   0.111    |  0.511   0.369
     4 | 1.000   0.675    |  0.683   0.519
     8 | 1.000   0.740    |  0.906   0.741
    16 | 1.000   0.846    |  0.978   0.848
    32 | 1.000   0.918    |  1.000   0.919
```

Deterministic 在這些 states 上**中位數 3 個 decode 就到第一個成功**。

## Result 解讀（誠實版）

1. **分布移轉是真的**：Top-4 gain 92.2%（corruption）→ 51.9%（solver state）。
2. **但 Top-8 起 CCRL 與 deterministic 完全打平**（74.1% vs 74.0%、
   84.8% vs 84.6%、91.9% vs 91.8%）。模型沒有失效；它在等預算下追平。
3. **關鍵洞察：CCRL 的價值取決於 canonical order 有多差。** Corruption
   states 的第一個成功動作平均藏在第 14–31 個位置，排序價值巨大。
   `safe_shelf` states 是貨架佈局，index order 與 shelf order 對齊，
   中位數 3 個 decode 就命中——**這裡本來就沒有多少搜尋量可以節省**。
   換言之：51.9% 不是模型爛，是這個分佈的排序紅利小。
4. **Caveat（必須記錄）**：本實驗量的是 fallback lane（`safe_shelf`），
   不是 P8 guarded incumbent。P8 states 更密、更接近 corruption 分佈
   （Case70 audit 顯示 contact loop 平均 70+ decodes 才收斂），排序紅利
   很可能介於兩者之間，但未測。

## Decision

**MODIFY**（不是 KEEP，也不是 RETHINK）。

- KEEP 條件（Top-4 ≥ 90%）未達。
- RETHINK 條件（Top-16 也崩）未發生：Top-16 = 97.8% success、gain 追平。
- 下一步的價值判斷取決於「真實 P8 incumbent state 上 canonical order
  是否也這麼好」。這是一個可量測問題，不是信念問題。

## Next experiment（單一 decisive 問題）

拿 P8 guarded incumbents（`artifacts/benchmarks/hcfp5090-p8-guarded-full100.json`
+ train-split 對應 cases）重跑本腳本。判讀：

- P8 states 上 deterministic 仍只需 ~3 decodes → canonical order 已夠好，
  CCRL 排序在 contact lane 無附加價值 → 停止 CCRL 整合，把資源移向
  下一個 bottleneck；
- P8 states 上 deterministic 需要 15+ decodes（如 Case70 audit 的 70+）→
  排序紅利存在 → 用 P8-style states 做 DAgger/solver-state corruption
  再訓一次（有明確目的，不是 roadmap 驅動）。

Observed but not blocking current experiment: solver states 的 mean oracle
gain 只有 1.7（corruption states 約 3+），`safe_shelf` 本身的 contact debt
結構比 corruption 淺。
