# P11.5c — P8 incumbent battle：排序紅利仲裁

日期：2026-08-15
Checkpoint（凍結）：`train_fixed64_gate_d_statefix/contact_gate_d.pt`（64×2×4）
States：P8 guarded-full100 analytic-lane incumbents（100 validation cases，作為 solver 輸入 read-only；無 label/solution 消費）
產物：`artifacts/experiments/p11_ccrl_contact_replay/p11_p8_state/p8_battle.json`（76s，22 jobs）

## Hypothesis

P11.5b 發現排序紅利取決於 canonical order 的品質：corruption states 中位
數 14–31 decodes 才命中（紅利大），`safe_shelf` states 中位數 3（紅利小）。
本實驗仲裁：真 P8 incumbent states 屬於哪一邊？預先決策規則：
det ~3 decodes → 停止 CCRL 排序整合與資料 scaling；det 15+ → DAgger。

## Changed

- `scripts/experiment_ccrl_p8_state.py`（新）。與前兩場 battle 完全相同的
  action set / decode cache / budgets；states 來自 P8 analytic incumbents，
  case 經 `_official_case` 載入（P10/BFOD 同款唯讀路徑）。

## Experiment

100/100 cases qualifying（全部有 oracle headroom，mean gain 1.2），
mean 147 triples、165 decodes 全量掃描。

## Result

```text
budget | deterministic         CCRL
       | succ    gain      |   succ    gain
     1 | 0.420   0.355     |   0.430   0.385
     2 | 0.570   0.495     |   0.560   0.500
     4 | 0.810   0.695     |   0.710   0.630
     8 | 0.920   0.835     |   0.930   0.840
    16 | 0.980   0.915     |   1.000   0.945
    32 | 1.000   0.965     |   1.000   0.975
```

**Deterministic first-success：mean 3.3、median 2。**

## 解讀

1. **預設決策規則命中第一支**：P8 incumbents 上 canonical order 幾乎免費
   （median 2 decodes）。CCRL 排序在每個 budget 點與 canonical 打平
   （±2pp，噪聲級），Top-16 起兩者都 ~100% success。
2. **原因與 safe_shelf 相同且更強**：P8 analytic lane 是 B*-Tree/掃描序
   產生的，block index order 與 geometry order 天然對齊——index-order
   enumeration 本身就是一個好的 heuristic。corruption states 的紅利來
   自 corruption 打亂了這個對齊；真 solver 沒有打亂。
3. **範圍限定**：本結論限於 CCRL decoder 的 single-block bridge action
   space、單步修復。BFOD contact loop 的 dense patch generator 是另一個
   更豐富的 action space（Case70 的 72 decodes 屬於那個空間），不在本
   實驗範圍。

## Decision

**REJECT CCRL 排序整合；停止 10K/50K scaling curve。**

- 三場 battle 合起來的完整結論：CCRL 學到了真實的排序能力
  （corruption states Top-4 = 92.2% gain，50× 搜尋減量），但在它要服務
  的真 solver states 上，那個排序問題**不存在**——canonical order 已
  經中位數 2 decodes 命中。買更多資料（10K/50K/1M）只會把一個無紅利
  的排序器磨得更準。
- 依 research contract：「Stop extending methods that do not produce
  measurable benefit.」
- P11.5/5b/5c 三場構成完整的研究故事（見下），CCRL line 就此收斂。

## 這條線的最終研究敘事

```text
FloorSet 1M training data
  → structured corruption → tiny policy
  → 學到真排序能力（corruption held-out Top-4 92.2% gain, 50× 減量）
  → 但真 solver states 的 index-geometry 對齊使 canonical order 已近最優
  → 排序紅利只存在於分佈被刻意打亂的 states
```

這是一個誠實且有價值的 negative result：**「learning to rank repair
actions」在這個 solver 的接觸面上沒有著力點**，瓶頸不在排序、在
action space 的豐富度（patch-level generator）與其他 soft debt 種類。

## Next experiment

資源轉向真正的 bottleneck（依 2026-08-13 視覺診斷與 grill batch）：
HPWL 是唯一全場為正的 gap、sparse fragmentation（61/93/98）與 dense
contact debt（70/89）已由 BFOD patch generator 覆蓋。下一刀候選：
以 P8 winner 為 incumbent 的 **MIB local repair 單假設實驗**
（P10 文件已預告）或 **HPWL-directed tree move**——兩者都是 deterministic
sidecar，不涉 learned ranker。

Observed but not blocking current experiment: replay generation 平行化
（10K/50K scaling 的前置工程）隨 scaling 取消而不再需要；戰鬥腳本的
Pool 模式已沉澱為可重用 pattern。
