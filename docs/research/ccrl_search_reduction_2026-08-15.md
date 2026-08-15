# P11.5 — Decisive experiment：CCRL Top-K vs deterministic Contact search

日期：2026-08-15
Checkpoint（凍結）：`train_fixed64_gate_d_statefix/contact_gate_d.pt`（64×2×4，50k steps）
資料：`full_2k_512/heldout.replay.jsonl.gz`，分層抽樣 60/kind = 180 states（512 個未見 sources）
產物：`artifacts/experiments/p11_ccrl_contact_replay/p11_search_reduction/heldout_battle.json`

## Hypothesis

停止 error anatomy。單一問題：**模型推薦 Top-K 能否用大幅較少的 decode
保留 deterministic Contact enumeration 的修復能力？** 此答案決定 CCRL
繼續、重想或停止。

## Changed

- `scripts/experiment_ccrl_search_reduction.py`（新）。凍結 checkpoint、masks、
  decoder、replay cache。每個 state 兩臂看到**完全相同**的 mask 合法
  action set（平均 113 個 (target, anchor, side) triples × lazy budget ≈ 216
  個 decode 的全量掃描），唯一差別是排序：
  - deterministic：canonical index order（member → anchor → side）
  - ccrl：model 分數 Top-K（lazy budget escalation 規則兩臂相同）
- 量測：decode budget 1/2/4/8/16/32 的 success 與 recovered oracle
  grouping-gain、decodes-to-first-success、model forward 時間。

## Experiment

180 held-out states（C0/C1/C2 各 60），oracle = 全 action set 最佳 debt。
模型 forward 8 ms/state。全量 deterministic 掃描 = 平均 216 decodes。

## Result

```text
budget | deterministic        CCRL
       | succ    gain     |  succ    gain
     1 | 0.117   0.111    |  0.689   0.678
     2 | 0.189   0.181    |  0.894   0.867
     4 | 0.244   0.231    |  0.950   0.922
     8 | 0.361   0.339    |  0.983   0.953
    16 | 0.528   0.511    |  1.000   0.978
    32 | 0.694   0.678    |  1.000   0.983
```

- **CCRL Top-4 = 92.2% oracle gain，用 4 個 decode；deterministic 要
  ~14（中位）到 31（平均）個 decode 才到第一個成功，且到 32 個 decode
  也只有 67.8% gain。**
- Top-1 已 67.8% gain；Top-16 = 97.8%。
- 每 state 只需 8 ms 模型 forward 取代 O(100+) exact decodes 的排序成本。
- 分 kind：C0/C1 Top-8 即 100% success；C2 Top-4 = 85% gain、Top-8 = 90.8%
  gain、Top-16 = 93.3%——C2 仍是相對弱項但曲線仍在爬。

## Decision

**KEEP。** 研究主張成立：structured corruption 學到的 repair policy 以
~1/8～1/50 的 decode 量保留 ≥92% 的 deterministic contact repair 能力，
模型 forward 成本可忽略。這是 CCRL 作為「搜尋減量器」的直接證據，
不是 auxiliary-task 分數。

對 P11 後續的直接推論：

1. **不修 role-swap / anchor / mask**——Top-4 曲線已足夠，剩餘尾巴
   （4→16 之間的 5.6%）不值得在接進 loop 前處理。
2. **接入點確定**：`CCRL fast path → 失敗才 deterministic fallback`。
   Top-16 保證 success=100%，fallback 觸發率 <5%。
3. **停止 lab-metric 追高**（Top-4 exact recall、budget canonicalization
   等）；後續判準改為 frozen contact-loop QoR + decode budget。

## Next experiment

唯一 decisive 問題：**在真 solver state 上還成立嗎？** 目前測的是
corrupted clean placement（lab state）。下一步拿 train-split 的 P8
incumbent placements（未見於模型訓練、非 validation），跑同一個
battle 腳本量：
```text
solver-state CCRL Top-4 gain vs deterministic Top-N gain
```
若 solver state 上 Top-4 仍 ≥90% deterministic gain → 接進
contact-loop sidecar（HCFP_CHECKPOINT 式 opt-in）；
若顯著崩 → 分佈移轉問題（DAgger / solver-state corruption），屆時
error analysis 才有目的。

Observed but not blocking current experiment: C2 的 gain 曲線在 Top-8
後趨平（0.908→0.950），oracle 本身與 teacher 幾乎重合，剩餘差距是
C2 decoder 的 action-space 尾巴，不是排序問題。
