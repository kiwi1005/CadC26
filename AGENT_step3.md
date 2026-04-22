這份計劃的核心是：**把目前開發節奏從「修 pipeline / 調 repair」拉回「訓練是否真的帶來方法貢獻」的研究節奏。**

---

# Agent 指導計劃書：Training-first Research Recovery Plan

## 0. 任務定位

你現在接手的任務不是繼續修一般工程細節，也不是繼續調 repair 參數。

目前 Sprint 2 Pivot 已經證明：

* semantic candidate coverage = 1.0
* relaxed candidate coverage = 1.0
* semantic rollout completion rate = 1.0
* repair success rate = 0.8
* contest quick validate = true
* contest feasible cases = 4/5
* fallback fraction = 0.2

這代表 semantic-first 主鏈已經能跑，但目前仍缺少一個關鍵證據：

> 訓練出來的 policy / prior 是否真的改善了 floorplanning，而不是只是讓 pipeline 有 checkpoint 和報告。

因此本輪任務的主軸是：

**建立完整 training evidence，驗證 learned semantic prior 是否真的優於 heuristic / untrained / teacher-hint-only baseline，並判斷它對 downstream repair/finalizer 是否有實質幫助。**

原始技術藍圖本來就規劃：先做可驗證的 legalizer baseline，再接 ML initializer，最後才做 search / ensemble / second-stage ML。現在的狀態已經到了該檢查 ML initializer 是否真的有貢獻的節點。

---

## 1. 目前不要做的事

本輪禁止以下行為：

1. 不要繼續泛泛地「改善 repair」
2. 不要只調 hyperparameter 追求 smoke 指標變好
3. 不要改 scorer
4. 不要改 official / strict / public scoring rules
5. 不要擴張 B/C branch
6. 不要把 fallback fraction 下降當成唯一成果
7. 不要只產生 checkpoint 卻沒有 ablation
8. 不要只跑 validation 0–4 就宣稱 training 有效
9. 不要把 untrained / heuristic / teacher-hint baseline 混在一起比較
10. 不要同時改 rollout、repair、training loss、candidate ranking 四層

本輪研究問題要清楚：

> learned semantic prior 是否真的能改善候選品質，並讓 repair/finalizer 更容易產生 hard-feasible 或更低 fallback 的 layout？

---

## 2. 本輪核心假設

### Research Question

Semantic-first 主鏈已經打通，但目前 BC / AWBC 只完成最小接線。
我們需要知道：

> 訓練後的 semantic policy 是否比 heuristic、untrained model、teacher-hint-only baseline 更能產生可被 finalizer 修好的 floorplan？

### Hypothesis

若 BC / AWBC 的 semantic prior 真的有效，則它應該至少在以下其中一層產生改善：

1. proposal-level 改善

   * semantic action 命中率提升
   * priority / placement order 更接近 teacher
   * candidate ranking 更穩定

2. rollout-level 改善

   * placed fraction 維持 1.0
   * no-progress fallback 減少
   * semantic rollout 中的 violation profile 改善

3. downstream-level 改善

   * repair success rate 提升
   * fallback fraction 降低
   * strict candidate coverage 提升
   * contest feasible cases 增加
   * validation-1 dense case 從 fallback-required 變成 repaired-feasible

4. competition-level 改善

   * official-first score 改善
   * large-case feasibility 或 boundary / overlap failure 減少

---

## 3. Training Evidence 等級

請用下面等級定義訓練成果，不要把「有 checkpoint」當作訓練成功。

### T0：Plumbing Evidence

表示訓練流程能跑，但不能證明模型有效。

要求：

* train command 可以跑完
* checkpoint 產生
* loss 有記錄
* config / seed / dataset slice 有記錄
* report 產生

### T1：Learning Evidence

表示模型真的學到資料中的訊號。

要求至少滿足：

* 可以 overfit 小資料集，例如 16 或 32 cases
* training loss 明顯下降
* validation loss 或 action accuracy 有合理變化
* untrained model 與 trained model 有明確差距

### T2：Proposal Evidence

表示 learned prior 對 semantic proposal 有幫助。

要求比較：

* heuristic semantic baseline
* untrained model
* BC trained model
* AWBC trained model

觀察：

* semantic candidate quality
* action / priority accuracy
* teacher hint usage
* candidate ranking accuracy
* semantic rollout violation profile

### T3：Downstream Evidence

表示 learned prior 對 finalizer / repair 有幫助。

觀察：

* repair success rate
* contest feasible cases
* fallback fraction
* overlap failure count
* protected constraint violations
* strict candidate coverage

### T4：Generalization Evidence

表示方法有競賽價值。

觀察：

* validation 0–20
* dense / fallback cases
* 90–99 或 90–120 large-case slice
* official-first score
* feasibility rate
* runtime

本輪最低目標是達到 **T2 + 初步 T3**。
T4 可以作為 stretch goal。

---

## 4. 本輪工作分期

## Phase 0：Training Audit

目標：確認目前 BC / AWBC 到底跑到什麼程度。

請檢查並輸出：

1. 現有 training scripts
2. 現有 configs
3. 現有 checkpoints
4. 現有 reports
5. 是否真的跑完整 epoch
6. 是否只跑 smoke
7. 是否有 validation metrics
8. 是否有 untrained baseline
9. 是否有 heuristic baseline
10. 是否能 resume
11. 是否能固定 seed 重跑

請產出：

```text
artifacts/research/training_audit.md
```

內容必須包含：

* available training entrypoints
* dataset slice used
* number of epochs / steps actually run
* checkpoint list
* train / val metrics
* missing metrics
* whether current evidence is T0 / T1 / T2 / T3

如果發現目前 training 只是 smoke，請直接寫明。

---

## Phase 1：Small-data Overfit Test

目標：先證明模型真的能學。

### 固定資料

請建立一個小資料訓練 slice：

* train cases: 16 或 32
* validation cases: 8
* 固定 seed
* 固定 semantic candidate mode

### 必跑模型

至少跑：

1. BC semantic model
2. AWBC semantic model

### 必須比較

1. untrained model
2. heuristic semantic baseline
3. trained BC
4. trained AWBC

### 指標

至少記錄：

* train loss
* validation loss
* action accuracy 或 priority accuracy
* semantic candidate coverage
* semantic rollout completion
* semantic placed fraction
* violation profile
* checkpoint step
* runtime

### 成功條件

至少滿足：

* trained model 明顯優於 untrained model
* small train slice 可以 overfit
* rollout 不因 learned model 退化
* metrics 可重跑一致

### 失敗條件

若以下任一成立，請停止擴大訓練：

* loss 不下降
* trained 與 untrained 沒差
* trained model 造成 rollout 退化
* metrics 不可重現
* checkpoint 無法載入 downstream pipeline

請產出：

```text
artifacts/research/small_overfit_bc.json
artifacts/research/small_overfit_awbc.json
artifacts/research/small_overfit_summary.md
```

---

## Phase 2：Meaningful Training Run

目標：跑一個不只是 smoke 的正式訓練。

### 資料設定

請至少建立三個 slice：

1. small-dev

   * 32–64 cases
   * 用於快速驗證

2. medium-dev

   * 200–1000 cases
   * 用於確認模型真的能學泛化訊號

3. hard-dev

   * dense / fallback / repair-failed cases
   * 如果目前資料不足，先從 validation failure 和 internal train failure 中建立

### 模型

至少跑：

* BC semantic
* AWBC semantic

可選：

* BC + class weighting
* AWBC + failure-bucket weighting

### 必須固定

* commit
* config
* seed
* dataset slice
* candidate mode
* checkpoint path
* evaluation command

### 指標

訓練指標：

* train loss
* val loss
* action accuracy
* priority accuracy
* semantic candidate ranking quality
* teacher hint agreement

rollout 指標：

* semantic candidate coverage
* relaxed candidate coverage
* strict candidate coverage
* semantic rollout completion rate
* avg semantic placed fraction
* no-progress fallback count
* violation profile

downstream 指標：

* repair success rate
* contest feasible cases
* fallback fraction
* overlap failure count
* protected constraint violations
* runtime

請產出：

```text
artifacts/research/training_bc_semantic_medium.json
artifacts/research/training_awbc_semantic_medium.json
artifacts/research/training_run_summary.md
```

---

## Phase 3：Ablation：Training 是否真的有用

目標：回答「訓練是否帶來方法進展」。

### 必須比較的 variants

固定同一批 evaluation slice，比較：

1. heuristic-only semantic candidate
2. untrained model semantic prior
3. BC trained model
4. AWBC trained model
5. teacher-hint oracle 或 teacher-assisted upper bound，如果現有系統支援

### 固定 evaluation slice

至少包含：

* validation 0–4
* validation-1 dense case
* repair failed / fallback cases
* 若可行，加 validation 0–20
* 若可行，加 large-case 90–99

### 必須輸出 comparison table

欄位至少包含：

* variant
* semantic candidate coverage
* strict candidate coverage
* semantic rollout completion
* avg semantic placed fraction
* repair success rate
* contest feasible cases
* fallback fraction
* official / strict score
* overlap failure count
* boundary violations
* grouping violations
* MIB violations
* runtime

請產出：

```text
artifacts/research/semantic_training_ablation.json
artifacts/research/semantic_training_ablation.md
```

### 判斷規則

如果 BC / AWBC 無法優於 heuristic-only 或 untrained model，請不要繼續擴大訓練。

此時要回報：

* 是資料問題？
* 是 label 問題？
* 是 model capacity 問題？
* 是 candidate interface 問題？
* 是 finalizer 抹掉了 prior 的差異？
* 是 metric 沒有觀察到 prior 的作用？

---

## Phase 4：Finalizer Interaction Study

目標：判斷 training prior 是否被 repair/finalizer 吃掉。

這一階段非常重要，因為目前 repair/finalizer 很可能把不同 prior 的差異都壓平。

請對每個 variant 分別記錄：

1. before repair layout metrics
2. after shape normalize
3. after overlap resolve
4. after shelf fallback
5. after strict finalization
6. final official / strict result

### 必須回答

* trained model 在 repair 前是否更好？
* repair 後差異是否消失？
* fallback 是否讓所有 variant 變得一樣？
* dense case 失敗是 prior 問題還是 finalizer 問題？
* validation-1 需要 fallback 的原因是否和 prior 有關？

請產出：

```text
artifacts/research/finalizer_interaction_trace.json
artifacts/research/finalizer_interaction_summary.md
```

### 關鍵結論分類

最後必須把結果歸成以下其中一類：

#### 結論 A：Training 有效，finalizer 保留了差異

下一步可以擴大 training，並往 large-case / soft constraint 推。

#### 結論 B：Training 有效，但 finalizer 抹掉差異

下一步應先改 finalizer interface，讓 prior 影響 placement / reinsertion，而不是盲目擴大訓練。

#### 結論 C：Training 無效

下一步應回到 label、feature、loss 或 candidate generation，不要再跑大訓練。

#### 結論 D：Training 指標有效，但 downstream 無效

下一步要重定義 training objective，讓它對 repair success / feasibility / boundary 有關。

---

## Phase 5：Dense-case Research Experiment

只有在 Phase 3 / Phase 4 判斷完後，才開始這個階段。

目標：把目前 dense-case failure 拉成真正的方法研究。

### Research Question

Current semantic rollout can produce complete layouts, but dense cases still require fallback.
Does obstacle-aware constructive reinsertion improve dense-case hard-feasibility compared with post-hoc repair?

### Hypothesis

目前 finalizer 失敗的原因不是 candidate recall，而是 dense overlap cluster 沒有在 constructive stage 被合理重插。
如果加入 obstacle-aware reinsertion，validation-1 應該能從 fallback-required 變成 repaired-feasible。

### Scope

只允許修改：

* repair/finalizer
* minimal geometry helpers
* optional constructive reinsertion helper

禁止修改：

* scorer
* semantic rollout
* BC/AWBC training
* official evaluator
* dashboard contract

### Metrics

* validation-1 是否仍需要 fallback
* repair success rate
* contest feasible cases
* fallback fraction
* overlap failure count
* protected constraint violations
* runtime

### Success

* validation-1 no longer requires fallback
* contest feasible 4/5 → 5/5
* fallback fraction 0.2 → 0
* no new protected constraint violations
* runtime no material regression

### Kill Criteria

* validation-1 無改善
* 改善只來自更激進 shelf fallback
* protected constraints regression
* runtime unstable
* 必須同時修改 rollout 或 scorer 才能成立

請產出：

```text
artifacts/research/dense_case_reinsertion_experiment.json
artifacts/research/dense_case_reinsertion_summary.md
```

---

## 5. Agent 分工建議

若使用多 agent，請固定角色，不要全部都去改同一條 pipeline。

### Agent A：Training Auditor

任務：

* 檢查目前 training 是否只是 smoke
* 確認 checkpoint / config / metrics
* 補 training evidence classification

不可做：

* 修改 repair
* 修改 scorer
* 修改 rollout

交付：

```text
training_audit.md
```

---

### Agent B：Training Runner

任務：

* 跑 small overfit
* 跑 medium training
* 固定 seed / config / slice
* 產出 BC / AWBC training metrics

不可做：

* 改 solver
* 改 finalizer
* 改 scorer

交付：

```text
small_overfit_summary.md
training_run_summary.md
```

---

### Agent C：Ablation Analyst

任務：

* 比較 heuristic / untrained / BC / AWBC / teacher
* 產出 proposal、rollout、downstream 指標
* 判斷 training 是否有效

不可做：

* 為了讓結果好看去改模型或 repair

交付：

```text
semantic_training_ablation.md
```

---

### Agent D：Finalizer Interaction Analyst

任務：

* 做 before/after repair trace
* 判斷 prior 是否被 finalizer 抹掉
* 定位 dense failure 是 prior 問題還是 finalizer 問題

交付：

```text
finalizer_interaction_summary.md
```

---

### Agent E：Method Experimenter

只在前面結果清楚後啟動。

任務：

* obstacle-aware constructive reinsertion
* dense-case failure experiment

不可做：

* 大範圍調參
* 動 scorer
* 動 training

交付：

```text
dense_case_reinsertion_summary.md
```

---

## 6. 統一執行原則

所有 agent 都必須遵守：

1. 每個任務先寫 research question
2. 每個任務只改一個主機制
3. 每個任務必須有 baseline comparison
4. 每個任務必須有 kill criteria
5. 每個任務必須產出 artifact
6. 不允許只回報「指標變好」
7. 必須解釋變好的機制
8. 必須保留失敗案例
9. 不允許修改 scorer 來讓結果變好
10. 不允許把 fallback 當成方法貢獻

---

## 7. 下一輪最小可執行任務

如果只能先派一個 agent，請派這個：

```text
Task: Training Evidence Audit and Small Overfit Verification

Context:
Sprint 2 semantic-first pivot has established semantic candidate coverage = 1.0 and semantic rollout completion = 1.0 on validation 0–4. BC and AWBC currently support candidate_mode=semantic and produce checkpoints/reports, but it is unclear whether training was run beyond smoke level or whether learned priors improve downstream floorplanning.

Research Question:
Does the current BC/AWBC semantic training setup actually learn a useful prior, or is the current improvement mostly from semantic candidate design and repair/finalizer?

Hypothesis:
If the training setup is valid, BC/AWBC should overfit a small fixed slice and outperform untrained / heuristic baselines on proposal-level and rollout-level metrics. If it cannot overfit or downstream metrics do not change, then training is not yet contributing meaningful method progress.

Scope:
- Audit existing training entrypoints, configs, checkpoints, and reports.
- Run small-data overfit for BC semantic and AWBC semantic.
- Compare against heuristic semantic baseline and untrained model.
- Do not modify scorer.
- Do not modify repair/finalizer.
- Do not modify semantic rollout.
- Do not optimize soft constraints in this task.

Fixed evaluation:
- Train slice: 16 or 32 cases
- Validation slice: 8 cases
- Rollout smoke: validation 0–4
- Dense/fallback case: validation-1

Metrics:
- train loss
- validation loss
- action / priority accuracy
- semantic candidate coverage
- strict candidate coverage
- semantic rollout completion
- avg semantic placed fraction
- violation profile
- repair success rate
- fallback fraction
- contest feasible cases
- runtime

Success Criteria:
- Training runs are reproducible with fixed seed/config/slice.
- Trained BC/AWBC clearly outperform untrained model on training slice.
- At least one trained model improves proposal-level or rollout-level metrics over heuristic or untrained baseline.
- Downstream metrics are reported even if they do not improve.
- Artifacts clearly classify evidence level as T0/T1/T2/T3.

Kill Criteria:
- Training cannot overfit small slice.
- Trained model is indistinguishable from untrained model.
- Checkpoint cannot be loaded into rollout path.
- Metrics are not reproducible.
- Any improvement requires changing scorer, rollout, or repair.

Deliverables:
- artifacts/research/training_audit.md
- artifacts/research/small_overfit_bc.json
- artifacts/research/small_overfit_awbc.json
- artifacts/research/small_overfit_summary.md
- artifacts/research/semantic_training_ablation_initial.md

Final report must answer:
1. Was current training only smoke-level?
2. Can BC/AWBC overfit a small slice?
3. Does trained prior beat untrained / heuristic proposal?
4. Does trained prior survive downstream repair/finalizer?
5. Should we invest in larger training, change labels/loss, or focus on finalizer?
```

---

## 8. 最終判斷

目前你看到的問題是對的：**agent 可能沒有真正跑完整 training，或者跑了也沒有證明 training 對方法有用。**

所以這份計劃的重點不是叫 agent「多訓練一點」，而是要求它回答：

> 訓練是否真的產生了可用 prior？
> 這個 prior 是否能改善 rollout？
> 這個 prior 是否能通過 finalizer 保留下來？
> 如果不能，問題在 training、label、feature、candidate interface，還是 finalizer？

只有這些問題被回答後，才值得進入大訓練、large-case generalization、soft-constraint learning 或 ensemble。
