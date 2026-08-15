# P11.4 Contact Gate D — 損之又損修正計畫

## 結論

目前 Gate D 是 `REJECT`，但不能直接解讀為 CCRL 無法泛化。固定模型在 held-out
的 Top-4 inverse recall 只有 `53.89%`，decoded exact success `61.82%`，grouping
recovery `61.99%`；同時 train 與 held-out 表現幾乎相同，表示模型尚未學好完整
training mapping，而不是典型 source overfit。

第一個應移除的混淆因素是 dynamic state 與 exact decoder 的 Contact truth 不同：
現有 replay 先把 placement normalize/cast 成 float32，再以 zero tolerance 重建
contacts；decoder 則使用原始 placement。全量檢查 held-out replay 時，contact edge
set 有 `93.2%` 不同，component partition 有 `89.1%` 不同。

第二個 correctness defect 是 `group_component_id` 在每個 group 都從 0 重新編號，
但 model 以全域相等判斷 `same_component` 與 component size，會製造跨 group 的假
component 關係。

## 範圍鎖定

只修上述兩個 correctness defects，然後重跑 Gate D。保持：

- 同一份 2,000 train / 512 held-out source manifest；
- 64 hidden、2 layers、4 heads、FFN 4x；
- 50,000 steps、learning rate `2e-3`、seed `5090`；
- C0/C1/C2、factorized inverse-action loss、Top-4 exact decode；
- 現有 hard-feasibility verifier 與 Gate D thresholds。

不加入新 expert、critic、router、DAgger、oracle relabeling、runtime integration 或
larger Transformer。

## 執行順序

### G0 — Exact Contact state parity

最小修正：`RepairState.placement` 繼續保存 normalized float32 continuous features，
但 `build_repair_state()` 可接收同一盤面的 exact/raw Contact placement，並只用它
推導 `contact_edges` 與 grouping components。不要以 tolerance 掩蓋 precision
問題，也不要修改全域 `normalize_xywh()`。

驗證：

1. 加一個 precision regression：normalized float32 會失去 zero-gap contact，但
   exact Contact placement 仍產生與 decoder 相同的 edges/components。
2. 載入固定 manifest 的既有 replay；用每筆已保存的 raw float64
   `decoder_placement` 重建離散 state。重建後的 contact edges 與 exact placement
   必須 `100%` 相同，target group component partition 也必須 `100%` 相同。

任一 parity 未達 100%，停止，不訓練。

### G1 — Collision-free component semantics

最小修正：component identity 必須包含 group identity，不能讓不同 group 的
`component 0` 被 model 視為同一 component。優先在現有一維表示中配置全域唯一
ID；只有實際資料證明 group membership 可重疊且一維表示失真時，才改為
obligation-local 表示。

驗證：建立兩個各自含 `component 0` 的 disconnected groups，要求：

- 跨 group `same_component == false`；
- component size 只計算同 group、同 component members；
- 同 group 內的 component 關係維持不變。

### G2 — Cache reuse and fixed rerun

保留 source manifest、train/held-out replay 與 generation files。既有 replay 已保存
normalized `state.placement` 與 raw float64 `decoder_placement`；在 loader 內以後者
重建 Contact edges/components，不重跑 corruption，也不改寫舊 cache。只有現有
record 無法導出 100% parity 時，才重新生成 cache。

訓練前先確認：cache SHA-256、source counts、source hashes、row counts、generation
failure buckets 都與舊 run 相同；任何 source split drift 都停止。新的 checkpoint 與
report 寫入新 output directory，不覆寫舊 run。

以完全相同模型與 training recipe 重跑，報告：

- train 與 held-out factorized NLL、Top-1/Top-4 inverse recall；
- held-out Top-4 decoded exact success 與 hard-feasibility；
- grouping recovery versus inverse；
- C0/C1/C2 分項；
- exact-state parity 與 component-collision assertions。

## 決策

- `KEEP`：held-out Top-4 recall `>= 80%`、decoded exact success `>= 99%`、grouping
  recovery `>= 90%`，且 correctness gates 全通過。
- `REJECT-generalization`：training mapping 已明顯學會，但 held-out 未達 gate。
- `REJECT-fit`：train 與 held-out 仍同步偏低；不要以 larger model 續命，先結束
  此 frozen Gate D 結論。
- `BLOCKED-correctness`：任一 exact parity、component semantics、split 或 hard
  feasibility check 失敗；不得把結果解讀成模型泛化證據。

OMP 只負責長時間 parity audit、必要時的 cache regeneration、training、evaluation
與 failure buckets；主線負責兩個 correctness fixes、targeted tests、artifact
contract 與最終 Gate 決策。
