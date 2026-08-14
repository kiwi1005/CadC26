# CCRL：Constraint-Conditioned Corruption-and-Repair Learning

日期：2026-08-14  
分支：`research/ccrl-structured-repair`  
基準：`main@ce3a2ca3b4d10091da1fe4895a802723c05da7e1`

## 1. 結論

這個提案**可行**，而且比目前只對已生成 contact candidates 排名的 32-wide ranker 更充分利用 FloorSet-Lite 資料。

不過，第一版不應直接做成四個完整模型串接，也不應要求模型還原唯一的 `(x,y,w,h)`。建議收斂成：

> **官方 clean placement → 結構化 corruption → shared dynamic graph encoder → factorized expert action heads → deterministic decoder → exact verifier → cooperative repair loop。**

可行性分級：

| 部分 | 可行性 | 原因 |
| --- | --- | --- |
| Contact structured repair | 高 | deterministic Contact Loop 已有五案 QoR 證據，可直接充當 teacher/oracle |
| Boundary structured repair | 中高 | 官方直接給 boundary bitmask，且已有 deterministic boundary skeleton decoder；仍需解 side-order 多解問題 |
| MIB structured repair | 中高 | shape corruption 容易定義，已有 local MIB decoder；需處理 shape 與 local packing 的耦合 |
| Topology structured repair | 中 | 有 `tree_sol` 與 B*-Tree decoder，但 tree serialization 多解，不能只學唯一 edge target |
| Joint multi-expert loop | 中 | shared state 和 exact loop 可行；router、expert conflict、runtime 需分階段驗證 |

第一個正式 milestone 應只做 **Contact CCRL**。Boundary、MIB、Topology 必須先各自通過 deterministic corruption-repair oracle gate，才加入 shared model。

## 2. 提案中需要修正的假設

### 2.1 FloorSet-Lite 已有 clean placement

目前 `src/hcfp/floorset_lite.py` 會從官方 payload 讀入：

- `fp_sol`：clean placement；
- `metrics_sol`：baseline area / HPWL；
- `tree_sol`：B*-Tree edge labels。

所以第一版不需要先靠 solver 建立全部 Clean Placement Pool。官方 `fp_sol` 可作為主要 clean source，P7/P8/contact-loop winners 則作為 later-stage augmentation。

仍需做 per-expert eligibility：

- Contact sample：被選中的 group 在 clean placement 必須 connected；
- Boundary sample：被選中的 boundary obligation 在 clean placement 必須滿足；
- MIB sample：被選中的 MIB group 必須 shape-uniform；
- Topology sample：`tree_sol` 必須能通過 B*-Tree parser。

### 2.2 「拿掉 block」不能真的刪除 block metadata

模型仍必須看到 area、net、group、MIB、boundary、fixed/preplaced 等 metadata。Corruption 應改變 geometry visibility 或 relation，而不是從 case 中移除 node。

建議狀態：

```text
node exists = true
geometry_observed = false / corrupted
repair_target = true
```

Preplaced 永遠不能被 corruption 移動。Fixed-shape 可以移動，但 `(w,h)` 必須固定。

### 2.3 不學唯一原始座標

Floorplanning 有大量等價解。把 gold `(x,y,w,h)` 當唯一答案，會讓模型背座標與 serialization。

CCRL 應學共同 action language：

```text
expert_kind
obligation_id
target block / subtree
anchor block / side / group
relation
shape specification
patch budget
```

最終 geometry 由 deterministic decoder 產生，再由 exact verifier 與 QoR critic 判定。

### 2.4 不先訓練 router

目前 BFOD 已經證明錯誤 routing 會吃掉有限 decode budget。第一版 router 應是 deterministic residual-debt routing：

```text
disconnected group → contact
boundary miss → boundary
MIB shape debt → MIB
invalid or weak topology → topology
```

只有當各 expert 都能在同一 state 上產生可比較 outcomes，才用 `best Δcost by expert` 訓練 router。

## 3. 建議模型架構

## 3.1 新增 RepairState，而不是改 FloorplanCase

`FloorplanCase` 保留官方 immutable input。新增 dynamic state：

```python
@dataclass(frozen=True)
class RepairState:
    case: FloorplanCase
    placement: Tensor              # [N,4]
    geometry_observed: Tensor      # [N]
    repair_target: Tensor          # [N]
    mobility_mask: Tensor          # [N]
    shape_mobility_mask: Tensor    # [N]
    current_contact: Tensor        # sparse edge table
    group_component_id: Tensor     # [N]
    boundary_missing: Tensor       # [N,4]
    mib_shape_class: Tensor        # [N]
    round_index: int
    corruption_kind: str | None
    corruption_level: int
```

不把 mutable placement 塞進 `FloorplanCase`，可避免污染官方 input adapter、checkpoint 與現有 runtime。

## 3.2 Shared Dynamic Graph Encoder

建議新增獨立 `RepairModel`，不要第一版就擴充 `HCFPModel`。

### Static node features

重用目前 SceneEncoder 的：

- log area / sqrt area；
- b2b weighted degree；
- pin weight / centroid / spread；
- fixed / preplaced；
- boundary bits；
- group / MIB membership；
- target geometry validity。

### Dynamic node features

新增：

- current normalized `(x,y,w,h)`；
- current center / log aspect；
- geometry observed / masked；
- repair target；
- current boundary debt；
- current group component size；
- current MIB shape class；
- mobility / shape mobility；
- accepted-lock confidence；
- round and corruption severity embedding。

### Dynamic pair features

新增：

- current relative dx/dy；
- edge gap / overlap；
- current exact side contact type；
- same group / same MIB；
- b2b weight；
- component equality；
- required relation / candidate relation；
- blocker/conflict flag；
- current B*-Tree relation if available。

### Encoder size

第一輪：

```text
d_model = 192
layers = 4
heads = 6
FFN multiplier = 4
```

N 最大 120，dense pair-biased attention 可接受。只有 Contact CCRL oracle 通過後才測 `d=256, layers=6`。

## 3.3 Factorized Expert Heads

### Contact head

```text
group pointer
→ component-pair pointer
→ bridge pointer
→ anchor pointer
→ side classifier
→ patch-budget classifier
```

### Boundary head

```text
boundary target pointer
→ required side/corner
→ witness/anchor pointer
→ side-order slot
→ patch-budget classifier
```

### MIB head

```text
MIB-group pointer
→ anchor/canonical shape pointer
→ target-member pointer
→ shape variant
→ patch-budget classifier
```

### Topology head

```text
subtree/root pointer
→ target parent pointer
→ branch/axis classifier
→ local move type
```

各 head 只輸出 action distribution，不輸出完整 placement。

## 3.4 Common Repair Action IR

所有 expert 共用：

```python
class ExpertKind(Enum):
    CONTACT = "contact"
    BOUNDARY = "boundary"
    MIB = "mib"
    TOPOLOGY = "topology"

@dataclass(frozen=True)
class RepairAction:
    expert: ExpertKind
    obligation_id: str
    target_ids: tuple[int, ...]
    anchor_ids: tuple[int, ...]
    relation: str
    shape_spec: tuple[float, float] | None
    patch_budget: int
    score: float
    corruption_id: str | None
```

不要再讓各 family 以不一致的 nested dict 傳遞 action。

## 4. Structured Corruption 設計

## 4.1 原則

1. Corruption 必須 deterministic under seed。
2. Preplaced geometry 永遠不動。
3. Fixed-shape block 可移動，但不可改 shape。
4. 優先保持 hard feasible，讓模型學 soft/topology repair，而不是 legality cleanup。
5. 不留下可直接讀出原座標的精確空洞。
6. 同一 clean source 的所有 corruptions 必須屬於同一 train/held-out split。

## 4.2 Contact curriculum

| Level | Corruption | Target action |
| --- | --- | --- |
| C0 | 將 singleton group member產生微小 gap | bridge / anchor / side |
| C1 | 將一個 movable member 移到合法 free slot | component pair + reinsertion |
| C2 | 對 2-4 block patch 重切，斷開一條 contact | patch action |
| C3 | detach 一個 movable component/subtree | component bridge action |
| C4 | 兩個 contact debt + blocker | multi-step repair rollout |

第一輪只做 C0-C2。

## 4.3 Boundary curriculum

- B0：沿 inward normal 將 witness 移入 interior；
- B1：交換一個 side witness 與低-degree filler；
- B2：破壞 2-4 block side order；
- B3：corner + group joint corruption。

## 4.4 MIB curriculum

- M0：area-preserving log-aspect perturbation；
- M1：將一個 member 改為同面積錯誤 shape；
- M2：shape + local blocker repack；
- M3：MIB + grouping joint corruption。

## 4.5 Topology curriculum

- T0：single edge / sibling swap；
- T1：detach one leaf/subtree；
- T2：4-8 block subtree reinsert；
- T3：topology + region/anchor corruption。

## 4.6 防止 trivial leakage

單純將 B 從原位置拿走會留下 B 形狀的洞，模型可能只學「填洞」。需要混合：

- patch re-slicing，讓 hole 不保留原輪廓；
- D4 transforms；
- random equivalent clean placements；
- mask target geometry；
- solver-generated residual states；
- corruption 後加入低幅度 legal filler movement。

## 5. Loss 與訓練流程

## 5.1 Loss

不要使用單一 gold action cross-entropy。建議：

```text
L = L_expert
  + L_obligation
  + L_group/component
  + L_target_pointer
  + L_anchor_pointer
  + L_relation
  + L_patch_budget
  + λ_rank L_listwise(decoded top-K outcomes)
  + λ_value L_Δcost
  + λ_feasible L_action_feasibility
```

對多個等價 action，使用 acceptable-action set 的 marginal NLL，或依 deterministic decoder outcomes 產生 listwise target。

## 5.2 三階段訓練

### Phase A：synthetic inverse action

從 clean placement 產 corruption，已知 inverse action。先讓模型學基本 relation。

### Phase B：oracle relabeling

對 corrupted state 枚舉 bounded actions，由 exact decoder/scorer 選可接受 action set。避免只模仿 corruption generator 的唯一 inverse。

### Phase C：DAgger / rollout

讓模型在 common loop 中產生 state，再由 deterministic oracle relabel。這一步縮小 synthetic corruption 與真實 solver residual state 的分布差距。

## 5.3 第一輪訓練 gate

Contact-only：

- 2K-10K clean training sources；
- 每 source 4-16 corruptions；
- split by source ID；
- C0-C2 curriculum；
- Top-4 action recall；
- decoded exact-feasible rate；
- recovered grouping debt；
- decode reduction at equal QoR。

## 6. Runtime / cooperative loop

```text
incumbent
→ build RepairState
→ deterministic router chooses active expert(s)
→ RepairModel proposes top-K RepairAction
→ decoder registry materializes placements
→ exact verifier
→ exact/observable QoR critic
→ incumbent guard / beam
→ update dynamic state
→ repeat
```

第一版只啟用 Contact head。其他 heads 即使存在，也保持 checkpoint capability gate 關閉。

Production gate 仍為：

- hard feasibility 100%；
- zero per-case regression through incumbent guard；
- held-out unique wins；
- large15/full100 gain；
- runtime/VRAM 可接受；
- no validation ID/fingerprint routing。

## 7. Current code review

## 7.1 值得保留的基礎

### Exact verifier 與 runtime fallback

`verify.py` 已清楚分離 hard 與 soft semantics；`runtime.py` 保留官方 solve contract、hard-feasible fallback 與 fail-closed behavior。這兩層不應重寫。

### FloorSet-Lite loader

`floorset_lite.py` 已有 no-copy training stream、visible/test path guard、`fp_sol/metrics_sol/tree_sol` 解析。這正是 CCRL data foundation。

### Provenance / replay

`data.py` 與 `replay.py` 已有 sample serialization、hash、candidate identity 與 exact score欄位。CCRL 應重用設計模式，但建立獨立 RepairReplay schema。

### Pair-biased Transformer

現有 `SceneEncoder` 已支援 pair-biased Graph Transformer，可重用 static feature construction與 attention block概念。

### Deterministic contact loop

`contact_patch.py` 與 `experiment_bfod_v1.py` 已證明 local contact repair和 exact common loop有價值，是第一個 teacher/oracle。

## 7.2 High-priority redesign

### H1：模型沒有 dynamic placement state

目前 `SceneEncoder.forward(case)` 只讀 immutable `FloorplanCase`。同一 case 的 clean/corrupted/current state會得到相同 embedding，無法做 repair policy。

**修正**：新增 `RepairStateEncoder(case, state)`，不要把 current placement塞進 static Case。

### H2：ContactPolicy 是 post-construction ranker，不是 generator

`contact_policy.py` 的 features包含 `grouping_delta_fraction`與`boundary_delta_fraction`，這些必須先把 candidate decode出來才能知道。它無法直接決定 group/component/bridge/anchor。

**修正**：保留為 baseline ranker；新 Contact head從 dynamic state直接產 factorized action。

### H3：contact patch mobility mask 過度保守

`contact_patch._protected()` 把 `fixed | preplaced | any MIB member` 全部視為不可動。

- Preplaced不可動是正確的；
- Fixed-shape僅 `(w,h)` 固定，位置可以動；
- MIB member不應全部不可動，只需維持一致 shape obligation。

**修正**：拆成 position mobility、shape mobility與relation protection三種 mask。

### H4：action schema 是 ad hoc dict

`experiment_bfod_v1.py` 中 contact/joint/tree/region family使用不同 details dict，難以做 shared model、serialization、mask與 replay。

**修正**：先建立 `RepairAction` / `RepairCandidate` / `RepairOutcome` IR。

### H5：case-specific group priority

`_contact_candidates()` 使用 `{3:0, 4:1, 1:2, 2:3}` 的 group priority。這是診斷 case導向，對 unseen cases不具語義。

**修正**：完全移除 group ID priority，改用 runtime-visible group debt、component size、bridge mobility、estimated repair cost與 learned score。

### H6：現有 learned orchestration 過度集中

`learned.py` 同時管理 checkpoint、candidate families、tail guards、region、B*-Tree、contact與selection，已形成高度耦合的單檔 orchestration。

**修正**：CCRL 先放在獨立 `hcfp.repair` sidecar；promotion時只在 `select_official_from_analysis` 加一個 capability-gated call，不把多個 expert直接再塞進 `learned.py`。

### H7：visible Case70 參與 checkpoint selection

`train_bfod_contact_policy.py` 雖然沒有使用 Case70 teacher label，但用 Case70 input signature收集資料，並以 Case70 QoR決定最佳 checkpoint與 early stop。這不適合作為 general CCRL training protocol。

**修正**：training/held-out split只能來自 FloorSet training source；visible cases只在模型與 hyperparameters凍結後做 evaluation。

## 7.3 Medium-priority redesign

### M1：private cross-module imports

`contact_patch.py`、`boundary_skeleton.py`、`mib_patch.py` 直接依賴 `_Item`、`_partition`、`_closed_patch`等 private symbol。

**修正**：抽出 public `repair/decoders/packing.py` API。

### M2：generator output受迭代順序影響

Contact generator按 obligation、patch size、side迭代，達到 `max_candidates`就 early return。Top-K pool會因枚舉順序改變，已出現 pool-sensitive結果。

**修正**：先枚舉輕量 `RepairAction`，再以 predecode heuristic/learned policy排序，最後 decode bounded top-K。

### M3：production guard與research loop語義不同

Production guard採 no-regression；研究 common loop也使用相同嚴格 admission。這保證安全，但可能阻擋需兩步完成的 joint repair。

**修正**：保留 production hard guard；在 training-only beam中允許有界 temporary trade-off，final state仍須 no-regression。

### M4：exception observability不足

高層 `runtime.py`/`learned.py` 使用廣泛 exception fallback。提交安全，但研究時會把 decoder/model錯誤靜默變成 fallback。

**修正**：production行為不改；repair sidecar提供 typed failure reason與 telemetry。

### M5：測試目前只驗 API，不驗 learning semantics

`test_contact_policy.py`只測 finite feature與checkpoint round-trip。缺少：

- source split isolation；
- corruption leakage；
- action mask correctness；
- equivalent action labels；
- one-state overfit；
- rollout exact feasibility；
- D4 equivariance；
- deterministic replay。

## 8. 建議程式架構

```text
src/hcfp/repair/
  __init__.py
  schema.py                 RepairState/Action/Candidate/Outcome
  state.py                  dynamic geometry/contact/debt features
  actions.py                action masks, canonicalization, serialization
  critic.py                 exact/observable outcome metrics
  router.py                 deterministic first, learned later
  loop.py                   beam/common loop
  replay.py                 repair replay + provenance
  model.py                  shared encoder + expert heads
  losses.py
  dataset.py
  corruption/
    base.py
    contact.py
    boundary.py
    mib.py
    topology.py
  decoders/
    base.py
    packing.py
    contact.py
    boundary.py
    mib.py
    topology.py

scripts/
  audit_ccrl_clean_pool.py
  generate_ccrl_replay.py
  train_ccrl.py
  eval_ccrl.py

 tests/
  test_repair_schema.py
  test_structured_corruption.py
  test_repair_decoders.py
  test_repair_model.py
  test_repair_loop.py
  test_repair_replay.py
```

第一輪不移動現有檔案。新 public decoder通過 parity後，既有 contact/boundary/MIB modules再逐步改成 wrapper。

## 9. 實行計劃

## P0：資料與 action feasibility

1. Audit 1K-10K `fp_sol`，統計各 expert clean eligibility。
2. 定義 Repair IR 與 split/provenance。
3. 實作 Contact C0-C2 corruptions。
4. 驗證 corruption deterministic、hard-safe、非 trivial hole。
5. 驗證 deterministic decoder能恢復 debt。

Gate：

- >=95% selected clean contact samples可產一個 valid corruption；
- >=80% C0-C1 corruption有 exact-feasible inverse action；
- preplaced/fixed semantics 100% preserved。

## P1：Contact CCRL model

1. Dynamic RepairState encoder。
2. Contact factorized heads。
3. Synthetic inverse-action overfit。
4. Oracle relabel / acceptable action set。
5. 2K-10K source held-out evaluation。

Gate：

- Top-4 valid action recall >=80%；
- decoded hard-feasible >=99%；
- recovered grouping debt >= deterministic heuristic的90%；
- equal QoR下decode count降低 >=50%。

## P2：Contact rollout / DAgger

1. Model-in-the-loop rollout。
2. Oracle relabel failed states。
3. Mix synthetic與real residual states。
4. Compare deterministic、ranker、CCRL generator。

Gate：

- training-held-out unique wins或等QoR更低runtime；
- no visible-case-specific routing；
- 5 diagnostic cases不退步。

## P3：Boundary與MIB

只在各自 deterministic corruption oracle通過後加入 head。先單 expert，再 joint obligation。

## P4：Topology與router

Topology最後加入。Router只有在至少兩個 expert有held-out unique wins後訓練。

## P5：Promotion

Shadow checkpoint → large15 → full100 → repeated runtime → A100 → capability-gated production integration。

## 10. Stop conditions

立即暫停方向：

- corruption只能靠原位置洞口還原；
- exact coordinate regression好，但decoded QoR無改善；
- Contact C0-C2在training-held-out無法恢復group debt；
- shared model不優於 deterministic/contact ranker baseline；
- runtime cost大於QoR價值且Top-K routing無法回收；
- gain只出現在visible diagnostics。

## 11. 分支規則

`research/ccrl-structured-repair` 第一階段只接受：

- architecture/data audit；
- Repair IR；
- structured corruption；
- Contact-only model/decoder/evaluation；
- focused tests與research docs。

不在此階段修改：

- official solve contract；
- submission wrapper；
- default production flags；
- existing P7/P8 incumbent path；
- Boundary/MIB/Topology neural promotion。
