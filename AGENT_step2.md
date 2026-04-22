## Sprint 2 Pivot：Semantic-First Puzzle Placement → Repair → Strict Finalization

> 目標調整：
> 不再把 early-stage learning 目標設成「每一步都 hard-feasible」。
> 新策略是先讓模型學會 **semantic placement intent**，再透過 violation-aware training 與 repair/finalizer 把 layout 修成 final feasible。

FloorSet-Lite 的 final submission 仍然必須滿足 hard constraints，例如 area / dimension validity 與 overlap-free；infeasible solution 會得到固定 penalty (M=10)，而 feasible solution 才會依 HPWL gap、bbox area gap、soft violations 與 runtime 計分。
但這是 **final-output requirement**，不代表 training / intermediate rollout 每一步都必須 legal。這份指令就是基於這個 pivot 重寫。

---

# 0. 更新後核心方向

## 0.1 原本思路的問題

目前 repo 的架構已經成形，但遇到兩個核心瓶頸：

| 問題                            | 原因                                                        |
| ----------------------------- | --------------------------------------------------------- |
| candidate coverage 很低         | candidate generator 太早做 strict legality pruning           |
| rollout 只放 1–2 blocks         | 一遇到 infeasible / no strict legal action 就 early terminate |
| model 很難學 block selection     | 可選 action 被 legality filter 刪太多                           |
| fallback packing 補完大部分 blocks | policy 沒有學到 constructive semantic behavior                |
| 開發時間花在合法化                     | 模型還沒學會「想怎麼放」就被迫先學「怎麼完全合法」                                 |

因此這一輪不再要求：

```text
每個 intermediate action 都必須 hard-feasible
```

而是改成：

```text
先學 semantic intent
再學 violation awareness
最後用 repair / finalizer 產生 hard-feasible output
```

---

# 1. 全部 Agents 共用的新原則

請每個 agent 都遵守以下規則。

```text
本專案目標是建立 Instruction-Aware Puzzle Placement policy。

新的核心策略：

1. Early learning 不強迫每一步 hard-feasible。
2. Candidate generator 必須支援 semantic / relaxed / strict 三種模式。
3. Rollout 必須支援 semantic / relaxed / strict 三種模式。
4. semantic mode 允許 overlap、允許 provisional positions、允許 bbox 暫時變大。
5. type-level legality 永遠保留，例如 invalid block id、invalid target type、invalid primitive argument 不可允許。
6. area / fixed dimension / preplaced metadata 原則上盡量保留，因為這些容易處理且 final 是 hard constraint。
7. overlap-free 不應在 early learning 作為 hard pruning gate。
8. final submission 仍然必須 hard-feasible。
9. repair / finalizer 可以存在，但必須報告 displacement、intent preservation、fallback fraction，不能默默吃掉 policy contribution。
10. 不可使用 validation/test target positions 作為直接答案。
11. 不可 reverse-engineer dataset generator。
12. 所有 reports 必須分開記錄：
    - semantic policy output
    - violation profile
    - repair effect
    - final strict feasibility
```

---

# 2. 新系統架構

更新後整體 pipeline 應該變成：

```text
FloorSet case
  ↓
data adapter
  ↓
role / instruction / constraint encoding
  ↓
semantic candidate generator
  ↓
semantic policy rollout
  ↓
provisional layout
  ↓
violation analyzer
  ↓
repair / finalizer
  ↓
strict hard-feasible layout
  ↓
official evaluator / contest optimizer
```

不要再把 pipeline 寫成：

```text
strict legal candidates
  ↓
strict legal rollout
  ↓
rollout 卡住
  ↓
fallback packing 補完
```

---

# 3. 重要概念：Intent Layer vs Physical Layer

請所有 agents 把 state 拆成兩層。

## 3.1 Intent / semantic layer

這一層允許 provisional infeasibility。

它表示：

```text
這個 block 想靠誰？
想靠哪個 terminal？
想碰哪個 boundary？
想跟哪個 group abut？
想維持哪個 MIB shape？
想當 anchor 還是 follower？
```

可包含：

```text
proposed_positions
proposed_shapes
attachment_intents
pin_pull_intents
boundary_intents
group_intents
mib_shape_intents
relative_order_intents
semantic_action_history
```

此層允許：

```text
overlap
bbox 暫時變大
group 還沒真正 shared-edge
boundary 只是接近
pin pull 只是方向正確
geometry 尚未 legal
```

---

## 3.2 Physical / committed layer

這一層用於 final checking。

它表示：

```text
目前真正提交的位置是什麼？
是否 overlap-free？
area 是否滿足？
fixed / preplaced 是否符合？
能否交給官方 evaluator？
```

可包含：

```text
committed_positions
hard_feasibility_report
soft_violation_report
repair_history
finalizer_status
official_cost
```

---

# 4. Constraint relaxation policy

不是所有 constraints 都要放鬆。請依照以下表格處理。

| Constraint 類型           | Early semantic mode | Relaxed mode          | Strict mode           |
| ----------------------- | ------------------- | --------------------- | --------------------- |
| primitive type legality | 不放鬆                 | 不放鬆                   | 不放鬆                   |
| valid block / target id | 不放鬆                 | 不放鬆                   | 不放鬆                   |
| block availability      | 不放鬆                 | 不放鬆                   | 不放鬆                   |
| area target             | 盡量保留                | 保留，允許 tiny error 記錄   | 必須滿足                  |
| fixed dimension         | 保留                  | 保留                    | 必須滿足                  |
| preplaced location      | 優先保留                | 優先保留                  | 必須滿足或依官方版本處理          |
| MIB shape               | 半硬，優先一致             | penalty + repair      | 必須修正或最小 violation     |
| overlap-free            | 放鬆                  | 允許但記錄 overlap         | 必須滿足                  |
| boundary                | soft intent         | soft reward / penalty | final soft constraint |
| grouping abutment       | soft intent         | soft reward / penalty | final soft constraint |
| HPWL / bbox             | scoring only        | scoring / reward      | final objective       |

關鍵原則：

```text
type constraints 不放鬆
geometry overlap early 放鬆
final feasibility strict enforcement
```

---

# 5. 新增模式定義

## 5.1 `candidate_mode`

所有 candidate generator 必須支援：

```text
candidate_mode = semantic | relaxed | strict
```

### `semantic`

目的：最大化 learning signal 與 action recall。

```text
- high recall
- only type-level legality
- overlap allowed
- bbox expansion allowed
- attach intent allowed even if not immediately legal
- boundary intent can be approximate
- grouping intent can be approximate
- no aggressive geometry pruning
```

用途：

```text
behavior cloning
pseudo trajectory matching
early rollout training
semantic policy evaluation
```

---

### `relaxed`

目的：保留 high recall，但開始讓模型看到 infeasibility cost。

```text
- overlap allowed but scored
- soft violations tracked
- catastrophic actions filtered
- no immediate termination on infeasible state
- cheap repair optionally allowed every K steps
```

用途：

```text
scheduled rollout
feedback learning
AWBC
offline improvement
repair-aware training
```

---

### `strict`

目的：final submission safety。

```text
- no overlap
- area / dimension hard checks
- fixed / preplaced enforced
- geometry projection / repair allowed
- used only for finalizer / contest optimizer / official validation
```

用途：

```text
final inference
contest optimizer
official evaluator
submission preparation
```

---

## 5.2 `rollout_mode`

所有 rollout engine 必須支援：

```text
rollout_mode = semantic | relaxed | strict
```

### `semantic_rollout`

```text
- uses semantic candidates
- does not terminate on overlap
- every block should receive a proposed position / intent
- outputs provisional layout
- reports violation profile
```

### `relaxed_rollout`

```text
- uses relaxed candidates
- tracks violation vector
- may call local repair every K steps
- prioritizes completion over strict feasibility
- outputs provisional layout + violation report
```

### `strict_rollout`

```text
- uses finalizer / strict candidates
- outputs hard-feasible positions if possible
- can fallback only after policy + finalizer fail
```

---

# 6. 更新後 repo 結構建議

請在既有 repo 上新增 / 修改以下模組。

```text
src/
  puzzleplace/
    actions/
      candidates.py        # add candidate_mode
      masks.py             # mode-aware masks
      executor.py          # allow provisional placement
    rollout/
      semantic.py          # new semantic rollout
      relaxed.py           # new relaxed rollout
      strict.py            # strict final rollout
      greedy.py            # can wrap mode-specific rollout
      beam.py              # can wrap mode-specific rollout
    repair/
      __init__.py
      shape_normalizer.py
      overlap_resolver.py
      shelf_packer.py
      intent_preserver.py
      finalizer.py
    eval/
      violation.py         # violation profile for provisional layout
      metrics.py           # semantic / repair / final metrics
    train/
      dataset_bc.py        # mode-aware candidate matching
      train_bc.py
    feedback/
      rewards.py           # violation-aware rewards
scripts/
  check_candidate_coverage.py
  rollout_validate.py
  repair_validate.py
  run_sprint2_pivot_smoke.sh
  make_sprint2_pivot_summary.py
docs/
  00_environment.md
  01_dataset_adapter.md
  02_action_schema.md
  03_pseudo_trajectory.md
  04_behavior_cloning.md
  05_semantic_rollout.md
  06_repair_finalizer.md
  07_strict_evaluation.md
  08_known_failure_modes.md
```

---

# 7. Orchestrator Agent 指令

請把以下段落交給總控 agent。

```text
你是 Sprint 2 Pivot orchestrator。

目前 repo 已經完成基本架構，但 early-stage strict feasibility 造成 candidate recall 低、rollout early termination、policy 學不到 semantic placement 行為。

本 sprint 的新方向是 semantic-first：

Phase 1:
  先建立 semantic candidate generator 與 semantic rollout。
  不要求 intermediate hard feasibility。
  要求所有 blocks 都能產生 placement intent。

Phase 2:
  建立 violation analyzer 與 repair/finalizer。
  把 provisional layout 修成 hard-feasible layout。
  report displacement / intent preservation。

Phase 3:
  contest optimizer 改成:
    semantic rollout → repair/finalizer → strict evaluator → fallback only if needed

你要分派以下 agents：

1. Agent 6P：Multi-mode candidate generator
2. Agent 9P：Semantic / relaxed rollout engine
3. Agent 15：Repair and finalizer
4. Agent 10P：Semantic / violation / repair metrics
5. Agent 8P：BC training update for semantic candidates
6. Agent 12P：Contest optimizer pivot
7. Agent 13P：CI smoke workflow
8. Agent 14P：Docs expansion

Sprint 2 Pivot 最小完成條件：

1. semantic candidate coverage >= 0.95 on validation case 0–4
2. relaxed candidate coverage >= 0.80 on validation case 0–4
3. semantic rollout completion = 5/5 on validation case 0–4
4. semantic rollout reports overlap / area / soft violation profile instead of early stopping
5. repair_validate shows overlap pairs / total overlap area decrease after repair
6. at least 1 validation case becomes hard-feasible after repair without final deterministic fallback
7. contest optimizer still passes smoke validation
8. final fallback fraction is explicitly reported
9. docs and CI are updated
```

---

# 8. Agent 6P：Multi-mode Candidate Generator

## 任務名稱

```text
Agent 6P — Multi-mode Candidate Generator: Semantic / Relaxed / Strict
```

## 任務目標

把目前 candidate generator 從 single strict-ish mode 改成三種模式：

```text
semantic
relaxed
strict
```

早期重點不再是 strict legal coverage，而是：

```text
semantic candidate coverage >= 0.95
relaxed candidate coverage >= 0.80
strict coverage only diagnostic
```

---

## 指令

```text
請修改：

src/puzzleplace/actions/candidates.py
src/puzzleplace/actions/masks.py
src/puzzleplace/actions/executor.py
scripts/check_candidate_coverage.py
tests/test_candidates.py

新增 candidate_mode argument：

generate_candidates(state, primitive=None, mode="semantic", max_per_primitive=None)

mode behavior:

1. semantic:
   - only type-level legality
   - do not prune by overlap
   - do not require immediate hard feasibility
   - generate high-recall candidates
   - record candidate.intent_type and candidate.semantic_score if possible

2. relaxed:
   - generate most semantic candidates
   - compute violation estimates:
       overlap_risk
       area_error
       boundary_distance
       grouping_distance
       mib_inconsistency
   - do not filter unless catastrophic

3. strict:
   - existing or improved strict legal candidates
   - no overlap allowed
   - hard dimension / area enforced
   - used by finalizer / contest path only
```

---

## Candidate families to implement

### 8.1 `SET_SHAPE`

```text
For every unplaced or shape-unassigned block:
  - square shape
  - ratio bins:
      1:1
      1:2
      2:1
      1:3
      3:1
      2:3
      3:2
  - exact area normalization
  - fixed/preplaced dimensions if available
  - MIB reference shape if any group member already shaped
```

Semantic mode：

```text
all shape candidates allowed if area roughly valid
```

Strict mode：

```text
must satisfy area tolerance / fixed dimension
```

---

### 8.2 `SNAP_TO_BOUNDARY`

Semantic mode：

```text
Generate boundary intent even if final bbox is provisional.

For each boundary block:
  - required edge / corner candidates
  - use provisional bbox if exists
  - if no bbox, use origin bbox
  - align to:
      origin
      current bbox edge
      high-weight terminal axis
      high-weight placed neighbor axis
```

Relaxed mode：

```text
same as semantic, but compute boundary_distance
```

Strict mode：

```text
candidate must actually touch final or provisional bbox boundary after finalization
```

---

### 8.3 `ATTACH`

Semantic mode：

```text
For each unplaced block b:
  For each placed or semantic-placed block p:
    generate candidate attach intent:
      TOUCH_LEFT
      TOUCH_RIGHT
      TOUCH_TOP
      TOUCH_BOTTOM
    with alignment variants:
      ALIGN_CENTER
      ALIGN_LOW_EDGE
      ALIGN_HIGH_EDGE
      ALIGN_TO_PIN_AXIS
      ALIGN_TO_NEIGHBOR_AXIS

Do not reject candidate only because it overlaps another block.
```

Prioritize but do not exclusively filter:

```text
1. same grouping cluster
2. high b2b edge
3. same MIB group
4. follower → leader / anchor
5. terminal direction compatible
6. generic placed anchor
```

---

### 8.4 `PULL_TO_PIN`

Semantic mode：

```text
For each block with external connectivity:
  choose top-k terminals
  generate:
    center near pin
    left of pin
    right of pin
    above pin
    below pin
  overlap allowed
```

Relaxed mode：

```text
compute pin distance improvement
```

Strict mode：

```text
candidate may require projection / repair
```

---

### 8.5 `PLACE_RELATIVE`

This is an action-level fallback, not final fallback packing.

Semantic mode：

```text
For every unplaced block:
  relative to bbox:
    right strip
    top strip
    left strip
    bottom strip
  relative to placed anchors:
    right
    above
    below
    left
```

Use this to ensure semantic rollout can always continue.

---

### 8.6 `FREEZE`

修改語意：

```text
FREEZE means the block itself should not move.
A frozen block can still be used as attach target.
FREEZE must not prevent other blocks from attaching to it.
FREEZE should not be selected too early.
```

---

## Coverage script update

`scripts/check_candidate_coverage.py` 必須同時輸出：

```json
{
  "semantic_coverage": 0.0,
  "relaxed_coverage": 0.0,
  "strict_coverage": 0.0,
  "teacher_hint_coverage": 0.0,
  "coverage_by_primitive": {},
  "missing_examples": [
    {
      "case_id": 0,
      "step": 12,
      "mode": "semantic",
      "primitive": "ATTACH",
      "reason": "target_not_generated | relation_not_generated | param_tolerance | masked_out"
    }
  ]
}
```

---

## Agent 6P 驗收命令

```bash
python scripts/check_candidate_coverage.py \
  --split validation \
  --case-ids 0 1 2 3 4 \
  --modes semantic relaxed strict \
  --output artifacts/reports/sprint2_pivot_candidate_coverage.json
```

## Agent 6P Pass criteria

```text
semantic candidate coverage >= 0.95
relaxed candidate coverage >= 0.80
teacher-hint coverage remains >= 0.98
strict coverage is reported but not used as early pass gate
missing examples are categorized
```

---

# 9. Agent 9P：Semantic / Relaxed Rollout Engine

## 任務名稱

```text
Agent 9P — Semantic and Relaxed Rollout Without Early Termination
```

## 任務目標

修改 rollout，不要因 intermediate infeasibility 提前停止。
首先要做到：

```text
semantic rollout 5/5 validation cases complete
```

即每個 block 都能得到 proposed position / placement intent。

---

## 指令

```text
請修改或新增：

src/puzzleplace/rollout/semantic.py
src/puzzleplace/rollout/relaxed.py
src/puzzleplace/rollout/strict.py
src/puzzleplace/rollout/greedy.py
src/puzzleplace/rollout/beam.py
src/puzzleplace/actions/executor.py
src/puzzleplace/eval/violation.py
scripts/rollout_validate.py
tests/test_rollout_smoke.py

新增 rollout_mode argument：

rollout(case, model=None, mode="semantic" | "relaxed" | "strict")
```

---

## BoardState 必須新增或確認以下欄位

```text
positions: list[Box | None]
proposed_positions: list[Box | None]
shape_assigned: list[bool]
semantic_placed: list[bool]
physically_placed: list[bool]
frozen: list[bool]
step: int
history: list[Action]
violation_profile: ViolationProfile
```

定義：

```text
semantic_placed:
  block 已有 proposed placement intent，不要求 hard feasible

physically_placed:
  block 已有 committed physical position，可能用於 strict check

shape_assigned:
  block 已有 w,h

frozen:
  block 自己不可被移動，但仍可被其他 block attach
```

---

## Semantic rollout lifecycle

```text
INIT:
  1. if preplaced exists:
       seed it as first anchor
     else:
       choose first anchor by:
         boundary + high p2b
         high b2b degree hub
         largest area
  2. SET_SHAPE
  3. PLACE_RELATIVE at origin or preplaced location
  4. mark semantic_placed

MAIN_LOOP:
  while not all blocks semantic_placed:
    1. generate semantic candidates
    2. score by model if available, otherwise heuristic
    3. choose action
    4. apply action provisionally
    5. update semantic_placed / proposed_positions
    6. compute violation profile
    7. do not terminate on overlap
    8. if no progress for 3 steps:
         force PLACE_RELATIVE for one unplaced block

END:
  output provisional layout
  output violation profile
```

---

## No-progress invariant

每一步都要記錄：

```json
{
  "step": 10,
  "primitive": "ATTACH",
  "before_semantic_placed": 5,
  "after_semantic_placed": 6,
  "progress_made": true,
  "overlap_pairs": 3,
  "total_overlap_area": 102.4
}
```

禁止：

```text
NOOP as active rollout action
FREEZE repeatedly without placement progress
terminating only because overlap exists
terminating after 1–2 blocks unless max_steps reached
```

---

## Relaxed rollout

Relaxed rollout 比 semantic rollout 多做：

```text
- compute overlap risk
- compute repair distance estimate
- optionally call light repair every K steps
- use relaxed candidates
- still prioritize completion
```

---

## Agent 9P 驗收命令

```bash
python scripts/rollout_validate.py \
  --split validation \
  --case-ids 0 1 2 3 4 \
  --rollout-mode semantic \
  --output artifacts/reports/sprint2_pivot_semantic_rollout.json
```

```bash
python scripts/rollout_validate.py \
  --split validation \
  --case-ids 0 1 2 3 4 \
  --rollout-mode relaxed \
  --output artifacts/reports/sprint2_pivot_relaxed_rollout.json
```

## Agent 9P Pass criteria

```text
semantic rollout completion = 5/5
semantic placed fraction = 1.0 average
no case terminates after only 1–2 blocks
violation profile is reported
relaxed rollout placed fraction >= 0.90
strict feasibility is not required for semantic rollout
```

---

# 10. Agent 15：Repair / Finalizer

## 任務名稱

```text
Agent 15 — Provisional Layout Repair and Strict Finalizer
```

## 任務目標

建立一個 module，把 semantic / relaxed rollout 的 provisional layout 修成 hard-feasible layout。

這個 finalizer 不是黑箱 fallback packing。它必須：

```text
1. 儘量保留 policy intent
2. 最小化 displacement
3. 修 overlap / area / dimension
4. 報告 repair 做了什麼
```

---

## 指令

```text
請新增：

src/puzzleplace/repair/__init__.py
src/puzzleplace/repair/shape_normalizer.py
src/puzzleplace/repair/overlap_resolver.py
src/puzzleplace/repair/shelf_packer.py
src/puzzleplace/repair/intent_preserver.py
src/puzzleplace/repair/finalizer.py
scripts/repair_validate.py
tests/test_repair_finalizer.py
```

---

## Repair input

```text
FloorSetCase
provisional positions
semantic action history
constraint metadata
role labels
```

---

## Repair output

```text
repaired positions
RepairReport
```

---

## RepairReport schema

```python
@dataclass
class RepairReport:
    hard_feasible_before: bool
    hard_feasible_after: bool

    overlap_pairs_before: int
    overlap_pairs_after: int
    total_overlap_area_before: float
    total_overlap_area_after: float

    area_violations_before: int
    area_violations_after: int
    dimension_violations_before: int
    dimension_violations_after: int

    moved_block_count: int
    changed_block_fraction: float
    mean_displacement: float
    max_displacement: float

    preserved_attach_intents: int
    destroyed_attach_intents: int
    preserved_boundary_intents: int
    destroyed_boundary_intents: int
    intent_preservation_rate: float

    final_fallback_used: bool
    final_fallback_block_count: int
```

---

## Repair strategy MVP

### Step 1：Shape normalization

```text
- enforce area target for soft blocks
- enforce fixed / preplaced dimensions
- enforce MIB same shape if possible
```

### Step 2：Anchor locking

```text
priority locked:
  1. preplaced
  2. fixed-shape dimensions
  3. high-degree anchors
  4. boundary anchors
```

### Step 3：Overlap resolver

```text
sort blocks by role priority:
  anchors move least
  followers move more

for each overlapping pair:
  try local shifts:
    right
    up
    left
    down
  choose shift with:
    minimal displacement
    least intent destruction
    no new large overlap
```

### Step 4：Shelf fallback for unresolved blocks

```text
If local resolver fails:
  move unresolved block to right shelf or top shelf
  preserve shape
  avoid overlap
  record as shelf_fallback, not silent repair
```

### Step 5：Intent preservation measurement

Measure after repair:

```text
- attach intent still shares edge?
- boundary intent still touches required edge/corner?
- pin-pull distance worsened?
- group blocks still near / connected?
```

---

## Agent 15 驗收命令

```bash
python scripts/repair_validate.py \
  --input artifacts/reports/sprint2_pivot_semantic_rollout.json \
  --case-ids 0 1 2 3 4 \
  --output artifacts/reports/sprint2_pivot_repair_validate.json
```

## Agent 15 Pass criteria

```text
total_overlap_area_after < total_overlap_area_before
overlap_pairs_after < overlap_pairs_before
at least 1 case hard_feasible_after = true
mean_displacement is reported
intent_preservation_rate is reported
final_fallback_used is reported
```

---

# 11. Agent 10P：Metrics and Reports Update

## 任務名稱

```text
Agent 10P — Semantic / Violation / Repair Metrics
```

## 任務目標

更新 metrics，不再只看 final feasible rate。
必須分成四層：

```text
semantic learning
infeasibility profile
repair profile
final strict profile
```

---

## 指令

```text
請修改：

src/puzzleplace/eval/metrics.py
src/puzzleplace/eval/reports.py
src/puzzleplace/eval/violation.py
scripts/make_sprint2_pivot_summary.py
```

---

## 必須新增 metrics

### Semantic metrics

```json
{
  "semantic_candidate_coverage": 0.0,
  "relaxed_candidate_coverage": 0.0,
  "strict_candidate_coverage": 0.0,
  "semantic_rollout_completion_rate": 0.0,
  "avg_semantic_placed_fraction": 0.0,
  "primitive_accuracy": 0.0,
  "arg1_top5": 0.0,
  "arg2_top5": 0.0,
  "role_action_consistency": 0.0
}
```

### Infeasibility metrics

```json
{
  "overlap_pair_count": 0,
  "total_overlap_area": 0.0,
  "max_overlap_area": 0.0,
  "area_error_total": 0.0,
  "area_violation_count": 0,
  "dimension_violation_count": 0,
  "boundary_distance_total": 0.0,
  "group_fragmentation": 0,
  "mib_shape_inconsistency": 0
}
```

### Repair metrics

```json
{
  "repair_success_rate": 0.0,
  "hard_feasible_before_repair": 0,
  "hard_feasible_after_repair": 0,
  "overlap_area_reduction": 0.0,
  "overlap_pair_reduction": 0,
  "mean_displacement": 0.0,
  "max_displacement": 0.0,
  "changed_block_fraction": 0.0,
  "intent_preservation_rate": 0.0,
  "repair_destroyed_attach_intents": 0,
  "repair_destroyed_boundary_intents": 0
}
```

### Final metrics

```json
{
  "final_hard_feasible_rate": 0.0,
  "official_cost": null,
  "hpwl_gap": null,
  "bbox_area_gap": null,
  "soft_violations_relative": null,
  "final_fallback_fraction": 0.0,
  "policy_contribution_ratio": 0.0
}
```

---

## Sprint summary format

`sprint2_pivot_summary.md` 必須回答：

```text
1. Semantic learning 是否過線？
   - semantic coverage
   - relaxed coverage
   - semantic rollout completion

2. Infeasibility 有多嚴重？
   - overlap pairs
   - total overlap area
   - area / dimension violations

3. Repair 有沒有改善？
   - before / after overlap
   - before / after hard feasibility
   - displacement
   - intent preservation

4. Final submission path 是否可行？
   - official evaluator smoke
   - fallback fraction
   - policy contribution ratio

5. 下一個 blocker 是什麼？
```

---

# 12. Agent 8P：BC Training Update for Semantic Candidates

## 任務名稱

```text
Agent 8P — Behavior Cloning with Semantic Candidates
```

## 任務目標

更新 BC training，不再因 strict candidate miss 錯誤懲罰模型。
BC 的第一目標是學 semantic action，不是一步 legal placement。

---

## 指令

```text
請修改：

src/puzzleplace/train/dataset_bc.py
src/puzzleplace/train/train_bc.py
src/puzzleplace/models/losses.py
configs/bc_small.yaml
scripts/train_bc_small.py
```

---

## Training changes

### 12.1 Use semantic candidates by default

```text
BC training candidate_mode = semantic
validation also report relaxed / strict coverage separately
```

### 12.2 Candidate miss handling

如果 expert action 不在 candidate set：

```text
do not count it as normal CE failure
mark as candidate_miss
exclude from action CE
include in candidate generator diagnostic
```

### 12.3 Loss breakdown

必須報：

```text
primitive_loss
arg1_loss
arg2_loss
relation_loss
param_loss
role_aux_loss
semantic_ranking_loss
candidate_miss_rate
```

### 12.4 Add violation auxiliary prediction

Optional but recommended：

```text
Given state + action, predict:
  overlap_risk_bin
  boundary_progress
  grouping_progress
  mib_progress
```

這能為 relaxed training 做準備。

---

## Agent 8P 驗收命令

```bash
python scripts/train_bc_small.py \
  --case-ids 0 \
  --max-traces 1 \
  --epochs 50 \
  --candidate-mode semantic \
  --overfit \
  --output artifacts/reports/sprint2_pivot_bc_overfit.json
```

## Agent 8P Pass criteria

```text
train loss decreases
primitive accuracy > 0.80 on tiny overfit
arg1 top-5 > 0.80 on tiny overfit
candidate_miss_rate is reported
semantic candidate miss rate is low
strict candidate miss rate can be high but diagnostic only
```

---

# 13. Agent 12P：Contest Optimizer Pivot

## 任務名稱

```text
Agent 12P — Contest Optimizer = Semantic Rollout + Finalizer + Last-resort Fallback
```

## 任務目標

改寫 contest optimizer，不要求 policy 每步 strict feasible。
新的 solve flow：

```text
semantic rollout
  ↓
repair / finalizer
  ↓
strict hard check
  ↓
return repaired layout if feasible
  ↓
only then final fallback packing
```

---

## 指令

```text
請修改：

src/puzzleplace/optimizer/contest.py
contest_optimizer.py
scripts/rollout_validate.py
artifacts/reports/agent12_contest_summary.md
```

---

## solve() 必須遵守

```text
1. 不可使用 target positions 作為答案。
2. 先跑 semantic_rollout。
3. semantic_rollout 必須嘗試為所有 blocks 產生 proposed positions。
4. 跑 finalizer.repair。
5. 如果 repaired layout hard-feasible，返回 repaired layout。
6. 如果 repair 失敗，才使用 deterministic fallback packing。
7. 無論是否 fallback，都必須報告：
   - semantic_completed
   - semantic_placed_fraction
   - hard_feasible_before_repair
   - hard_feasible_after_repair
   - repair_displacement
   - intent_preservation_rate
   - final_fallback_used
   - final_fallback_block_fraction
```

---

## Agent 12P 驗收命令

```bash
python scripts/rollout_validate.py \
  --split validation \
  --case-ids 0 1 2 3 4 \
  --use-contest-optimizer \
  --output artifacts/reports/sprint2_pivot_contest_optimizer.json
```

## Agent 12P Pass criteria

```text
contest optimizer smoke still runs
semantic_completed is reported
repair_success is reported
final_fallback_used is reported
at least 1 case returns repaired layout without final fallback if Agent 15 succeeds
fallback block fraction is lower than previous baseline if possible
```

---

# 14. Agent 11P：Feedback / AWBC Update

## 任務名稱

```text
Agent 11P — Violation-aware Feedback and AWBC
```

## 任務目標

更新 feedback，不再只把 infeasible 當 terminal failure。
改成讓 violation vector 成為 learning signal。

---

## 指令

```text
請修改：

src/puzzleplace/feedback/rewards.py
src/puzzleplace/feedback/replay_buffer.py
src/puzzleplace/feedback/improve_bc.py
scripts/collect_rollout_buffer.py
scripts/train_awbc.py
```

---

## Reward changes

新增 reward breakdown：

```python
RewardBreakdown:
    semantic_progress
    block_placed_progress
    attach_intent_reward
    boundary_intent_reward
    pin_pull_reward
    grouping_progress
    mib_shape_progress

    overlap_pair_penalty
    overlap_area_penalty
    area_error_penalty
    repair_distance_penalty
    intent_destruction_penalty

    final_feasible_bonus
    final_fallback_penalty
```

重要：

```text
In semantic / relaxed training:
  overlap penalty exists but should not terminate rollout immediately.

In strict final evaluation:
  hard infeasible still fails final check.
```

---

## Replay buffer update

每個 transition 必須保存：

```text
candidate_mode
rollout_mode
violation_profile_before
violation_profile_after
repair_report if available
semantic_completed
final_feasible
fallback_used
```

---

## Agent 11P Pass criteria

```text
AWBC can train from semantic / relaxed rollout buffer
reward breakdown is logged
overlap-aware penalty does not reduce semantic completion to zero
fallback penalty is included
```

---

# 15. Agent 13P：CI Workflow

## 任務名稱

```text
Agent 13P — CI Workflow with Tiny Synthetic Case
```

## 任務目標

補真正 CI，但不要在 CI 下載大型資料集。

---

## 指令

```text
請新增：

.github/workflows/smoke.yml
scripts/ci_smoke.sh
tests/fixtures/tiny_case.py
artifacts/reports/agent13_ci_summary.md
```

---

## CI must run

```text
pytest tests/test_geometry.py
pytest tests/test_action_schema.py
pytest tests/test_candidates.py
pytest tests/test_policy_shapes.py
pytest tests/test_repair_finalizer.py
pytest tests/test_rollout_smoke.py
```

---

## Tiny fixture requirements

```text
3–5 blocks
1 boundary block
1 grouping pair
1 MIB pair
simple b2b edges
simple p2b terminal
known feasible simple layout
known overlapping provisional layout for repair test
```

---

## Agent 13P Pass criteria

```bash
bash scripts/ci_smoke.sh
```

```text
local CI smoke passes
.github/workflows/smoke.yml exists
tests do not download large FloorSet data
CI runtime target < 10 minutes
```

---

# 16. Agent 14P：Docs Expansion

## 任務名稱

```text
Agent 14P — Update Docs for Semantic-first Pipeline
```

## 任務目標

把目前單一 research manual 擴成 step-by-step docs。

---

## 指令

```text
請新增：

docs/00_environment.md
docs/01_dataset_adapter.md
docs/02_action_schema.md
docs/03_pseudo_trajectory.md
docs/04_behavior_cloning.md
docs/05_semantic_rollout.md
docs/06_repair_finalizer.md
docs/07_strict_evaluation.md
docs/08_known_failure_modes.md
```

---

## 每份 doc 必須包含

```text
Purpose
Required previous steps
Commands
Expected outputs
Key metrics
Common failure modes
How to debug
Next step
```

---

## 特別要求

### `docs/05_semantic_rollout.md`

必須解釋：

```text
semantic rollout vs strict rollout
why overlap is allowed during early learning
how to read violation profile
why semantic completion matters
```

### `docs/06_repair_finalizer.md`

必須解釋：

```text
repair is not final fallback packing
how displacement is measured
how intent preservation is measured
how to debug infeasible after repair
```

### `docs/08_known_failure_modes.md`

必須包含：

```text
semantic candidate coverage low
relaxed coverage low
strict coverage low but semantic okay
rollout terminates after 1 block
SET_SHAPE never followed by placement
FREEZE selected too early
candidate exists but masked out
executor applies action but semantic_placed not updated
semantic rollout complete but impossible to repair
repair destroys all policy intent
fallback packing hides policy failure
official evaluator mismatch
```

---

# 17. 統一 smoke command

請新增：

```text
scripts/run_sprint2_pivot_smoke.sh
```

內容：

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "[1/7] Candidate coverage: semantic / relaxed / strict"
python scripts/check_candidate_coverage.py \
  --split validation \
  --case-ids 0 1 2 3 4 \
  --modes semantic relaxed strict \
  --output artifacts/reports/sprint2_pivot_candidate_coverage.json

echo "[2/7] Tiny BC overfit with semantic candidates"
python scripts/train_bc_small.py \
  --case-ids 0 \
  --max-traces 1 \
  --epochs 10 \
  --candidate-mode semantic \
  --overfit \
  --output artifacts/reports/sprint2_pivot_bc_overfit.json

echo "[3/7] Semantic rollout validation"
python scripts/rollout_validate.py \
  --split validation \
  --case-ids 0 1 2 3 4 \
  --rollout-mode semantic \
  --output artifacts/reports/sprint2_pivot_semantic_rollout.json

echo "[4/7] Relaxed rollout validation"
python scripts/rollout_validate.py \
  --split validation \
  --case-ids 0 1 2 3 4 \
  --rollout-mode relaxed \
  --output artifacts/reports/sprint2_pivot_relaxed_rollout.json

echo "[5/7] Repair validation"
python scripts/repair_validate.py \
  --semantic-rollout artifacts/reports/sprint2_pivot_semantic_rollout.json \
  --case-ids 0 1 2 3 4 \
  --output artifacts/reports/sprint2_pivot_repair_validate.json

echo "[6/7] Contest optimizer smoke"
python scripts/rollout_validate.py \
  --split validation \
  --case-ids 0 1 2 3 4 \
  --use-contest-optimizer \
  --output artifacts/reports/sprint2_pivot_contest_optimizer.json

echo "[7/7] Make summary"
python scripts/make_sprint2_pivot_summary.py \
  --candidate artifacts/reports/sprint2_pivot_candidate_coverage.json \
  --bc artifacts/reports/sprint2_pivot_bc_overfit.json \
  --semantic-rollout artifacts/reports/sprint2_pivot_semantic_rollout.json \
  --relaxed-rollout artifacts/reports/sprint2_pivot_relaxed_rollout.json \
  --repair artifacts/reports/sprint2_pivot_repair_validate.json \
  --contest artifacts/reports/sprint2_pivot_contest_optimizer.json \
  --output artifacts/reports/sprint2_pivot_summary.md
```

---

# 18. Sprint 2 Pivot 最終驗收標準

## Phase 1：Learning viability

| 指標                                    |                    最低要求 |
| ------------------------------------- | ----------------------: |
| semantic candidate coverage           |               `>= 0.95` |
| relaxed candidate coverage            |               `>= 0.80` |
| semantic rollout completion           | `5/5` on validation 0–4 |
| avg semantic placed fraction          |                   `1.0` |
| BC primitive accuracy on tiny overfit |                `> 0.80` |
| BC arg1 top-5 on tiny overfit         |                `> 0.80` |

---

## Phase 2：Repair viability

| 指標                                       |              最低要求 |
| ---------------------------------------- | ----------------: |
| overlap pairs after repair               | `< before repair` |
| total overlap area after repair          | `< before repair` |
| at least one repaired case hard feasible |          `>= 1/5` |
| repair displacement reported             |          required |
| intent preservation reported             |          required |

---

## Phase 3：Submission path viability

| 指標                                  |                        最低要求 |
| ----------------------------------- | --------------------------: |
| contest optimizer smoke             |                        pass |
| semantic completed reported         |                    required |
| repair success reported             |                    required |
| final fallback fraction reported    |                    required |
| at least one case no final fallback | ideal but depends on repair |

---

# 19. Debug priority guide

## 19.1 Candidate coverage low

檢查：

```text
1. expert action token 與 candidate token 是否同一格式？
2. primitive enum 是否一致？
3. arg1 / arg2 indexing 是否一致？
4. relation normalization 是否一致？
5. params tolerance 是否太嚴？
6. semantic mode 是否不小心用了 strict pruning？
7. candidate 生成了但被 mask 掉？
8. teacher hint 是否污染 heuristic / semantic mode？
```

---

## 19.2 Semantic rollout still stops early

檢查：

```text
1. semantic_placed 是否正確更新？
2. SET_SHAPE 是否沒有後續 placement action？
3. FREEZE 是否太早阻止 block 被移動？
4. frozen block 是否仍可當 attach target？
5. no-progress fallback 是否觸發？
6. PLACE_RELATIVE 是否對每個 unplaced block 都存在？
7. max_steps 是否太小？
8. executor 是否 apply 成功但 BoardState 沒更新？
```

---

## 19.3 Semantic rollout complete but impossible to repair

檢查：

```text
1. overlap 是否過於嚴重？
2. 所有 blocks 是否都堆在 origin？
3. PLACE_RELATIVE 是否沒有 bbox expansion？
4. shape normalization 是否失敗？
5. fixed / preplaced anchors 是否被錯誤移動？
6. repair shelf fallback 是否啟用？
7. repair displacement 是否過大？
```

---

## 19.4 Repair feasible but destroys all intent

檢查：

```text
1. repair 是否直接 shelf-pack 所有 blocks？
2. attach intent preservation rate 是否很低？
3. boundary intent 是否被破壞？
4. group cluster 是否被拆散？
5. displacement 是否過大？
6. finalizer 是否變成 silent fallback?
```

---

## 19.5 Contest optimizer feasible but policy contribution low

檢查：

```text
1. final_fallback_block_fraction
2. semantic_placed_fraction
3. repair_changed_block_fraction
4. intent_preservation_rate
5. fallback 是否過早觸發
6. policy output 是否被全部丟棄
```

---

# 20. 分支與提交規範

建議分支：

```text
sprint2-pivot/candidate-modes
sprint2-pivot/semantic-rollout
sprint2-pivot/repair-finalizer
sprint2-pivot/metrics
sprint2-pivot/contest-optimizer
sprint2-pivot/ci-docs
```

commit style：

```text
[agent6P] add semantic relaxed strict candidate modes
[agent9P] add semantic rollout without overlap termination
[agent15] add overlap repair finalizer
[agent10P] report violation and intent preservation metrics
[agent8P] train BC with semantic candidate mode
[agent12P] route contest optimizer through finalizer
[agent13P] add smoke CI workflow
[agent14P] expand semantic-first docs
```

每個 patch 必須包含：

```text
What changed
Why it was needed
Before metrics
After metrics
Commands run
Known remaining failures
```

---

# 21. 給 agents 的最終訊息

```text
重要方向調整：

我們不再要求 early-stage policy 每一步都 hard-feasible。
這個要求目前讓 candidate generator 太窄、rollout 太早停止、模型學不到 semantic placement behavior。

新的策略是：

1. Semantic-first:
   先學 action intent。
   candidate_mode=semantic。
   overlap allowed。
   只保留 type-level legality。

2. Violation-aware:
   rollout 不因 overlap 立即終止。
   overlap / area / boundary / grouping / MIB violations 變成 feedback signal。

3. Repair/finalizer:
   將 provisional layout 修成 hard-feasible。
   報告 displacement 與 intent preservation。
   不允許 repair 默默吃掉 policy contribution。

4. Strict final:
   contest optimizer 最終仍必須 hard-feasible。
   final fallback packing 只作最後保底，而且必須報告 fallback fraction。

Sprint 2 Pivot 的成功標準不是 final score 最佳，而是：
- semantic candidate coverage high
- semantic rollout complete
- violation profile measurable
- repair reduces infeasibility
- policy contribution remains visible
```

---

# 22. 最精簡的執行順序

如果只能先做最重要的部分，請照這個順序：

```text
1. Agent 6P:
   add candidate_mode=semantic/relaxed/strict
   target semantic coverage >= 0.95

2. Agent 9P:
   add semantic rollout
   target validation 0–4 completion = 5/5

3. Agent 15:
   add repair/finalizer
   target overlap reduction and at least 1 hard-feasible repaired case

4. Agent 10P:
   update reports
   make semantic / violation / repair / final metrics visible

5. Agent 12P:
   contest optimizer = semantic rollout + finalizer + last fallback

6. Agent 8P:
   retrain BC using semantic candidates

7. Agent 13P / 14P:
   CI and docs
```

---

# 23. 一句話總結

更新後完整方向是：

> **不要讓 feasibility 在模型還沒學會 puzzle behavior 前就成為瓶頸。先讓 policy 學會 semantic placement intent，再用 violation-aware feedback 與 repair/finalizer 把它修成可提交的 hard-feasible floorplan。**
