# HELP

## 目前狀態摘要（evidence-backed）
- **這個資料夾目前更像研究/協作工作區，不是已啟動的實作 repo。**
  - 證據：root 僅見 `.omx/`、`.skills/`、`AGENT.md`、`GUIDE.md`、`HELP.md`、`andrej-karpathy-skills-main/`。
  - 證據：未找到 `package.json`、`pyproject.toml`、`go.mod`、`Cargo.toml`、`Makefile`、`tsconfig.json`、`requirements.txt`。
  - 證據：未找到 `src/`、`tests/`、`scripts/`、`configs/`、`app/` 目錄。

- **`AGENT.md` 存在但目前是空檔。**
  - 證據：`AGENT.md` 為 `0 bytes`。

- **`GUIDE.md` 是完整的研究驗證計畫，不是可直接執行的 build/run 指南。**
  - 證據：`GUIDE.md` 共 `1057` 行。
  - 證據：章節明確覆蓋 `Dataset 與 evaluation setting`、`E0–E7`、`MVP scope`、`Milestone 1–6`、`Final recommendation`。

- **Karpathy-guidelines 在此任務主要意味著：先盤點、避免臆測、避免先蓋大框架。**
  - 證據：`~/.codex/skills/karpathy-guidelines/SKILL.md` 強調 `Think Before Coding`、`Simplicity First`、`Surgical Changes`、`Goal-Driven Execution`。
  - 推論：在未確認目標前，最安全的工作是文件整理、缺口記錄、與下一步選單，而不是 speculative implementation。

- **目前不是 git repository。**
  - 證據：`git status` 回報 `fatal: not a git repository`。
  - 影響：目前不適合要求 worker 以 commit-based workflow 交付程式變更；若要持續多人協作，之後最好明確決定是否初始化 git。

## 可安全先做的低風險工作
1. **把 `GUIDE.md` 轉成 operator-friendly 的執行入口。**
   - 例如整理成：
     - milestone checklist
     - experiment matrix
     - open questions / unblock list

2. **補齊 repo-level 導航文件。**
   - 最低風險的是補充：
     - 非空 `AGENT.md`
     - `README.md`
     - 更精煉的 `HELP.md`

3. **先定義最小可交付，而不是先寫框架。**
   - 建議先選一個最小目標：
     - 僅整理研究計畫
     - 建立最小實驗 scaffold
     - 建立 evaluator / pseudo-trajectory prototype

## 目前缺的關鍵資訊
1. **此 repo 的主要產出是什麼？**
   - A. 研究計畫整理
   - B. 實驗骨架/scaffold
   - C. 真正開始做 dataset/model/evaluator 實作
   - D. 單純整理 skill / agent workflow

2. **`GUIDE.md` 接下來應扮演哪種角色？**
   - 保持研究 memo
   - 拆成 PRD / milestones / task list
   - 轉成 implementation spec

3. **`AGENT.md` 空白是否刻意？**
   - 若不是刻意，應補 repo-level working agreement、build/test 入口、與禁止事項。

4. **是否要把這個資料夾正式變成可協作 repo？**
   - 若答案是 yes，至少需要：
     - git init / 遠端策略
     - README
     - 選定技術棧
     - 最小可執行命令

## 建議下一步選單（從低風險到較高風險）
### Option 1 — 研究整理模式（最低風險）
- 目標：把 `GUIDE.md` 拆成 `milestones + metrics + open questions`。
- 輸出：純文件；不碰 speculative code。
- 適合現在：**是**。

### Option 2 — 最小實驗啟動模式
- 先補這四個答案再動手：
  1. dataset 來源
  2. 預計語言/框架
  3. 第一個可執行命令
  4. 第一個驗證目標（建議 `E0 pseudo-trajectory sanity check`）
- 適合現在：**只有在你明確要開始實作時才適合**。

### Option 3 — 協作/倉庫整理模式
- 先補：`README.md`、非空 `AGENT.md`、git repo 初始化策略。
- 好處：後續 worker 才有穩定入口與提交流程。
- 風險：需要你確認這個資料夾是否真的要成為長期 repo。

## 目前建議（依 Karpathy-guidelines）
- **先不要做大型 implementation。**
- **先把目標類型講清楚。**
- 在未確認之前，最合理的持續工作是：
  1. evidence-backed inventory
  2. 文件導覽/決策清單
  3. 將 `GUIDE.md` 轉成更可執行的任務分解

## 本次 HELP 更新重點
- 新增 evidence-backed 現況摘要：說明此資料夾目前是研究/協作工作區，而非已啟動的實作 repo。
- 新增 operator 視角的低風險工作清單：只建議文件整理、導覽補強、與最小交付定義。
- 新增明確 open questions：主產出、GUIDE.md 角色、AGENT.md 是否刻意留白、是否需要正式 repo 化。
- 新增 next-step menu：研究整理 / 最小實驗啟動 / 協作整理 三種路徑。
- 新增環境限制：此目錄目前不是 git repository，且本輪依指示不初始化 git。

## 本輪盤點依據
- `AGENT.md` — 空檔
- `GUIDE.md` — 研究驗證計畫主文件
- `HELP.md` — operator help / open questions
- `.omx/context/team-intake-20260420T111200Z.md` — team intake snapshot
- `~/.codex/skills/karpathy-guidelines/SKILL.md` — 行為約束來源
- root directory listing / manifest search / source-dir search / `git status` 結果
