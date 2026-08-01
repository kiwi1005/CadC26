# HCFP-5090：從零設計的 GPU 原生資料驅動式 Floorplanning 系統

**全名：Heterogeneous Collective FloorPlanner for RTX 5090**  
**中文名：異質區塊集體動力 Floorplanner**  
**文件版本：v0.9 Research Architecture**  
**調研截止日：2026-08-01**

> 本文件提出的是一套 **greenfield 新方案**。核心演算法、模型表示、推論流程、投影器、fallback 與軟體架構均從零設計；不依賴既有 DGLPR、Step7ML-P、Sequence-Pair、B*-tree、既有 legalizer、既有 candidate generator 或既有 Pareto gate。傳統方法只作為文獻比較與設計證據，不是執行時依賴。

## 證據層級

本套文件刻意區分四種內容：

1. **官方規格**：以 IntelLabs/FloorSet `iccad2026contest` 的 v10 README 與 evaluator 原始碼為準。
2. **已同行審查文獻**：Nature Communications、ICCAD、DAC、TCAD、NeurIPS、ICML 等正式出版物。
3. **2025–2026 預印本／近期工作**：僅作趨勢與風險參考，文件中均明確標為 preprint。
4. **HCFP-5090 設計提案**：本文件的新架構，尚未經實驗驗證；所有性能數字均是工程目標，不是已達成結果。

## 一句話技術定位

HCFP-5090 將每個不同面積、形狀與約束的 block 視為異質 agent，讓整個 block population 在 GPU 上透過 **多尺度 learned collective dynamics** 同步演化；當局部動力陷入停滯時，由 **event-triggered reconfiguration** 重建局部拓撲；最後以 **Batched Disjunctive Projection（BDP）** 選擇矩形非重疊析取方向並平行求解稀疏凸投影，得到可由 exact verifier 接受的 floorplan。

## 壓縮檔內容

| 檔案 | 用途 |
|---|---|
| `HCFP5090_完整技術報告.md` | 全部章節合併版，適合由頭閱讀 |
| `00_執行摘要與技術決策.md` | 技術結論、go/no-go 與最小可行版本 |
| `01_官方問題定義與評分規格.md` | v10 規格、hard/soft constraints、評分導向 |
| `02_文獻Survey_2026.md` | classical、GPU、RL、diffusion、swarm、learned physics、projection survey |
| `03_研究缺口與方法定位.md` | 為何需要 HCFP-5090、創新邊界與不可過度宣稱處 |
| `04_HCFP5090_完整系統架構.md` | 端到端模組圖、控制流、候選 population 管理 |
| `05_狀態表示與網路架構.md` | scene encoder、initializer、region/global fields、ranker |
| `06_集體動力學與事件式重配置.md` | typed interaction、latching、progressive contraction、ETR |
| `07_GPU原生析取約束投影.md` | BDP 數學模型、PDHG/ADMM、方向 beam、fallback |
| `08_資料工程與訓練策略.md` | 1M data shards、label、curriculum、DAgger、自我改進 |
| `09_RTX5090效能工程.md` | BF16/FP8/FP32 policy、CUDA Graph、Triton 優先順序 |
| `10_推論流程與提交封裝.md` | solve API、推論 state machine、portable profile |
| `11_實驗設計與消融.md` | 指標、ablation、acceptance gates、統計方法 |
| `12_里程碑_人力_風險.md` | 20 日 contest MVP 與 12 週完整研究版 |
| `13_API_資料結構_偽碼.md` | dataclass、module contract、核心偽碼 |
| `14_創新點與論文框架.md` | 可能論文主張、章節結構、需要證明的假設 |
| `CHECKLIST.md` | 開工、訓練、評測、submission freeze 檢查表 |
| `config/*.yaml` | RTX 5090 full 與 portable CPU 參考設定 |
| `pseudocode/*.py` | 模型、訓練、投影、推論骨架（非可直接執行產品碼） |
| `diagrams/*.mmd` | Mermaid 架構與流程圖 |
| `references/*` | BibTeX 與來源分級清單 |
| `MANIFEST.sha256` | 檔案完整性校驗 |

## 閱讀順序

先讀 `00 → 01 → 02 → 03 → 04` 形成整體判斷；實作者接著讀 `05 → 10 → 13`；訓練與實驗負責人讀 `08、11、12`；寫論文時讀 `14`。

## 核心決策摘要

- **不用離散 packing representation 作主幹**：不以 Sequence-Pair/B*-tree 表示搜索空間。
- **不用逐 block policy 作主幹**：所有 block 同步更新，避免 early-placement error accumulation。
- **不用 diffusion 作主幹**：採可控、短步、可觀察能量下降的 collective dynamics；生成多樣性由 population initializer 與事件分支提供。
- **不用「先生成、最後再補救」的弱 legalizer**：hard projection 是架構的一級元件，方向預測與投影結果會回饋訓練。
- **GPU 不是單一路徑加速器，而是 population search engine**：同時執行 32–96 條 trajectories 與 4–16 條 projection beams。
- **任何模型失敗都不能輸出 cost=10**：從零實作 deterministic shelf fallback；無 fixed outline 使此策略可保 hard feasibility。

## 重要限制

截至本文件調研日，公開 contest README 說明了 scoring 與 runtime normalization，但沒有公開承諾 final evaluation 使用 RTX 5090 或任何 GPU。故本方案提供 `5090_full` 與 `portable_cpu` 兩個 profile；是否允許 GPU、模型大小、套件與啟動時間，必須另外向主辦方確認。[R01][R02]
