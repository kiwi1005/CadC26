# HCFP-5090 檢查表

## 規格與資料

- [ ] pin official FloorSet commit / evaluator hash
- [ ] solve() 支援 `target_positions=None`
- [ ] hard/soft constraints parity tests 全通過
- [ ] official 100 validation 未進訓練
- [ ] internal split 有 topology/constraint stratification
- [ ] shard manifest 與 checksum 完成

## Greenfield 邊界

- [ ] runtime dependency 不含舊 DGLPR/Step7ML-P/SP/B*-tree legalizer
- [ ] 新 fallback 獨立實作
- [ ] 新 verifier 獨立實作且與官方對帳
- [ ] dependency audit/lockfile 完成

## Geometry / BDP

- [ ] soft area parameterization數值安全
- [ ] preplaced/fixed exact overwrite
- [ ] all boundary bitmasks測試
- [ ] cluster contact tree/connected union測試
- [ ] MIB compatibility interval測試
- [ ] direction beam/cycle recovery測試
- [ ] exact overlap margin測試
- [ ] fallback 100% hard-feasible

## 模型與訓練

- [ ] block permutation test
- [ ] D4 transform/inverse test
- [ ] initializer mode collapse監控
- [ ] one-step recovery promotion gate
- [ ] multi-step rollout無 NaN/divergence
- [ ] event replay包含 hard negatives
- [ ] direction labels來自 post-projection outcomes
- [ ] PVR報 top-1 regret與calibration
- [ ] self-improvement保留 gold/analytic teacher mixture

## RTX 5090

- [ ] CUDA 12.8+ / Blackwell環境確認
- [ ] neural BF16、geometry/BDP FP32
- [ ] static N/K/constraint buckets
- [ ] torch.compile graph breaks = 0 或有說明
- [ ] CUDA Graph warmup/capture測試
- [ ] inner loop無 `.item()`/CPU sync
- [ ] inner loop無 Shapely/NetworkX/SciPy/Python loops
- [ ] Triton只針對 profiler hotspot
- [ ] p50/p95/p99 per bucket報告
- [ ] cold-start與warm-start分開

## 評測與提交

- [ ] 5 seeds full validation
- [ ] 106–120 bucket單列
- [ ] 新增 cost=10 cases = 0
- [ ] exact verifier後才可更新 incumbent
- [ ] timeout 時回 best incumbent/fallback
- [ ] GPU unavailable自動切 portable profile
- [ ] checkpoint/config/code/evaluator hash打包
- [ ] 無網路下載依賴
- [ ] 100-case dry run
- [ ] submission freeze後只修封裝
