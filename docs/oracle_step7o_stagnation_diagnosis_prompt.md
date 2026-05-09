# Oracle Review Request: CadC26 Step7 stagnation diagnosis and next architecture plan

You are reviewing `/home/hwchen/PROJ/CadC26`, a Python sidecar research repo for ICCAD 2026 FloorSet-Lite floorplanning experiments. Please act as an external research/architecture reviewer. The user feels the project is stuck: many Step7 experiments produce artifacts and gates, but visible objective improvement remains tiny or zero. We need a blunt diagnosis of why the current approach has little effect and a concrete next plan.

Please inspect the attached compact evidence file, docs, summaries, and implementation files. The key current milestone is Step7O Phase2 prior calibration, which completed with `decision=keep_prior_report_only`, `phase3_gate_open=false`, `gnn_rl_gate_open=false` because the concentration gate failed even though winner/archive preservation and regression reduction passed.

The central question: are we just filtering/ranking/perturbing the same weak candidate universe instead of creating a causally useful move/repack architecture? If yes, what should change?

Please produce:

1. Short verdict: the real bottleneck and why the user sees “原地踏步”.
2. Root-cause tree grounded in attached evidence.
3. Stop/keep/rework decisions for Step7L, Step7M, Step7N, Step7O.
4. Recommended next lane name and hypothesis.
5. Phase-by-phase executable plan with artifacts, files/scripts/tests, pass/fail gates, and kill criteria.
6. RL/GNN decision: keep closed or reopen; if reopen, define exact model target and minimum corpus metrics.
7. What to demote/delete as diagnostic-only.
8. Traps to avoid: micro-delta false positives, case024/025 concentration, validation leakage, scalar objective soup, candidate ranker without generator, soft-regression hiding behind HPWL.

Answer in zh-TW if possible. Be critical and specific; avoid generic “try a bigger model / tune hyperparameters” advice.
