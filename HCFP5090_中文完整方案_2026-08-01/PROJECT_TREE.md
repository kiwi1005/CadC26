# 建議 standalone repository tree

```text
hcfp5090/
├── pyproject.toml
├── hcfp/
│   ├── api/
│   ├── data/
│   ├── geometry/
│   ├── model/
│   ├── dynamics/
│   ├── projection/
│   ├── verify/
│   ├── runtime/
│   └── fallback/
├── train/
├── eval/
├── tests/
├── configs/
└── submission/optimizer.py
```

此 tree 是 greenfield contract；禁止以舊專案 package 作 runtime dependency。
