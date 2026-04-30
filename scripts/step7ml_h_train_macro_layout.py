#!/usr/bin/env python3
"""Train Step7ML-H supervised macro closure layout baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.ml.supervised_macro_layout import train_macro_layout_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout-prior",
        type=Path,
        default=Path("artifacts/research/step7ml_g_layout_prior_examples.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/research"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models/step7ml_h"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()
    result = train_macro_layout_model(
        args.layout_prior,
        args.output_dir,
        args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
