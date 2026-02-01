from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _torch_weights_only_workaround() -> None:
    """PyTorch 2.6 changed torch.load(weights_only=True) default.

    Some third-party checkpoints (e.g., Lightning/GluonTS) require allowlisting
    certain globals for weights-only loading. We only allowlist functools.partial,
    which is commonly used in these checkpoints.
    """

    try:
        import torch
        from functools import partial

        if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([partial])
    except Exception:
        # If torch isn't installed yet (or older versions), do nothing.
        return

import numpy as np


def _project_root() -> Path:
    # scripts/run_all_experiments.py -> project root
    return Path(__file__).resolve().parents[1]


def main() -> int:
    _torch_weights_only_workaround()

    parser = argparse.ArgumentParser(description="Run all experiments (E1–E5) with all available models.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lookback", type=int, default=52)
    parser.add_argument("--test-weeks", type=int, default=39)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default=None, help="Path to Walmart_Sales.csv (defaults to data/Walmart_Sales.csv)")
    parser.add_argument("--output-dir", type=str, default=None, help="Outputs folder (defaults to outputs/)")
    parser.add_argument("--shock-pct", type=float, default=0.15)
    args = parser.parse_args()

    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.common import TEST_WEEKS, load_data, make_features, validate_split_consistency
    from src.experiments import (
        collect_summary_metrics,
        run_E1_walk_forward,
        run_E2_last_39_weeks,
        run_E3_loso,
        run_E4_train35_test10low,
        run_E5_unemployment_shock,
    )
    from src.models import (
        DeepARForecaster,
        LSTMForecaster,
        ProphetForecaster,
        SarimaxForecaster,
        TransformerForecaster,
    )

    np.random.seed(args.seed)

    data_path = Path(args.data) if args.data else (project_root / "data" / "Walmart_Sales.csv")
    output_dir = Path(args.output_dir) if args.output_dir else (project_root / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "seed": args.seed,
        "lags": [1, 2, 4, 8, 52],
        "rollings": [4, 8, 12],
        "exog_cols": ["Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"],
        "add_calendar": True,
        "lookback": args.lookback,
        "test_weeks": args.test_weeks if args.test_weeks is not None else TEST_WEEKS,
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }

    print(f"Project root: {project_root}")
    print(f"Data path: {data_path}")
    print(f"Outputs: {output_dir}")

    df = load_data(data_path)
    _, feature_cols = make_features(
        df,
        lags=config["lags"],
        rollings=config["rollings"],
        add_calendar=config["add_calendar"],
        group_col="Store",
    )
    config["feature_cols"] = feature_cols

    models = [
        SarimaxForecaster(),
        ProphetForecaster(),
        DeepARForecaster(),
        LSTMForecaster(),
        TransformerForecaster(),
    ]

    run_E1_walk_forward(models, df, config)
    run_E2_last_39_weeks(models, df, config)

    models_global = [m for m in models if getattr(m, "is_global", False)]
    run_E3_loso(models_global, df, config)
    run_E4_train35_test10low(models_global, df, config)

    run_E5_unemployment_shock(models, df, config, shock_pct=float(args.shock_pct))

    summary = collect_summary_metrics(output_dir)
    summary_path = output_dir / "experiments" / "summary_metrics.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    print(f"Wrote summary: {summary_path}")
    validation = validate_split_consistency(output_dir, test_weeks=config["test_weeks"])
    print(f"Split validation: {validation}")
    if not validation["ok"]:
        print("Split validation failed:", validation["issues"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
