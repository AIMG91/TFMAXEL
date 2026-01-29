from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: Tuple[str, ...] = (
    "Store",
    "Date",
    "Weekly_Sales",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
)

EXOG_COLUMNS: Tuple[str, ...] = (
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
)

DEFAULT_LAGS: Tuple[int, ...] = (1, 2, 4, 8, 52)
DEFAULT_ROLLINGS: Tuple[int, ...] = (4, 8, 12)


@dataclass(frozen=True)
class SplitConfig:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def as_dict(self) -> Dict[str, str]:
        return {
            "train_start": self.train_start.strftime("%Y-%m-%d"),
            "train_end": self.train_end.strftime("%Y-%m-%d"),
            "val_start": self.val_start.strftime("%Y-%m-%d"),
            "val_end": self.val_end.strftime("%Y-%m-%d"),
            "test_start": self.test_start.strftime("%Y-%m-%d"),
            "test_end": self.test_end.strftime("%Y-%m-%d"),
        }


def load_data(csv_path: str | os.PathLike) -> pd.DataFrame:
    """Load Walmart weekly sales dataset and perform basic validation.

    - Ensures required columns exist
    - Parses Date to datetime
    - Sorts by Store, Date
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    date_raw = df["Date"].astype(str).str.strip()
    # Intento 1: dd/mm/yyyy
    parsed_1 = pd.to_datetime(date_raw, format="%d/%m/%Y", errors="coerce")
    # Intento 2: mm/dd/yyyy
    parsed_2 = pd.to_datetime(date_raw, format="%m/%d/%Y", errors="coerce")
    # Intento 3: parseo flexible
    parsed_3 = pd.to_datetime(date_raw, errors="coerce")
    df["Date"] = parsed_1.fillna(parsed_2).fillna(parsed_3)
    if df["Date"].isna().any():
        n_bad = int(df["Date"].isna().sum())
        raise ValueError(f"Found {n_bad} rows with unparseable Date")

    df["Store"] = df["Store"].astype(int)
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    return df


def temporal_split(
    df: pd.DataFrame,
    *,
    val_weeks: int = 8,
    test_weeks: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitConfig]:
    """Global temporal split by unique weekly dates (same cutoffs for all stores).

    This avoids leakage by ensuring the split is purely time-based.

    Returns train/val/test dataframes and a SplitConfig.
    """

    if "Date" not in df.columns:
        raise ValueError("df must contain Date")

    unique_dates = pd.Series(pd.to_datetime(df["Date"]).unique()).sort_values().reset_index(drop=True)
    if len(unique_dates) < (val_weeks + test_weeks + 10):
        raise ValueError(
            "Not enough unique dates for requested split. "
            f"Have {len(unique_dates)} dates, need at least {val_weeks + test_weeks + 10}."
        )

    test_start = unique_dates.iloc[-test_weeks]
    val_start = unique_dates.iloc[-(test_weeks + val_weeks)]
    train_start = unique_dates.iloc[0]

    train_end = unique_dates[unique_dates < val_start].iloc[-1]
    val_end = unique_dates[unique_dates < test_start].iloc[-1]
    test_end = unique_dates.iloc[-1]

    split_cfg = SplitConfig(
        train_start=pd.Timestamp(train_start),
        train_end=pd.Timestamp(train_end),
        val_start=pd.Timestamp(val_start),
        val_end=pd.Timestamp(val_end),
        test_start=pd.Timestamp(test_start),
        test_end=pd.Timestamp(test_end),
    )

    train_df = df[(df["Date"] >= split_cfg.train_start) & (df["Date"] <= split_cfg.train_end)].copy()
    val_df = df[(df["Date"] >= split_cfg.val_start) & (df["Date"] <= split_cfg.val_end)].copy()
    test_df = df[(df["Date"] >= split_cfg.test_start) & (df["Date"] <= split_cfg.test_end)].copy()

    return train_df, val_df, test_df, split_cfg


def make_features(
    df: pd.DataFrame,
    *,
    lags: Sequence[int] = DEFAULT_LAGS,
    rollings: Sequence[int] = DEFAULT_ROLLINGS,
    add_calendar: bool = True,
    group_col: str = "Store",
) -> Tuple[pd.DataFrame, List[str]]:
    """Create leakage-safe features.

    - Target lags per Store
    - Rolling stats computed on y shifted by 1 (only past)
    - Exogenous variables aligned by Date (oracle exog is assumed available at prediction time)
    - Optional calendar features

    Returns (df_with_features, feature_columns)
    """

    required = set(REQUIRED_COLUMNS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for features: {missing}")

    out = df.sort_values([group_col, "Date"]).copy()
    g = out.groupby(group_col, group_keys=False)

    # Lags
    for k in lags:
        out[f"lag_{k}"] = g["Weekly_Sales"].shift(k)

    # Rolling stats on shifted target (so the current y is never used)
    y_shift_1 = g["Weekly_Sales"].shift(1)
    for w in rollings:
        out[f"roll_mean_{w}"] = (
            y_shift_1.rolling(w).mean().reset_index(level=0, drop=True)
        )
        out[f"roll_std_{w}"] = (
            y_shift_1.rolling(w).std().reset_index(level=0, drop=True)
        )

    # Calendar
    if add_calendar:
        iso = out["Date"].dt.isocalendar()
        out["weekofyear"] = iso.week.astype(int)
        out["month"] = out["Date"].dt.month.astype(int)
        out["year"] = out["Date"].dt.year.astype(int)

    feature_cols: List[str] = []
    feature_cols.extend([f"lag_{k}" for k in lags])
    for w in rollings:
        feature_cols.extend([f"roll_mean_{w}", f"roll_std_{w}"])
    feature_cols.extend(list(EXOG_COLUMNS))
    if add_calendar:
        feature_cols.extend(["weekofyear", "month", "year"])

    return out, feature_cols


def temporal_split_by_date(
    df: pd.DataFrame,
    *,
    train_end: Optional[pd.Timestamp] = None,
    test_weeks: Optional[int] = None,
    test_start: Optional[pd.Timestamp] = None,
    test_end: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Temporal split by explicit dates or last N weeks.

    Returns train/test dataframes and a dict with date boundaries.
    """

    if "Date" not in df.columns:
        raise ValueError("df must contain Date")

    unique_dates = (
        pd.Series(pd.to_datetime(df["Date"]).unique()).sort_values().reset_index(drop=True)
    )

    if test_weeks is not None and (test_start is None or test_end is None):
        if len(unique_dates) < (test_weeks + 2):
            raise ValueError("Not enough dates for requested test_weeks")
        test_start = unique_dates.iloc[-test_weeks]
        test_end = unique_dates.iloc[-1]

    if test_start is None:
        raise ValueError("test_start is required if test_weeks is not provided")

    if test_end is None:
        test_end = unique_dates.iloc[-1]

    if train_end is None:
        train_end = unique_dates[unique_dates < test_start].iloc[-1]

    train_df = df[df["Date"] <= train_end].copy()
    test_df = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)].copy()

    split_info = {
        "train_end": pd.Timestamp(train_end).strftime("%Y-%m-%d"),
        "test_start": pd.Timestamp(test_start).strftime("%Y-%m-%d"),
        "test_end": pd.Timestamp(test_end).strftime("%Y-%m-%d"),
    }

    return train_df, test_df, split_info


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, sMAPE."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    denom = np.abs(y_true) + np.abs(y_pred)
    # Avoid division by zero: if denom==0, define contribution as 0
    smape = np.where(denom == 0, 0.0, 200.0 * np.abs(y_true - y_pred) / denom)
    smape = float(np.mean(smape))

    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape}


def save_outputs(
    *,
    model_name: str,
    predictions: pd.DataFrame,
    metrics_global: pd.DataFrame,
    metrics_by_store: pd.DataFrame,
    output_dir: str | os.PathLike = "outputs",
) -> Dict[str, str]:
    """Save standardized outputs for a model.

    Expected columns in predictions: Store, Date, y_true, y_pred, model
    """

    output_dir = Path(output_dir)
    pred_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pred_path = pred_dir / f"{model_name}_predictions.csv"
    global_path = metrics_dir / f"{model_name}_metrics_global.csv"
    store_path = metrics_dir / f"{model_name}_metrics_by_store.csv"

    predictions.to_csv(pred_path, index=False)
    metrics_global.to_csv(global_path, index=False)
    metrics_by_store.to_csv(store_path, index=False)

    return {
        "predictions": str(pred_path),
        "metrics_global": str(global_path),
        "metrics_by_store": str(store_path),
    }


def write_metadata(
    metadata_path: str | os.PathLike,
    *,
    split_cfg: SplitConfig,
    feature_cols: Sequence[str],
    seed: int,
    libs: Optional[Dict[str, str]] = None,
    notes: Optional[Dict[str, str]] = None,
) -> None:
    """Write outputs/metadata.json with split, features, seed, and library versions."""

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(seed),
        "split": split_cfg.as_dict(),
        "features": list(feature_cols),
        "libs": libs or {},
        "notes": notes or {},
    }

    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
