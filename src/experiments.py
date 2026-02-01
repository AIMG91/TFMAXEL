from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.common import evaluate_predictions, get_E2_test_mask, make_features, temporal_split_by_date


def _ensure_dirs(base_dir: Path, exp_id: str) -> Dict[str, Path]:
    exp_dir = base_dir / "experiments" / exp_id
    pred_dir = exp_dir
    metrics_dir = exp_dir
    fig_dir = exp_dir / "figures"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return {"exp": exp_dir, "pred": pred_dir, "metrics": metrics_dir, "fig": fig_dir}


def _compute_metrics_frames(pred_df: pd.DataFrame, model_name: str, group: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_global_df, metrics_by_store_df, _ = evaluate_predictions(
        pred_df,
        group_keys=["Store"],
        model_name=model_name,
        group_label=group,
    )
    return metrics_global_df, metrics_by_store_df


def _save_experiment_outputs(
    *,
    exp_dirs: Dict[str, Path],
    exp_id: str,
    model_name: str,
    pred_df: pd.DataFrame,
    metrics_global_df: pd.DataFrame,
    metrics_by_store_df: pd.DataFrame,
) -> None:
    pred_path = exp_dirs["pred"] / f"{model_name}_predictions.csv"
    global_path = exp_dirs["metrics"] / f"{model_name}_metrics_global.csv"
    store_path = exp_dirs["metrics"] / f"{model_name}_metrics_by_store.csv"

    pred_df.to_csv(pred_path, index=False)
    metrics_global_df.to_csv(global_path, index=False)
    metrics_by_store_df.to_csv(store_path, index=False)


def _plot_standard(pred_df: pd.DataFrame, model_name: str, fig_dir: Path, title_suffix: str = "") -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    top_stores = (
        pred_df.groupby("Store")["y_true"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
        .index
        .tolist()
    )

    for store in top_stores:
        g = pred_df[pred_df["Store"] == store].sort_values("Date")
        plt.figure(figsize=(10, 4))
        plt.plot(g["Date"], g["y_true"], label="y_true")
        plt.plot(g["Date"], g["y_pred"], label="y_pred")
        plt.title(f"Store {store} — {model_name} {title_suffix}".strip())
        plt.xlabel("Date")
        plt.ylabel("Weekly_Sales")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"{model_name}_plot_store_{store}.png", dpi=150)
        plt.close()

    errors = pred_df["y_true"] - pred_df["y_pred"]
    plt.figure(figsize=(8, 4))
    sns.histplot(errors, bins=30, kde=True)
    plt.title("Error distribution (y_true - y_pred)")
    plt.xlabel("Error")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{model_name}_plot_error_dist.png", dpi=150)
    plt.close()


def _prepare_features(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, List[str]]:
    df_feat, feature_cols = make_features(
        df,
        lags=config["lags"],
        rollings=config["rollings"],
        add_calendar=config.get("add_calendar", True),
        group_col="Store",
    )
    return df_feat, feature_cols


def _safe_fit_predict(model, train_df: pd.DataFrame, context_df: pd.DataFrame, future_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    model.fit(train_df, config)
    pred = model.predict(context_df, future_df, config)
    return pred


def run_E1_walk_forward(models: Iterable, df: pd.DataFrame, config: Dict) -> None:
    exp_id = "E1"
    base_dir = Path(config.get("output_dir", "outputs"))
    exp_dirs = _ensure_dirs(base_dir, exp_id)

    _, test_df, split_info = temporal_split_by_date(df, test_weeks=config["test_weeks"])
    test_dates = sorted(test_df["Date"].unique())

    for model in models:
        model_name = model.name
        all_preds = []
        for date in test_dates:
            train_slice = df[df["Date"] < date].copy()
            if config.get("rolling_window_weeks"):
                unique_dates = pd.Series(train_slice["Date"].unique()).sort_values()
                cutoff = unique_dates.iloc[-config["rolling_window_weeks"]]
                train_slice = train_slice[train_slice["Date"] >= cutoff]

            future_slice = df[df["Date"] == date].copy()

            local_cfg = dict(config)
            local_cfg["prediction_length"] = 1

            try:
                pred = _safe_fit_predict(model, train_slice, train_slice, future_slice, local_cfg)
            except ImportError as exc:
                print(f"[E1] Skipping {model_name}: {exc}")
                break

            pred = pred.merge(
                future_slice[["Store", "Date", "Weekly_Sales"]],
                on=["Store", "Date"],
                how="left",
            ).rename(columns={"Weekly_Sales": "y_true"})
            pred["model"] = model_name
            all_preds.append(pred)

        if not all_preds:
            continue

        pred_df = pd.concat(all_preds, ignore_index=True)
        metrics_global_df, metrics_by_store_df = _compute_metrics_frames(pred_df, model_name)
        _save_experiment_outputs(
            exp_dirs=exp_dirs,
            exp_id=exp_id,
            model_name=model_name,
            pred_df=pred_df,
            metrics_global_df=metrics_global_df,
            metrics_by_store_df=metrics_by_store_df,
        )
        _plot_standard(pred_df, model_name, exp_dirs["fig"], title_suffix="E1")


def run_E2_last_39_weeks(models: Iterable, df: pd.DataFrame, config: Dict) -> Dict[str, str]:
    exp_id = "E2"
    base_dir = Path(config.get("output_dir", "outputs"))
    exp_dirs = _ensure_dirs(base_dir, exp_id)

    test_mask, split_info = get_E2_test_mask(df, test_weeks=config["test_weeks"])
    test_df = df.loc[test_mask].copy()
    train_df = df.loc[~test_mask].copy()
    test_dates = sorted(pd.to_datetime(test_df["Date"]).unique())

    for model in models:
        model_name = model.name
        local_cfg = dict(config)
        local_cfg["prediction_length"] = len(test_dates)

        try:
            pred = _safe_fit_predict(model, train_df, train_df, test_df, local_cfg)
        except ImportError as exc:
            print(f"[E2] Skipping {model_name}: {exc}")
            continue

        pred_df = pred.merge(
            test_df[["Store", "Date", "Weekly_Sales"]],
            on=["Store", "Date"],
            how="left",
        ).rename(columns={"Weekly_Sales": "y_true"})
        pred_df["model"] = model_name
        metrics_global_df, metrics_by_store_df = _compute_metrics_frames(pred_df, model_name)
        _save_experiment_outputs(
            exp_dirs=exp_dirs,
            exp_id=exp_id,
            model_name=model_name,
            pred_df=pred_df,
            metrics_global_df=metrics_global_df,
            metrics_by_store_df=metrics_by_store_df,
        )
        _plot_standard(pred_df, model_name, exp_dirs["fig"], title_suffix="E2")

    return split_info


def run_E3_loso(models_global: Iterable, df: pd.DataFrame, config: Dict) -> None:
    exp_id = "E3"
    base_dir = Path(config.get("output_dir", "outputs"))
    exp_dirs = _ensure_dirs(base_dir, exp_id)

    train_base, test_df, split_info = temporal_split_by_date(df, test_weeks=config["test_weeks"])
    test_dates = sorted(test_df["Date"].unique())

    for model in models_global:
        model_name = model.name
        all_preds = []
        for store in df["Store"].unique():
            train_df = train_base[train_base["Store"] != store].copy()
            context_df = train_base[train_base["Store"] == store].copy()
            future_df = test_df[test_df["Store"] == store].copy()

            local_cfg = dict(config)
            local_cfg["prediction_length"] = len(test_dates)

            try:
                pred = _safe_fit_predict(model, train_df, context_df, future_df, local_cfg)
            except ImportError as exc:
                print(f"[E3] Skipping {model_name}: {exc}")
                break

            pred = pred.merge(
                future_df[["Store", "Date", "Weekly_Sales"]],
                on=["Store", "Date"],
                how="left",
            ).rename(columns={"Weekly_Sales": "y_true"})
            pred["model"] = model_name
            all_preds.append(pred)

        if not all_preds:
            continue

        pred_df = pd.concat(all_preds, ignore_index=True)
        metrics_global_df, metrics_by_store_df = _compute_metrics_frames(pred_df, model_name)
        _save_experiment_outputs(
            exp_dirs=exp_dirs,
            exp_id=exp_id,
            model_name=model_name,
            pred_df=pred_df,
            metrics_global_df=metrics_global_df,
            metrics_by_store_df=metrics_by_store_df,
        )
        _plot_standard(pred_df, model_name, exp_dirs["fig"], title_suffix="E3")


def run_E4_train35_test10low(models_global: Iterable, df: pd.DataFrame, config: Dict) -> None:
    exp_id = "E4"
    base_dir = Path(config.get("output_dir", "outputs"))
    exp_dirs = _ensure_dirs(base_dir, exp_id)

    train_base, test_df, split_info = temporal_split_by_date(df, test_weeks=config["test_weeks"])
    test_dates = sorted(test_df["Date"].unique())

    store_means = train_base.groupby("Store")["Weekly_Sales"].mean().sort_values()
    test_group = set(store_means.head(10).index.tolist())
    train_group = set(store_means.index.tolist()) - test_group

    for model in models_global:
        model_name = model.name
        local_cfg = dict(config)
        local_cfg["prediction_length"] = len(test_dates)

        train_df = train_base[train_base["Store"].isin(train_group)].copy()
        context_train_group = train_base[train_base["Store"].isin(train_group)].copy()
        context_test_group = train_base[train_base["Store"].isin(test_group)].copy()

        future_train_group = test_df[test_df["Store"].isin(train_group)].copy()
        future_test_group = test_df[test_df["Store"].isin(test_group)].copy()

        try:
            model.fit(train_df, local_cfg)
        except ImportError as exc:
            print(f"[E4] Skipping {model_name}: {exc}")
            continue

        pred_test = model.predict(context_test_group, future_test_group, local_cfg)
        pred_test = pred_test.merge(
            future_test_group[["Store", "Date", "Weekly_Sales"]],
            on=["Store", "Date"],
            how="left",
        ).rename(columns={"Weekly_Sales": "y_true"})
        pred_test["model"] = model_name

        pred_train = model.predict(context_train_group, future_train_group, local_cfg)
        pred_train = pred_train.merge(
            future_train_group[["Store", "Date", "Weekly_Sales"]],
            on=["Store", "Date"],
            how="left",
        ).rename(columns={"Weekly_Sales": "y_true"})
        pred_train["model"] = model_name

        metrics_global_test, metrics_by_store_test = _compute_metrics_frames(pred_test, model_name, group="TEST_GROUP")
        metrics_global_train, metrics_by_store_train = _compute_metrics_frames(pred_train, model_name, group="TRAIN_GROUP")

        metrics_global_df = pd.concat([metrics_global_train, metrics_global_test], ignore_index=True)
        metrics_by_store_df = pd.concat([metrics_by_store_train, metrics_by_store_test], ignore_index=True)

        _save_experiment_outputs(
            exp_dirs=exp_dirs,
            exp_id=exp_id,
            model_name=model_name,
            pred_df=pred_test,
            metrics_global_df=metrics_global_df,
            metrics_by_store_df=metrics_by_store_df,
        )
        _plot_standard(pred_test, model_name, exp_dirs["fig"], title_suffix="E4_TEST_GROUP")


def run_E5_unemployment_shock(models: Iterable, df: pd.DataFrame, config: Dict, shock_pct: float = 0.15) -> None:
    exp_id = "E5"
    base_dir = Path(config.get("output_dir", "outputs"))
    exp_dirs = _ensure_dirs(base_dir, exp_id)

    train_df, test_df, split_info = temporal_split_by_date(df, test_weeks=config["test_weeks"])
    test_dates = sorted(test_df["Date"].unique())

    for model in models:
        model_name = model.name
        local_cfg = dict(config)
        local_cfg["prediction_length"] = len(test_dates)

        try:
            model.fit(train_df, local_cfg)
        except ImportError as exc:
            print(f"[E5] Skipping {model_name}: {exc}")
            continue

        pred_base = model.predict(train_df, test_df, local_cfg)
        pred_base = pred_base.merge(
            test_df[["Store", "Date", "Weekly_Sales"]],
            on=["Store", "Date"],
            how="left",
        ).rename(columns={"Weekly_Sales": "y_true"})

        shocked = test_df.copy()
        shocked["Unemployment"] = shocked["Unemployment"] * (1.0 + shock_pct)
        pred_shocked = model.predict(train_df, shocked, local_cfg)

        base_mean = pred_base.groupby("Store")["y_pred"].mean()
        shocked_mean = pred_shocked.groupby("Store")["y_pred"].mean()

        summary = pd.DataFrame({
            "Store": base_mean.index.astype(int),
            "mean_pred_base": base_mean.values,
            "mean_pred_shocked": shocked_mean.reindex(base_mean.index).values,
        })
        summary["delta_abs"] = summary["mean_pred_shocked"] - summary["mean_pred_base"]
        summary["delta_pct"] = np.where(
            summary["mean_pred_base"] == 0,
            0.0,
            100.0 * summary["delta_abs"] / summary["mean_pred_base"],
        )

        global_row = pd.DataFrame(
            {
                "Store": ["__global__"],
                "mean_pred_base": [summary["mean_pred_base"].mean()],
                "mean_pred_shocked": [summary["mean_pred_shocked"].mean()],
                "delta_abs": [summary["delta_abs"].mean()],
                "delta_pct": [summary["delta_pct"].mean()],
            }
        )
        summary_full = pd.concat([summary, global_row], ignore_index=True)

        summary_path = exp_dirs["exp"] / f"{model_name}_shock_summary.csv"
        summary_full.to_csv(summary_path, index=False)

        # Figures
        import matplotlib.pyplot as plt
        import seaborn as sns

        top10 = summary.sort_values("delta_pct", ascending=False).head(10)
        plt.figure(figsize=(8, 4))
        sns.barplot(data=top10, x="Store", y="delta_pct")
        plt.title("Top 10 stores by delta_pct")
        plt.tight_layout()
        plt.savefig(exp_dirs["fig"] / f"{model_name}_shock_top10.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        sns.histplot(summary["delta_pct"], bins=30, kde=True)
        plt.title("Distribution of delta_pct")
        plt.tight_layout()
        plt.savefig(exp_dirs["fig"] / f"{model_name}_shock_delta_dist.png", dpi=150)
        plt.close()


def collect_summary_metrics(output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    rows = []
    for metrics_file in output_dir.glob("experiments/*/*_metrics_global.csv"):
        exp_id = metrics_file.parent.name
        df = pd.read_csv(metrics_file)
        for _, row in df.iterrows():
            record = {
                "experiment_id": exp_id,
                "model_name": row.get("model"),
                "MAE": row.get("MAE"),
                "RMSE": row.get("RMSE"),
                "sMAPE": row.get("sMAPE"),
                "WAPE": row.get("WAPE"),
                "notes": row.get("group", ""),
            }
            rows.append(record)
    return pd.DataFrame(rows)
