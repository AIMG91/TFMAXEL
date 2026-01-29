from __future__ import annotations

from typing import Dict, List
from warnings import filterwarnings

import numpy as np
import pandas as pd

from .base import BaseForecaster


class DeepARForecaster(BaseForecaster):
    def __init__(self) -> None:
        super().__init__(name="deepar_exog", is_global=True)
        self._predictor = None
        self._freq: str | None = None
        self._prediction_length: int | None = None

    def fit(self, train_df: pd.DataFrame, config: Dict) -> None:
        filterwarnings("ignore")
        try:
            from gluonts.dataset.common import ListDataset
            from gluonts.dataset.field_names import FieldName
            from gluonts.torch.model.deepar import DeepAREstimator
        except Exception as exc:
            raise ImportError("GluonTS no está instalado. Instala con: pip install 'gluonts[torch]'") from exc

        exog_cols: List[str] = config["exog_cols"]
        prediction_length = int(config.get("prediction_length", 1))

        dates_all = train_df.sort_values("Date")["Date"].drop_duplicates().values
        freq = pd.infer_freq(pd.to_datetime(dates_all)) or "W-FRI"

        def build_series(store_df: pd.DataFrame):
            store_df = store_df.sort_values("Date")
            y = store_df["Weekly_Sales"].values.astype(float)
            exog = store_df[exog_cols].values.astype(float).T
            start = pd.Timestamp(store_df["Date"].iloc[0])
            return y, exog, start

        train_records = []
        for store, g in train_df.groupby("Store"):
            y, exog, start = build_series(g)
            train_records.append(
                {
                    FieldName.TARGET: y,
                    FieldName.START: start,
                    FieldName.FEAT_DYNAMIC_REAL: exog,
                    FieldName.ITEM_ID: str(int(store)),
                }
            )

        train_ds = ListDataset(train_records, freq=freq)

        estimator = DeepAREstimator(
            prediction_length=prediction_length,
            context_length=int(config.get("context_length", 52)),
            freq=freq,
            num_feat_dynamic_real=len(exog_cols),
            batch_size=int(config.get("batch_size", 32)),
            num_batches_per_epoch=int(config.get("num_batches_per_epoch", 50)),
            lr=float(config.get("lr", 1e-3)),
            num_layers=int(config.get("num_layers", 2)),
            hidden_size=int(config.get("hidden_size", 40)),
            dropout_rate=float(config.get("dropout_rate", 0.1)),
            scaling=True,
            num_parallel_samples=int(config.get("num_parallel_samples", 100)),
            trainer_kwargs={"max_epochs": int(config.get("epochs", 20))},
        )

        self._predictor = estimator.train(train_ds)
        self._freq = freq
        self._prediction_length = prediction_length

    def predict(self, context_df: pd.DataFrame, future_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        if self._predictor is None:
            raise RuntimeError("Model not fitted")

        from gluonts.dataset.common import ListDataset
        from gluonts.dataset.field_names import FieldName

        exog_cols: List[str] = config["exog_cols"]
        prediction_length = int(config.get("prediction_length", 1))

        pred_records = []
        for store, g_future in future_df.groupby("Store"):
            g_future = g_future.sort_values("Date")
            g_hist = context_df[context_df["Store"] == store].sort_values("Date")
            if g_hist.empty:
                continue

            combined = pd.concat([g_hist, g_future], axis=0)
            y = g_hist["Weekly_Sales"].values.astype(float)
            exog_full = combined[exog_cols].values.astype(float).T
            start = pd.Timestamp(combined["Date"].iloc[0])
            pred_records.append(
                {
                    FieldName.TARGET: y,
                    FieldName.START: start,
                    FieldName.FEAT_DYNAMIC_REAL: exog_full,
                    FieldName.ITEM_ID: str(int(store)),
                }
            )

        pred_ds = ListDataset(pred_records, freq=self._freq)
        store_preds = {}
        for forecast, item in zip(self._predictor.predict(pred_ds), pred_records):
            store_id = int(item[FieldName.ITEM_ID])
            store_preds[store_id] = forecast.mean

        preds = []
        for store, g in future_df.groupby("Store"):
            g = g.sort_values("Date")
            yhat = store_preds.get(int(store))
            if yhat is None:
                yhat = np.full(len(g), context_df["Weekly_Sales"].mean())
            else:
                yhat = np.asarray(yhat)[: len(g)]
                if len(yhat) < len(g):
                    yhat = np.pad(yhat, (0, len(g) - len(yhat)), constant_values=yhat[-1])
            preds.append(
                pd.DataFrame(
                    {
                        "Store": g["Store"].values,
                        "Date": g["Date"].values,
                        "y_pred": yhat,
                    }
                )
            )

        if not preds:
            return pd.DataFrame(columns=["Store", "Date", "y_pred"])

        return pd.concat(preds, ignore_index=True)
