from __future__ import annotations

from typing import Dict
from warnings import filterwarnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import BaseForecaster


class SarimaxForecaster(BaseForecaster):
    def __init__(self) -> None:
        super().__init__(name="sarimax_exog", is_global=False)
        self._models: Dict[int, object] = {}
        self._store_means: Dict[int, float] = {}
        self._global_mean: float = float("nan")

    def fit(self, train_df: pd.DataFrame, config: Dict) -> None:
        filterwarnings("ignore")
        exog_cols = config["exog_cols"]
        self._models = {}
        self._store_means = train_df.groupby("Store")["Weekly_Sales"].mean().to_dict()
        self._global_mean = float(train_df["Weekly_Sales"].mean())

        for store, g in train_df.groupby("Store"):
            y_train = g["Weekly_Sales"].astype(float)
            X_train = g[exog_cols].astype(float)
            try:
                model = SARIMAX(
                    y_train,
                    exog=X_train,
                    order=config.get("sarimax_order", (1, 1, 1)),
                    seasonal_order=config.get("sarimax_seasonal_order", (0, 0, 0, 0)),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False)
                self._models[int(store)] = res
            except Exception:
                # fallback handled in predict
                continue

    def predict(self, context_df: pd.DataFrame, future_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        exog_cols = config["exog_cols"]
        preds = []
        for store, g in future_df.groupby("Store"):
            g = g.sort_values("Date")
            X_future = g[exog_cols].astype(float)
            res = self._models.get(int(store))
            if res is None:
                mean_val = self._store_means.get(int(store), self._global_mean)
                yhat = np.full(len(g), mean_val)
            else:
                try:
                    forecast = res.get_forecast(steps=len(g), exog=X_future)
                    yhat = forecast.predicted_mean.values
                except Exception:
                    mean_val = self._store_means.get(int(store), self._global_mean)
                    yhat = np.full(len(g), mean_val)

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
