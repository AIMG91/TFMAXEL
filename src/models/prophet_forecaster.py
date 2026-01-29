from __future__ import annotations

from typing import Dict
from warnings import filterwarnings

import numpy as np
import pandas as pd

from .base import BaseForecaster


class ProphetForecaster(BaseForecaster):
    def __init__(self) -> None:
        super().__init__(name="prophet_regressors", is_global=False)
        self._models: Dict[int, object] = {}
        self._store_means: Dict[int, float] = {}
        self._global_mean: float = float("nan")

    def fit(self, train_df: pd.DataFrame, config: Dict) -> None:
        filterwarnings("ignore")
        try:
            from prophet import Prophet
        except Exception as exc:
            raise ImportError("Prophet no está instalado. Instala con: pip install prophet") from exc

        exog_cols = config["exog_cols"]
        self._models = {}
        self._store_means = train_df.groupby("Store")["Weekly_Sales"].mean().to_dict()
        self._global_mean = float(train_df["Weekly_Sales"].mean())

        for store, g in train_df.groupby("Store"):
            g = g.sort_values("Date")
            train_p = g[["Date", "Weekly_Sales"] + exog_cols].copy()
            train_p = train_p.rename(columns={"Date": "ds", "Weekly_Sales": "y"})
            try:
                m = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                )
                for col in exog_cols:
                    m.add_regressor(col)
                m.fit(train_p)
                self._models[int(store)] = m
            except Exception:
                continue

    def predict(self, context_df: pd.DataFrame, future_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        exog_cols = config["exog_cols"]
        preds = []
        for store, g in future_df.groupby("Store"):
            g = g.sort_values("Date")
            test_p = g[["Date"] + exog_cols].copy().rename(columns={"Date": "ds"})
            m = self._models.get(int(store))
            if m is None:
                mean_val = self._store_means.get(int(store), self._global_mean)
                yhat = np.full(len(g), mean_val)
            else:
                try:
                    forecast = m.predict(test_p)
                    yhat = forecast["yhat"].values
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
