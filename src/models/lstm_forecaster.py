from __future__ import annotations

from typing import Dict, List, Tuple
from warnings import filterwarnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import BaseForecaster


class LSTMForecaster(BaseForecaster):
    def __init__(self) -> None:
        super().__init__(name="lstm_exog", is_global=True)
        self._model = None
        self._scaler_x: StandardScaler | None = None
        self._scaler_y: StandardScaler | None = None
        self._feature_cols: List[str] = []
        self._lookback: int = 52
        self._train_mean: float = 0.0
        self._device = None

    def _build_sequences(
        self, df_in: pd.DataFrame, feature_cols: List[str], lookback: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        sequences = []
        targets = []
        df_in = df_in.sort_values(["Store", "Date"]).copy()
        for store, g in df_in.groupby("Store"):
            g = g.sort_values("Date")
            X = self._scaler_x.transform(g[feature_cols].values)
            y = self._scaler_y.transform(g[["Weekly_Sales"]].values).ravel()
            for t in range(lookback, len(g)):
                sequences.append(X[t - lookback : t])
                targets.append(y[t])
        return np.array(sequences), np.array(targets)

    def fit(self, train_df: pd.DataFrame, config: Dict) -> None:
        filterwarnings("ignore")
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except Exception as exc:
            raise ImportError("PyTorch no está instalado. Instala con: pip install torch") from exc

        # If the incoming dataframe doesn't already contain engineered features,
        # build them here (keeps experiments compatible with raw df input).
        feature_cols = list(config.get("feature_cols", []))
        has_features = bool(feature_cols) and all(c in train_df.columns for c in feature_cols)
        if not has_features:
            from src.common import make_features

            train_df, feature_cols = make_features(
                train_df,
                lags=config.get("lags"),
                rollings=config.get("rollings"),
                add_calendar=bool(config.get("add_calendar", True)),
                group_col="Store",
            )

        # Drop rows with NaNs introduced by lags/rolling before fitting scalers.
        train_df = train_df.dropna(subset=list(feature_cols) + ["Weekly_Sales"]).copy()
        if train_df.empty:
            raise ValueError("Training dataframe is empty after feature engineering / NaN drop.")

        self._feature_cols = list(feature_cols)

        requested_lookback = int(config.get("lookback", 52))
        max_lookback = max(1, len(train_df))
        effective_lookback = min(requested_lookback, max_lookback)
        if effective_lookback < requested_lookback:
            print(
                f"[LSTM] Reducing lookback from {requested_lookback} to {effective_lookback}"
                f" due to limited training rows ({len(train_df)} after dropna)."
            )

        self._lookback = effective_lookback
        self._train_mean = float(train_df["Weekly_Sales"].mean())

        self._scaler_x = StandardScaler()
        self._scaler_y = StandardScaler()
        self._scaler_x.fit(train_df[self._feature_cols].values)
        self._scaler_y.fit(train_df[["Weekly_Sales"]].values)

        X_train, y_train = self._build_sequences(train_df, self._feature_cols, self._lookback)
        if len(X_train) == 0:
            raise ValueError(
                "Not enough data to build LSTM sequences (check lookback / feature NaNs / data length)."
            )

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            ),
            batch_size=int(config.get("batch_size", 64)),
            shuffle=True,
        )

        class LSTMRegressor(nn.Module):
            def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                return self.fc(last).squeeze(-1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        model = LSTMRegressor(input_size=X_train.shape[-1]).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-3)))

        epochs = int(config.get("epochs", 20))
        for _ in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        self._model = model

    def _compute_feature_vector(
        self,
        y_hist: List[float],
        date: pd.Timestamp,
        exog_row: pd.Series,
        config: Dict,
    ) -> List[float]:
        lags = config["lags"]
        rollings = config["rollings"]
        add_calendar = bool(config.get("add_calendar", True))
        exog_cols = config["exog_cols"]

        feat = {}
        for k in lags:
            feat[f"lag_{k}"] = y_hist[-k] if len(y_hist) >= k else np.nan
        for w in rollings:
            if len(y_hist) >= w:
                window = np.array(y_hist[-w:], dtype=float)
                feat[f"roll_mean_{w}"] = float(window.mean())
                feat[f"roll_std_{w}"] = float(window.std(ddof=0))
            else:
                feat[f"roll_mean_{w}"] = np.nan
                feat[f"roll_std_{w}"] = np.nan

        for c in exog_cols:
            feat[c] = float(exog_row[c])

        if add_calendar:
            iso = pd.Timestamp(date).isocalendar()
            feat["weekofyear"] = int(iso.week)
            feat["month"] = int(pd.Timestamp(date).month)
            feat["year"] = int(pd.Timestamp(date).year)

        vec = [feat.get(c, np.nan) for c in self._feature_cols]
        return vec

    def predict(self, context_df: pd.DataFrame, future_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        if self._model is None or self._scaler_x is None or self._scaler_y is None:
            raise RuntimeError("Model not fitted")

        import torch

        preds = []
        for store, g_future in future_df.groupby("Store"):
            g_future = g_future.sort_values("Date")
            g_hist = context_df[context_df["Store"] == store].sort_values("Date")
            if g_hist.empty:
                continue

            # Compute feature history from actuals
            y_hist = g_hist["Weekly_Sales"].tolist()
            feat_hist = []
            for _, row in g_hist.iterrows():
                feat_hist.append(
                    self._compute_feature_vector(
                        y_hist[: g_hist.index.get_loc(row.name) + 1],
                        row["Date"],
                        row,
                        config,
                    )
                )

            store_preds = []
            for _, row in g_future.iterrows():
                feat_vec = self._compute_feature_vector(y_hist, row["Date"], row, config)
                feat_hist.append(feat_vec)

                seq = np.array(feat_hist[-self._lookback :], dtype=float)
                # Fill NaNs with train mean
                seq = np.nan_to_num(seq, nan=self._train_mean)
                seq_scaled = self._scaler_x.transform(seq)
                xb = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    yhat_scaled = self._model(xb).cpu().numpy().ravel()[0]
                yhat = self._scaler_y.inverse_transform([[yhat_scaled]])[0][0]
                y_hist.append(float(yhat))
                store_preds.append(float(yhat))

            preds.append(
                pd.DataFrame(
                    {
                        "Store": g_future["Store"].values,
                        "Date": g_future["Date"].values,
                        "y_pred": store_preds,
                    }
                )
            )

        if not preds:
            return pd.DataFrame(columns=["Store", "Date", "y_pred"])

        return pd.concat(preds, ignore_index=True)
