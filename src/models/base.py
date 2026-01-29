from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class BaseForecaster:
    name: str
    is_global: bool = False

    def fit(self, train_df: pd.DataFrame, config: Dict) -> None:
        raise NotImplementedError

    def predict(
        self,
        context_df: pd.DataFrame,
        future_df: pd.DataFrame,
        config: Dict,
    ) -> pd.DataFrame:
        raise NotImplementedError
