from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet

from .base import EnergyForecastModel


class ProphetForecaster(EnergyForecastModel):
    name = "prophet"

    def __init__(self) -> None:
        self.model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:  # type: ignore[override]
        raise NotImplementedError(
            "Use fit_from_dataframe with timestamp and target columns for Prophet."
        )

    def fit_from_dataframe(
        self, df: pd.DataFrame, timestamp_col: str = "timestamp", target_col: str = "energy_kwh"
    ) -> None:
        df_prophet = pd.DataFrame(
            {"ds": pd.to_datetime(df[timestamp_col]), "y": df[target_col].values}
        )
        self.model.fit(df_prophet)

    def predict(  # type: ignore[override]
        self, x: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError(
            "Use predict_horizon with future timestamps for Prophet."
        )

    def predict_horizon(self, periods: int, freq: str = "H") -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast

    def save(self, path: Path) -> None:  # type: ignore[override]
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:  # type: ignore[override]
        self.model = joblib.load(path)

    def get_params(self) -> Dict[str, Any]:  # type: ignore[override]
        return {}

