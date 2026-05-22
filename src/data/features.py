from __future__ import annotations

import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["dayofyear"] = ts.dt.dayofyear
    df["month"] = ts.dt.month

    # cyclic encodings for hour and dayofweek
    hour_rad = df["hour"].to_numpy(dtype=float) / 24.0 * 2.0 * np.pi
    dow_rad = df["dayofweek"].to_numpy(dtype=float) / 7.0 * 2.0 * np.pi
    df["hour_sin"] = np.sin(hour_rad)
    df["hour_cos"] = np.cos(hour_rad)
    df["dow_sin"] = np.sin(dow_rad)
    df["dow_cos"] = np.cos(dow_rad)
    return df


def add_price_and_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "price_per_kwh" not in df.columns:
        raise ValueError("price_per_kwh column is required")
    df["cost"] = df["energy_kwh"] * df["price_per_kwh"]
    # simple lag features
    df = df.sort_values(["building_id", "timestamp"])
    for lag in [1, 24]:
        df[f"energy_kwh_lag_{lag}"] = (
            df.groupby("building_id")["energy_kwh"].shift(lag)
        )
    return df

