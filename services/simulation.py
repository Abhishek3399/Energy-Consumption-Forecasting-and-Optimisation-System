from __future__ import annotations

from typing import Dict

import pandas as pd


def apply_simulation_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Applies scenario transformations to a feature dataframe.
    """
    if df is None or df.empty:
        return df

    work = df.copy()
    scenario = (scenario or "").strip().lower()

    if scenario == "summer":
        if "temperature_c" in work.columns:
            work["temperature_c"] = pd.to_numeric(work["temperature_c"], errors="coerce").fillna(0.0) + 5.0
    elif scenario == "winter":
        if "temperature_c" in work.columns:
            work["temperature_c"] = pd.to_numeric(work["temperature_c"], errors="coerce").fillna(0.0) - 5.0
    elif scenario == "weekday":
        if "occupancy" in work.columns:
            work["occupancy"] = pd.to_numeric(work["occupancy"], errors="coerce").fillna(0.0) * 1.15
    elif scenario == "weekend":
        if "occupancy" in work.columns:
            work["occupancy"] = pd.to_numeric(work["occupancy"], errors="coerce").fillna(0.0) * 0.75
    elif scenario == "office":
        if "occupancy" in work.columns:
            work["occupancy"] = pd.to_numeric(work["occupancy"], errors="coerce").fillna(0.0) * 1.10
        if "price_per_kwh" in work.columns:
            work["price_per_kwh"] = pd.to_numeric(work["price_per_kwh"], errors="coerce").fillna(0.0) * 1.05
    elif scenario == "factory":
        if "occupancy" in work.columns:
            work["occupancy"] = pd.to_numeric(work["occupancy"], errors="coerce").fillna(0.0) * 1.30
        if "price_per_kwh" in work.columns:
            work["price_per_kwh"] = pd.to_numeric(work["price_per_kwh"], errors="coerce").fillna(0.0) * 1.08

    return work

