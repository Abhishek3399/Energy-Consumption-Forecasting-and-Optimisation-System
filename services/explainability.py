from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def get_feature_importance(model: object, data: pd.DataFrame) -> Dict[str, float]:
    """
    Lightweight feature-importance approximation.
    This is intentionally decoupled from core model internals for compatibility.
    """
    if data is None or data.empty:
        return {"temperature": 0.45, "occupancy": 0.30, "hour": 0.15, "price": 0.10}

    work = data.copy()
    imp = {
        "temperature": 0.25,
        "occupancy": 0.25,
        "hour": 0.25,
        "price": 0.25,
    }
    if "temperature_c" in work.columns and "energy_kwh" in work.columns:
        c = np.corrcoef(
            pd.to_numeric(work["temperature_c"], errors="coerce").fillna(0.0),
            pd.to_numeric(work["energy_kwh"], errors="coerce").fillna(0.0),
        )[0, 1]
        imp["temperature"] = float(abs(c)) if np.isfinite(c) else imp["temperature"]

    if "occupancy" in work.columns and "energy_kwh" in work.columns:
        c = np.corrcoef(
            pd.to_numeric(work["occupancy"], errors="coerce").fillna(0.0),
            pd.to_numeric(work["energy_kwh"], errors="coerce").fillna(0.0),
        )[0, 1]
        imp["occupancy"] = float(abs(c)) if np.isfinite(c) else imp["occupancy"]

    if "timestamp" in work.columns:
        ts = pd.to_datetime(work["timestamp"], errors="coerce")
        hour = ts.dt.hour.fillna(0)
        if "energy_kwh" in work.columns:
            c = np.corrcoef(hour, pd.to_numeric(work["energy_kwh"], errors="coerce").fillna(0.0))[0, 1]
            imp["hour"] = float(abs(c)) if np.isfinite(c) else imp["hour"]

    if "price_per_kwh" in work.columns and "energy_kwh" in work.columns:
        c = np.corrcoef(
            pd.to_numeric(work["price_per_kwh"], errors="coerce").fillna(0.0),
            pd.to_numeric(work["energy_kwh"], errors="coerce").fillna(0.0),
        )[0, 1]
        imp["price"] = float(abs(c)) if np.isfinite(c) else imp["price"]

    total = sum(max(v, 0.0) for v in imp.values()) or 1.0
    return {k: float(max(v, 0.0) / total) for k, v in imp.items()}

