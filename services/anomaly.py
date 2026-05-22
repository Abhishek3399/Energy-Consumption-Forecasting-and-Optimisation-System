from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def detect_anomalies(df: pd.DataFrame, z_threshold: float = 3.0) -> Dict[str, List[Dict[str, Any]]]:
    """
    Z-score anomaly detector for energy spikes/drops.
    """
    if df is None or df.empty or "energy_kwh" not in df.columns:
        return {"anomalies": []}

    work = df.copy()
    work["energy_kwh"] = pd.to_numeric(work["energy_kwh"], errors="coerce")
    work = work.dropna(subset=["energy_kwh"])
    if len(work) < 10:
        return {"anomalies": []}

    mean = float(work["energy_kwh"].mean())
    std = float(work["energy_kwh"].std(ddof=0))
    if std <= 1e-9:
        return {"anomalies": []}

    z = (work["energy_kwh"] - mean) / std
    mask = z.abs() >= z_threshold

    anomalies: List[Dict[str, Any]] = []
    for _, row in work.loc[mask].iterrows():
        val = float(row["energy_kwh"])
        t = row.get("timestamp")
        anomalies.append(
            {
                "timestamp": pd.to_datetime(t).isoformat() if t is not None else None,
                "value": val,
                "type": "spike" if val > mean else "drop",
            }
        )
    return {"anomalies": anomalies}

