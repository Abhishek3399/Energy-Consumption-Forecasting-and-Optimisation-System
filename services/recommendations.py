from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def generate_recommendations(data: pd.DataFrame | None) -> Dict[str, List[str]]:
    """
    Rule-based recommendations module.
    Kept lightweight and independent from core forecast/optimize flows.
    """
    recs: List[str] = [
        "Reduce HVAC usage during peak hours",
        "Shift energy usage to off-peak times",
    ]

    if data is None or data.empty:
        return {"recommendations": recs}

    work = data.copy()
    if "price_per_kwh" in work.columns and "energy_kwh" in work.columns:
        hi_price = float(work["price_per_kwh"].quantile(0.8))
        hi_usage = float(work["energy_kwh"].quantile(0.8))
        risky = work[(work["price_per_kwh"] >= hi_price) & (work["energy_kwh"] >= hi_usage)]
        if len(risky) > 0:
            recs.append("High cost and high usage overlap detected; prioritize load shifting automation.")

    if "temperature_c" in work.columns:
        t95 = float(work["temperature_c"].quantile(0.95))
        recs.append(f"Set adaptive cooling strategy when temperature exceeds {t95:.1f} C.")

    return {"recommendations": recs}

