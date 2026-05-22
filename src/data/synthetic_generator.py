from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


def _generate_base_datetime_index(
    start: str = "2022-01-01", end: Optional[str] = None, periods: Optional[int] = None
) -> pd.DatetimeIndex:
    if periods is not None:
        return pd.date_range(start=start, periods=periods, freq="H")
    if end is None:
        end = (datetime.fromisoformat(start) + timedelta(days=365 * 3)).strftime(
            "%Y-%m-%d"
        )
    return pd.date_range(start=start, end=end, freq="H")


def generate_synthetic_energy_data(
    start: str = "2022-01-01",
    years: int = 3,
    seed: int = 42,
    building_ids: int = 3,
) -> pd.DataFrame:
    """
    Generate realistic multi-building hourly synthetic dataset.

    Columns:
    - building_id
    - timestamp
    - energy_kwh
    - temperature_c
    - occupancy
    - is_holiday
    - price_per_kwh
    """
    np.random.seed(seed)

    periods = years * 365 * 24
    idx = _generate_base_datetime_index(start=start, periods=periods)

    records = []
    for b in range(building_ids):
        # base seasonal patterns
        daily_pattern = np.sin(2 * np.pi * (idx.hour / 24.0 - 0.25))  # peak late afternoon
        weekly_pattern = np.where(idx.dayofweek < 5, 1.0, 0.7)  # lower on weekends
        seasonal_pattern = 0.6 + 0.4 * np.sin(
            2 * np.pi * (idx.dayofyear / 365.0 - 0.1)
        )  # yearly seasonality

        # temperature: colder winters, warmer summers
        temperature = 10 + 10 * np.sin(
            2 * np.pi * (idx.dayofyear / 365.0 - 0.3)
        ) + np.random.normal(0, 3, size=len(idx))

        # occupancy: higher in business hours, lower at night
        base_occ = np.where((idx.hour >= 8) & (idx.hour <= 18), 1.0, 0.2)
        weekly_occ = np.where(idx.dayofweek < 5, 1.0, 0.5)
        occupancy = base_occ * weekly_occ + np.random.normal(0, 0.05, size=len(idx))
        occupancy = np.clip(occupancy, 0, 1.2)

        # holiday calendar (simple heuristic: fewer occupied days)
        is_weekend = idx.dayofweek >= 5
        # random public-holiday-like days
        random_holidays = np.random.binomial(1, 0.02, size=len(idx)).astype(bool)
        is_holiday = np.logical_or(is_weekend, random_holidays)

        # price signal: higher during peak hours and high demand
        base_price = 0.12
        peak_hours = ((idx.hour >= 17) & (idx.hour <= 21)).astype(float)
        seasonal_price = 0.02 * seasonal_pattern
        price_per_kwh = base_price + 0.05 * peak_hours + seasonal_price

        # energy consumption: combination of patterns, temp, occupancy, noise
        temp_sensitivity = 0.4 + 0.2 * np.random.rand()
        occ_sensitivity = 2.0 + 1.0 * np.random.rand()
        base_load = 5 + 2 * np.random.rand()

        energy = (
            base_load
            + 1.5 * daily_pattern
            + 1.0 * seasonal_pattern
            + temp_sensitivity * np.maximum(0, 25 - temperature)  # heating
            + temp_sensitivity * np.maximum(0, temperature - 20)  # cooling
            + occ_sensitivity * occupancy
        )
        # reduce on holidays
        energy = np.where(is_holiday, energy * 0.7, energy)

        # add noise and non-negativity
        energy += np.random.normal(0, 0.5, size=len(idx))
        energy = np.clip(energy, 0.5, None)

        df_b = pd.DataFrame(
            {
                "building_id": b + 1,
                "timestamp": idx,
                "energy_kwh": energy,
                "temperature_c": temperature,
                "occupancy": occupancy,
                "is_holiday": is_holiday.astype(int),
                "price_per_kwh": price_per_kwh,
            }
        )
        records.append(df_b)

    df = pd.concat(records, ignore_index=True)
    return df


if __name__ == "__main__":
    df = generate_synthetic_energy_data()
    print(df.head())

