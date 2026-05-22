from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class ForecastRequest(BaseModel):
    building_id: int
    horizon_hours: int = 24
    mode: str = "future"  # "future" | "test"


class ForecastResponse(BaseModel):
    building_id: int
    horizon_hours: int
    model_type: str = "lstm"
    timestamps: List[datetime] = []
    forecast_kwh: List[float] = []
    predictions: List[float] = []
    actuals: List[float] = []
    models: Optional[dict] = None
    metrics: Optional[dict] = None


class OptimizationRequest(BaseModel):
    building_id: int
    horizon_hours: int = 24
    peak_limit_kw: float
    comfort_min_kw: float
    comfort_max_kw: float
    equipment_min_kw: float
    equipment_max_kw: float


class OptimizationResponse(BaseModel):
    building_id: int
    horizon_hours: int
    strategy: str
    timestamps: List[datetime]
    optimized_load_kwh: List[float]
    original_cost: float
    expected_cost: float


class HistoricalDataResponse(BaseModel):
    building_id: int
    timestamps: List[datetime]
    energy_kwh: List[float]
    temperature_c: List[float]
    occupancy: List[float]


class MetricsResponse(BaseModel):
    model_type: str
    mae: float
    rmse: float
    mape: float
    r2: float

