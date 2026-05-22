"""
Run the API as:

    uvicorn backend.api:app --reload --port 8000

This module imports the main FastAPI app from `src/backend/main.py` and
adds upload-oriented endpoints for forecasting on user-provided datasets.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import Body, File, HTTPException, UploadFile

from services.anomaly import detect_anomalies
from services.explainability import get_feature_importance
from services.recommendations import generate_recommendations
from services.simulation import apply_simulation_scenario
from src.backend.db import EnergyRecord, get_db
from src.backend.forecast_from_df import forecast_from_dataframe
from src.backend.main import app

logger = logging.getLogger("energy_api.extensions")


@app.post("/forecast/upload")
async def forecast_from_upload(
    file: Optional[UploadFile] = File(default=None),
    horizon_hours: int = 24,
    mode: str = "future",
    records: Optional[List[Dict[str, Any]]] = Body(default=None),
) -> Dict[str, Any]:
    """
    Forecast from a user-provided dataset.

    Accepts either:
    - multipart/form-data with a CSV file field named `file`
    - JSON body with `records` as a list of row dicts (each dict is a record)
    """
    try:
        if file is not None:
            if not file.filename or not file.filename.lower().endswith(".csv"):
                raise HTTPException(status_code=400, detail="Please upload a .csv file.")
            raw = await file.read()
            try:
                df = pd.read_csv(io.BytesIO(raw))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Malformed CSV: {e}")
        else:
            if records is None:
                raise HTTPException(
                    status_code=400,
                    detail="Provide a CSV file upload or JSON `records` payload.",
                )
            try:
                df = pd.DataFrame.from_records(records)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid records payload: {e}")

        result = forecast_from_dataframe(df_raw=df, horizon_hours=int(horizon_hours), mode=mode)
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")


def _historical_df_from_db(building_id: int = 1, max_rows: int = 24 * 45) -> pd.DataFrame:
    db = next(get_db())
    try:
        rows = (
            db.query(EnergyRecord)
            .filter(EnergyRecord.building_id == building_id)
            .order_by(EnergyRecord.timestamp.desc())
            .limit(max_rows)
            .all()
        )
        rows = list(reversed(rows))
    finally:
        db.close()

    return pd.DataFrame(
        [
            {
                "building_id": r.building_id,
                "timestamp": r.timestamp,
                "energy_kwh": r.energy_kwh,
                "temperature_c": r.temperature_c,
                "occupancy": r.occupancy,
                "is_holiday": r.is_holiday,
                "price_per_kwh": r.price_per_kwh,
            }
            for r in rows
        ]
    )


@app.get("/recommendations")
def recommendations(building_id: int = 1) -> Dict[str, Any]:
    try:
        df = _historical_df_from_db(building_id=building_id)
        return generate_recommendations(df)
    except Exception:
        logger.exception("Recommendations endpoint failed.")
        return {
            "recommendations": [
                "Reduce HVAC usage during peak hours",
                "Shift energy usage to off-peak times",
            ]
        }


@app.post("/what-if")
def what_if(payload: Dict[str, float] = Body(...), building_id: int = 1, horizon_hours: int = 24) -> Dict[str, Any]:
    try:
        temperature = float(payload.get("temperature", 22.0))
        occupancy = float(payload.get("occupancy", 0.5))
        price = float(payload.get("price", 0.12))

        df = _historical_df_from_db(building_id=building_id)
        if df.empty:
            return {"modified_predictions": []}

        if "temperature_c" in df.columns:
            df["temperature_c"] = temperature
        if "occupancy" in df.columns:
            df["occupancy"] = occupancy
        if "price_per_kwh" in df.columns:
            df["price_per_kwh"] = price

        out = forecast_from_dataframe(df_raw=df, horizon_hours=horizon_hours, mode="future")
        return {"modified_predictions": out.get("forecast_kwh", [])}
    except Exception:
        logger.exception("What-if endpoint failed.")
        return {"modified_predictions": []}


@app.get("/explainability")
def explainability(building_id: int = 1) -> Dict[str, Any]:
    try:
        df = _historical_df_from_db(building_id=building_id)
        importance = get_feature_importance(model=None, data=df)
        return {"feature_importance": importance}
    except Exception:
        logger.exception("Explainability endpoint failed.")
        return {
            "feature_importance": {
                "temperature": 0.45,
                "occupancy": 0.30,
                "hour": 0.15,
            }
        }


@app.get("/anomalies")
def anomalies(building_id: int = 1) -> Dict[str, Any]:
    try:
        df = _historical_df_from_db(building_id=building_id)
        return detect_anomalies(df)
    except Exception:
        logger.exception("Anomaly endpoint failed.")
        return {"anomalies": []}


@app.post("/simulate")
def simulate(payload: Dict[str, Any] = Body(...), building_id: int = 1, horizon_hours: int = 24) -> Dict[str, Any]:
    try:
        scenario = str(payload.get("scenario", "weekday"))
        df = _historical_df_from_db(building_id=building_id)
        if df.empty:
            return {"simulation_predictions": []}

        modified = apply_simulation_scenario(df, scenario=scenario)
        out = forecast_from_dataframe(df_raw=modified, horizon_hours=horizon_hours, mode="future")
        return {"simulation_predictions": out.get("forecast_kwh", [])}
    except Exception:
        logger.exception("Simulation endpoint failed.")
        return {"simulation_predictions": []}


