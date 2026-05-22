"""
Modular Plotly-based visualization components for the Energy Forecasting dashboard.
All charts accept DataFrames and options; they are backend-agnostic and update dynamically.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# --- Column name resolution (aliases) ---
def _ts_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).lower().strip() in ("timestamp", "time", "datetime", "date_time", "date"):
            return c
    return None


def _energy_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).lower().strip() in ("energy_kwh", "energy", "consumption", "load", "kwh"):
            return c
    return None


def _temp_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).lower().strip() in ("temperature_c", "temperature", "temp", "temp_c"):
            return c
    return None


def _price_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).lower().strip() in ("price_per_kwh", "price", "tariff", "rate"):
            return c
    return None


def ensure_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with standard column names for internal use."""
    out = df.copy()
    ts, en, temp, price = _ts_col(out), _energy_col(out), _temp_col(out), _price_col(out)
    if ts:
        out["timestamp"] = pd.to_datetime(out[ts], errors="coerce")
    if en:
        out["energy_kwh"] = pd.to_numeric(out[en], errors="coerce")
    if temp:
        out["temperature_c"] = pd.to_numeric(out[temp], errors="coerce")
    if price:
        out["price_per_kwh"] = pd.to_numeric(out[price], errors="coerce")
    if "temperature_c" not in out.columns:
        out["temperature_c"] = 20.0
    if "price_per_kwh" not in out.columns:
        out["price_per_kwh"] = 0.12
    return out.dropna(subset=["timestamp", "energy_kwh"]).sort_values("timestamp").reset_index(drop=True)


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort hourly normalization for time series.

    - Drops NaT timestamps
    - Sorts by timestamp
    - If cadence is inconsistent, resamples to hourly

    Notes:
    - For higher-frequency data (multiple rows per hour), `energy_kwh` is summed per hour.
    - For lower-frequency/missing hours, missing values are filled by interpolation/ffill.
    """
    work = ensure_standard_columns(df)
    if work.empty:
        return work

    work = work.set_index("timestamp").sort_index()
    diffs = work.index.to_series().diff().dropna()
    if diffs.empty:
        return work.reset_index()

    median_seconds = float(diffs.dt.total_seconds().median())
    # If not close to 1 hour (+/- 10 minutes), normalize to hourly.
    if not (3000.0 <= median_seconds <= 4200.0):
        # Determine if we likely have multiple samples per hour
        counts = work["energy_kwh"].resample("h").count()
        multi_per_hour = float(counts.median()) > 1.0

        agg_energy = "sum" if multi_per_hour else "mean"
        hourly = pd.DataFrame(index=work.resample("h").mean().index)
        hourly["energy_kwh"] = getattr(work["energy_kwh"].resample("h"), agg_energy)()

        for col in ("temperature_c", "price_per_kwh"):
            if col in work.columns:
                hourly[col] = work[col].resample("h").mean()

        # Fill gaps
        hourly["energy_kwh"] = hourly["energy_kwh"].interpolate(limit_direction="both")
        for col in ("temperature_c", "price_per_kwh"):
            if col in hourly.columns:
                hourly[col] = hourly[col].interpolate(limit_direction="both").ffill().bfill()

        work = hourly

    return work.reset_index().rename(columns={"index": "timestamp"})

# ---------------------------------------------------------------------------
# 1) Energy consumption time-series: historical vs predicted, with zoom & date range
# ---------------------------------------------------------------------------
def chart_energy_timeseries(
    df: pd.DataFrame,
    date_min: Optional[pd.Timestamp] = None,
    date_max: Optional[pd.Timestamp] = None,
    forecast_ts: Optional[pd.DatetimeIndex] = None,
    forecast_kwh: Optional[List[float]] = None,
    title: str = "Energy Consumption: Historical vs Predicted",
) -> go.Figure:
    work = ensure_standard_columns(df)
    if work.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False, font=dict(size=16))

    if date_min is not None:
        work = work[work["timestamp"] >= date_min]
    if date_max is not None:
        work = work[work["timestamp"] <= date_max]
    if work.empty:
        return go.Figure().add_annotation(text="No data in selected date range", showarrow=False, font=dict(size=16))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=work["timestamp"],
            y=work["energy_kwh"],
            mode="lines",
            name="Historical",
            line=dict(color="#1f77b4", width=1.5),
        )
    )
    if forecast_ts is not None and forecast_kwh is not None and len(forecast_ts) == len(forecast_kwh):
        fig.add_trace(
            go.Scatter(
                x=forecast_ts,
                y=forecast_kwh,
                mode="lines+markers",
                name="Predicted",
                line=dict(color="#ff7f0e", width=2, dash="dot"),
                marker=dict(size=4),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Energy (kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        xaxis_rangeslider_visible=True,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 2) Forecast vs actual comparison (when actuals overlap forecast period)
# ---------------------------------------------------------------------------
def chart_forecast_vs_actual(
    actual_ts: pd.DatetimeIndex,
    actual_kwh: List[float],
    forecast_ts: pd.DatetimeIndex,
    forecast_kwh: List[float],
    title: str = "Forecast vs Actual",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual_ts,
            y=actual_kwh,
            mode="lines+markers",
            name="Actual",
            line=dict(color="#2ca02c", width=2),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_ts,
            y=forecast_kwh,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#d62728", width=2, dash="dash"),
            marker=dict(size=6),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Energy (kWh)",
        height=400,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# 3) Energy usage heatmap: hour-of-day vs day-of-week
# ---------------------------------------------------------------------------
def chart_energy_heatmap(
    df: pd.DataFrame,
    title: str = "Energy Usage by Hour of Day & Day of Week",
) -> go.Figure:
    work = ensure_standard_columns(df)
    if work.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False, font=dict(size=16))

    work = work.copy()
    work["hour"] = work["timestamp"].dt.hour
    work["dayofweek"] = work["timestamp"].dt.dayofweek
    heat_data = work.groupby(["dayofweek", "hour"])["energy_kwh"].mean().reset_index()
    pivot = heat_data.pivot(index="dayofweek", columns="hour", values="energy_kwh")
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.index = [day_names[i] if i < 7 else str(i) for i in pivot.index]

    fig = px.imshow(
        pivot,
        labels=dict(x="Hour of day", y="Day of week", color="Avg kWh"),
        aspect="auto",
        color_continuous_scale="Viridis",
        x=[int(c) for c in pivot.columns],
        y=pivot.index,
    )
    fig.update_layout(
        title=title,
        height=380,
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# 4) Energy cost breakdown (bar or by category/time period)
# ---------------------------------------------------------------------------
def chart_cost_breakdown(
    df: pd.DataFrame,
    breakdown_by: str = "month",
    title: str = "Energy Cost Breakdown",
) -> go.Figure:
    work = ensure_standard_columns(df)
    if work.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False, font=dict(size=16))

    work = work.copy()
    hour = work["timestamp"].dt.hour
    work["cost"] = work["energy_kwh"] * work["price_per_kwh"]
    if breakdown_by == "month":
        work["period"] = work["timestamp"].dt.month
        agg = work.groupby("period")["cost"].sum().reset_index()
        agg["period"] = agg["period"].astype(str).str.zfill(2)
        agg["label"] = "Month " + agg["period"]
    elif breakdown_by == "dayofweek":
        work["period"] = work["timestamp"].dt.dayofweek
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        agg = work.groupby("period")["cost"].sum().reset_index()
        agg["label"] = agg["period"].map(lambda i: day_names[i] if i < 7 else str(i))
    else:
        work["period"] = work["timestamp"].dt.date
        agg = work.groupby("period")["cost"].sum().reset_index()
        agg["label"] = agg["period"].astype(str)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=agg["label"],
            y=agg["cost"],
            text=[f"${v:.2f}" for v in agg["cost"]],
            textposition="outside",
            marker_color=px.colors.qualitative.Set3[: len(agg)],
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title="Cost ($)",
        height=360,
        template="plotly_white",
        showlegend=False,
    )
    return fig


def chart_cost_pie(
    df: pd.DataFrame,
    breakdown_by: str = "month",
    title: str = "Energy Cost by Period",
) -> go.Figure:
    work = ensure_standard_columns(df)
    if work.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False, font=dict(size=16))

    work = work.copy()
    work["cost"] = work["energy_kwh"] * work["price_per_kwh"]
    if breakdown_by == "month":
        work["period"] = work["timestamp"].dt.month
        agg = work.groupby("period")["cost"].sum().reset_index()
        agg["label"] = "Month " + agg["period"].astype(str).str.zfill(2)
    else:
        work["period"] = work["timestamp"].dt.dayofweek
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        agg = work.groupby("period")["cost"].sum().reset_index()
        agg["label"] = agg["period"].map(lambda i: day_names[i] if i < 7 else str(i))

    fig = go.Figure(
        data=[
            go.Pie(
                labels=agg["label"],
                values=agg["cost"],
                hole=0.4,
                textinfo="label+percent",
                hovertemplate="%{label}<br>Cost: $%{value:.2f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=380,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    return fig


# ---------------------------------------------------------------------------
# 5) Peak load detection with threshold and alerts
# ---------------------------------------------------------------------------
def chart_peak_load(
    df: pd.DataFrame,
    threshold_kw: float,
    title: str = "Peak Load Detection",
) -> go.Figure:
    work = ensure_standard_columns(df)
    if work.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False, font=dict(size=16))

    above = work["energy_kwh"] >= threshold_kw
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=work["timestamp"],
            y=work["energy_kwh"],
            mode="lines",
            name="Energy (kWh)",
            line=dict(color="#1f77b4", width=1.2),
        )
    )
    if above.any():
        fig.add_trace(
            go.Scatter(
                x=work.loc[above, "timestamp"],
                y=work.loc[above, "energy_kwh"],
                mode="markers",
                marker=dict(color="red", size=8, symbol="triangle-up"),
                name="Above threshold",
            )
        )
    fig.add_hline(
        y=threshold_kw,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Threshold: {threshold_kw:.1f} kW",
        annotation_position="right",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Energy (kWh)",
        height=360,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 6) Model performance comparison (MAE, RMSE, MAPE, R²)
# ---------------------------------------------------------------------------
def chart_model_performance(
    metrics_by_model: Dict[str, Dict[str, float]],
    metric_keys: List[str] = None,
    title: str = "Model Performance Comparison",
) -> go.Figure:
    if metric_keys is None:
        metric_keys = ["mae", "rmse", "mape", "r2"]
    rows = []
    for model_name, metrics in metrics_by_model.items():
        for m in metric_keys:
            v = metrics.get(m)
            if v is not None:
                rows.append({"Model": model_name, "Metric": m.upper(), "Value": v})
    if not rows:
        fig = go.Figure()
        fig.add_annotation(text="No metrics available", showarrow=False, font=dict(size=16))
        return fig

    perf_df = pd.DataFrame(rows)
    fig = px.bar(
        perf_df,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set2,
        text_auto=".3f",
    )
    fig.update_layout(
        title=title,
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Metric",
        yaxis_title="Value",
    )
    fig.update_traces(textposition="outside")
    return fig


# ---------------------------------------------------------------------------
# 7) Energy savings: before vs after optimization
# ---------------------------------------------------------------------------
def chart_energy_savings(
    original_cost: float,
    optimized_cost: float,
    title: str = "Energy Cost: Before vs After Optimization",
) -> go.Figure:
    savings = original_cost - optimized_cost
    savings_pct = (savings / original_cost * 100.0) if original_cost > 0 else 0.0
    df = pd.DataFrame(
        {"Scenario": ["Before optimization", "After optimization"], "Cost ($)": [original_cost, optimized_cost]}
    )
    fig = px.bar(
        df,
        x="Scenario",
        y="Cost ($)",
        color="Cost ($)",
        color_continuous_scale=["#2ca02c", "#1f77b4"],
        text_auto=".2f",
    )
    fig.update_layout(
        title=f"{title} — Savings: ${savings:.2f} ({savings_pct:.1f}%)",
        height=380,
        template="plotly_white",
        showlegend=False,
        xaxis_tickangle=-15,
    )
    fig.update_traces(textposition="outside")
    return fig


# ---------------------------------------------------------------------------
# 8) Weather vs energy correlation
# ---------------------------------------------------------------------------
def chart_weather_energy_correlation(
    df: pd.DataFrame,
    title: str = "Temperature vs Energy Consumption",
) -> go.Figure:
    work = ensure_standard_columns(df)
    if work.empty or "temperature_c" not in work.columns:
        fig = go.Figure()
        fig.add_annotation(text="No temperature or energy data available", showarrow=False, font=dict(size=16))
        return fig

    work = work.dropna(subset=["temperature_c", "energy_kwh"])
    if work.empty:
        return go.Figure().add_annotation(text="No valid data points", showarrow=False, font=dict(size=16))

    fig = px.scatter(
        work,
        x="temperature_c",
        y="energy_kwh",
        trendline="ols",
        trendline_color_override="crimson",
        labels={"temperature_c": "Temperature (°C)", "energy_kwh": "Energy (kWh)"},
        opacity=0.6,
    )
    fig.update_layout(
        title=title,
        height=400,
        template="plotly_white",
        showlegend=True,
    )
    return fig
