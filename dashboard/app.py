"""
AI-Driven Energy Consumption Forecasting and Optimization — Streamlit Dashboard.
Organized into: Monitoring | Forecasting | Optimization Insights | Model Analytics.
All visualizations update dynamically with API data or uploaded CSV.
"""
import sys
from pathlib import Path
import os

# Allow running as: streamlit run dashboard/app.py (from repo root)
if Path(__file__).resolve().parent not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from viz_components import (
    ensure_standard_columns,
    resample_to_hourly,
    chart_energy_timeseries,
    chart_forecast_vs_actual,
    chart_energy_heatmap,
    chart_cost_breakdown,
    chart_cost_pie,
    chart_peak_load,
    chart_model_performance,
    chart_energy_savings,
    chart_weather_energy_correlation,
)

DEFAULT_API_BASE = os.getenv("ENERGY_API_BASE", "http://127.0.0.1:8000")
API_BASE = st.sidebar.text_input("Backend API URL", value=DEFAULT_API_BASE, help="Example: http://127.0.0.1:8000")
REQUEST_TIMEOUT = 10.0

st.set_page_config(
    page_title="Energy Forecasting & Optimization",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI-Driven Energy Consumption Forecasting and Optimization")
st.caption("Monitoring • Forecasting • Optimization • Model Analytics")

# --- utilities ---
def _closest_timestamp_row(
    df: pd.DataFrame,
    target_ts: pd.Timestamp,
    tolerance_hours: float = 0.5,
) -> pd.Series | None:
    """
    Return the closest row (by timestamp) to target_ts if within tolerance_hours.
    Safely handles NaT timestamps and non-datetime dtypes.
    """
    if df is None or df.empty or "timestamp" not in df.columns:
        return None

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"])
    if work.empty:
        return None

    target_ts = pd.to_datetime(target_ts, errors="coerce")
    if pd.isna(target_ts):
        return None

    # Compute absolute difference in hours using total_seconds()
    time_diff_hours = (work["timestamp"] - target_ts).dt.total_seconds().abs() / 3600.0
    if time_diff_hours.isna().all():
        return None

    idx = time_diff_hours.idxmin()
    if pd.isna(idx):
        return None
    if float(time_diff_hours.loc[idx]) <= float(tolerance_hours):
        return work.loc[idx]
    return None

# --- Backend status / friendly guidance ---
backend_ok = False
backend_err = None
try:
    ping = requests.get(f"{API_BASE}/health", timeout=2.5)
    if ping.ok:
        try:
            backend_payload = ping.json()
        except Exception:
            backend_payload = {}
        backend_ok = backend_payload.get("status") == "ok" or backend_payload == {}
        if not backend_ok and backend_payload:
            backend_err = f"Backend health returned status={backend_payload.get('status')}"
    else:
        backend_err = f"Health check returned HTTP {ping.status_code}"
except requests.exceptions.RequestException as e:
    backend_err = str(e)

status_col1, status_col2 = st.columns([2, 3])
with status_col1:
    if backend_ok:
        st.success(f"Backend health OK: `{API_BASE}`")
    else:
        st.error("Backend not reachable (health check failed).")
with status_col2:
    if not backend_ok:
        st.caption(
            "Start the backend from the project root:\n\n"
            "`uvicorn main:app --reload --port 8000`\n\n"
            "Then refresh this page. "
            "If you changed the port/host, update **Backend API URL** in the sidebar (or set `ENERGY_API_BASE` to match)."
        )
        if backend_err:
            with st.expander("Connection details", expanded=False):
                st.code(backend_err)

# --- Sidebar: configuration and data source ---
st.sidebar.markdown("### Configuration")
building_id = st.sidebar.number_input("Building ID", min_value=1, value=1, step=1)
horizon = st.sidebar.slider("Forecast horizon (hours)", 6, 48, 24)
mode_label = st.sidebar.toggle(
    "Mode: Evaluate on Test Data",
    value=True,
    help="On: compare predictions vs actuals from test split. Off: forecast future horizon.",
)
forecast_mode = "test" if mode_label else "future"

st.sidebar.markdown("---")
st.sidebar.subheader("Data source")
st.sidebar.caption("Use API historical data or upload a CSV to drive all charts.")
hourly_normalize = st.sidebar.toggle(
    "Normalize to hourly cadence (recommended)",
    value=True,
    help="If timestamps are irregular or higher-frequency, resample to hourly for stable analytics and forecast-vs-actual matching.",
)
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
required_note = (
    "Required columns: timestamp, energy_kwh (or energy/consumption/load). "
    "Optional: temperature_c, humidity, occupancy, price_per_kwh, is_holiday, building_id."
)
st.sidebar.caption(required_note)

# --- Resolve working dataset: uploaded CSV or API historical ---
working_df = None
forecast_ts = None
forecast_kwh = None
forecast_metrics = None
test_pred = None
test_actual = None
test_ts = None
models_metrics = {}
source_label = "API"
if not backend_ok:
    source_label = "Demo (backend offline)"

if uploaded is not None:
    try:
        df_up = pd.read_csv(uploaded)
        cols_lower = {str(c).lower().strip() for c in df_up.columns}
        has_ts = "timestamp" in cols_lower or any(a in cols_lower for a in ("time", "datetime", "date_time", "date"))
        has_energy = "energy_kwh" in cols_lower or any(a in cols_lower for a in ("energy", "consumption", "load", "kwh"))
        if has_ts and has_energy:
            working_df = ensure_standard_columns(df_up)
            if hourly_normalize:
                working_df = resample_to_hourly(working_df)
            source_label = "Uploaded CSV"
            if "uploaded_forecast" in st.session_state:
                uf = st.session_state["uploaded_forecast"]
                forecast_ts = pd.to_datetime(uf["response"]["timestamps"])
                forecast_kwh = uf["response"]["forecast_kwh"]
                forecast_metrics = uf["response"].get("metrics")
                test_pred = uf["response"].get("predictions")
                test_actual = uf["response"].get("actuals")
                test_ts_raw = uf["response"].get("timestamps")
                test_ts = pd.to_datetime(test_ts_raw) if test_ts_raw else None
                models_metrics = uf["response"].get("models", {})
        else:
            st.sidebar.warning("CSV must include timestamp and energy column to drive visualizations.")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")

if working_df is None and backend_ok:
    try:
        hist = requests.get(f"{API_BASE}/historical/{building_id}", timeout=REQUEST_TIMEOUT)
        if hist.ok:
            h = hist.json()
            working_df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(h["timestamps"], errors="coerce"),
                    "energy_kwh": h["energy_kwh"],
                    "temperature_c": h["temperature_c"],
                    "occupancy": h["occupancy"],
                }
            )
            working_df = ensure_standard_columns(working_df)
            if hourly_normalize:
                working_df = resample_to_hourly(working_df)
            if "price_per_kwh" not in working_df.columns:
                working_df["price_per_kwh"] = 0.12
        else:
            st.sidebar.warning(f"Historical API returned {hist.status_code}.")
    except requests.exceptions.RequestException:
        st.sidebar.warning(f"Backend not reachable at {API_BASE} (health ok but historical failed). Using demo data.")

if working_df is None and not backend_ok:
    st.sidebar.warning(
        f"Backend unavailable at {API_BASE}. "
        "Using demo data. Start backend with `uvicorn main:app --reload --port 8000` (or update Backend API URL)."
    )

if working_df is None:
    # Minimal demo data so sections still render
    dates = pd.date_range(end=pd.Timestamp.now(), periods=7 * 24, freq="h")
    working_df = pd.DataFrame({
        "timestamp": dates,
        "energy_kwh": 10 + 5 * np.sin(np.arange(len(dates)) / 6) + np.random.rand(len(dates)) * 2,
        "temperature_c": 18 + 5 * np.sin(np.arange(len(dates)) / 12),
        "price_per_kwh": 0.12,
    })
    working_df = ensure_standard_columns(working_df)

# Forecast from API (Run Forecast button) — merge into global forecast state when used
if working_df is not None and source_label == "API" and "last_forecast" in st.session_state:
    fc = st.session_state["last_forecast"]
    if fc.get("building_id") == building_id and fc.get("horizon_hours") == horizon:
        forecast_ts = pd.to_datetime(fc["timestamps"])
        forecast_kwh = fc["forecast_kwh"]
        forecast_metrics = fc.get("metrics")
        test_pred = fc.get("predictions")
        test_actual = fc.get("actuals")
        test_ts_raw = fc.get("timestamps")
        test_ts = pd.to_datetime(test_ts_raw) if test_ts_raw else None
        models_metrics = fc.get("models", {})

# --- Uploaded CSV: preview and "Run Forecast" ---
if uploaded is not None and working_df is not None:
    st.sidebar.markdown("---")
    if st.sidebar.button("Run Forecast on Uploaded CSV"):
        if not backend_ok:
            st.sidebar.error(f"Backend not reachable at {API_BASE}. Check /health and start the backend.")
        else:
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                resp = requests.post(
                    f"{API_BASE}/forecast/upload",
                    files=files,
                    params={"horizon_hours": horizon, "mode": forecast_mode},
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.ok:
                    out = resp.json()
                    st.session_state["uploaded_forecast"] = {"response": out, "df": working_df.copy()}
                    forecast_ts = pd.to_datetime(out["timestamps"])
                    forecast_kwh = out["forecast_kwh"]
                    forecast_metrics = out.get("metrics")
                    test_pred = out.get("predictions")
                    test_actual = out.get("actuals")
                    test_ts_raw = out.get("timestamps")
                    test_ts = pd.to_datetime(test_ts_raw) if test_ts_raw else None
                    models_metrics = out.get("models", {})
                    st.sidebar.success("Forecast generated.")
                    st.sidebar.json(out.get("metrics", {}))
                else:
                    st.sidebar.error(f"Forecast failed ({resp.status_code}).")
            except requests.exceptions.RequestException as e:
                st.sidebar.error(f"Cannot reach backend: {e}")

    with st.expander("Uploaded data preview", expanded=False):
        st.dataframe(working_df.head(20), use_container_width=True)

# ========== SECTION 1: MONITORING ==========
st.markdown("---")
st.markdown("## 1. Monitoring & Historical Analytics")
st.caption(f"Data source: {source_label}")

if working_df is not None and not working_df.empty:
    ts_min, ts_max = working_df["timestamp"].min(), working_df["timestamp"].max()
    try:
        d_min = ts_min.date() if hasattr(ts_min, "date") else pd.Timestamp(ts_min).date()
        d_max = ts_max.date() if hasattr(ts_max, "date") else pd.Timestamp(ts_max).date()
    except Exception:
        d_min = datetime.now().date() - timedelta(days=7)
        d_max = datetime.now().date()
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        date_min = st.date_input("From date", value=d_min, min_value=d_min, max_value=d_max, key="date_min")
    with col_d2:
        date_max = st.date_input("To date", value=d_max, min_value=d_min, max_value=d_max, key="date_max")
    date_min_ts = pd.Timestamp(date_min) if date_min else None
    date_max_ts = pd.Timestamp(date_max) + timedelta(days=1) if date_max else None

    st.markdown("#### Energy consumption: historical vs predicted")
    fig_ts = chart_energy_timeseries(
        working_df,
        date_min=date_min_ts,
        date_max=date_max_ts,
        forecast_ts=forecast_ts,
        forecast_kwh=forecast_kwh,
        title="Energy Consumption (Historical vs Predicted)",
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("#### Energy usage heatmap (hour of day × day of week)")
    fig_heat = chart_energy_heatmap(working_df, title="Energy Usage by Hour and Day of Week")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("#### Peak load detection")
    peak_threshold = st.slider(
        "Peak threshold (kW)",
        float(working_df["energy_kwh"].min()),
        float(working_df["energy_kwh"].max()),
        float(working_df["energy_kwh"].quantile(0.9)),
        key="peak_threshold",
    )
    fig_peak = chart_peak_load(working_df, peak_threshold, title="Peak Load Detection")
    st.plotly_chart(fig_peak, use_container_width=True)
    above_count = (working_df["energy_kwh"] >= peak_threshold).sum()
    if above_count > 0:
        st.warning(f"**{above_count}** readings above threshold ({peak_threshold:.1f} kW).")

    st.markdown("#### Energy cost breakdown")
    cost_by = st.radio("Breakdown by", ["month", "dayofweek"], horizontal=True, key="cost_breakdown")
    c1, c2 = st.columns(2)
    with c1:
        fig_cost_bar = chart_cost_breakdown(working_df, breakdown_by=cost_by, title="Cost by period (bar)")
        st.plotly_chart(fig_cost_bar, use_container_width=True)
    with c2:
        fig_cost_pie = chart_cost_pie(working_df, breakdown_by=cost_by, title="Cost share (pie)")
        st.plotly_chart(fig_cost_pie, use_container_width=True)

    st.markdown("#### Weather vs energy correlation")
    fig_weather = chart_weather_energy_correlation(working_df, title="Temperature vs Energy Consumption")
    st.plotly_chart(fig_weather, use_container_width=True)
else:
    st.info("No data available for monitoring. Upload a CSV or ensure the backend is running and has historical data.")

# ========== SECTION 2: FORECASTING ==========
st.markdown("---")
st.markdown("## 2. Forecasting Analytics")

col_f1, col_f2 = st.columns(2)
with col_f1:
    st.subheader("Run forecast (LSTM)")
    if st.button("Run Forecast"):
        if not backend_ok:
            st.error(f"Backend not reachable at {API_BASE}. Health check: /health")
        else:
            try:
                resp = requests.post(
                    f"{API_BASE}/forecast",
                    json={"building_id": building_id, "horizon_hours": horizon, "mode": forecast_mode},
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.ok:
                    out = resp.json()
                    st.session_state["last_forecast"] = out
                    st.success("Forecast completed.")
                    st.json(out.get("metrics", {}))
                    test_pred = out.get("predictions")
                    test_actual = out.get("actuals")
                    test_ts_raw = out.get("timestamps")
                    test_ts = pd.to_datetime(test_ts_raw) if test_ts_raw else None
                    models_metrics = out.get("models", {})
                else:
                    st.error(f"Forecast failed ({resp.status_code}).")
            except requests.exceptions.RequestException as e:
                st.error(f"Backend unreachable: {e}")

with col_f2:
    st.subheader("Forecast vs actual comparison")
    st.caption("When forecast and overlapping actuals exist, they are compared below.")

if (
    test_pred is not None
    and test_actual is not None
    and test_ts is not None
    and len(test_pred) > 0
    and len(test_actual) > 0
):
    st.markdown("#### Forecast vs actual (test evaluation)")
    n = min(len(test_pred), len(test_actual), len(test_ts))
    cmp_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(test_ts[:n]),
            "actual": pd.to_numeric(pd.Series(test_actual[:n]), errors="coerce"),
            "predicted": pd.to_numeric(pd.Series(test_pred[:n]), errors="coerce"),
        }
    ).dropna()
    if not cmp_df.empty:
        cmp_df = cmp_df.sort_values("timestamp")
        st.line_chart(cmp_df.set_index("timestamp")[["actual", "predicted"]], use_container_width=True)
    else:
        st.info("No overlapping actual data available")
else:
    st.info("No overlapping actual data available")

# ========== SECTION 3: OPTIMIZATION INSIGHTS ==========
st.markdown("---")
st.markdown("## 3. Optimization Insights & Energy Savings")

col_o1, col_o2 = st.columns(2)
with col_o1:
    peak = st.number_input("Peak limit (kW)", value=15.0, key="opt_peak")
    comfort_min = st.number_input("Comfort min (kW)", value=5.0, key="opt_comfort_min")
    comfort_max = st.number_input("Comfort max (kW)", value=20.0, key="opt_comfort_max")
with col_o2:
    equip_min = st.number_input("Equipment min (kW)", value=0.0, key="opt_equip_min")
    equip_max = st.number_input("Equipment max (kW)", value=25.0, key="opt_equip_max")

if st.button("Run Optimization"):
    if not backend_ok:
        st.error(f"Backend not reachable at {API_BASE}. Health check: /health")
    else:
        try:
            resp = requests.post(
                f"{API_BASE}/optimize",
                json={
                    "building_id": building_id,
                    "horizon_hours": horizon,
                    "peak_limit_kw": peak,
                    "comfort_min_kw": comfort_min,
                    "comfort_max_kw": comfort_max,
                    "equipment_min_kw": equip_min,
                    "equipment_max_kw": equip_max,
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.ok:
                st.session_state["last_optimization"] = resp.json()
                st.success("Optimization completed.")
            else:
                st.error(f"Optimization failed ({resp.status_code}).")
        except requests.exceptions.RequestException as e:
            st.error(f"Backend unreachable: {e}")

if "last_optimization" in st.session_state:
    opt = st.session_state["last_optimization"]
    fig_savings = chart_energy_savings(
        opt["original_cost"],
        opt["expected_cost"],
        title="Energy Cost: Before vs After Optimization",
    )
    st.plotly_chart(fig_savings, use_container_width=True)
else:
    st.info("Run optimization to see cost before vs after and savings.")

# ========== SECTION 4: MODEL ANALYTICS ==========
st.markdown("---")
st.markdown("## 4. Model Performance Comparison")

metrics_by_model = {}
if models_metrics:
    metrics_by_model = {k.upper(): v for k, v in models_metrics.items()}
elif forecast_metrics:
    metrics_by_model["LSTM"] = forecast_metrics

if metrics_by_model:
    rows = []
    for model_name, m in metrics_by_model.items():
        rows.append(
            {
                "Model": model_name,
                "MAE": float(m.get("mae", np.nan)),
                "RMSE": float(m.get("rmse", np.nan)),
                "MAPE": float(m.get("mape", np.nan)),
                "R2": float(m.get("r2", np.nan)),
            }
        )
    perf_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(perf_df, use_container_width=True)
    st.bar_chart(perf_df[["MAE", "RMSE", "MAPE"]], use_container_width=True)
    if len(metrics_by_model) == 1 and "LSTM" in metrics_by_model:
        st.info("Only LSTM model available. Add more models for comparison.")
else:
    st.info("Run a forecast to see model performance metrics.")

# ========== SECTION 5: AI RECOMMENDATIONS ==========
st.markdown("---")
st.markdown("## 5. AI Recommendations")
if backend_ok:
    try:
        rec_resp = requests.get(
            f"{API_BASE}/recommendations",
            params={"building_id": building_id},
            timeout=REQUEST_TIMEOUT,
        )
        if rec_resp.ok:
            recs = rec_resp.json().get("recommendations", [])
            if recs:
                for rec in recs:
                    st.info(str(rec))
            else:
                st.caption("No recommendations available.")
        else:
            st.warning(f"Recommendations API returned {rec_resp.status_code}.")
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not load recommendations: {e}")
else:
    st.caption("Recommendations unavailable while backend is offline.")

# ========== SECTION 6: WHAT-IF ANALYSIS ==========
st.markdown("---")
st.markdown("## 6. What-if Analysis")
wf_col1, wf_col2, wf_col3 = st.columns(3)
with wf_col1:
    wf_temp = st.slider("Temperature (C)", min_value=-5.0, max_value=45.0, value=24.0, step=0.5)
with wf_col2:
    wf_occ = st.slider("Occupancy", min_value=0.0, max_value=2.0, value=0.8, step=0.05)
with wf_col3:
    wf_price = st.slider("Price per kWh", min_value=0.01, max_value=1.00, value=0.12, step=0.01)

if st.button("Run What-if Scenario"):
    if not backend_ok:
        st.warning("Backend offline; cannot run what-if analysis.")
    else:
        try:
            wf_resp = requests.post(
                f"{API_BASE}/what-if",
                params={"building_id": building_id, "horizon_hours": horizon},
                json={"temperature": wf_temp, "occupancy": wf_occ, "price": wf_price},
                timeout=REQUEST_TIMEOUT,
            )
            if wf_resp.ok:
                modified = wf_resp.json().get("modified_predictions", [])
                if modified:
                    wf_df = pd.DataFrame(
                        {
                            "step": list(range(1, len(modified) + 1)),
                            "modified_prediction": modified,
                        }
                    ).set_index("step")
                    st.line_chart(wf_df, use_container_width=True)
                else:
                    st.info("No modified predictions returned.")
            else:
                st.warning(f"What-if API returned {wf_resp.status_code}.")
        except requests.exceptions.RequestException as e:
            st.warning(f"What-if request failed: {e}")

# ========== SECTION 7: EXPLAINABILITY ==========
st.markdown("---")
st.markdown("## 7. Explainability")
if backend_ok:
    try:
        ex_resp = requests.get(
            f"{API_BASE}/explainability",
            params={"building_id": building_id},
            timeout=REQUEST_TIMEOUT,
        )
        if ex_resp.ok:
            fi = ex_resp.json().get("feature_importance", {})
            if fi:
                fi_df = pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())})
                fi_df = fi_df.sort_values("importance", ascending=False).set_index("feature")
                st.bar_chart(fi_df, use_container_width=True)
            else:
                st.info("No feature importance available.")
        else:
            st.warning(f"Explainability API returned {ex_resp.status_code}.")
    except requests.exceptions.RequestException as e:
        st.warning(f"Explainability request failed: {e}")
else:
    st.caption("Explainability unavailable while backend is offline.")

# ========== SECTION 8: ANOMALY DETECTION ==========
st.markdown("---")
st.markdown("## 8. Anomaly Detection")
if backend_ok:
    try:
        an_resp = requests.get(
            f"{API_BASE}/anomalies",
            params={"building_id": building_id},
            timeout=REQUEST_TIMEOUT,
        )
        if an_resp.ok:
            anomalies = an_resp.json().get("anomalies", [])
            if anomalies:
                an_df = pd.DataFrame(anomalies)
                st.dataframe(an_df, use_container_width=True)
                st.warning(f"Detected {len(an_df)} anomalies.")
            else:
                st.success("No anomalies detected.")
        else:
            st.warning(f"Anomalies API returned {an_resp.status_code}.")
    except requests.exceptions.RequestException as e:
        st.warning(f"Anomaly detection request failed: {e}")
else:
    st.caption("Anomaly detection unavailable while backend is offline.")

# ========== SECTION 9: SIMULATION MODE ==========
st.markdown("---")
st.markdown("## 9. Simulation Mode")
sim_scenario = st.selectbox(
    "Scenario",
    ["summer", "winter", "weekday", "weekend", "office", "factory"],
    index=2,
)
if st.button("Run Simulation"):
    if not backend_ok:
        st.warning("Backend offline; cannot run simulation.")
    else:
        try:
            sim_resp = requests.post(
                f"{API_BASE}/simulate",
                params={"building_id": building_id, "horizon_hours": horizon},
                json={"scenario": sim_scenario},
                timeout=REQUEST_TIMEOUT,
            )
            if sim_resp.ok:
                preds = sim_resp.json().get("simulation_predictions", [])
                if preds:
                    sim_df = pd.DataFrame(
                        {
                            "step": list(range(1, len(preds) + 1)),
                            "simulation_prediction": preds,
                        }
                    ).set_index("step")
                    st.line_chart(sim_df, use_container_width=True)
                else:
                    st.info("No simulation predictions returned.")
            else:
                st.warning(f"Simulation API returned {sim_resp.status_code}.")
        except requests.exceptions.RequestException as e:
            st.warning(f"Simulation request failed: {e}")
