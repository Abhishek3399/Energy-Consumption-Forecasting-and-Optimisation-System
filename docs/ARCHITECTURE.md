## System Architecture Overview

- **Data Layer**: Synthetic/historical data with weather, occupancy, holidays, and prices (`src/data`).
- **Forecasting Layer**: LSTM, Transformer, Prophet baseline with evaluation and SHAP explainability (`src/models`).
- **Optimization Layer**: OR-Tools LP and RL environment for adaptive strategies (`src/optimization`).
- **Real-Time Layer**: MQTT-style simulator emitting smart meter readings (`src/realtime`).
- **Backend API**: FastAPI service exposing forecasting, optimization, historical data, and metrics (`src/backend`).
- **Dashboard**: Streamlit UI for visualization, model comparison, and decision support (`dashboard/app.py`).

