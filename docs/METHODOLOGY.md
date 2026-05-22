## Problem Statement

Design and implement an AI-driven system that forecasts short- and long-term building electricity consumption and recommends cost-optimal operating strategies while respecting comfort and equipment constraints.

## Literature Review (Summary)

- **Classical time-series** (ARIMA, SARIMA) provide strong baselines but struggle with nonlinear effects of weather, occupancy, and prices.
- **Deep learning** (LSTM, Transformer) improves forecasting of complex temporal dependencies and exogenous inputs.
- **Probabilistic forecasting and Prophet** offer interpretable, seasonality-aware baselines.
- **Optimization-based demand response** uses linear/convex programming to minimize cost under constraints.
- **Reinforcement learning** (PPO/DQN) enables adaptive control in uncertain, dynamic environments with partial observability.

## Methodology

- Generate/ingest multi-year hourly data with weather, occupancy, holidays, and price signals.
- Engineer time-of-day, calendar, lag, and price-derived features; build sliding windows for supervised training.
- Train LSTM and Transformer sequence models and a Prophet baseline; evaluate using MAE, RMSE, MAPE, and \(R^2\).
- Use SHAP-based explainability to analyze feature contributions and model behavior.
- Formulate a linear program to minimize energy cost with comfort, peak, and equipment constraints using OR-Tools.
- Define an RL environment that adjusts load based on forecasts, prices, and solar, trained with PPO/DQN via RLlib.
- Serve forecasting and optimization endpoints via FastAPI and expose analytics through a Streamlit dashboard.

