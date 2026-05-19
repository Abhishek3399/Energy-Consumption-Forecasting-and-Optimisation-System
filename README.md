## AI-Driven Energy Consumption Forecasting and Optimization System

This project is a modular, production-ready, end-to-end system for forecasting and optimizing building electricity consumption using AI/ML, optimization, and real-time simulation.

Key components:
- **Data pipeline** for synthetic/historical data with weather, occupancy, prices, holidays.
- **Forecasting models**: LSTM, Transformer, Prophet baseline.
- **Optimization engine**: Linear Programming (OR-Tools) and Reinforcement Learning (RLlib PPO/DQN).
- **Real-time simulation** with MQTT-style streaming.
- **FastAPI backend** with database persistence.
- **Streamlit dashboard** for monitoring, analysis, and decision support.

### How to run

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and adjust settings as needed (database URL, MQTT broker, secret key).

4. **Start the backend first** from the project root:

   ```bash
   uvicorn main:app --reload --port 8000
   ```

   - Open `http://localhost:8000/docs` in your browser to verify the API (try the `/forecast` endpoint).

5. In a separate terminal, start the Streamlit dashboard:

   ```bash
   streamlit run dashboard/app.py
   ```

   - The dashboard is configured to call the backend at `http://localhost:8000`.
   - You can upload your own CSV and run forecasts using the sidebar uploader (uses `POST /forecast/upload`).

### One-command dev run (auto-start backend + dashboard)

From the project root:

```bash
python run_project.py
```

This starts the backend (uvicorn on port 8000) and the Streamlit dashboard together.

> If port `8000` is already in use on your machine, you can change the port (e.g., to `8001`) by:
> - Starting the backend with `uvicorn main:app --reload --port 8001`
> - Updating **Backend API URL** in the Streamlit sidebar, or setting `ENERGY_API_BASE=http://127.0.0.1:8001`

Detailed architecture, methodology, and deployment guide are provided in `docs/` and inline throughout the codebase.

