## Local Development

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and adjust settings as needed (database URL, MQTT broker, secret key).

4. Start the FastAPI backend:

```bash
uvicorn src.backend.main:app --reload --port 8000
```

5. Run the Streamlit dashboard in a separate terminal:

```bash
streamlit run dashboard/app.py
```

## Docker (Example)

- Create a multi-stage Dockerfile with one service for the FastAPI backend and another for the Streamlit dashboard.
- Use environment variables to configure database and MQTT broker endpoints.
- Expose ports 8000 (FastAPI) and 8501 (Streamlit).

## Cloud Deployment

- Containerize the backend and dashboard and deploy to a managed container service (e.g., AWS ECS/Fargate, Azure Container Apps, or GCP Cloud Run).
- Use a managed PostgreSQL instance instead of SQLite for production.
- Optionally deploy an MQTT broker (e.g., Eclipse Mosquitto) as a separate service.
- Configure HTTPS and authentication for the API and dashboard using a gateway/load balancer and an identity provider.

