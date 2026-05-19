"""
Backend entrypoint for local development.

This file exists so you can run exactly:

    uvicorn main:app --reload --port 8000

It re-exports the FastAPI app that includes:
- core API: /forecast, /optimize, /historical/{building_id}
- upload API: /forecast/upload
"""

from backend.api import app  # noqa: F401

