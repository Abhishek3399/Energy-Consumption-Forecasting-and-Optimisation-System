"""
Run backend + dashboard together (Windows-friendly).

Usage:
  python run_project.py

This starts:
  - FastAPI backend on http://127.0.0.1:8000
  - Streamlit dashboard

Stop with Ctrl+C.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
import json
import socket
from urllib.parse import urlparse
from urllib.request import urlopen, Request


ROOT = Path(__file__).resolve().parent


def _popen(cmd: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=os.environ.copy(),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )


def main() -> int:
    python = sys.executable

    # Normalize API base (used by both backend and Streamlit)
    api_base = os.getenv("ENERGY_API_BASE", "http://127.0.0.1:8000")
    parsed = urlparse(api_base)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    api_base = f"http://{host}:{port}"
    os.environ["ENERGY_API_BASE"] = api_base

    def _backend_health_url() -> str:
        return f"{api_base}/health"

    def _wait_for_backend_health(timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                req = Request(_backend_health_url(), headers={"Accept": "application/json"})
                with urlopen(req, timeout=2.0) as resp:
                    payload = json.loads(resp.read().decode("utf-8", errors="replace"))
                if payload.get("status") == "ok":
                    return True
            except Exception:
                time.sleep(0.5)
        return False

    backend_cmd = [
        python,
        "-m",
        "uvicorn",
        "main:app",
        "--reload",
        "--port",
        str(port),
        "--host",
        host,
    ]
    dashboard_cmd = [python, "-m", "streamlit", "run", "dashboard/app.py"]

    # If a backend is already up, don't start another one (avoids port conflicts).
    if _wait_for_backend_health(timeout_s=2.0):
        print(f"Backend already healthy at {api_base}.")
        backend = None
    else:
        # Basic port check: if something is listening but health is failing, fail fast.
        try:
            sock = socket.create_connection((host, port), timeout=1.0)
            sock.close()
            port_listening = True
        except Exception:
            port_listening = False

        if port_listening:
            print(f"Port {port} is already in use, but backend health is not OK at {api_base}/health.")
            return 1

        print("Starting backend:", " ".join(backend_cmd))
        backend = _popen(backend_cmd)

        print("Waiting for backend health...")
        if not _wait_for_backend_health(timeout_s=30.0):
            print(f"Backend did not become healthy at {api_base}/health. Check FastAPI logs.")
            try:
                if backend is not None:
                    backend.terminate()
            except Exception:
                pass
            return 1

    print("Starting dashboard:", " ".join(dashboard_cmd))
    dashboard = _popen(dashboard_cmd)

    try:
        while True:
            time.sleep(0.5)
            b = backend.poll() if backend is not None else None
            d = dashboard.poll()
            if b is not None:
                print(f"Backend exited with code {b}.")
                return b
            if d is not None:
                print(f"Dashboard exited with code {d}.")
                return d
    except KeyboardInterrupt:
        print("Stopping...")
        for p in (dashboard, backend):
            try:
                p.terminate()
            except Exception:
                pass
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

