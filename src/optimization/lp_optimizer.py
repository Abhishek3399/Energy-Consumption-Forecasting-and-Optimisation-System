from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from ortools.linear_solver import pywraplp


@dataclass
class LPOptimizationConfig:
    peak_limit_kw: float
    comfort_min_kw: float
    comfort_max_kw: float
    equipment_min_kw: float
    equipment_max_kw: float


def run_lp_optimization(
    forecast_load: np.ndarray,
    price_per_kwh: np.ndarray,
    cfg: LPOptimizationConfig,
) -> Dict[str, np.ndarray]:
    """
    Simple deterministic optimization:
    - Variables: adjusted load per hour.
    - Objective: minimize total cost sum(price * load).
    - Constraints: comfort range, equipment bounds, peak limit.
    """
    n = len(forecast_load)
    if n <= 0:
        raise ValueError("forecast_load must be non-empty")

    # Align price vector length to forecast horizon
    if len(price_per_kwh) != n:
        if len(price_per_kwh) == 0:
            price_per_kwh = np.full(n, 0.12, dtype=float)
        elif len(price_per_kwh) < n:
            pad = np.full(n - len(price_per_kwh), float(price_per_kwh[-1]), dtype=float)
            price_per_kwh = np.concatenate([price_per_kwh.astype(float), pad], axis=0)
        else:
            price_per_kwh = price_per_kwh[:n].astype(float)

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("Failed to create OR-Tools solver")

    x: List[pywraplp.Variable] = []
    for t in range(n):
        var = solver.NumVar(
            cfg.equipment_min_kw,
            min(cfg.equipment_max_kw, cfg.peak_limit_kw),
            f"x_{t}",
        )
        x.append(var)
        # comfort constraints: load cannot be below/above comfort bounds too much
        solver.Add(x[t] >= cfg.comfort_min_kw)
        solver.Add(x[t] <= cfg.comfort_max_kw)
        # peak constraint (explicit; upper bound above also enforces it)
        solver.Add(x[t] <= cfg.peak_limit_kw)

    # NOTE: Avoid solver.Max(x) here; it's not supported in all OR-Tools linear solvers.

    # objective
    objective = solver.Objective()
    for t in range(n):
        objective.SetCoefficient(x[t], float(price_per_kwh[t]))
    objective.SetMinimization()

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("No optimal solution found for LP optimization")

    optimized = np.array([var.solution_value() for var in x], dtype=float)
    cost = float(np.dot(optimized, price_per_kwh))

    return {
        "optimized_load": optimized,
        "expected_cost": cost,
        "original_cost": float(np.dot(forecast_load, price_per_kwh)),
    }

