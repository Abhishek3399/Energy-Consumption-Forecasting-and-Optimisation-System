from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    # Robustness: avoid backend 500s if a model emits NaNs/Infs.
    mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    if not np.any(mask):
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "r2": float("nan")}
    y_true_flat = y_true_flat[mask]
    y_pred_flat = y_pred_flat[mask]

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = mean_squared_error(y_true_flat, y_pred_flat, squared=False)
    mape = np.mean(
        np.abs((y_true_flat - y_pred_flat) / np.clip(np.abs(y_true_flat), 1e-6, None))
    ) * 100.0
    r2 = r2_score(y_true_flat, y_pred_flat)

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape), "r2": float(r2)}

