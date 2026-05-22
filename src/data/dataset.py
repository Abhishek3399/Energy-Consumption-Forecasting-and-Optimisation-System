from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


@dataclass
class WindowConfig:
    input_window: int = 24
    forecast_horizon: int = 24


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.data = data.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data)

    def __getitem__(self, idx: int):  # type: ignore[override]
        x = self.data[idx]
        y = self.targets[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def _build_windows(
    feature_matrix: np.ndarray,
    target: np.ndarray,
    cfg: WindowConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    total_len = len(feature_matrix)
    full_window = cfg.input_window + cfg.forecast_horizon
    for start in range(0, total_len - full_window + 1):
        end = start + full_window
        xs.append(feature_matrix[start : start + cfg.input_window])
        ys.append(target[start + cfg.input_window : end])
    return np.stack(xs), np.stack(ys)


def create_train_val_test_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "energy_kwh",
    cfg: WindowConfig | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[
    SlidingWindowDataset,
    SlidingWindowDataset,
    SlidingWindowDataset,
    StandardScaler,
]:
    if cfg is None:
        cfg = WindowConfig()

    df_sorted = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    features = df_sorted[feature_cols].values
    target = df_sorted[target_col].values

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    x_all, y_all = _build_windows(features_scaled, target, cfg)

    n_total = len(x_all)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    x_train, y_train = x_all[:n_train], y_all[:n_train]
    x_val, y_val = x_all[n_train : n_train + n_val], y_all[n_train : n_train + n_val]
    x_test, y_test = x_all[n_train + n_val :], y_all[n_train + n_val :]

    return (
        SlidingWindowDataset(x_train, y_train),
        SlidingWindowDataset(x_val, y_val),
        SlidingWindowDataset(x_test, y_test),
        scaler,
    )

