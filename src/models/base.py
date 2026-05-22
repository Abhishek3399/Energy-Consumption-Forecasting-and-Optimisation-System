from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import numpy as np


class EnergyForecastModel(ABC):
    """Abstract base class for all forecasting models."""

    name: str = "base"

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        ...

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        ...

