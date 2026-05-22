from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class RLEnvConfig:
    horizon: int
    peak_limit_kw: float
    comfort_min_kw: float
    comfort_max_kw: float
    max_action_kw: float = 5.0


class EnergyEnv(gym.Env):
    """
    Simple RL environment for load adjustment.

    Observation: [forecast_load_t, price_t, solar_kw_t]
    Action: delta load in [-max_action_kw, max_action_kw]
    Reward: -(price * adjusted_load) - penalty for violating comfort/peak.
    """

    metadata = {"render_modes": []}

    def __init__(self, forecast: np.ndarray, price: np.ndarray, solar: np.ndarray, cfg: RLEnvConfig):
        super().__init__()
        assert len(forecast) == len(price) == len(solar)
        self.forecast = forecast
        self.price = price
        self.solar = solar
        self.cfg = cfg
        self.t = 0

        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-cfg.max_action_kw,
            high=cfg.max_action_kw,
            shape=(1,),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [self.forecast[self.t], self.price[self.t], self.solar[self.t]],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.t = 0
        return self._get_obs(), {}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        delta = float(np.clip(action[0], -self.cfg.max_action_kw, self.cfg.max_action_kw))
        base_load = self.forecast[self.t] - self.solar[self.t]
        adjusted = max(base_load + delta, 0.0)

        # enforce peak and comfort via penalties
        price = self.price[self.t]
        cost = price * adjusted

        penalty = 0.0
        if adjusted > self.cfg.peak_limit_kw:
            penalty += (adjusted - self.cfg.peak_limit_kw) * price * 2.0
        if adjusted < self.cfg.comfort_min_kw:
            penalty += (self.cfg.comfort_min_kw - adjusted) * price * 1.5
        if adjusted > self.cfg.comfort_max_kw:
            penalty += (adjusted - self.cfg.comfort_max_kw) * price * 1.5

        reward = -(cost + penalty)

        self.t += 1
        terminated = self.t >= len(self.forecast) or self.t >= self.cfg.horizon
        truncated = False
        info = {"adjusted_load": adjusted, "solar": self.solar[self.t - 1]}

        if terminated:
            return self._get_obs(), float(reward), True, truncated, info
        else:
            return self._get_obs(), float(reward), False, truncated, info

