"""The interface every sensor shares, and the suite that runs them at their own rates."""
from __future__ import annotations

import math
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml


def load_sensor_params(path: Optional[str] = None) -> dict:
    if path is None:
        path = str(Path(__file__).resolve().parents[2] / "config" / "sensors.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class MountPose:
    """Sensor pose in the vehicle frame: rear-axle origin, x forward, y left."""
    x: float
    y: float
    yaw: float      # rad

    @classmethod
    def from_config(cls, entry: dict) -> "MountPose":
        return cls(entry["x"], entry["y"], math.radians(entry["yaw_deg"]))

    def to_vehicle(self, points: np.ndarray) -> np.ndarray:
        """Points (N, 2) from the sensor's frame to the vehicle frame."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.column_stack([self.x + c * points[:, 0] - s * points[:, 1],
                                self.y + s * points[:, 0] + c * points[:, 1]])

    def in_map(self, state) -> tuple:
        """(x, y, yaw) of the sensor in the map frame, for a vehicle state."""
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        return (state.x + c * self.x - s * self.y, state.y + s * self.x + c * self.y, state.yaw + self.yaw)


def map_to_vehicle(points: np.ndarray, state) -> np.ndarray:
    """Points (..., 2) from the map frame to the vehicle frame."""
    c, s = math.cos(state.yaw), math.sin(state.yaw)
    dx, dy = points[..., 0] - state.x, points[..., 1] - state.y
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=-1)


@dataclass(frozen=True, eq=False)
class Measurement:
    """Base of every sensor's output. eq=False: subclasses hold arrays."""
    sensor: str
    t: float        # s, when it was measured


class Sensor(ABC):
    def __init__(self, name: str, mount: MountPose, rate_hz: float):
        self.name, self.mount, self.rate_hz = name, mount, rate_hz

    @abstractmethod
    def measure(self, scenario, state, t: float, rng: np.random.Generator) -> Measurement:
        """One measurement of `scenario` from a vehicle in `state` (x, y, yaw) at time t."""


class SensorSuite:
    """Runs sensors at their own rates. Each sensor draws from its own random stream, derived
    from the seed and its name, so turning one sensor on or off doesn't change another's noise."""

    def __init__(self, sensors: Sequence[Sensor], seed: int = 0):
        self.sensors = list(sensors)
        self.by_name: Dict[str, Sensor] = {s.name: s for s in self.sensors}
        self._rngs = [np.random.default_rng([seed, zlib.crc32(s.name.encode())]) for s in self.sensors]
        self._ticks = [0] * len(self.sensors)

    def step(self, scenario, state, t: float) -> List[Measurement]:
        """Measurements from every sensor due at time t. Call once per sim tick, from t = 0."""
        out = []
        for i, sensor in enumerate(self.sensors):
            if t + 1e-9 >= self._ticks[i] / sensor.rate_hz:
                out.append(sensor.measure(scenario, state, t, self._rngs[i]))
                self._ticks[i] = math.floor(t * sensor.rate_hz + 1e-9) + 1
        return out
