"""Optional 2D lidar: 360-degree ray cast against the obstacle polygons."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import Measurement, MountPose, Sensor, load_sensor_params
from .scene import cast_rays, scene_arrays, within_reach


@dataclass(frozen=True, eq=False)
class LidarMeasurement(Measurement):
    angles: np.ndarray = None   # (N,) rad in the sensor frame
    ranges: np.ndarray = None   # (N,) m; nan = no return

    def points(self) -> np.ndarray:
        """Returns as (K, 2) points in the sensor frame."""
        hit = np.isfinite(self.ranges)
        return np.column_stack([self.ranges[hit] * np.cos(self.angles[hit]),
                                self.ranges[hit] * np.sin(self.angles[hit])])


class Lidar2D(Sensor):
    def __init__(self, name: str, mount: MountPose, rate_hz: float, beams: int, max_range: float, noise_std: float):
        super().__init__(name, mount, rate_hz)
        self.angles = np.linspace(-math.pi, math.pi, beams, endpoint=False)
        self.max_range, self.noise_std = max_range, noise_std

    def measure(self, scenario, state, t: float, rng: np.random.Generator) -> LidarMeasurement:
        scene = scene_arrays(scenario)
        x, y, yaw = self.mount.in_map(state)
        origin = np.array([x, y])
        world = self.angles + yaw
        ranges = cast_rays(origin, np.column_stack([np.cos(world), np.sin(world)]),
                           *within_reach(origin, self.max_range, scene.obstacle_a, scene.obstacle_b), self.max_range)
        if self.noise_std > 0.0:
            ranges = ranges + rng.normal(0.0, self.noise_std, len(ranges))
        return LidarMeasurement(self.name, t, self.angles, np.where(np.isfinite(ranges), ranges, np.nan))


def make_lidar(cfg: Optional[dict] = None) -> Lidar2D:
    cfg = cfg if cfg is not None else load_sensor_params()["lidar"]
    return Lidar2D(cfg["name"], MountPose.from_config(cfg), cfg["rate_hz"], cfg["beams"], cfg["range"], cfg["noise_std"])
