"""Ultrasonic sensors: nearest obstacle in a cone, or nothing."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .base import Measurement, MountPose, Sensor, load_sensor_params
from .scene import nearest_in_cone, scene_arrays, within_reach


@dataclass(frozen=True, eq=False)
class UltrasonicMeasurement(Measurement):
    distance: Optional[float] = None    # m from the sensor; None = nothing in range, or a dropout


class UltrasonicSensor(Sensor):
    def __init__(self, name: str, mount: MountPose, rate_hz: float, cone_deg: float, range_min: float,
                 range_max: float, noise_std: float, dropout_prob: float):
        super().__init__(name, mount, rate_hz)
        self.cone_deg = cone_deg
        self.range_min, self.range_max = range_min, range_max
        self.noise_std, self.dropout_prob = noise_std, dropout_prob

    def measure(self, scenario, state, t: float, rng: np.random.Generator) -> UltrasonicMeasurement:
        scene = scene_arrays(scenario)
        x, y, yaw = self.mount.in_map(state)
        origin = np.array([x, y])
        true = nearest_in_cone(origin, yaw, math.radians(self.cone_deg) / 2.0,
                               *within_reach(origin, self.range_max, scene.obstacle_a, scene.obstacle_b))
        if not self.range_min <= true <= self.range_max:
            return UltrasonicMeasurement(self.name, t, None)
        if self.dropout_prob > 0.0 and rng.random() < self.dropout_prob:
            return UltrasonicMeasurement(self.name, t, None)
        noise = rng.normal(0.0, self.noise_std) if self.noise_std > 0.0 else 0.0
        return UltrasonicMeasurement(self.name, t, true + noise)


def make_ultrasonic_sensors(cfg: Optional[dict] = None) -> List[UltrasonicSensor]:
    """The configured ring plus the side-looking pair. A sensor's own entry overrides the shared settings."""
    cfg = cfg if cfg is not None else load_sensor_params()["ultrasonic"]
    sensors = []
    for entry in list(cfg["sensors"]) + list(cfg["side_sensors"]):
        merged = {**cfg, **entry}
        sensors.append(UltrasonicSensor(entry["name"], MountPose.from_config(entry), merged["rate_hz"],
                                        merged["cone_deg"], merged["range"][0], merged["range"][1],
                                        merged["noise_std"], merged["dropout_prob"]))
    return sensors
