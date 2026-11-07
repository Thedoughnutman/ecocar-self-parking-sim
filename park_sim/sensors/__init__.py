"""Synthetic 2D sensors: line cameras, ultrasonics and an optional lidar. See docs/sensors.md."""
from __future__ import annotations

from typing import List, Optional

from .base import Measurement, MountPose, Sensor, SensorSuite, load_sensor_params
from .lidar2d import Lidar2D, LidarMeasurement, make_lidar
from .line_camera import FALSE_POSITIVE, LineCamera, LineCameraMeasurement, PaintWear, make_line_cameras
from .ultrasonic import UltrasonicMeasurement, UltrasonicSensor, make_ultrasonic_sensors

__all__ = ["FALSE_POSITIVE", "Lidar2D", "LidarMeasurement", "LineCamera", "LineCameraMeasurement", "Measurement",
           "MountPose", "PaintWear", "Sensor", "SensorSuite", "UltrasonicMeasurement", "UltrasonicSensor",
           "load_sensor_params", "make_lidar", "make_line_cameras", "make_sensors", "make_suite",
           "make_ultrasonic_sensors"]


def make_sensors(cfg: Optional[dict] = None) -> List[Sensor]:
    """Every sensor config/sensors.yaml enables."""
    cfg = cfg if cfg is not None else load_sensor_params()
    sensors: List[Sensor] = []
    if cfg["line_camera"]["enabled"]:
        sensors.extend(make_line_cameras(cfg["line_camera"]))
    if cfg["ultrasonic"]["enabled"]:
        sensors.extend(make_ultrasonic_sensors(cfg["ultrasonic"]))
    if cfg["lidar"]["enabled"]:
        sensors.append(make_lidar(cfg["lidar"]))
    return sensors


def make_suite(cfg: Optional[dict] = None, seed: int = 0) -> SensorSuite:
    return SensorSuite(make_sensors(cfg), seed)
