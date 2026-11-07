"""Safety monitor, separate from the planner: standard library + numpy, and no ground truth.

SafetyMonitor sweeps the car's footprint along the remaining planned path, as far as it could need
to stop, and reports whether an ultrasonic or lidar return falls inside. Everything is in the frame of
the estimated pose, the same one the controller follows. StreamWatchdog turns real stream ages into
the supervisor's SensorHealth inputs; the stale check itself stays in the supervisor.
SteeringMonitor reports steering that isn't following its command. See docs/safety.md.

An ultrasonic reading is a distance with no bearing: the object is somewhere on an arc across the cone.
A reading counts if the whole arc is in the sweep. A parked car at the edge of a cone is on the arc and
outside the sweep, so it doesn't read as something in the path, wherever it sits. The price is range: the
rear inner sensors only get their whole arc inside at about 1.3 m, and the side and corner sensors
hardly ever. So within close_range_m any part of the arc in the sweep counts: an arc that short can't
be far from where the object really is (docs/safety.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .supervisor import SensorHealth


def stop_distance(speed: float, decel: float, delay: float, jerk: float) -> float:
    """Meters to stop from `speed` with `delay` s before the brakes act, then a ramp to `decel`."""
    v = abs(speed)
    return v * (delay + decel / (2.0 * jerk)) + v * v / (2.0 * decel)


@dataclass(frozen=True)
class SafetyConfig:
    body_rear: float            # m behind the rear axle
    body_front: float           # m ahead of the rear axle
    body_half_width: float      # m
    sweep_step_m: float
    lateral_margin_m: float
    end_margin_m: float
    standoff_m: float
    sensor_latency_s: float
    brake_dead_time_s: float
    brake_jerk: float
    service_decel: float
    ttc_emergency_s: float
    standstill_speed: float
    confirm_hits: int
    clear_readings: int
    hit_gate_m: float
    clear_tolerance_m: float
    arc_samples: int
    close_range_m: float

    @classmethod
    def from_params(cls, safety: dict, vehicle, standstill_speed: float) -> "SafetyConfig":
        """`safety` is the safety: section of park_params.yaml."""
        return cls(vehicle.rear_overhang, vehicle.wheelbase + vehicle.front_overhang, vehicle.width / 2.0,
                   safety["sweep_step_m"], safety["lateral_margin_m"], safety["end_margin_m"], safety["standoff_m"],
                   safety["sensor_latency_s"], vehicle.brake_dead_time, vehicle.brake_jerk,
                   vehicle.brake_service_decel, safety["ttc_emergency_s"], standstill_speed,
                   safety["confirm_hits"], safety["clear_readings"], safety["hit_gate_m"], safety["clear_tolerance_m"],
                   safety["arc_samples"], safety["close_range_m"])


@dataclass(frozen=True)
class SensorSpec:
    """Where a range sensor is and what it can see. Mount in the vehicle frame, rear-axle origin."""
    name: str
    x: float
    y: float
    yaw: float          # rad
    half_cone: float    # rad
    range_min: float    # m; nearer than this it reads None: the blind zone
    range_max: float    # m


@dataclass(frozen=True)
class RangeReading:
    sensor: str
    distance: Optional[float]   # m; None = nothing seen OR something in the blind zone


@dataclass(frozen=True, eq=False)
class ScanReading:
    """A lidar scan as points in the same frame as the pose, and what each beam covered."""
    points: np.ndarray      # (K, 2) returns
    origin: Tuple[float, float]
    angles: np.ndarray      # (N,) rad in that frame, ascending
    ranges: np.ndarray      # (N,) m; nan = no return


@dataclass(frozen=True, eq=False)
class SafetyOutput:
    obstacle_in_path: bool
    emergency_brake: bool
    ttc_s: Optional[float]      # s to the nearest return in the sweep, at this speed; None if none, or stopped
    gap_m: Optional[float]      # m along the path to that return
    stop_distance_m: float
    horizon_m: float            # stop distance + standoff, how far along the path the sweep may reach
    sweep: np.ndarray           # (K, 3) x, y, yaw of the swept footprints; the first is the car now
    hazards: np.ndarray         # (H, 2) returns inside the sweep


@dataclass
class _Return:
    """A sensor's latest return. It stays when the sensor goes quiet with it in the blind zone."""
    arc: np.ndarray             # (A, 2) everywhere along the cone it could be
    distance: float             # m
    x: float                    # on the sensor's axis
    y: float
    hits: int = 1
    misses: int = 0

class SafetyMonitor:
    def __init__(self, cfg: SafetyConfig, sensors: Sequence[SensorSpec] = ()):
        self.cfg = cfg
        self.specs: Dict[str, SensorSpec] = {s.name: s for s in sensors}
        self.returns: Dict[str, _Return] = {}
        self._scan: Optional[ScanReading] = None

    # ---------- inputs ----------
    def _sensor_at(self, spec: SensorSpec, pose: Tuple[float, float, float]) -> Tuple[float, float, float]:
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return pose[0] + c * spec.x - s * spec.y, pose[1] + s * spec.x + c * spec.y, pose[2] + spec.yaw

    def observe(self, pose: Tuple[float, float, float], readings: Iterable[RangeReading],
                scan: Optional[ScanReading] = None) -> None:
        """Take in the readings that arrived this tick."""
        for reading in readings:
            self._take(pose, reading)
        if scan is not None:
            self._scan = scan
            self._clear_by_scan(scan)

    def _take(self, pose: Tuple[float, float, float], reading: RangeReading) -> None:
        cfg, spec = self.cfg, self.specs[reading.sensor]
        x, y, yaw = self._sensor_at(spec, pose)
        known = self.returns.get(reading.sensor)
        if reading.distance is not None:
            px, py = x + reading.distance * math.cos(yaw), y + reading.distance * math.sin(yaw)
            same = known is not None and math.hypot(px - known.x, py - known.y) <= cfg.hit_gate_m
            angles = yaw + np.linspace(-spec.half_cone, spec.half_cone, cfg.arc_samples)
            arc = np.column_stack([x + reading.distance * np.cos(angles), y + reading.distance * np.sin(angles)])
            self.returns[reading.sensor] = _Return(arc, reading.distance, px, py, known.hits + 1 if same else 1)
        elif known is not None and math.hypot(known.x - x, known.y - y) >= spec.range_min:
            known.misses += 1       # None with the point in view: it is gone. In the blind zone, None says nothing
            if known.misses >= cfg.clear_readings:
                del self.returns[reading.sensor]
        self._clear_by_neighbor(pose, reading, spec, (x, y, yaw))

    def _clear_by_neighbor(self, pose, reading: RangeReading, spec: SensorSpec, at: Tuple[float, float, float]) -> None:
        """A sensor with a clear view of a point another sensor is holding in its blind zone."""
        cfg = self.cfg
        for name, kept in list(self.returns.items()):
            if name == reading.sensor:
                continue
            other = self.specs[name]
            ox, oy, _ = self._sensor_at(other, pose)
            if math.hypot(kept.x - ox, kept.y - oy) >= other.range_min:
                continue
            dx, dy = kept.x - at[0], kept.y - at[1]
            rng = math.hypot(dx, dy)
            bearing = abs((math.atan2(dy, dx) - at[2] + math.pi) % (2.0 * math.pi) - math.pi)
            if not (spec.range_min <= rng <= spec.range_max and bearing <= spec.half_cone):
                continue
            if reading.distance is None or reading.distance > rng + cfg.clear_tolerance_m:
                kept.misses += 1
                if kept.misses >= cfg.clear_readings:
                    del self.returns[name]

    def _clear_by_scan(self, scan: ScanReading) -> None:
        """Returns the lidar's beams pass straight through."""
        cfg = self.cfg
        if len(scan.angles) == 0:
            return
        for name, kept in list(self.returns.items()):
            dx, dy = kept.x - scan.origin[0], kept.y - scan.origin[1]
            wrapped = (math.atan2(dy, dx) - scan.angles[0]) % (2.0 * math.pi)
            step = 2.0 * math.pi / len(scan.angles)
            measured = scan.ranges[int(round(wrapped / step)) % len(scan.angles)]
            if math.isnan(measured) or measured > math.hypot(dx, dy) + cfg.clear_tolerance_m:
                kept.misses += 1
                if kept.misses >= cfg.clear_readings:
                    del self.returns[name]

    # ---------- the check ----------
    def horizon(self, speed: float) -> Tuple[float, float]:
        """(stop distance, how far along the path to sweep) at this speed."""
        cfg = self.cfg
        d = stop_distance(speed, cfg.service_decel, cfg.sensor_latency_s + cfg.brake_dead_time_s, cfg.brake_jerk)
        return d, d + cfg.standoff_m

    def sweep(self, pose: Tuple[float, float, float], path: np.ndarray, horizon: float) -> Tuple[np.ndarray, np.ndarray]:
        """Poses to sweep the body over and their distance along the path: the car now, then the
        remaining path (N, 3) resampled to sweep_step_m and cut at the horizon."""
        now = np.array(pose, dtype=float)[None]
        if len(path) == 0:
            return now, np.zeros(1)
        all_poses = np.vstack([now, path])
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(all_poses[:, :2], axis=0).T))])
        within = s <= horizon
        within[0] = True
        poses, s = all_poses[within], s[within]
        keep = np.concatenate([[True], np.diff(np.floor(s / self.cfg.sweep_step_m)) > 0.0])
        keep[-1] = True
        return poses[keep], s[keep]

    def first_inside(self, poses: np.ndarray, points: np.ndarray) -> np.ndarray:
        """For each point (P, 2), the index of the first pose whose body plus margins holds it; -1 if none."""
        cfg = self.cfg
        c, s = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
        dx, dy = points[:, 0] - poses[:, 0, None], points[:, 1] - poses[:, 1, None]
        along, side = c * dx + s * dy, -s * dx + c * dy
        inside = ((along >= -cfg.body_rear - cfg.end_margin_m) & (along <= cfg.body_front + cfg.end_margin_m)
                  & (np.abs(side) <= cfg.body_half_width + cfg.lateral_margin_m))
        return np.where(inside.any(axis=0), inside.argmax(axis=0), -1)

    def _hazards(self, poses: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """The returns that are in the sweep wherever they could be, as points, and the nearest one's
        distance along the path."""
        points, gaps = [], []
        for r in self.returns.values():
            if r.hits < self.cfg.confirm_hits:
                continue
            first = self.first_inside(poses, r.arc)
            inside = first >= 0
            if inside.all() or (inside.any() and r.distance <= self.cfg.close_range_m):
                points.append(r.arc[inside])
                gaps.append(float(s[first[inside]].min()))
        if self._scan is not None and len(self._scan.points):
            first = self.first_inside(poses, self._scan.points)
            if (first >= 0).any():
                points.append(self._scan.points[first >= 0])
                gaps.append(float(s[first[first >= 0]].min()))
        if not points:
            return np.zeros((0, 2)), None
        return np.vstack(points), min(gaps)

    def check(self, pose: Tuple[float, float, float], speed: float, path: np.ndarray,
              planned_speed: float = 0.0) -> SafetyOutput:
        """pose: estimated (x, y, yaw); speed: m/s, signed; path: the path_ahead(max_length) to drive,
        (N, 3); planned_speed: what the car was last asked to do, so that a car about to pull away
        looks as far as a moving one."""
        cfg = self.cfg
        stop, horizon = self.horizon(max(abs(speed), abs(planned_speed)))
        poses, s = self.sweep(pose, path, horizon)
        hazards, gap = self._hazards(poses, s)
        ttc = gap / abs(speed) if gap is not None and abs(speed) > cfg.standstill_speed else None
        emergency = ttc is not None and ttc < cfg.ttc_emergency_s
        return SafetyOutput(gap is not None, emergency, ttc, gap, stop, horizon, poses, hazards)

    def look_ahead(self, speed: float, planned_speed: float = 0.0) -> float:
        """How much remaining path check() needs at this speed."""
        return self.horizon(max(abs(speed), abs(planned_speed)))[1]


def _same_value(a, b) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, np.ndarray):
        return a.size > 0 and np.array_equal(a, b, equal_nan=True)
    return a == b


class StreamWatchdog:
    """Remembers when each stream last delivered something new. A reading identical to the last while
    the car moves isn't new: real readings carry noise, so that is a stuck sensor with fresh timestamps.
    A stuck "nothing" can't be told from a real one. At a stop, repeats are normal and count as new."""

    def __init__(self, names: Iterable[str], t0: float = 0.0):
        self.last: Dict[str, float] = {name: t0 for name in names}
        self._values: Dict[str, object] = {}

    def heard(self, name: str, t: float, value=None, moving: bool = False) -> None:
        if moving and _same_value(value, self._values.get(name)):
            return
        self._values[name] = value
        self.last[name] = t

    def health(self, extra: Sequence[SensorHealth] = ()) -> Tuple[SensorHealth, ...]:
        """The supervisor's inputs. The supervisor alone decides what is too old."""
        return tuple(SensorHealth(name, True, t) for name, t in self.last.items()) + tuple(extra)


class SteeringMonitor:
    """Unhealthy once the actual angle has sat away from the command, not moving toward it, for fault_time.
    The rate limit and the dead time don't trigger it: the wheels are moving by then, or soon after."""

    def __init__(self, error_rad: float, stuck_rate_rad_s: float, fault_time_s: float):
        self.error, self.stuck_rate, self.fault_time = error_rad, stuck_rate_rad_s, fault_time_s
        self.healthy = True
        self._since: Optional[float] = None
        self._last: Optional[float] = None

    def update(self, t: float, dt: float, command: float, actual: float) -> bool:
        rate = abs(actual - self._last) / dt if self._last is not None else math.inf
        self._last = actual
        if abs(command - actual) > self.error and rate < self.stuck_rate:
            if self._since is None:
                self._since = t
            if t - self._since >= self.fault_time:
                self.healthy = False
        else:
            self._since = None
        return self.healthy
