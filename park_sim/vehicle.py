"""Kinematic bicycle model of the LYRIQ with actuator dynamics.

Reference point is the rear-axle center. State: x, y, yaw, speed v (signed,
negative = reverse) and actual road-wheel angle delta.

    x_dot   = v cos(yaw)
    y_dot   = v sin(yaw)
    yaw_dot = v / L * tan(delta)

Actuators (placeholders until measured on the car):
  steering: dead time -> rate limit -> first-order lag -> mechanical lock
  speed:    first-order lag toward the commanded speed, accel/decel limited

Optional effects, off by default: tire scrub (sideways slip at standstill and crawl) and road grade
(a disturbance in the speed loop).

Brakes: step(..., brake_decel) asks for a deceleration on top of the speed loop, through a dead time
and a jerk limit. A run that never asks is untouched. Injected steering faults: steer_limit (EPS
saturation) and steer_stuck (command not followed), both off by default.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

GRAVITY = 9.81          # m/s^2
ROLLING_SPEED = 1e-3    # m/s, below this the car counts as stopped for tire scrub


def max_steer_from_turning_circle(turning_circle_curb: float, wheelbase: float,
                                  track: float, tire_width: float) -> float:
    """Bicycle-equivalent max road-wheel angle [rad] from a curb-to-curb circle.

    The curb-to-curb radius is traced by the outer edge of the outer front tire.
    Outer front wheel center radius: R_o = R_curb - tire_width/2
    R_o^2 = L^2 + (R_rear + track/2)^2  ->  R_rear (rear-axle-center radius)
    delta = atan(L / R_rear)
    """
    r_outer = turning_circle_curb / 2.0 - tire_width / 2.0
    r_rear = math.sqrt(r_outer ** 2 - wheelbase ** 2) - track / 2.0
    return math.atan(wheelbase / r_rear)


@dataclass
class VehicleParams:
    wheelbase: float
    length: float
    width: float
    front_overhang: float
    rear_overhang: float
    max_steer: float            # rad, mechanical lock (bicycle equivalent)
    planner_steer: float        # rad, what the planner is allowed to use
    steer_rate: float           # rad/s
    steer_tau: float            # s
    steer_dead_time: float      # s
    max_speed: float            # m/s
    accel: float                # m/s^2
    decel: float                # m/s^2
    speed_tau: float            # s
    shift_time: float           # s
    width_mirrors: float = 0.0
    height: float = 0.0         # m, visualization only (the planner works in 2D)
    track: float = 0.0          # m, wheel speeds for odometry only
    tire_scrub_gain: float = 0.0            # m/s of sideways slip per rad of road-wheel angle; 0 = off
    tire_scrub_crawl_speed: float = 0.5     # m/s, scrub fades to nothing at this speed
    tire_scrub_min_steer_rate: float = 0.0  # rad/s, at standstill only wheels turning faster than this scrub
    road_grade_percent: float = 0.0         # % rise along map +x; 0 = off
    brake_dead_time: float = 0.0            # s, request to first pressure
    brake_jerk: float = math.inf            # m/s^3, how fast the deceleration builds
    brake_service_decel: float = 0.0        # m/s^2, a safety stop
    brake_emergency_decel: float = 0.0      # m/s^2, an emergency stop

    @property
    def min_turn_radius(self) -> float:
        """Rear-axle turning radius at the planner's steering limit."""
        return self.wheelbase / math.tan(self.planner_steer)

    def footprint_local(self, margin: float = 0.0) -> np.ndarray:
        """Body rectangle corners in the rear-axle frame (CCW), 4x2."""
        xr = -self.rear_overhang - margin
        xf = self.wheelbase + self.front_overhang + margin
        hw = self.width / 2.0 + margin
        return np.array([[xr, -hw], [xf, -hw], [xf, hw], [xr, hw]])

    def footprint(self, x: float, y: float, yaw: float, margin: float = 0.0) -> np.ndarray:
        local = self.footprint_local(margin)
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + np.array([x, y])

    def center_offset(self) -> float:
        """Distance from rear axle to geometric body center, along heading."""
        return (self.wheelbase + self.front_overhang - self.rear_overhang) / 2.0

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "VehicleParams":
        if path is None:
            path = str(Path(__file__).resolve().parents[1] / "config" / "lyriq_vehicle.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        g, st, lo, re, br = cfg["geometry"], cfg["steering"], cfg["longitudinal"], cfg["realism"], cfg["brake"]
        if st.get("max_road_wheel_angle_deg"):
            max_steer = math.radians(st["max_road_wheel_angle_deg"])
        else:
            max_steer = max_steer_from_turning_circle(
                g["turning_circle_curb"], g["wheelbase"], g["track"], g["tire_width"])
        return cls(
            wheelbase=g["wheelbase"], length=g["length"], width=g["width"],
            front_overhang=g["front_overhang"], rear_overhang=g["rear_overhang"],
            width_mirrors=g.get("width_mirrors", 0.0), height=g["height"],
            max_steer=max_steer,
            planner_steer=max_steer * st["planner_steer_fraction"],
            steer_rate=math.radians(st["rate_limit_deg_s"]),
            steer_tau=st["time_constant_s"], steer_dead_time=st["dead_time_s"],
            max_speed=lo["max_speed_mps"], accel=lo["accel_mps2"], decel=lo["decel_mps2"],
            speed_tau=lo["speed_time_constant_s"], shift_time=lo["shift_time_s"],
            track=g["track"],
            tire_scrub_gain=re["tire_scrub_gain"], tire_scrub_crawl_speed=re["tire_scrub_crawl_speed"],
            tire_scrub_min_steer_rate=math.radians(re["tire_scrub_min_steer_rate_deg_s"]),
            road_grade_percent=re["road_grade_percent"],
            brake_dead_time=br["dead_time_s"], brake_jerk=br["jerk_mps3"],
            brake_service_decel=br["service_decel_mps2"], brake_emergency_decel=br["emergency_decel_mps2"],
        )


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    v: float = 0.0
    delta: float = 0.0


@dataclass
class Vehicle:
    """Plant: kinematic bicycle + steering and speed actuator models."""
    params: VehicleParams
    state: VehicleState
    dt: float = 0.02
    _steer_buf: deque = field(default_factory=deque, init=False)
    _steer_rl: float = field(default=0.0, init=False)   # rate-limited intermediate
    steer_limit: Optional[float] = field(default=None, init=False)  # rad, EPS saturation fault; None = off
    steer_stuck: bool = field(default=False, init=False)            # the wheels ignore steering commands
    _brake_buf: deque = field(default_factory=deque, init=False)    # requests inside the dead time
    _brake_now: float = field(default=0.0, init=False)              # m/s^2, deceleration reached so far
    _braking: bool = field(default=False, init=False)

    def __post_init__(self):
        n = max(1, int(round(self.params.steer_dead_time / self.dt)))
        self._steer_buf = deque([self.state.delta] * n, maxlen=n)
        self._steer_rl = self.state.delta
        n = max(1, int(round(self.params.brake_dead_time / self.dt)))
        self._brake_buf = deque([0.0] * n, maxlen=n)

    def step(self, steer_cmd: float, speed_cmd: float, brake_decel: float = 0.0) -> VehicleState:
        """brake_decel: m/s^2 asked of the brakes, on top of the speed loop; 0 = not braking."""
        p, s, dt = self.params, self.state, self.dt

        # Steering: dead time -> rate limit -> first-order lag -> lock
        delta_before = s.delta
        if not self.steer_stuck:
            delayed = self._steer_buf[0]
            limit = p.max_steer if self.steer_limit is None else min(p.max_steer, self.steer_limit)
            self._steer_buf.append(float(np.clip(steer_cmd, -limit, limit)))
            max_d = p.steer_rate * dt
            self._steer_rl += float(np.clip(delayed - self._steer_rl, -max_d, max_d))
            alpha = dt / (p.steer_tau + dt)
            s.delta += alpha * (self._steer_rl - s.delta)
            s.delta = float(np.clip(s.delta, -p.max_steer, p.max_steer))

        # Speed: first-order lag, limited by accel/decel
        v_before = s.v
        speed_cmd = float(np.clip(speed_cmd, -p.max_speed, p.max_speed))
        dv = (speed_cmd - s.v) * dt / (p.speed_tau + dt)
        speeding_up = abs(speed_cmd) > abs(s.v) and speed_cmd * s.v >= 0
        lim = (p.accel if speeding_up else p.decel) * dt
        s.v += float(np.clip(dv, -lim, lim))
        if p.road_grade_percent and speed_cmd != 0.0:   # at a zero command the brakes hold the car
            s.v -= GRAVITY * p.road_grade_percent / 100.0 * math.cos(s.yaw) * dt
        if brake_decel > 0.0 or self._braking:
            self._brake(brake_decel, v_before)

        # Kinematics (exact integration over the step for constant v, delta)
        ds = s.v * dt
        k = math.tan(s.delta) / p.wheelbase
        if abs(k) < 1e-9:
            s.x += ds * math.cos(s.yaw)
            s.y += ds * math.sin(s.yaw)
        else:
            dyaw = ds * k
            s.x += (math.sin(s.yaw + dyaw) - math.sin(s.yaw)) / k
            s.y += (-math.cos(s.yaw + dyaw) + math.cos(s.yaw)) / k
            s.yaw += dyaw
        s.yaw = (s.yaw + math.pi) % (2 * math.pi) - math.pi
        if p.tire_scrub_gain:
            self._scrub(abs(s.delta - delta_before) / dt)
        return s

    def _brake(self, request: float, v_before: float) -> None:
        """The request acts after the dead time, building at the jerk limit, and can only slow the car:
        it never reverses it, and a stronger slowdown from the speed loop stands."""
        p, s = self.params, self.state
        target = self._brake_buf[0]
        self._brake_buf.append(request)
        step = p.brake_jerk * self.dt
        self._brake_now += float(np.clip(target - self._brake_now, -step, step))
        self._braking = self._brake_now > 0.0 or any(r > 0.0 for r in self._brake_buf)
        braked = max(abs(v_before) - self._brake_now * self.dt, 0.0)
        if abs(s.v) > braked:
            s.v = math.copysign(braked, v_before)

    def _scrub(self, steer_rate: float) -> None:
        """Sideways slip, unseen by wheel speeds and yaw rate: proportional to the road-wheel angle,
        fading with speed. A car at standstill only slips while its wheels are being turned."""
        p, s = self.params, self.state
        rolling = abs(s.v) > ROLLING_SPEED
        if not rolling and steer_rate <= p.tire_scrub_min_steer_rate:
            return
        slip = p.tire_scrub_gain * s.delta * max(0.0, 1.0 - abs(s.v) / p.tire_scrub_crawl_speed) * self.dt
        s.x -= slip * math.sin(s.yaw)
        s.y += slip * math.cos(s.yaw)
