"""Path-following controller for parking maneuvers.

Splits the planned path into constant-direction segments. For each one:
  lateral:      pure pursuit about the rear axle (same law forward and reverse;
                the target point is simply behind the car when reversing)
  longitudinal: speed limited by distance-to-go so the car stops at cusps/goal, and by how
                fast the plan asks the steering to change (controller.steer_slew_window_m)
At each cusp: stop, hold for the gear-change time, pre-steer for the next
segment, then continue. Output is (road-wheel angle cmd, signed speed cmd).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .geometry import wrap_angle
from .hybrid_astar import PlannedPath
from .vehicle import VehicleParams, VehicleState


@dataclass
class _Segment:
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    s: np.ndarray        # cumulative arc length
    direction: int

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def start_steer(self, wheelbase: float) -> float:
        """Road-wheel angle matching the segment's initial curvature."""
        n = min(len(self.x) - 1, 4)
        if n < 1 or self.s[n] < 1e-6:
            return 0.0
        k = wrap_angle(self.yaw[n] - self.yaw[0]) / self.s[n] * self.direction
        return math.atan(wheelbase * k)

    def steering_speed_cap(self, vehicle: VehicleParams, window: float, decel: float) -> np.ndarray:
        """Speed limit at each path point, m/s. A plan's curvature can jump where a real wheel has to slew, and at
        full speed the car then runs wide of the path with no steering left to come back. So: slow enough to
        make the steering change the plan asks for within `window` meters, and start slowing in time to be there.
        Stretches with no big change stay at top speed."""
        n = len(self.x)
        cap = np.full(n, vehicle.max_speed)
        if n < 2:
            return cap
        ds = np.maximum(np.diff(self.s), 1e-6)
        k = np.array([wrap_angle(a) for a in np.diff(self.yaw)]) / ds * self.direction
        steer = np.clip(np.arctan(vehicle.wheelbase * np.append(k, k[-1])), -vehicle.max_steer, vehicle.max_steer)
        for i in range(n):
            j = int(np.searchsorted(self.s, self.s[i] + window))
            change = float(np.max(np.abs(steer[i:j + 1] - steer[i])))
            if change > 1e-6:
                cap[i:j + 1] = np.minimum(cap[i:j + 1], vehicle.steer_rate * window / change)
        for i in range(n - 2, -1, -1):
            cap[i] = min(cap[i], math.sqrt(cap[i + 1] ** 2 + 2.0 * decel * (self.s[i + 1] - self.s[i])))
        return cap


class ParkingController:
    IDLE, SHIFTING, DRIVING, DONE = "idle", "shifting", "driving", "done"

    def __init__(self, path: PlannedPath, vehicle: VehicleParams, cfg: dict,
                 initial_gear: int = 1, standstill_steer: bool = True):
        self.p, self.c = vehicle, cfg
        self.segments: List[_Segment] = []
        for a, b, d in path.segments():
            x, y, yaw = path.x[a:b + 1], path.y[a:b + 1], path.yaw[a:b + 1]
            s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
            self.segments.append(_Segment(x, y, yaw, s, d))
        self._speed_caps = [self._speed_cap_for(seg) for seg in self.segments]
        self.seg_i = 0
        self.gear = initial_gear
        self.gear_changes = 0
        self.state = self.IDLE
        self.timer = 0.0
        self.near_i = 0
        self.standstill_steer = standstill_steer
        self.last_steer = 0.0
        self.stopping = False   # latched once the stop point is reached (no re-creep)
        # Telemetry only: written by update(), never read back by the control law
        self.lookahead: Optional[tuple] = None   # (x, y) pure-pursuit target, None when not driving
        # m left on the current segment (drives the speed profile)
        self.seg_dist_to_go = self.segments[0].length if self.segments else 0.0

    def _speed_cap_for(self, seg: _Segment) -> np.ndarray:
        return seg.steering_speed_cap(self.p, self.c["steer_slew_window_m"], self._profile_decel)

    @property
    def _profile_decel(self) -> float:
        return 0.5 * self.p.decel

    @property
    def seg(self) -> Optional[_Segment]:
        return self.segments[self.seg_i] if self.seg_i < len(self.segments) else None

    @property
    def dist_to_go(self) -> float:
        """Meters left on the whole path: current segment plus all later ones."""
        if self.seg is None:
            return 0.0
        return self.seg_dist_to_go + sum(s.length for s in self.segments[self.seg_i + 1:])

    def distance_left_is_within(self, distance: float) -> bool:
        """On the last segment, with at most this many meters to go."""
        return self.seg_i == len(self.segments) - 1 and self.state == self.DRIVING and self.seg_dist_to_go <= distance

    def move_final_stop(self, along: float) -> None:
        """Move the end of the last segment along its own direction of travel (+ = further), keeping
        its line. Points past a pulled-in stop are dropped."""
        seg = self.segments[-1]
        target = seg.s[-1] + along
        keep = max(int(np.searchsorted(seg.s, target)), 1)
        c, s = math.cos(seg.yaw[-1]), math.sin(seg.yaw[-1])
        ex = seg.x[-1] + along * seg.direction * c
        ey = seg.y[-1] + along * seg.direction * s
        seg.x = np.append(seg.x[:keep], ex)
        seg.y = np.append(seg.y[:keep], ey)
        seg.yaw = np.append(seg.yaw[:keep], seg.yaw[-1])
        seg.s = np.append(seg.s[:keep], target)
        self._speed_caps[-1] = self._speed_cap_for(seg)
        self.near_i = min(self.near_i, len(seg.x) - 1)

    def _nearest_ahead(self, seg, x: float, y: float) -> int:
        """Index of the nearest path point, searching forward from near_i (monotonic search window)."""
        lo, hi = self.near_i, min(len(seg.x), self.near_i + 40)
        return lo + int(np.argmin(np.hypot(seg.x[lo:hi] - x, seg.y[lo:hi] - y)))

    def remaining_path(self, max_length: float, at: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """The path still to drive from the nearest point, across cusps, up to max_length meters along
        it: (N, 3) x, y, yaw. Empty when finished. Telemetry for the safety monitor, not used by control.
        `at` is where the car is now. update() doesn't run while the car is stopped, so near_i alone is
        where it was when the brakes came on; this finds the nearest point without changing near_i."""
        parts, left = [], max_length
        for k in range(self.seg_i, len(self.segments)):
            seg = self.segments[k]
            first = 0
            if k == self.seg_i:
                first = self.near_i if at is None else self._nearest_ahead(seg, at[0], at[1])
            last = max(int(np.searchsorted(seg.s, seg.s[first] + left, side="right")), first + 1)
            parts.append(np.column_stack([seg.x[first:last], seg.y[first:last], seg.yaw[first:last]]))
            left -= seg.s[last - 1] - seg.s[first]
            if left <= 0.0:
                break
        return np.concatenate(parts) if parts else np.zeros((0, 3))

    def gear_changes_with(self, path: PlannedPath) -> int:
        """How many gear changes the maneuver would end up with if it switched to `path` now."""
        first_shift = 1 if path.direction[0] != self.gear else 0
        return self.gear_changes + first_shift + path.n_direction_changes

    def continued_on(self, path: PlannedPath) -> "ParkingController":
        """A controller for `path` that takes over from this one, keeping its gear and gear count."""
        fresh = ParkingController(path, self.p, self.c, initial_gear=self.gear)
        fresh.gear_changes = self.gear_changes
        fresh.last_steer = self.last_steer
        return fresh

    def _begin_segment(self) -> None:
        self.near_i = 0
        self.stopping = False
        if self.seg.direction != self.gear:
            self.state, self.timer = self.SHIFTING, 0.0
        else:
            self.state = self.DRIVING

    def _predict(self, st: VehicleState) -> VehicleState:
        """Project the pose ahead by the steering delay so commands land on time."""
        T = self.p.steer_dead_time + self.p.steer_tau
        ds = st.v * T
        k = math.tan(st.delta) / self.p.wheelbase
        if abs(k) < 1e-9:
            x, y, yaw = st.x + ds * math.cos(st.yaw), st.y + ds * math.sin(st.yaw), st.yaw
        else:
            yaw = st.yaw + ds * k
            x = st.x + (math.sin(yaw) - math.sin(st.yaw)) / k
            y = st.y + (-math.cos(yaw) + math.cos(st.yaw)) / k
        return VehicleState(x, y, yaw, st.v, st.delta)

    def update(self, st: VehicleState, dt: float):
        """Returns (steer_cmd, speed_cmd)."""
        if self.state == self.DONE:
            return self.last_steer, 0.0
        if self.state == self.IDLE:
            self._begin_segment()

        seg = self.seg
        if self.state == self.SHIFTING:
            self.lookahead, self.seg_dist_to_go = None, seg.length
            # Hold stopped; change gear after the shift time; pre-steer if allowed
            target = seg.start_steer(self.p.wheelbase) if self.standstill_steer else self.last_steer
            self.last_steer = target
            if abs(st.v) < 0.02:
                self.timer += dt
            if self.timer >= self.p.shift_time:
                self.gear = seg.direction
                self.gear_changes += 1
                self.state = self.DRIVING
            return target, 0.0

        # ---- DRIVING ----
        d = seg.direction
        st = self._predict(st)
        # nearest point (monotonic search window)
        self.near_i = self._nearest_ahead(seg, st.x, st.y)
        i = self.near_i

        # distance to go: arc remaining, refined by projection onto end tangent
        ex, ey, eyaw = seg.x[-1], seg.y[-1], seg.yaw[-1]
        proj = ((ex - st.x) * math.cos(eyaw) + (ey - st.y) * math.sin(eyaw)) * d
        s_rem = seg.length - seg.s[i]
        if s_rem < 1.0:
            s_rem = proj

        # lookahead target
        ld = max(self.c["lookahead_min"], self.c["lookahead_gain"] * abs(st.v))
        tx = ty = None
        for j in range(i, len(seg.x)):
            if math.hypot(seg.x[j] - st.x, seg.y[j] - st.y) >= ld:
                tx, ty = seg.x[j], seg.y[j]
                break
        if tx is None:     # extend past the segment end along its tangent
            ext = max(ld - math.hypot(ex - st.x, ey - st.y), 0.0) + 0.3
            tx, ty = ex + d * ext * math.cos(eyaw), ey + d * ext * math.sin(eyaw)
        self.lookahead, self.seg_dist_to_go = (float(tx), float(ty)), float(s_rem)
        c, s = math.cos(st.yaw), math.sin(st.yaw)
        lx = c * (tx - st.x) + s * (ty - st.y)
        ly = -s * (tx - st.x) + c * (ty - st.y)
        curv = 2.0 * ly / max(lx * lx + ly * ly, 1e-6)
        steer = float(np.clip(math.atan(self.p.wheelbase * curv), -self.p.max_steer, self.p.max_steer))
        self.last_steer = steer

        # speed profile
        decel = self._profile_decel
        v_max = float(self._speed_caps[self.seg_i][i])
        v_stop = math.sqrt(max(2 * decel * max(s_rem, 0.0), 0.0))
        v_t = min(v_max, v_stop)
        v_t = max(v_t, min(self.c["min_creep_speed"], 1.5 * max(s_rem, 0.0)))
        settled = s_rem < 3 * self.c["stop_tol"] and abs(st.v) < 0.03
        if s_rem <= self.c["stop_tol"] or settled:
            self.stopping = True
        if self.stopping:
            v_t = 0.0
            if abs(st.v) < 0.02:
                self.seg_i += 1
                if self.seg is None:
                    self.state = self.DONE
                    self.lookahead, self.seg_dist_to_go = None, 0.0
                else:
                    self._begin_segment()
                    self.seg_dist_to_go = self.seg.length
        return steer, d * v_t
