"""Slot detectors: turn what the car has seen into a list of parking slots.

The simulator only uses the SlotDetector interface. IdealSlotDetector reads the true lot.
LineSlotDetector, SpaceSlotDetector and FusedSlotDetector work from camera and ultrasonic
measurements, in the odometry frame, and keep slots as tracks (slot_tracker.py). See
docs/slot_detection.md.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .sensors.base import SensorSuite
from .sensors.lidar2d import LidarMeasurement
from .sensors.line_camera import LineCameraMeasurement
from .sensors.ultrasonic import UltrasonicMeasurement
from .slot_tracker import SlotDetection, SlotTrack, SlotTracker, corner_covariance
from .supervisor import SlotCandidate


class SlotDetector(ABC):
    """Each tick, call observe() with the tick's sensor measurements (if any), then update() with
    the car's pose (anything with x, y, yaw): the true one, or the estimate if uses_estimated_pose.
    Slot corners come back in that same frame."""
    uses_estimated_pose = False

    def __init__(self) -> None:
        self._slots: Dict[int, SlotCandidate] = {}   # insertion order = order first seen

    def observe(self, measurements: Sequence) -> None:
        """This tick's sensor measurements. Detectors that read the true lot ignore them."""

    def set_selected(self, slot_id: Optional[int]) -> None:
        """The slot the car is committed to, or None. Detectors that refine it can use this."""

    def confirm_times(self) -> List[float]:
        """For tracking detectors: seconds from first detection to confirmation, per confirmed slot."""
        return []

    @abstractmethod
    def update(self, t: float, pose) -> Tuple[SlotCandidate, ...]:
        """Take in this tick's data and return every slot currently on offer."""

    @property
    def slots(self) -> Tuple[SlotCandidate, ...]:
        return tuple(self._slots.values())


class IdealSlotDetector(SlotDetector):
    """Reads the true lot. A free stall becomes a candidate once its whole opening has been
    within detection_range of the car's side and the rear axle is at least pass_margin past
    the far end of the opening. Occupied stalls are never reported. No occlusion or noise."""

    def __init__(self, stalls: Iterable, vehicle, cfg: dict):
        super().__init__()
        self._waiting = [s for s in stalls if not s.occupied]
        self._in_range = set()
        self._range, self._margin = cfg["detection_range"], cfg["pass_margin"]
        self._rear = -vehicle.rear_overhang
        self._front = vehicle.wheelbase + vehicle.front_overhang
        self._half_width = vehicle.width / 2.0

    def update(self, t: float, pose) -> Tuple[SlotCandidate, ...]:
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        still_waiting = []
        for stall in self._waiting:
            opening = [(c * (px - pose.x) + s * (py - pose.y), -s * (px - pose.x) + c * (py - pose.y))
                       for px, py in stall.corners[:2]]
            if all(self._distance_to_side(*p) <= self._range for p in opening):
                self._in_range.add(stall.slot_id)
            if stall.slot_id in self._in_range and max(p[0] for p in opening) <= -self._margin:
                self._slots[stall.slot_id] = SlotCandidate(
                    slot_id=stall.slot_id, confirmed=True, corners=stall.corners,
                    entry_side=stall.entry_side, slot_type=stall.slot_type, occupied=False,
                    confidence=1.0, first_seen_t=t)
            else:
                still_waiting.append(stall)
        self._waiting = still_waiting
        return self.slots

    def _distance_to_side(self, lx: float, ly: float) -> float:
        """Distance from a point in the car's frame to the body side nearest it."""
        along = max(self._rear - lx, 0.0, lx - self._front)
        return math.hypot(along, abs(ly) - self._half_width)


# ---------- merging camera segments into lines ----------
class MergedLine:
    """Camera segments that lie along one painted line, with a total-least-squares fit."""

    def __init__(self, a: np.ndarray, b: np.ndarray, usable_a: bool, usable_b: bool):
        self._starts: List[np.ndarray] = []
        self._ends: List[np.ndarray] = []
        self._usable: List[Tuple[bool, bool]] = []
        self._n, self._sx, self._sy, self._sxx, self._sxy, self._syy = 0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.cx = self.cy = self.ux = self.uy = 0.0     # current fit, as floats for the merge loop
        self.add(a, b, usable_a, usable_b)

    def add(self, a: np.ndarray, b: np.ndarray, usable_a: bool, usable_b: bool) -> None:
        self._starts.append(a)
        self._ends.append(b)
        self._usable.append((usable_a, usable_b))
        for x, y in (a.tolist(), b.tolist()):
            self._n += 1
            self._sx += x
            self._sy += y
            self._sxx += x * x
            self._sxy += x * y
            self._syy += y * y
        n = self._n
        self.cx, self.cy = self._sx / n, self._sy / n
        angle = 0.5 * math.atan2(2.0 * (self._sxy - self._sx * self._sy / n),
                                 (self._sxx - self._sx * self._sx / n) - (self._syy - self._sy * self._sy / n))
        self.ux, self.uy = math.cos(angle), math.sin(angle)
        self._positions = None

    @property
    def n_segments(self) -> int:
        return len(self._starts)

    @property
    def center(self) -> np.ndarray:
        return np.array([self.cx, self.cy])

    @property
    def direction(self) -> np.ndarray:
        """Unit vector along the line; its sign has no meaning."""
        return np.array([self.ux, self.uy])

    def along(self, points: np.ndarray) -> np.ndarray:
        """Positions along the line, from its center."""
        return (points[..., 0] - self.cx) * self.ux + (points[..., 1] - self.cy) * self.uy

    def span(self) -> Tuple[float, float]:
        """Range of the segments along the line, ignoring stray ends."""
        if self._positions is None:
            self._positions = np.sort(self.along(np.array(self._starts + self._ends)))
        count = len(self._positions)
        return float(self._positions[int(0.05 * (count - 1))]), float(self._positions[int(math.ceil(0.95 * (count - 1)))])

    def _low_high(self) -> Tuple[float, float]:
        if self._positions is None:
            self._positions = np.sort(self.along(np.array(self._starts + self._ends)))
        return float(self._positions[0]), float(self._positions[-1])

    def distance_to(self, point: np.ndarray) -> float:
        return abs((point[0] - self.cx) * -self.uy + (point[1] - self.cy) * self.ux)

    def real_end(self, toward: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """The line's end on the side toward a point: (unit vector along the line toward it, its position
        from the center), or a None position if no usable end was seen. Ends cut by a camera footprint
        or the ego body don't count."""
        center, u = self.center, self.direction
        unit = u if (toward - center) @ u >= 0.0 else -u
        starts = (np.array(self._starts) - center) @ unit
        ends = (np.array(self._ends) - center) @ unit
        usable = np.array(self._usable)
        use_start = starts >= ends
        near = np.where(use_start, starts, ends)
        samples = near[np.where(use_start, usable[:, 0], usable[:, 1])]
        if len(samples) == 0:
            return unit, None
        # The real end is the furthest one seen; pieces after a paint gap stop short of it. A high
        # percentile keeps single noisy ends out, and the median of what's near it averages the noise.
        top = np.percentile(samples, 90)
        return unit, float(np.median(samples[samples >= top - 0.15]))


def merge_collinear(starts: np.ndarray, ends: np.ndarray, usable: np.ndarray, cfg: dict) -> List[MergedLine]:
    """Greedy grouping, longest segments first, so short noisy ones join well-founded lines."""
    lengths = np.hypot(*(ends - starts).T)
    min_dot, max_offset, max_gap = math.cos(math.radians(cfg["merge_angle_deg"])), cfg["merge_offset"], cfg["merge_gap"]
    lines: List[MergedLine] = []
    start_xy, end_xy = starts.tolist(), ends.tolist()
    for k in np.argsort(-lengths).tolist():
        (ax, ay), (bx, by) = start_xy[k], end_xy[k]
        size = max(float(lengths[k]), 1e-9)
        hx, hy = (bx - ax) / size, (by - ay) / size
        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
        for line in lines:
            ux, uy = line.ux, line.uy
            if abs(hx * ux + hy * uy) < min_dot or abs((mx - line.cx) * -uy + (my - line.cy) * ux) > max_offset:
                continue
            low, high = line._low_high()
            sa = (ax - line.cx) * ux + (ay - line.cy) * uy
            sb = (bx - line.cx) * ux + (by - line.cy) * uy
            if min(sa, sb) <= high + max_gap and max(sa, sb) >= low - max_gap:
                line.add(starts[k], ends[k], bool(usable[k, 0]), bool(usable[k, 1]))
                break
        else:
            lines.append(MergedLine(starts[k], ends[k], bool(usable[k, 0]), bool(usable[k, 1])))
    return lines


# ---------- sources of detections ----------
class _LineSource:
    """Finds slots between stall dividers in the last window of camera segments."""

    def __init__(self, suite: SensorSuite, vehicle, cfg: dict):
        self.cfg = cfg["lines"]
        self.window_s = cfg["window_s"]
        self.suite, self.vehicle = suite, vehicle
        self._window = deque()  # (t, starts (K, 2), ends (K, 2), usable (K, 2))
        self.lines: List[MergedLine] = []

    def add(self, m: LineCameraMeasurement, pose) -> None:
        if len(m.segments) == 0:
            return
        points = m.segments.reshape(-1, 2)
        usable = ~self._cut_by_view(points, self.suite.by_name[m.sensor].footprint).reshape(-1, 2)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        odom = np.column_stack([pose.x + c * points[:, 0] - s * points[:, 1],
                                pose.y + s * points[:, 0] + c * points[:, 1]]).reshape(-1, 2, 2)
        self._window.append((m.t, odom[:, 0], odom[:, 1], usable))

    def _cut_by_view(self, points: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        """Whether each point lies at the edge of the camera's footprint or of the ego body."""
        margin = self.cfg["end_clip_margin"]
        edges = np.roll(footprint, -1, axis=0) - footprint
        normals = np.column_stack([-edges[:, 1], edges[:, 0]])
        normals = normals / np.hypot(*normals.T)[:, None]
        inside = ((points[:, None, :] - footprint[None]) * normals[None]).sum(axis=2)
        at_footprint = np.any(inside <= margin, axis=1)
        v = self.vehicle
        gap_x = np.maximum(-v.rear_overhang - points[:, 0], points[:, 0] - (v.wheelbase + v.front_overhang))
        gap_y = np.abs(points[:, 1]) - v.width / 2.0
        return at_footprint | (np.maximum(gap_x, gap_y) <= margin)

    def detect(self, t: float, pose) -> List[SlotDetection]:
        while self._window and t - self._window[0][0] > self.window_s:
            self._window.popleft()
        self.lines = []
        if self._window:
            starts = np.concatenate([w[1] for w in self._window])
            ends = np.concatenate([w[2] for w in self._window])
            usable = np.concatenate([w[3] for w in self._window])
            self.lines = [ln for ln in merge_collinear(starts, ends, usable, self.cfg)
                          if ln.n_segments >= self.cfg["min_segments"]]
        car = np.array([pose.x, pose.y])
        travel = np.array([math.cos(pose.yaw), math.sin(pose.yaw)])
        dividers = self._dividers(car, travel)
        detections = []
        for side in (-1.0, 1.0):
            row = sorted((d for d in dividers if d["side"] == side), key=lambda d: d["along"])
            for first, second in zip(row, row[1:]):
                detection = self._fit_slot(first, second, travel)
                if detection is not None:
                    detections.append(detection)
        return detections

    def _dividers(self, car: np.ndarray, travel: np.ndarray) -> List[dict]:
        """Lines roughly square to the direction of travel, with a real end on the car's side."""
        max_along = math.sin(math.radians(self.cfg["max_divider_skew_deg"]))
        out = []
        for line in self.lines:
            lo, hi = line.span()
            if abs(line.direction @ travel) > max_along or hi - lo < self.cfg["min_divider_length"]:
                continue
            unit, end = line.real_end(car)
            if end is None:
                continue
            entry, at_junction = self._junction(line.center + end * unit, line, unit)
            offset = entry - car
            cross = travel[0] * offset[1] - travel[1] * offset[0]
            out.append({"inward": -unit, "entry": entry, "junction": at_junction, "side": math.copysign(1.0, cross),
                        "along": float(offset @ travel)})
        return out

    def _junction(self, end: np.ndarray, divider: MergedLine, unit: np.ndarray) -> Tuple[np.ndarray, bool]:
        """The divider's aisle end, moved to where a crossing line (tick, corner) meets it, if there is one."""
        best, best_distance = None, self.cfg["junction_radius"]
        max_dot = math.sin(math.radians(self.cfg["junction_angle_deg"]))
        center = divider.center
        for line in self.lines:
            u = line.direction
            if line is divider or abs(u @ unit) > max_dot:
                continue
            distance = line.distance_to(end)
            lo, hi = line.span()
            along = line.along(end[None])[0]
            if distance > best_distance or along < lo - best_distance or along > hi + best_distance:
                continue
            normal = np.array([-u[1], u[0]])
            best = center + ((line.center - center) @ normal / (unit @ normal)) * unit
            best_distance = distance
        return (best, True) if best is not None else (end, False)

    def _fit_slot(self, first: dict, second: dict, travel: np.ndarray) -> Optional[SlotDetection]:
        cfg = self.cfg
        if first["inward"] @ second["inward"] < math.cos(math.radians(cfg["parallel_deg"])):
            return None
        inward = first["inward"] + second["inward"]
        inward = inward / np.hypot(*inward)
        across = np.array([-inward[1], inward[0]])
        if across @ travel < 0.0:
            across = -across
        e1, e2 = first["entry"], second["entry"]
        width = float((e2 - e1) @ across)
        if not cfg["spacing"][0] <= width <= cfg["spacing"][1] or abs((e2 - e1) @ inward) > cfg["end_align"]:
            return None
        mouth = (e1 + e2) / 2.0
        e1, e2 = mouth - across * width / 2.0, mouth + across * width / 2.0
        depth, depth_seen = self._depth(mouth, inward, width)
        corners = np.array([e1, e2, e2 + inward * depth, e1 + inward * depth])
        sigma_along = cfg["sigma_junction"] if first["junction"] and second["junction"] else cfg["sigma_end"]
        entry = corner_covariance(inward, sigma_along, cfg["sigma_lateral"])
        rear = corner_covariance(inward, cfg["sigma_rear_seen"] if depth_seen else cfg["sigma_rear_assumed"],
                                 cfg["sigma_rear_lateral"])
        return SlotDetection(corners, np.array([entry, entry, rear, rear]), "lines")

    def _depth(self, mouth: np.ndarray, inward: np.ndarray, width: float) -> Tuple[float, bool]:
        """Distance to a rear line across the slot, or the standard depth."""
        max_dot = math.sin(math.radians(self.cfg["junction_angle_deg"]))
        lo_depth, hi_depth = self.cfg["depth_range"]
        for line in self.lines:
            if abs(line.direction @ inward) > max_dot:
                continue
            depth = float((line.center - mouth) @ inward)
            if not lo_depth <= depth <= hi_depth:
                continue
            lo, hi = line.span()
            if lo - width / 2.0 <= line.along((mouth + inward * depth)[None])[0] <= hi + width / 2.0:
                return depth, True
        return self.cfg["standard_depth"], False


class _SpaceSource:
    """Finds the space between parked cars from the side-looking ultrasonics during search. The
    opening is placed from the neighbors' bumpers. The sensors can't reach the back of a stall, so if
    the gap reads clear as far as they can see, its depth is the configured one. A gap needs a car at
    each end, so space at the very start or end of a row isn't found."""

    def __init__(self, suite: SensorSuite, cfg: dict):
        self.cfg = cfg["space"]
        self.suite = suite
        self.max_cone = cfg["occupancy"]["max_cone_deg"]
        self._emit = cfg["tracker"]["confirm_detections"] + 1
        self._sides: Dict[str, dict] = {}
        self._pending: List[SlotDetection] = []

    def add(self, m: UltrasonicMeasurement, pose) -> None:
        sensor = self.suite.by_name[m.sensor]
        if abs(math.sin(sensor.mount.yaw)) < 0.9 or sensor.cone_deg > self.max_cone:
            return
        x, y, yaw = sensor.mount.in_map(pose)
        here = np.array([x, y])
        state = self._sides.setdefault(m.sensor, {"tail": deque(maxlen=self.cfg["tail_readings"]), "open_from": None,
                                                  "gap": None, "emitted": 0})
        if m.distance is None:
            if state["open_from"] is None and state["gap"] is None and len(state["tail"]) >= self.cfg["min_wall_readings"]:
                state["open_from"] = state["tail"][-1]
            return
        travel = np.array([math.cos(pose.yaw), math.sin(pose.yaw)])
        if state["open_from"] is not None:
            start = state["open_from"]
            if (here - start[0]) @ travel >= self.cfg["min_open_length"]:     # else a dropout, not a gap
                state["gap"] = {"start": start, "end": (here, m.distance), "before": [r for _, r in state["tail"]],
                                "after": [], "travel": travel, "axis": np.array([math.cos(yaw), math.sin(yaw)]),
                                "half": math.radians(sensor.cone_deg) / 2.0}
                state["emitted"] = 0
            state["open_from"] = None
        state["tail"].append((here, m.distance))
        gap = state["gap"]
        if gap is not None:     # each reading on the far car is one more detection of the gap
            gap["after"].append(m.distance)
            self._pending.extend(self._fit_gap(gap))
            state["emitted"] += 1
            if state["emitted"] >= self._emit:
                state["gap"] = None

    def _fit_gap(self, gap: dict) -> List[SlotDetection]:
        cfg, travel, axis = self.cfg, gap["travel"], gap["axis"]
        (p1, r1), (p2, r2) = gap["start"], gap["end"]
        # A cone keeps touching a car until it has moved r sin(half) past the car's edge
        low = float(p1 @ travel) - r1 * math.sin(gap["half"]) + cfg["neighbor_clearance"]
        high = float(p2 @ travel) + r2 * math.sin(gap["half"]) - cfg["neighbor_clearance"]
        inner = high - low
        if inner < cfg["min_width"]:
            return []
        count = max(int(inner // cfg["min_width"]), 1)
        pitch = inner / count
        width = min(pitch, cfg["stall_width"])
        entry_line = axis * (float(p1 @ axis) + float(np.percentile(gap["before"] + gap["after"], 25)) - cfg["bumper_setback"])
        cov = corner_covariance(travel, cfg["sigma_width"], cfg["sigma_depth"])
        out = []
        for k in range(count):
            middle = low + (k + 0.5) * pitch
            e1 = entry_line + travel * (middle - width / 2.0)
            e2 = entry_line + travel * (middle + width / 2.0)
            out.append(SlotDetection(np.array([e1, e2, e2 + axis * cfg["depth"], e1 + axis * cfg["depth"]]),
                                     np.array([cov] * 4), "space"))
        return out

    def detect(self, t: float, pose) -> List[SlotDetection]:
        out, self._pending = self._pending, []
        return out


# ---------- the tracking detectors ----------
class TrackingSlotDetector(SlotDetector):
    """Slots from any mix of detection sources, tracked in the odometry frame. A slot is offered once
    its track is confirmed, the car has passed it, and the ultrasonics (and lidar) have read across it
    and found nothing inside. The selected slot keeps being refined from the lines around it."""
    uses_estimated_pose = True

    def __init__(self, sources: Sequence[str], suite: SensorSuite, vehicle, cfg: dict):
        super().__init__()
        self.suite, self.cfg = suite, cfg
        self.tracker = SlotTracker(cfg["tracker"])
        self._lines = _LineSource(suite, vehicle, cfg) if "lines" in sources else None
        self._space = _SpaceSource(suite, cfg) if "space" in sources else None
        self._pending: List = []
        self._next_fit = 0.0
        self._sonar: deque = deque()    # (t, x, y, axis_x, axis_y, range or nan, half cone, max range)
        self._lidar: deque = deque(maxlen=cfg["occupancy"]["lidar_scans"])
        self._selected: Optional[SlotTrack] = None
        self._driven = 0.0              # m along the estimate so far
        self._last_here: Optional[np.ndarray] = None
        self._seen_at: dict = {}        # track id: (its last update, distance driven then)

    def observe(self, measurements: Sequence) -> None:
        self._pending.extend(measurements)

    def set_selected(self, slot_id: Optional[int]) -> None:
        self._selected = next((tr for tr in self.tracker.tracks if tr.slot_id == slot_id), None) if slot_id is not None else None

    def confirm_times(self) -> List[float]:
        return [tr.confirmed_t - tr.first_seen_t for tr in self.tracker.tracks if tr.confirmed]

    def update(self, t: float, pose) -> Tuple[SlotCandidate, ...]:
        for m in self._pending:
            self._ingest(m, pose)
        self._pending = []
        if t + 1e-9 >= self._next_fit:
            self._next_fit = t + self.cfg["fit_period_s"]
            self._fit(t, pose)
        return self.slots

    def _ingest(self, m, pose) -> None:
        if isinstance(m, LineCameraMeasurement):
            if self._lines is not None:
                self._lines.add(m, pose)
        elif isinstance(m, UltrasonicMeasurement):
            sensor = self.suite.by_name[m.sensor]
            if sensor.cone_deg <= self.cfg["occupancy"]["max_cone_deg"]:
                x, y, yaw = sensor.mount.in_map(pose)
                self._sonar.append((m.t, x, y, math.cos(yaw), math.sin(yaw), math.nan if m.distance is None else m.distance,
                                    math.radians(sensor.cone_deg) / 2.0, sensor.range_max))
                while m.t - self._sonar[0][0] > self.cfg["occupancy"]["history_s"]:
                    self._sonar.popleft()
            if self._space is not None and self._selected is None:
                self._space.add(m, pose)
        elif isinstance(m, LidarMeasurement):
            points = self.suite.by_name[m.sensor].mount.to_vehicle(m.points())
            c, s = math.cos(pose.yaw), math.sin(pose.yaw)
            self._lidar.append(np.column_stack([pose.x + c * points[:, 0] - s * points[:, 1],
                                                pose.y + s * points[:, 0] + c * points[:, 1]]))

    def _fit(self, t: float, pose) -> None:
        detections: List[SlotDetection] = []
        for source in (self._lines, self._space):
            if source is not None:
                detections.extend(source.detect(t, pose))
        if self._selected is None:
            self.tracker.step(t, detections)
            self._refresh_offers(t, pose)
        else:
            self._refine(self._selected, t)
            self._publish(self._selected, t)

    # ---------- offering ----------
    def _refresh_offers(self, t: float, pose) -> None:
        sonar = np.array(self._sonar) if self._sonar else np.zeros((0, 8))
        lidar = np.concatenate(self._lidar) if self._lidar else np.zeros((0, 2))
        travel = np.array([math.cos(pose.yaw), math.sin(pose.yaw)])
        here = np.array([pose.x, pose.y])
        occ = self.cfg["occupancy"]
        if self._last_here is not None:
            self._driven += float(np.hypot(*(here - self._last_here)))
        self._last_here = here
        for track in self.tracker.tracks:
            if not track.confirmed:
                continue
            track.passed = track.passed or float(((track.corners[:2] - here) @ travel).max()) <= -self.cfg["pass_margin"]
            track.readings, track.returns_inside = self._occupancy(track, sonar, lidar)
            if track.readings >= occ["min_readings"]:   # with too few readings, what was last found stands: a parked car hasn't forgotten what it saw
                track.free = track.returns_inside < occ["min_hits"]
            free = bool(track.free)
            in_range = self._in_range(track, here)
            if track.passed and free and in_range:
                self._publish(track, t)
            elif not free or not in_range:
                self._slots.pop(track.slot_id, None)

    def _in_range(self, track: SlotTrack, here: np.ndarray) -> bool:
        """Whether a slot can still be offered: near enough, and not driven too far since last seen. Position error grows with distance driven, not
        with time, so a stopped car doesn't lose what it has in view. Near again, or seen again, it comes back with its old id."""
        seen = self._seen_at.get(track.track_id)
        if seen is None or track.last_update_t > seen[0]:
            seen = self._seen_at[track.track_id] = (track.last_update_t, self._driven)
        return (float(np.hypot(*(track.center - here))) <= self.cfg["offer_max_distance_m"]
                and self._driven - seen[1] <= self.cfg["offer_max_driven_m"])

    def _occupancy(self, track: SlotTrack, sonar: np.ndarray, lidar: np.ndarray) -> Tuple[int, int]:
        """(readings across the slot, returns inside it). An ultrasonic can't tell where in its cone a
        return came from, so it is placed on the axis, and the edges of the slot, where cone spread
        could have picked up a car next door, aren't used."""
        occ = self.cfg["occupancy"]
        entry, across = track.corners[0], track.corners[1] - track.corners[0]
        width = float(np.hypot(*across))
        across = across / width
        inward = track.corners[2:].mean(axis=0) - track.corners[:2].mean(axis=0)
        depth = float(np.hypot(*inward))
        inward = inward / depth
        readings = hits = 0
        rows = sonar[sonar[:, 3:5] @ inward >= math.cos(math.radians(occ["axis_tolerance_deg"]))] if len(sonar) else sonar
        if len(rows):
            rel = rows[:, 1:3] - entry
            to_entry = -(rel @ inward)
            spread = occ["margin"] + np.maximum(to_entry * np.tan(rows[:, 6]) - occ["neighbor_gap"], 0.0)
            lateral = rel @ across
            readings = int(np.count_nonzero((to_entry > 0.0) & (to_entry <= rows[:, 7])
                                            & (lateral >= spread) & (lateral <= width - spread)))
            rows = rows[np.isfinite(rows[:, 5])]
            tip = rows[:, 1:3] + rows[:, 3:5] * rows[:, 5, None] - entry
            spread = occ["margin"] + np.maximum(rows[:, 5] * np.tan(rows[:, 6]) - occ["neighbor_gap"], 0.0)
            x, y = tip @ across, tip @ inward
            hits = int(np.count_nonzero((x >= spread) & (x <= width - spread) & (y >= occ["entry_margin"])
                                        & (y <= depth - occ["margin"])))
        if len(lidar):
            rel = lidar - entry
            x, y = rel @ across, rel @ inward
            hits += int(np.count_nonzero((x >= occ["margin"]) & (x <= width - occ["margin"])
                                         & (y >= occ["entry_margin"]) & (y <= depth - occ["margin"])))
        return readings, hits

    def _publish(self, track: SlotTrack, t: float) -> None:
        known = self._slots.get(track.slot_id)
        faces_up = (track.corners[:2].mean(axis=0) - track.corners[2:].mean(axis=0))[1] > 0.0
        confidence = self.cfg["confidence_lines"] if track.source == "lines" else self.cfg["confidence_space"]
        self._slots[track.slot_id] = SlotCandidate(
            slot_id=track.slot_id, confirmed=True, corners=tuple(map(tuple, track.corners.tolist())),
            entry_side="+y" if faces_up else "-y", slot_type="perpendicular", occupied=False, confidence=confidence,
            first_seen_t=known.first_seen_t if known is not None else t)

    # ---------- refinement during the maneuver ----------
    def _refine(self, track: SlotTrack, t: float) -> None:
        """Correct the selected slot's corners from the lines now in view, one slot axis at a time. A
        divider corrects the corners across it, but only near the stretch of it in view: a short line's
        small tilt would become a large error several meters away. A tick or rear line corrects the
        corners along the slot. With one divider in view, the other side is placed one slot width away."""
        if self._lines is None:
            return
        cfg = self.cfg["refinement"]
        self.tracker.predict(track, t)
        origin = track.corners[0].copy()
        across = track.corners[1] - origin
        width = float(np.hypot(*across))
        across = across / width
        inward = track.corners[2:].mean(axis=0) - track.corners[:2].mean(axis=0)
        slot_depth = float(np.hypot(*inward))
        inward = inward / slot_depth
        parallel, square = math.cos(math.radians(cfg["line_angle_deg"])), math.sin(math.radians(cfg["line_angle_deg"]))
        radius, reach, sigma = cfg["line_radius"], cfg["line_reach"], cfg["sigma_line"]
        sides_seen = set()
        for line in self._lines.lines:
            lo, hi = line.span()
            if hi - lo < cfg["line_min_length"]:
                continue
            ends = line.center + np.outer((lo, hi), line.direction) - origin
            lateral, depth = ends @ across, ends @ inward
            if depth[0] > depth[1]:
                lateral, depth = lateral[::-1], depth[::-1]
            dot = abs(line.direction @ inward)
            if dot >= parallel:
                for side, offset in (((0, 3), 0.0), ((1, 2), width)):
                    if abs(lateral.mean() - offset) > radius or depth[1] < -reach or depth[0] > slot_depth + reach:
                        continue
                    sides_seen.add(side)
                    for corner, corner_depth in zip(side, (0.0, slot_depth)):
                        if depth[0] - reach <= corner_depth <= depth[1] + reach:
                            at = float(np.interp(corner_depth, depth, lateral))
                            self.tracker.update_along(track, corner, across, origin @ across + at, sigma)
            elif dot <= square and lateral.max() >= -radius and lateral.min() <= width + radius:
                line_depth = float(depth.mean())
                # The rear edge may be only an assumption, so any line in the rear-line range will do
                rear_low, rear_high = self.cfg["lines"]["depth_range"]
                corners = (0, 1) if abs(line_depth) <= radius else ((2, 3) if rear_low <= line_depth <= rear_high else ())
                for corner in corners:
                    self.tracker.update_along(track, corner, inward, origin @ inward + line_depth, sigma)
        if len(sides_seen) == 1:
            pairs, step = (((0, 1), (3, 2)), width) if (0, 3) in sides_seen else (((1, 0), (2, 3)), -width)
            for seen, other in pairs:
                self.tracker.update_along(track, other, across, track.corners[seen] @ across + step, cfg["sigma_other_side"])


class LineSlotDetector(TrackingSlotDetector):
    """Slots between painted stall dividers, with ultrasonics (and lidar) for occupancy."""

    def __init__(self, suite: SensorSuite, vehicle, cfg: dict):
        super().__init__(("lines",), suite, vehicle, cfg)


class SpaceSlotDetector(TrackingSlotDetector):
    """Slots in the space between parked cars, from the side ultrasonics, at lower confidence."""

    def __init__(self, suite: SensorSuite, vehicle, cfg: dict):
        super().__init__(("space",), suite, vehicle, cfg)


class FusedSlotDetector(TrackingSlotDetector):
    """Lines first; the space between cars where the lines don't give a slot."""

    def __init__(self, suite: SensorSuite, vehicle, cfg: dict):
        super().__init__(("lines", "space"), suite, vehicle, cfg)


def make_detector(name: str, scenario, suite: Optional[SensorSuite] = None) -> SlotDetector:
    """The detector for slot_detection.detector in the config. Only "ideal" needs no sensor suite."""
    cfg = scenario.cfg["slot_detection"]
    if name == "ideal":
        return IdealSlotDetector(scenario.stalls, scenario.vehicle, cfg)
    kinds = {"lines": LineSlotDetector, "space": SpaceSlotDetector, "fused": FusedSlotDetector}
    if name not in kinds:
        raise ValueError(f"unknown slot detector {name!r}, expected ideal, lines, space or fused")
    return kinds[name](suite, scenario.vehicle, cfg)
