"""An obstacle map built only from what the car has seen, for the planner. See docs/sensed_map.md.

Everything is in the frame of the pose it is given: the estimated pose on the car. The map never sees a lot,
a true pose, or a sensor simulation; a ROS node could feed it the same calls from its subscriptions.

Each cell holds log-odds of being occupied, and reads as one of:
  FREE      evidence says nothing is there
  OCCUPIED  evidence says something is
  UNKNOWN   no evidence either way. The planner treats it as blocked
  ASSUMED   unknown, but inside the slot being parked in. The planner may enter it: this is the one place
            free space is assumed rather than seen, because the sensors can't reach the back of a stall

Evidence:
  ultrasonics   inside the cone, short of the return: free, and any return placed there earlier is wiped, since
                the nearest thing in the cone is further out. No return at all may be a dropout, so it counts only
                once the sensor has read empty empty_readings times running, and then only against what is there. A return only says something is somewhere on the arc, so it is
                put on the arc cells not already known free. Only if the whole arc was known free must the
                object be new, and then every arc cell loses its free evidence at once, so the planner
                stays out even if the car never moves to look again. A return whose arc reaches past the selected
                slot's lines is ambiguous: it is most likely a neighbor at the edge of the cone, so it adds nothing to
                the slot or its margin, only to what's outside them and to cells already occupied
  line cameras  seeing paint means seeing the ground, so the triangle from the camera to a segment gets weak
                free evidence, too weak to clear a return. False segments last one frame, so stay unknown
  the car       the ground under the body is free
  tracked slots the slot in use is the ASSUMED area, with a margin
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

UNKNOWN, FREE, OCCUPIED, ASSUMED = 0, 1, 2, 3
_LOG_ODDS_ROUNDING = 1e-3   # the grid is float32, so a value set to a limit can read just under it


@dataclass(frozen=True)
class RangeSensor:
    """An ultrasonic's mount in the vehicle frame (rear-axle origin, x forward, y left) and its limits."""
    name: str
    x: float
    y: float
    yaw: float          # rad
    half_cone: float    # rad
    range_min: float    # m
    range_max: float    # m


@dataclass(frozen=True)
class BodyOutline:
    """The car's body in the vehicle frame."""
    rear: float         # m, negative: behind the rear axle
    front: float        # m
    half_width: float   # m


@dataclass(frozen=True)
class SensedMapConfig:
    resolution_m: float
    half_extent_m: Tuple[float, float]
    planning_window_m: float
    hit: float
    free: float
    camera_free: float
    camera_floor: float
    log_min: float
    log_max: float
    free_below: float
    occupied_above: float
    hit_band_m: float
    free_margin_m: float
    min_move_m: float
    min_turn_deg: float
    reading_change_m: float
    camera_min_move_m: float
    assumed_slot_margin_m: float
    max_paint_segments: int
    empty_readings: int
    slot_mark_min_log_odds: float

    @classmethod
    def from_params(cls, params: dict) -> "SensedMapConfig":
        odds = params["log_odds"]
        return cls(params["resolution_m"], tuple(params["half_extent_m"]), params["planning_window_m"],
                   odds["hit"], odds["free"], odds["camera_free"], odds["camera_floor"], odds["min"], odds["max"],
                   params["free_below"], params["occupied_above"], params["hit_band_m"], params["free_margin_m"],
                   params["min_move_m"], params["min_turn_deg"], params["reading_change_m"], params["camera_min_move_m"],
                   params["assumed_slot_margin_m"], params["max_paint_segments"], params["empty_readings"], params["slot_mark_min_log_odds"])


@dataclass(frozen=True, eq=False)
class SensedMapFrame:
    """The map at one moment, cropped to where there is evidence. For display and tests."""
    t: float
    x0: float               # m, the outer edge of cell (0, 0)
    y0: float               # m
    resolution: float       # m
    cells: np.ndarray       # (nx, ny), UNKNOWN | FREE | OCCUPIED | ASSUMED
    paint: np.ndarray       # (K, 2, 2) line segments seen

    def runs(self, kind: int) -> List[Tuple[float, float, float]]:
        """(x start, x end, y) of each unbroken run of `kind` along x, through cell centers in y."""
        out = []
        for j in range(self.cells.shape[1]):
            row = np.concatenate([[0], (self.cells[:, j] == kind).astype(np.int8), [0]])
            changes = np.flatnonzero(np.diff(row))
            y = self.y0 + (j + 0.5) * self.resolution
            out.extend((self.x0 + a * self.resolution, self.x0 + b * self.resolution, y)
                       for a, b in zip(changes[::2].tolist(), changes[1::2].tolist()))
        return out


class SensedMap:
    """Call update_body() and the add_ methods with each tick's data and that tick's pose (anything with x, y, yaw)."""

    def __init__(self, cfg: SensedMapConfig, sensors: Sequence[RangeSensor], cameras: Dict[str, Tuple[float, float]],
                 body: BodyOutline):
        self.cfg = cfg
        self.sensors = {s.name: s for s in sensors}
        self.cameras = dict(cameras)    # name: (x, y) mount in the vehicle frame
        self.body = body
        self.resolution = cfg.resolution_m
        self._log: Optional[np.ndarray] = None   # allocated around the first pose
        self._x0 = self._y0 = 0.0
        self._xc = self._yc = np.zeros(0)
        self._seen: Optional[List[int]] = None   # i0, i1, j0, j1 of every cell touched
        self._last_range: Dict[str, Tuple[float, float, float, Optional[float]]] = {}
        self._empty_in_a_row: Dict[str, int] = {}
        self._last_camera: Dict[str, Tuple[float, float]] = {}
        self._last_body: Optional[Tuple[float, float, float]] = None
        self._assumed: Optional[Tuple[int, int, int, int, np.ndarray]] = None
        self._slot: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = None   # center, across, inward, half sizes
        self._paint: deque = deque(maxlen=cfg.max_paint_segments)

    # ---------- the grid ----------
    def _ensure_grid(self, pose) -> None:
        if self._log is not None:
            return
        hx, hy = self.cfg.half_extent_m
        res = self.resolution
        nx, ny = int(round(2.0 * hx / res)), int(round(2.0 * hy / res))
        self._x0, self._y0 = float(pose.x) - hx, float(pose.y) - hy
        self._xc = self._x0 + (np.arange(nx) + 0.5) * res
        self._yc = self._y0 + (np.arange(ny) + 0.5) * res
        self._log = np.zeros((nx, ny), dtype=np.float32)

    def _window(self, xmin: float, xmax: float, ymin: float, ymax: float) -> Optional[Tuple[int, int, int, int]]:
        """Indices of the cells overlapping a box, or None if it is off the grid. Anything off the grid is lost."""
        nx, ny = self._log.shape
        res = self.resolution
        i0, i1 = max(int((xmin - self._x0) / res), 0), min(int(math.ceil((xmax - self._x0) / res)), nx)
        j0, j1 = max(int((ymin - self._y0) / res), 0), min(int(math.ceil((ymax - self._y0) / res)), ny)
        return (i0, i1, j0, j1) if i0 < i1 and j0 < j1 else None

    def _touch(self, window: Tuple[int, int, int, int]) -> None:
        if self._seen is None:
            self._seen = list(window)
        else:
            seen = self._seen
            seen[0], seen[1] = min(seen[0], window[0]), max(seen[1], window[1])
            seen[2], seen[3] = min(seen[2], window[2]), max(seen[3], window[3])

    def _offsets(self, window: Tuple[int, int, int, int], x: float, y: float) -> Tuple[np.ndarray, np.ndarray]:
        i0, i1, j0, j1 = window
        return (self._xc[i0:i1] - x)[:, None], (self._yc[j0:j1] - y)[None, :]

    # ---------- evidence ----------
    def update_body(self, pose) -> None:
        """The ground under the car is free."""
        self._ensure_grid(pose)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        last = self._last_body
        if last is not None and not self._moved(last, pose.x, pose.y, pose.yaw):
            return
        self._last_body = (pose.x, pose.y, pose.yaw)
        b = self.body
        reach = math.hypot(max(abs(b.rear), abs(b.front)), b.half_width)
        window = self._window(pose.x - reach, pose.x + reach, pose.y - reach, pose.y + reach)
        if window is None:
            return
        dx, dy = self._offsets(window, pose.x, pose.y)
        along, across = dx * c + dy * s, -dx * s + dy * c
        under = (along >= b.rear) & (along <= b.front) & (np.abs(across) <= b.half_width)
        i0, i1, j0, j1 = window
        self._log[i0:i1, j0:j1][under] = self.cfg.log_min
        self._touch(window)

    def _moved(self, last: Tuple[float, float, float], x: float, y: float, yaw: float) -> bool:
        turned = abs(math.remainder(yaw - last[2], 2.0 * math.pi))
        return math.hypot(x - last[0], y - last[1]) >= self.cfg.min_move_m or turned >= math.radians(self.cfg.min_turn_deg)

    def add_range(self, name: str, distance: Optional[float], pose) -> None:
        """One ultrasonic reading; distance None = no return. A sensor standing still adds a reading only if
        it changed: repeating one reading would pile evidence on whatever arc it happens to draw."""
        self._ensure_grid(pose)
        sensor, cfg = self.sensors[name], self.cfg
        if distance is None:    # one empty reading may be a dropout, and would clear the cone right through a parked car
            self._empty_in_a_row[name] = self._empty_in_a_row.get(name, 0) + 1
            if self._empty_in_a_row[name] < cfg.empty_readings:
                return
        else:
            self._empty_in_a_row[name] = 0
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        ox, oy = pose.x + c * sensor.x - s * sensor.y, pose.y + s * sensor.x + c * sensor.y
        yaw = pose.yaw + sensor.yaw
        last = self._last_range.get(name)
        if last is not None and not self._moved(last[:3], ox, oy, yaw) and not _reading_changed(last[3], distance, cfg.reading_change_m):
            return
        self._last_range[name] = (ox, oy, yaw, distance)
        reach = sensor.range_max if distance is None else min(distance + cfg.hit_band_m, sensor.range_max)
        window = self._window(ox - reach, ox + reach, oy - reach, oy + reach)
        if window is None:
            return
        dx, dy = self._offsets(window, ox, oy)
        ranges = np.hypot(dx, dy)
        in_cone = dx * math.cos(yaw) + dy * math.sin(yaw) >= ranges * math.cos(sensor.half_cone)
        i0, i1, j0, j1 = window
        cells = self._log[i0:i1, j0:j1]
        top = sensor.range_max if distance is None else distance - cfg.free_margin_m
        carved = in_cone & (ranges >= sensor.range_min) & (ranges <= top)
        if distance is None:    # may be a dropout, so it only counts against what is there
            cells[carved] = np.maximum(cells[carved] + cfg.free, cfg.log_min)
        else:                   # the nearest thing in the cone is further out: earlier returns placed here were wrong
            cells[carved] = np.maximum(np.minimum(cells[carved], 0.0) + cfg.free, cfg.log_min)
        if distance is not None:
            arc = in_cone & (np.abs(ranges - distance) <= cfg.hit_band_m)
            values = cells[arc]
            if len(values):
                ambiguous = self._hits_if_past_the_slot(window, arc, cells)
                explaining = values >= cfg.free_below
                if ambiguous is not None:
                    cells[ambiguous] = np.minimum(cells[ambiguous] + cfg.hit, cfg.log_max)
                elif explaining.any():  # a return from somewhere not known to be free explains itself; the rest stay free
                    cells[arc] = np.where(explaining, np.minimum(values + cfg.hit, cfg.log_max), values)
                else:                   # all known free, so something new is there. Its free evidence is out of date
                    cells[arc] = cfg.hit
        self._touch(window)

    def _hits_if_past_the_slot(self, window: Tuple[int, int, int, int], arc: np.ndarray, cells: np.ndarray) -> Optional[np.ndarray]:
        """The cells a return may mark if its arc reaches past the selected slot's lines to somewhere an object could be,
        or None if it doesn't and the usual rule applies. Such a return is explained out there: the arc from a neighbor at
        the edge of a wide cone runs right across an empty stall, and marking it draws a wall across the slot. Test against
        the lines, not the margin: a neighbor stands just past the margin, so its arc's tip can stop short of it. Only a cell
        an ultrasonic has cleared rules out an object: a camera seeing paint past a cell doesn't.
        The cost: something real in the slot goes unmarked until the car is close enough for the arc to fit between the lines."""
        if self._slot is None:
            return None
        cfg = self.cfg
        center, across, inward, half_w, half_d = self._slot
        dx, dy = self._offsets(window, float(center[0]), float(center[1]))
        sideways, deep = np.abs(dx * across[0] + dy * across[1]), np.abs(dx * inward[0] + dy * inward[1])
        margin = cfg.assumed_slot_margin_m
        past_margin = ~((sideways <= half_w + margin) & (deep <= half_d + margin))
        if not (arc & ~past_margin).any():      # nowhere near the slot, so nothing to protect
            return None
        possible = cells >= cfg.camera_floor - _LOG_ODDS_ROUNDING
        if not (arc & possible & ~((sideways <= half_w) & (deep <= half_d))).any():
            return None
        return (arc & possible & past_margin) | (arc & (cells >= cfg.occupied_above))

    def add_lines(self, name: str, segments: np.ndarray, pose) -> None:
        """One camera frame's painted segments, (K, 2, 2) in the vehicle frame."""
        self._ensure_grid(pose)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        mount = self.cameras[name]
        ox, oy = pose.x + c * mount[0] - s * mount[1], pose.y + s * mount[0] + c * mount[1]
        last = self._last_camera.get(name)
        if last is not None and math.hypot(ox - last[0], oy - last[1]) < self.cfg.camera_min_move_m:
            return
        self._last_camera[name] = (ox, oy)
        if len(segments) == 0:
            return
        px = pose.x + c * segments[..., 0] - s * segments[..., 1]
        py = pose.y + s * segments[..., 0] + c * segments[..., 1]
        for k in range(len(segments)):
            a, b = (float(px[k, 0]), float(py[k, 0])), (float(px[k, 1]), float(py[k, 1]))
            self._paint.append((a, b))
            self._see_ground((ox, oy), a, b)

    def _see_ground(self, o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> None:
        cfg = self.cfg
        window = self._window(min(o[0], a[0], b[0]), max(o[0], a[0], b[0]), min(o[1], a[1], b[1]), max(o[1], a[1], b[1]))
        if window is None:
            return
        dx, dy = self._offsets(window, o[0], o[1])
        ax, ay, bx, by = a[0] - o[0], a[1] - o[1], b[0] - o[0], b[1] - o[1]
        area = ax * by - ay * bx
        if abs(area) < 1e-9:
            return
        u = (dx * by - dy * bx) / area
        v = (ax * dy - ay * dx) / area
        i0, i1, j0, j1 = window
        cells = self._log[i0:i1, j0:j1]
        seen = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (cells > cfg.camera_floor) & (cells < cfg.occupied_above)
        cells[seen] = np.maximum(cells[seen] + cfg.camera_free, cfg.camera_floor)
        self._touch(window)

    def assume_slot(self, corners: Sequence[Tuple[float, float]]) -> None:
        """The slot the car is going to, as the detector has it now. Replaces any earlier one."""
        points = np.array(corners, dtype=float)
        margin = self.cfg.assumed_slot_margin_m
        center = points.mean(axis=0)
        across = points[1] - points[0]
        width = float(np.hypot(*across))
        across = across / width
        inward = points[2:].mean(axis=0) - points[:2].mean(axis=0)
        depth = float(np.hypot(*inward))
        inward = inward / depth
        half_w, half_d = width / 2.0 + margin, depth / 2.0 + margin
        reach = math.hypot(half_w, half_d)
        window = self._window(center[0] - reach, center[0] + reach, center[1] - reach, center[1] + reach)
        if window is None:
            self._assumed = self._slot = None
            return
        dx, dy = self._offsets(window, float(center[0]), float(center[1]))
        inside = (np.abs(dx * across[0] + dy * across[1]) <= half_w) & (np.abs(dx * inward[0] + dy * inward[1]) <= half_d)
        self._assumed = (*window, inside)
        self._slot = (center, across, inward, width / 2.0, depth / 2.0)
        self._touch(window)

    # ---------- reading the map ----------
    def _classes(self, window: Tuple[int, int, int, int]) -> np.ndarray:
        i0, i1, j0, j1 = window
        cells = self._log[i0:i1, j0:j1]
        out = np.full(cells.shape, UNKNOWN, dtype=np.int8)
        out[cells < self.cfg.free_below] = FREE
        out[cells >= self.cfg.occupied_above] = OCCUPIED
        if self._assumed is not None:
            a0, a1, b0, b1, inside = self._assumed
            x0, x1, y0, y1 = max(i0, a0), min(i1, a1), max(j0, b0), min(j1, b1)
            if x0 < x1 and y0 < y1:
                part = out[x0 - i0:x1 - i0, y0 - j0:y1 - j0]
                here = inside[x0 - a0:x1 - a0, y0 - b0:y1 - b0]
                part[(part == UNKNOWN) & here] = ASSUMED
                # Two long-range arcs crossing make a mark too. The detector has read this slot as free, so take more than that to overrule it
                part[(part == OCCUPIED) & here & (self._log[x0:x1, y0:y1] < self.cfg.slot_mark_min_log_odds)] = ASSUMED
        return out

    def cell_at(self, x: float, y: float) -> int:
        """What the map says about a point. Off the grid, or before any data: UNKNOWN."""
        if self._log is None:
            return UNKNOWN
        i, j = int((x - self._x0) // self.resolution), int((y - self._y0) // self.resolution)
        if not (0 <= i < self._log.shape[0] and 0 <= j < self._log.shape[1]):
            return UNKNOWN
        return int(self._classes((i, i + 1, j, j + 1))[0, 0])

    def occupied_in_sweep(self, poses: np.ndarray, lateral_margin: float, end_margin: float, min_log_odds: Optional[float] = None) -> int:
        """Cells with at least min_log_odds (default: occupied) inside the car's body at any of these poses, (N, 3) x, y, yaw, plus
        margins. Unknown doesn't count: the plan was already kept out of what was unknown then, and most of the map stays unknown. Only
        what has been seen stops a car."""
        level = self.cfg.occupied_above if min_log_odds is None else min_log_odds
        if self._log is None or len(poses) == 0:
            return 0
        b, found = self.body, set()
        reach = math.hypot(max(abs(b.rear), abs(b.front)) + end_margin, b.half_width + lateral_margin)
        for x, y, yaw in poses:
            window = self._window(x - reach, x + reach, y - reach, y + reach)
            if window is None:
                continue
            i0, i1, j0, j1 = window
            occupied = self._log[i0:i1, j0:j1] >= level
            if not occupied.any():
                continue
            dx, dy = self._offsets(window, float(x), float(y))
            c, s = math.cos(yaw), math.sin(yaw)
            along, across = dx * c + dy * s, -dx * s + dy * c
            inside = (along >= b.rear - end_margin) & (along <= b.front + end_margin) & (np.abs(across) <= b.half_width + lateral_margin)
            a, k = np.nonzero(occupied & inside)
            found.update(zip((a + i0).tolist(), (k + j0).tolist()))
        return len(found)

    def distance_field(self, points: Sequence[Tuple[float, float]]) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Distance to the nearest blocked cell (OCCUPIED or UNKNOWN), on the part of the map within
        planning_window_m of `points`: (field, (xmin, xmax, ymin, ymax)). Cell (i, j) is at
        xmin + (i + 0.5) * resolution, as in Scenario. The window's edge is a wall."""
        w = self.cfg.planning_window_m
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        window = self._window(min(xs) - w, max(xs) + w, min(ys) - w, max(ys) + w)
        if window is None:
            raise ValueError("the planning window is off the sensed map")
        i0, i1, j0, j1 = window
        cells = self._classes(window)
        blocked = (cells == UNKNOWN) | (cells == OCCUPIED)
        blocked[0, :] = blocked[-1, :] = blocked[:, 0] = blocked[:, -1] = True
        res = self.resolution
        field = distance_transform_edt(~blocked) * res - res / 2.0
        return field, (self._x0 + i0 * res, self._x0 + i1 * res, self._y0 + j0 * res, self._y0 + j1 * res)

    def frame(self, t: float) -> Optional[SensedMapFrame]:
        """A copy of the map as it stands, or None before any data."""
        if self._seen is None:
            return None
        i0, i1, j0, j1 = window = tuple(self._seen)
        res = self.resolution
        return SensedMapFrame(t, self._x0 + i0 * res, self._y0 + j0 * res, res, self._classes(window),
                              np.array(self._paint, dtype=float).reshape(-1, 2, 2))

    def counts(self) -> Dict[str, int]:
        """Cells of each kind."""
        if self._seen is None:
            return {"free": 0, "occupied": 0, "assumed": 0}
        cells = self._classes(tuple(self._seen))
        return {"free": int(np.count_nonzero(cells == FREE)), "occupied": int(np.count_nonzero(cells == OCCUPIED)),
                "assumed": int(np.count_nonzero(cells == ASSUMED))}


def _reading_changed(before: Optional[float], now: Optional[float], tolerance: float) -> bool:
    if before is None or now is None:
        return (before is None) != (now is None)
    return abs(before - now) >= tolerance
