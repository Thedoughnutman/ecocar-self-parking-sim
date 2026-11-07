"""Perpendicular parking lot scenario, obstacle map and collision checks.

Frame: x runs along the aisle, y points from the target row across the aisle.
The target row's stalls span y in [0, depth]; y = 0 is the stall end line
(curb side) and y = depth is the aisle-side line.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import yaml
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt, map_coordinates

from .geometry import polygon_distance, rect, wrap_angle
from .vehicle import VehicleParams


def load_params(path: Optional[str] = None) -> dict:
    if path is None:
        path = str(Path(__file__).resolve().parents[1] / "config" / "park_params.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class Pose:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Stall:
    """One marked stall of an aisle lot. The first two corners are the opening (aisle side)."""
    slot_id: int
    row: int                    # 0 = y in [0, depth], 1 = across the aisle
    index: int                  # position along the row, 0 at the entry end
    corners: Tuple[Tuple[float, float], ...]
    occupied: bool
    entry_side: str             # map direction from the stall out to the aisle: "+y" or "-y"
    slot_type: str = "perpendicular"

    @property
    def center(self) -> Tuple[float, float]:
        return (sum(c[0] for c in self.corners) / 4.0, sum(c[1] for c in self.corners) / 4.0)


def goal_from_corners(corners: Sequence[Tuple[float, float]], vehicle: VehicleParams,
                      maneuver: str) -> Pose:
    """Rear-axle goal centering the body in a stall whose first two corners are the opening."""
    cx, cy = sum(c[0] for c in corners) / 4.0, sum(c[1] for c in corners) / 4.0
    out = math.atan2((corners[0][1] + corners[1][1]) / 2.0 - cy,
                     (corners[0][0] + corners[1][0]) / 2.0 - cx)
    yaw = out if maneuver == "reverse_in" else wrap_angle(out + math.pi)
    off = vehicle.center_offset()
    return Pose(cx - off * math.cos(yaw), cy - off * math.sin(yaw), yaw)


@dataclass(frozen=True, eq=False)
class LotObject:
    """A pillar or a cart. Always in Scenario.obstacles, so sensors and clearance checks see it; the planner
    sees it only if known_to_planner."""
    kind: str
    polygon: np.ndarray
    known_to_planner: bool


@dataclass
class Scenario:
    obstacles: List[np.ndarray]                 # convex polygons
    lines: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    target_stall: Optional[np.ndarray]          # stall rectangle polygon; None until one is chosen
    stall_center: Optional[Tuple[float, float]]
    goal: Optional[Pose]                        # rear-axle goal pose
    start: Pose
    bounds: Tuple[float, float, float, float]   # xmin, xmax, ymin, ymax
    cfg: dict
    vehicle: VehicleParams
    res: float = 0.05
    parked_cars: List[np.ndarray] = field(default_factory=list)   # subset of obstacles
    curbs: List[np.ndarray] = field(default_factory=list)         # subset of obstacles
    lot_objects: List[LotObject] = field(default_factory=list)    # their polygons are a subset of obstacles
    stalls: List[Stall] = field(default_factory=list)             # aisle lots only
    aisle_y: float = 0.0                        # aisle centerline, aisle lots only
    aisle_end_x: float = 0.0                    # far edge of the last stall, aisle lots only
    _dist: np.ndarray = field(default=None, repr=False)
    _probe: np.ndarray = field(default=None, repr=False)
    _boxes: tuple = field(default=None, repr=False)     # obstacle bounding boxes (mins, maxes)
    _sensor_cache: dict = field(default_factory=dict, repr=False)   # see sensors/scene.py

    # ---------- distance field ----------
    def build_distance_field(self) -> None:
        xmin, xmax, ymin, ymax = self.bounds
        nx = int(math.ceil((xmax - xmin) / self.res))
        ny = int(math.ceil((ymax - ymin) / self.res))
        xs = xmin + (np.arange(nx) + 0.5) * self.res
        ys = ymin + (np.arange(ny) + 0.5) * self.res
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        occ = np.zeros((nx, ny), dtype=bool)
        for poly in self.obstacles:     # test only the cells inside each polygon's bounding box
            i0 = max(int((poly[:, 0].min() - xmin) / self.res) - 1, 0)
            i1 = min(int((poly[:, 0].max() - xmin) / self.res) + 2, nx)
            j0 = max(int((poly[:, 1].min() - ymin) / self.res) - 1, 0)
            j1 = min(int((poly[:, 1].max() - ymin) / self.res) + 2, ny)
            if i0 >= i1 or j0 >= j1:
                continue
            box = np.stack([gx[i0:i1, j0:j1].ravel(), gy[i0:i1, j0:j1].ravel()], axis=1)
            occ[i0:i1, j0:j1] |= MplPath(poly).contains_points(box, radius=1e-9).reshape(i1 - i0, j1 - j0)
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True   # map border is a wall
        self._dist = distance_transform_edt(~occ) * self.res - self.res / 2.0
        self._probe = _body_probe(self.vehicle)

    def clearance_fast(self, x: np.ndarray, y: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        """Approximate body clearance for many poses (vectorized), meters."""
        x, y, yaw = np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(yaw)
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        px = x[:, None] + c * self._probe[:, 0] - s * self._probe[:, 1]
        py = y[:, None] + s * self._probe[:, 0] + c * self._probe[:, 1]
        xmin, _, ymin, _ = self.bounds
        gi = (px - xmin) / self.res - 0.5
        gj = (py - ymin) / self.res - 0.5
        d = map_coordinates(self._dist, [gi.ravel(), gj.ravel()], order=1,
                            mode="constant", cval=-1.0).reshape(px.shape)
        return d.min(axis=1)

    def collision_free(self, x, y, yaw, margin: float) -> bool:
        return bool(np.all(self.clearance_fast(x, y, yaw) >= margin))

    def clearance_exact(self, x: float, y: float, yaw: float) -> float:
        body = self.vehicle.footprint(x, y, yaw)
        if self._boxes is None:
            self._boxes = (np.array([ob.min(axis=0) for ob in self.obstacles]),
                           np.array([ob.max(axis=0) for ob in self.obstacles]))
        # Bounding-box gaps are lower bounds, so nearest-first can stop early with the exact minimum
        gap = np.maximum(np.maximum(self._boxes[0] - body.max(axis=0), body.min(axis=0) - self._boxes[1]), 0.0)
        bound = np.hypot(gap[:, 0], gap[:, 1])
        best = math.inf
        for i in np.argsort(bound):
            if bound[i] >= best:
                break
            best = min(best, polygon_distance(body, self.obstacles[i]))
        return best

    # ---------- aisle lots ----------
    def nearest_stall(self, x: float, y: float) -> Stall:
        return min(self.stalls, key=lambda s: math.hypot(s.center[0] - x, s.center[1] - y))

    def with_target(self, stall: Stall) -> "Scenario":
        """The same lot, sharing its distance field, with `stall` as the parking target."""
        return replace(self, target_stall=np.array(stall.corners), stall_center=stall.center,
                       goal=goal_from_corners(stall.corners, self.vehicle, self.cfg["scenario"]["maneuver"]))

    def planner_view(self) -> "Scenario":
        """The lot as the planner knows it: without any object not known_to_planner. The same lot if there
        are none. The distance field is built once per lot."""
        hidden = [o.polygon for o in self.lot_objects if not o.known_to_planner]
        if not hidden:
            return self
        view = self._sensor_cache.get("planner_view")
        if view is None:
            known = [ob for ob in self.obstacles if not any(ob is h for h in hidden)]
            view = replace(self, obstacles=known, lot_objects=[o for o in self.lot_objects if o.known_to_planner],
                           _dist=None, _boxes=None, _probe=None, _sensor_cache={})
            view.build_distance_field()
            self._sensor_cache["planner_view"] = view
        return replace(view, target_stall=self.target_stall, stall_center=self.stall_center, goal=self.goal)


def _body_probe(v: VehicleParams) -> np.ndarray:
    """Probe points covering the body: perimeter every 0.1 m + interior grid."""
    xr, xf = -v.rear_overhang, v.wheelbase + v.front_overhang
    hw = v.width / 2.0
    per = []
    for a, b in [((xr, -hw), (xf, -hw)), ((xf, -hw), (xf, hw)),
                 ((xf, hw), (xr, hw)), ((xr, hw), (xr, -hw))]:
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.1))
        t = np.linspace(0, 1, n, endpoint=False)
        per.append(np.stack([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])], 1))
    ix, iy = np.meshgrid(np.arange(xr + 0.5, xf, 0.5), np.arange(-hw + 0.5, hw, 0.5))
    inner = np.stack([ix.ravel(), iy.ravel()], 1)
    return np.vstack(per + [inner])


def scene_from_distance_field(dist: np.ndarray, bounds: Tuple[float, float, float, float], res: float, goal: Pose,
                              start: Pose, cfg: dict, vehicle: VehicleParams) -> Scenario:
    """A planning scene with no lot behind it: all it knows is a distance field. It has no obstacles, so
    only the planner's fast clearance checks mean anything on it."""
    return Scenario(obstacles=[], lines=[], target_stall=None, stall_center=None, goal=goal, start=start, bounds=bounds,
                    cfg=cfg, vehicle=vehicle, res=res, _dist=dist, _probe=_body_probe(vehicle))


def add_target_obstacles(sc: Scenario, stall: Stall, specs: Sequence[dict],
                         rng: np.random.Generator) -> Scenario:
    """The same lot with obstacles placed relative to `stall`, on an aisle lot. `specs` come from a scenario
    file's `obstacles:`; rng draws where a cart stands. Sizes are in the config's lot_objects section.

    neighbor_over_line: the stall next door holds a car over_line_m across the line into `stall`. It replaces
                        any car there. It is a parked car like the others, so the planner knows it.
    pillar:             outside `stall`, its face on the stall line, at the aisle-side corner.
    cart:               inside `stall`, where the parked car's body would go.
    A pillar or cart is unknown to the planner unless its spec says known_to_planner: true.

    side: far is the +x side of the stall, near the -x side; a neighbor at the end of a row goes on the other."""
    params, objects_cfg = sc.cfg, sc.cfg["lot_objects"]
    cars, stalls, objects = list(sc.parked_cars), list(sc.stalls), list(sc.lot_objects)
    (x_near, y_open), (x_far, _), (_, y_back), _ = stall.corners
    into_stall = math.copysign(1.0, y_back - y_open)
    for spec in specs:
        kind = spec["kind"]
        toward = 1.0 if spec.get("side", "far") == "far" else -1.0
        if kind == "neighbor_over_line":
            row = {st.index: st for st in stalls if st.row == stall.row}
            if stall.index + int(toward) not in row:
                toward = -toward
            neighbor = row[stall.index + int(toward)]
            xs, ys = [c[0] for c in neighbor.corners], [c[1] for c in neighbor.corners]
            cars = [c for c in cars if not (min(xs) <= c[:, 0].mean() <= max(xs) and min(ys) <= c[:, 1].mean() <= max(ys))]
            line_x = x_far if toward > 0 else x_near
            pc = params["scenario"]["parked_car"]
            cars.append(rect(line_x + toward * (pc["width"] / 2.0 - spec["over_line_m"]), (y_open + y_back) / 2.0,
                             math.pi / 2.0, pc["length"], pc["width"]))
            stalls[stalls.index(neighbor)] = replace(neighbor, occupied=True)
        elif kind == "pillar":
            cfg = objects_cfg["pillar"]
            line_x = x_far if toward > 0 else x_near
            objects.append(LotObject(kind, rect(line_x + toward * cfg["size_m"] / 2.0,
                                                y_open - into_stall * cfg["aisle_offset_m"], 0.0, cfg["size_m"], cfg["size_m"]),
                                     bool(spec.get("known_to_planner", False))))
        elif kind == "cart":
            cfg = objects_cfg["cart"]
            objects.append(LotObject(kind, rect((x_near + x_far) / 2.0 + rng.uniform(*cfg["across_range_m"]),
                                                y_open + into_stall * rng.uniform(*cfg["depth_range_m"]),
                                                rng.uniform(0.0, math.pi), cfg["length_m"], cfg["width_m"]),
                                     bool(spec.get("known_to_planner", False))))
        else:
            raise ValueError(f"unknown obstacle kind {kind!r}")
    s = replace(sc, obstacles=cars + list(sc.curbs) + [o.polygon for o in objects], parked_cars=cars, stalls=stalls,
                lot_objects=objects, _dist=None, _boxes=None, _probe=None, _sensor_cache={})
    s.build_distance_field()
    return s


def _parked_car(rng: np.random.Generator, pc: dict, cx: float, cy: float, yaw: float) -> np.ndarray:
    return rect(cx + rng.uniform(-1, 1) * pc["lateral_jitter"] * abs(math.sin(yaw)),
                cy + rng.uniform(-1, 1) * pc["depth_jitter"] * abs(math.sin(yaw)),
                yaw + math.radians(rng.uniform(-1, 1) * pc["yaw_jitter_deg"]),
                pc["length"], pc["width"])


def _row_markings(marks: dict, n: int, w: float, y_open: float, y_back: float) -> list:
    """Rear stall line, one per stall, and short ticks at the aisle end of each stall line."""
    lines = []
    if marks["rear_line"]:
        lines.extend(((i * w, y_back), ((i + 1) * w, y_back)) for i in range(n))
    if marks["aisle_ticks"]:
        half = marks["tick_length"] / 2.0
        lines.extend(((i * w - half, y_open), (i * w + half, y_open)) for i in range(n + 1))
    return lines


def make_perpendicular(vehicle: VehicleParams, params: dict,
                       rng: Optional[np.random.Generator] = None,
                       start: Optional[Pose] = None) -> Scenario:
    rng = rng if rng is not None else np.random.default_rng(0)
    sc, sr = params["scenario"], params["start_region"]
    w, d, aisle = sc["stall_width"], sc["stall_depth"], sc["aisle_width"]
    n, tgt = sc["n_stalls"], sc["target_stall"]
    pc = sc["parked_car"]

    obstacles, lines = [], []

    # Target row (y in [0, d]) and opposite row (y in [d+aisle, 2d+aisle])
    rows = [(0.0, True)]
    if sc["opposite_row"]:
        rows.append((d + aisle, False))
    for y0, is_target_row in rows:
        for i in range(n):
            x0 = i * w
            if sc["markings"]["stall_lines"]:
                lines.append(((x0, y0), (x0, y0 + d)))
                if i == n - 1:
                    lines.append(((x0 + w, y0), (x0 + w, y0 + d)))
            if is_target_row and i == tgt:
                continue
            occupied = (is_target_row and abs(i - tgt) == 1 and sc["neighbors_occupied"]) \
                or rng.random() < sc["occupancy_prob"]
            if occupied:
                yaw = math.pi / 2 if rng.random() < 0.5 else -math.pi / 2
                obstacles.append(_parked_car(rng, pc, x0 + w / 2, y0 + d / 2, yaw))
        y_open, y_back = (y0 + d, y0) if is_target_row else (y0, y0 + d)
        lines.extend(_row_markings(sc["markings"], n, w, y_open, y_back))

    parked_cars = list(obstacles)

    # Curbs behind both rows
    total_x = n * w
    cg = sc["curb_gap"]
    curbs = [rect(total_x / 2, -cg - 0.15, 0.0, total_x + 12, 0.3)]
    if sc["opposite_row"]:
        curbs.append(rect(total_x / 2, 2 * d + aisle + cg + 0.15, 0.0, total_x + 12, 0.3))
    obstacles.extend(curbs)

    stall_cx = tgt * w + w / 2
    target_poly = rect(stall_cx, d / 2, math.pi / 2, d, w)
    off = vehicle.center_offset()
    if sc["maneuver"] == "reverse_in":
        goal = Pose(stall_cx, d / 2 - off, math.pi / 2)
    else:
        goal = Pose(stall_cx, d / 2 + off, -math.pi / 2)

    if start is None:
        start = Pose(stall_cx + rng.uniform(*sr["dx_from_stall"]),
                     d + aisle / 2 + rng.uniform(*sr["dy_from_aisle_center"]),
                     math.radians(rng.uniform(*sr["yaw_deg"])))

    ymax = (2 * d + aisle + cg + 1.0) if sc["opposite_row"] else (d + aisle + 2.0)
    bounds = (-6.0, total_x + 6.0, -cg - 1.0, ymax)
    s = Scenario(obstacles=obstacles, lines=lines, target_stall=target_poly,
                 stall_center=(stall_cx, d / 2), goal=goal, start=start, bounds=bounds,
                 cfg=params, vehicle=vehicle, res=sc["grid_resolution"],
                 parked_cars=parked_cars, curbs=curbs)
    s.build_distance_field()
    return s


def make_aisle_lot(vehicle: VehicleParams, params: dict,
                   rng: Optional[np.random.Generator] = None,
                   occupancy_prob: Optional[float] = None,
                   start: Optional[Pose] = None) -> Scenario:
    """A full aisle with stalls on both sides and the car entering at x < 0, heading +x.
    No stall is the target yet: Scenario.with_target() sets one once the driver has chosen."""
    rng = rng if rng is not None else np.random.default_rng(0)
    sc, al = params["scenario"], params["aisle_lot"]
    w, d, aisle = sc["stall_width"], sc["stall_depth"], sc["aisle_width"]
    n, pc, cg = al["stalls_per_row"], sc["parked_car"], sc["curb_gap"]
    prob = al["occupancy_prob"] if occupancy_prob is None else occupancy_prob

    occupied = rng.random(2 * n) < prob
    lo, hi = al["guaranteed_free"]
    short = int(rng.integers(lo, hi + 1)) - int(np.count_nonzero(~occupied))
    if short > 0:
        occupied[rng.choice(np.flatnonzero(occupied), short, replace=False)] = False

    stalls, parked_cars, lines = [], [], []
    for row, y0 in enumerate((0.0, d + aisle)):
        y_open, y_back = (y0 + d, y0) if row == 0 else (y0, y0 + d)
        for i in range(n):
            x0, sid = i * w, row * n + i
            if al["markings"]["stall_lines"]:
                lines.append(((x0, y0), (x0, y0 + d)))
            stalls.append(Stall(sid, row, i, ((x0, y_open), (x0 + w, y_open), (x0 + w, y_back), (x0, y_back)),
                                bool(occupied[sid]), "+y" if row == 0 else "-y"))
            if occupied[sid]:
                yaw = math.pi / 2 if rng.random() < 0.5 else -math.pi / 2
                parked_cars.append(_parked_car(rng, pc, x0 + w / 2, y0 + d / 2, yaw))
        if al["markings"]["stall_lines"]:
            lines.append(((n * w, y0), (n * w, y0 + d)))
        lines.extend(_row_markings(al["markings"], n, w, y_open, y_back))

    total_x = n * w
    xmin, xmax = -al["entry_distance"] - al["map_margin"], total_x + al["exit_margin"]
    curb_length, curb_x = xmax - xmin, (xmin + xmax) / 2.0
    curbs = [rect(curb_x, -cg - 0.15, 0.0, curb_length, 0.3),
             rect(curb_x, 2 * d + aisle + cg + 0.15, 0.0, curb_length, 0.3)]

    aisle_y = d + aisle / 2.0
    s = Scenario(obstacles=parked_cars + curbs, lines=lines, target_stall=None, stall_center=None,
                 goal=None, start=start or Pose(-al["entry_distance"], aisle_y, 0.0),
                 bounds=(xmin, xmax, -cg - 1.0, 2 * d + aisle + cg + 1.0),
                 cfg=params, vehicle=vehicle, res=sc["grid_resolution"],
                 parked_cars=parked_cars, curbs=curbs, stalls=stalls,
                 aisle_y=aisle_y, aisle_end_x=total_x)
    s.build_distance_field()
    return s
