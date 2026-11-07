"""Hybrid A* planner with Reeds-Shepp analytic expansion.

Search runs over (x, y, yaw) cells; each expansion drives a short exact arc
(bicycle model) forward or in reverse at one of n_steer steering angles.
Heuristic = max(obstacle-free Reeds-Shepp length, 2D obstacle-aware
Dijkstra distance). Every few expansions it tries a direct Reeds-Shepp shot
to the goal and accepts the first collision-free one.
"""
from __future__ import annotations

import heapq
import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import wrap_angle
from .reeds_shepp import reeds_shepp_length, reeds_shepp_paths
from .scenario import Pose, Scenario


@dataclass
class PlannedPath:
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    direction: np.ndarray            # +1 forward / -1 reverse, per sample
    expansions: int = 0
    plan_time_s: float = 0.0

    @property
    def n_direction_changes(self) -> int:
        return int(np.sum(self.direction[1:] != self.direction[:-1]))

    @property
    def length(self) -> float:
        return float(np.sum(np.hypot(np.diff(self.x), np.diff(self.y))))

    def segments(self) -> List[Tuple[int, int, int]]:
        """(start_idx, end_idx_inclusive, direction) for each constant-direction run."""
        segs, s = [], 0
        for i in range(1, len(self.direction)):
            if self.direction[i] != self.direction[i - 1]:
                segs.append((s, i - 1, int(self.direction[i - 1])))
                s = i - 1                     # cusp sample shared by both runs
        segs.append((s, len(self.direction) - 1, int(self.direction[-1])))
        return [(a, b, d) for a, b, d in segs if b > a]


class PlannerObserver:
    """Told what a search does, as it does it, for display. Nothing it does feeds back into the search, but it
    runs inside the planner's time limit, so keep it to recording: see SearchRecorder."""

    def on_search_start(self, start: Pose, target: Pose) -> None:
        """target: the staging pose the search aims for, in the same frame as start."""

    def on_expand(self, number: int, index: int, parent: Optional[int], x: float, y: float, g: float, f: float,
                  xs: np.ndarray, ys: np.ndarray) -> None:
        """Expansion `number` (from 1) took node `index`, reached from node `parent` along the arc xs, ys."""

    def on_shot(self, number: int, x: np.ndarray, y: np.ndarray, collision_free: bool) -> None:
        """An analytic shot was tried, from the node of expansion `number`."""

    def on_search_end(self, path: Optional[PlannedPath], expansions: int, elapsed_s: float, why: str) -> None:
        """why: found | start_blocked | time_limit | exhausted (out of nodes or of max_expansions)."""


@dataclass(frozen=True, eq=False)
class ExpandedNode:
    number: int
    index: int
    parent: Optional[int]
    x: float
    y: float
    g: float
    f: float
    xs: np.ndarray
    ys: np.ndarray


@dataclass(frozen=True, eq=False)
class TriedShot:
    number: int
    x: np.ndarray
    y: np.ndarray
    collision_free: bool


@dataclass(eq=False)
class SearchRecord:
    """One finished or running search, for showing afterwards."""
    start: Pose
    target: Pose
    expanded: List[ExpandedNode] = field(default_factory=list)
    shots: List[TriedShot] = field(default_factory=list)
    path: Optional[PlannedPath] = None
    expansions: int = 0
    elapsed_s: float = 0.0
    why: str = ""


class SearchRecorder(PlannerObserver):
    """Keeps each search to replay it later. It only appends references to what the search already built, so
    what it costs inside the time limit is a list append per event."""

    def __init__(self):
        self.records: List[SearchRecord] = []

    def on_search_start(self, start, target):
        self.records.append(SearchRecord(start, target))

    def on_expand(self, number, index, parent, x, y, g, f, xs, ys):
        self.records[-1].expanded.append(ExpandedNode(number, index, parent, x, y, g, f, xs, ys))

    def on_shot(self, number, x, y, collision_free):
        self.records[-1].shots.append(TriedShot(number, x, y, collision_free))

    def on_search_end(self, path, expansions, elapsed_s, why):
        record = self.records[-1]
        record.path, record.expansions, record.elapsed_s, record.why = path, expansions, elapsed_s, why

    def take(self) -> List[SearchRecord]:
        """The searches so far, leaving none."""
        records, self.records = self.records, []
        return records


_scoped_observer: ContextVar[Optional[PlannerObserver]] = ContextVar("planner_observer", default=None)


@contextmanager
def observing(observer: Optional[PlannerObserver]):
    """Planners made inside this block report to `observer`, for callers with layers between them and the planner."""
    token = _scoped_observer.set(observer)
    try:
        yield
    finally:
        _scoped_observer.reset(token)


@dataclass(order=True)
class _QItem:
    f: float
    idx: int = field(compare=False)


@dataclass
class _Node:
    x: float
    y: float
    yaw: float
    direction: int
    steer: float
    g: float
    parent: Optional[int]
    xs: np.ndarray
    ys: np.ndarray
    yaws: np.ndarray


def _arc(x, y, yaw, steer, direction, length, wheelbase, step=0.15):
    n = max(2, int(math.ceil(length / step)))
    s = np.linspace(length / n, length, n) * direction
    k = math.tan(steer) / wheelbase
    if abs(k) < 1e-9:
        xs = x + s * math.cos(yaw)
        ys = y + s * math.sin(yaw)
        yaws = np.full(n, yaw)
    else:
        yaws = yaw + s * k
        xs = x + (np.sin(yaws) - math.sin(yaw)) / k
        ys = y + (-np.cos(yaws) + math.cos(yaw)) / k
    return xs, ys, yaws


class HybridAStar:
    def __init__(self, scenario: Scenario, cfg: Optional[dict] = None, observer: Optional[PlannerObserver] = None):
        self.observer = observer if observer is not None else _scoped_observer.get()
        self.sc = scenario
        self.v = scenario.vehicle
        self.c = cfg if cfg is not None else scenario.cfg["planner"]
        self.radius = self.v.min_turn_radius
        steer_max = self.v.planner_steer
        self.steers = np.linspace(-steer_max, steer_max, self.c["n_steer"])
        # Final approach: a straight run of `final_straight` meters into the goal,
        # driven in `approach` direction (-1 reverse-in, +1 head-in). The search
        # targets the staging pose at the start of that straight.
        g = scenario.goal
        self.approach = -1 if scenario.cfg["scenario"]["maneuver"] == "reverse_in" else 1
        ls = self.c.get("final_straight", 0.0)
        self.final_straight = ls
        self.target = Pose(g.x - self.approach * ls * math.cos(g.yaw),
                           g.y - self.approach * ls * math.sin(g.yaw), g.yaw)
        self._build_holonomic()

    # ----- heuristic helpers -----
    def _build_holonomic(self, res: float = 0.5) -> None:
        xmin, xmax, ymin, ymax = self.sc.bounds
        nx, ny = int((xmax - xmin) / res) + 1, int((ymax - ymin) / res) + 1
        xs = xmin + np.arange(nx) * res
        ys = ymin + np.arange(ny) * res
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        from scipy.ndimage import map_coordinates
        d = map_coordinates(self.sc._dist, [((gx - xmin) / self.sc.res - 0.5).ravel(),
                                            ((gy - ymin) / self.sc.res - 0.5).ravel()],
                            order=1, mode="constant", cval=-1).reshape(nx, ny)
        free = d > 0.6
        cost = np.full((nx, ny), np.inf)
        gi = int(round((self.target.x - xmin) / res))
        gj = int(round((self.target.y - ymin) / res))
        gi, gj = min(max(gi, 0), nx - 1), min(max(gj, 0), ny - 1)
        cost[gi, gj] = 0.0
        pq = [(0.0, gi, gj)]
        moves = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                 (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
        while pq:
            cst, i, j = heapq.heappop(pq)
            if cst > cost[i, j]:
                continue
            for di, dj, w in moves:
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny and free[a, b]:
                    nc = cst + w * res
                    if nc < cost[a, b]:
                        cost[a, b] = nc
                        heapq.heappush(pq, (nc, a, b))
        self._holo, self._holo_res = cost, res

    def _h(self, x, y, yaw) -> float:
        g = self.target
        rs = reeds_shepp_length((x, y, yaw), (g.x, g.y, g.yaw), self.radius)
        xmin, _, ymin, _ = self.sc.bounds
        i = int(round((x - xmin) / self._holo_res))
        j = int(round((y - ymin) / self._holo_res))
        holo = self._holo[i, j] if 0 <= i < self._holo.shape[0] and 0 <= j < self._holo.shape[1] else math.inf
        if not math.isfinite(holo):
            holo = 0.0
        return max(rs, holo)

    def _key(self, x, y, yaw):
        r = self.c["xy_resolution"]
        nb = self.c["yaw_bins"]
        return (int(math.floor(x / r)), int(math.floor(y / r)),
                int(math.floor((wrap_angle(yaw) + math.pi) / (2 * math.pi) * nb)) % nb)

    def _path_cost(self, lengths, start_dir) -> float:
        c, cost, d_prev = self.c, 0.0, start_dir
        for l in lengths:
            if abs(l) < 1e-9:
                continue
            d = 1 if l > 0 else -1
            cost += abs(l) * (c["reverse_cost"] if d < 0 else 1.0)
            if d != d_prev:
                cost += c["switch_cost"]
            d_prev = d
        return cost

    # ----- main search -----
    def plan(self, start: Optional[Pose] = None,
             time_limit_s: Optional[float] = None) -> Optional[PlannedPath]:
        """Returns None if there is no path, or none was found within time_limit_s (wall clock)."""
        t0 = time.time()
        start = start or self.sc.start
        goal = self.target
        c = self.c
        margin = c["collision_margin"]
        observer = self.observer
        if observer is not None:
            observer.on_search_start(start, goal)
        if not self.sc.collision_free(start.x, start.y, start.yaw, margin * 0.5):
            if observer is not None:
                observer.on_search_end(None, 0, time.time() - t0, "start_blocked")
            return None

        nodes: List[_Node] = [_Node(start.x, start.y, start.yaw, 1, 0.0, 0.0, None,
                                    np.array([start.x]), np.array([start.y]), np.array([start.yaw]))]
        closed: Dict[tuple, int] = {}
        best_g: Dict[tuple, float] = {self._key(start.x, start.y, start.yaw): 0.0}
        pq = [_QItem(self._h(start.x, start.y, start.yaw), 0)]
        steer_max = self.v.planner_steer
        expansions = 0

        while pq and expansions < c["max_expansions"]:
            if time_limit_s is not None and time.time() - t0 >= time_limit_s:
                if observer is not None:
                    observer.on_search_end(None, expansions, time.time() - t0, "time_limit")
                return None
            item = heapq.heappop(pq)
            n = nodes[item.idx]
            key = self._key(n.x, n.y, n.yaw)
            if key in closed:
                continue
            closed[key] = item.idx
            expansions += 1
            if observer is not None:
                observer.on_expand(expansions, item.idx, n.parent, n.x, n.y, n.g, item.f, n.xs, n.ys)

            # Analytic expansion
            near = math.hypot(n.x - goal.x, n.y - goal.y) < 8.0
            if expansions % c["analytic_every"] == 0 or near:
                shot = self._try_shot(n, goal, margin, expansions)
                if shot is not None:
                    path = self._assemble(nodes, item.idx, shot, expansions, time.time() - t0)
                    if observer is not None:
                        observer.on_search_end(path, expansions, path.plan_time_s, "found")
                    return path

            for d in (1, -1):
                for st in self.steers:
                    xs, ys, yaws = _arc(n.x, n.y, n.yaw, st, d, c["arc_length"], self.v.wheelbase)
                    if not self.sc.collision_free(xs, ys, yaws, margin):
                        continue
                    nk = self._key(xs[-1], ys[-1], yaws[-1])
                    if nk in closed:
                        continue
                    g = n.g + c["arc_length"] * (c["reverse_cost"] if d < 0 else 1.0)
                    if d != n.direction:
                        g += c["switch_cost"]
                    g += c["steer_cost"] * abs(st) / steer_max * c["arc_length"]
                    g += c["steer_change_cost"] * abs(st - n.steer) / steer_max
                    if g >= best_g.get(nk, math.inf):
                        continue
                    best_g[nk] = g
                    nodes.append(_Node(xs[-1], ys[-1], yaws[-1], d, st, g, item.idx, xs, ys, yaws))
                    heapq.heappush(pq, _QItem(g + self._h(xs[-1], ys[-1], yaws[-1]), len(nodes) - 1))
        if observer is not None:
            observer.on_search_end(None, expansions, time.time() - t0, "exhausted")
        return None

    def _try_shot(self, n: _Node, goal: Pose, margin: float, number: int = 0):
        paths = reeds_shepp_paths((n.x, n.y, n.yaw), (goal.x, goal.y, goal.yaw),
                                  self.radius, step=0.15)
        if self.final_straight > 0:
            # arrive at the staging pose already moving in the approach direction
            paths = [p for p in paths
                     if [l for l in p.lengths if abs(l) > 1e-9][-1] * self.approach > 0]
        paths.sort(key=lambda p: self._path_cost(p.lengths, n.direction))
        for p in paths[:6]:
            free = self.sc.collision_free(p.x, p.y, p.yaw, margin)
            if self.observer is not None:
                self.observer.on_shot(number, p.x, p.y, bool(free))
            if free:
                return p
        return None

    def _assemble(self, nodes, idx, shot, expansions, dt) -> PlannedPath:
        chain = []
        while idx is not None:
            chain.append(nodes[idx])
            idx = nodes[idx].parent
        chain.reverse()
        xs, ys, yaws, dirs = [chain[0].x], [chain[0].y], [chain[0].yaw], [1]
        for nd in chain[1:]:
            xs.extend(nd.xs); ys.extend(nd.ys); yaws.extend(nd.yaws)
            dirs.extend([nd.direction] * len(nd.xs))
        xs.extend(shot.x[1:]); ys.extend(shot.y[1:]); yaws.extend(shot.yaw[1:])
        dirs.extend(shot.direction[1:].astype(int).tolist())
        if self.final_straight > 0:
            g = self.sc.goal
            n = max(2, int(math.ceil(self.final_straight / 0.1)))
            t = np.linspace(1.0 / n, 1.0, n)
            xs.extend((self.target.x + t * (g.x - self.target.x)).tolist())
            ys.extend((self.target.y + t * (g.y - self.target.y)).tolist())
            yaws.extend([g.yaw] * n)
            dirs.extend([self.approach] * n)
        dirs[0] = dirs[1] if len(dirs) > 1 else 1
        return PlannedPath(np.array(xs), np.array(ys), np.array(yaws), np.array(dirs),
                           expansions, dt)
