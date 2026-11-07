"""The lot as NumPy arrays, and the vectorized geometry the sensors share."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Sequence, Tuple

import numpy as np


@dataclass(frozen=True, eq=False)
class SceneArrays:
    lines_a: np.ndarray         # (L, 2) painted line starts, map frame
    lines_b: np.ndarray         # (L, 2) painted line ends
    obstacle_a: np.ndarray      # (M, 2) edges of every obstacle
    obstacle_b: np.ndarray
    car_a: np.ndarray           # (C, 2) edges of what hides paint from a camera: parked cars, pillars, carts
    car_b: np.ndarray


def _edges(polygons: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    polygons = list(polygons)
    if not polygons:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return (np.concatenate(polygons), np.concatenate([np.roll(p, -1, axis=0) for p in polygons]))


def _static_arrays(scenario) -> SceneArrays:
    cached = scenario._sensor_cache.get("arrays")
    if cached is None:
        lines = np.array(scenario.lines, dtype=float).reshape(-1, 2, 2)
        blockers = list(scenario.parked_cars) + [o.polygon for o in scenario.lot_objects]
        cached = SceneArrays(lines[:, 0], lines[:, 1], *_edges(scenario.obstacles), *_edges(blockers))
        scenario._sensor_cache["arrays"] = cached
    return cached


def scene_arrays(scenario) -> SceneArrays:
    """The lot, built once per lot, plus this tick's actors if there are any. Scenario.with_target()
    shares the result, since the lot is the same."""
    return scenario._sensor_cache.get("with_actors") or _static_arrays(scenario)


def set_actor_polygons(scenario, polygons: Sequence[np.ndarray]) -> None:
    """Moving obstacles for the sensors from now on: they return echoes and block sight, like parked
    cars. Call once per tick while they move; empty clears them. Only the actors' edges are new work."""
    cache = scenario._sensor_cache
    if len(polygons) == 0:
        cache.pop("with_actors", None)
        return
    static = _static_arrays(scenario)
    a, b = _edges(polygons)
    cache["with_actors"] = replace(static, obstacle_a=np.concatenate([static.obstacle_a, a]),
                                   obstacle_b=np.concatenate([static.obstacle_b, b]),
                                   car_a=np.concatenate([static.car_a, a]), car_b=np.concatenate([static.car_b, b]))


def within_reach(origin: np.ndarray, reach: float, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Only the segments whose bounding box comes within reach of the origin."""
    keep = np.all((np.minimum(a, b) <= origin + reach) & (np.maximum(a, b) >= origin - reach), axis=1)
    return a[keep], b[keep]


def cast_rays(origin: np.ndarray, directions: np.ndarray, a: np.ndarray, b: np.ndarray,
              max_range: float) -> np.ndarray:
    """Distance along each unit direction (N, 2) to the nearest segment a-b (M, 2); inf if none
    within max_range."""
    if len(a) == 0:
        return np.full(len(directions), math.inf)
    r, s, denom = _ray_segment(origin, directions, a, b)
    r = np.where((denom != 0.0) & (r >= 0.0) & (s >= 0.0) & (s <= 1.0), r, math.inf).min(axis=1)
    return np.where(r <= max_range, r, math.inf)


def sight_blocked(origin: np.ndarray, points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Whether a segment a-b (M, 2) crosses the straight line from origin to each point (P, 2)."""
    if len(a) == 0 or len(points) == 0:
        return np.zeros(len(points), dtype=bool)
    r, s, denom = _ray_segment(origin, points - origin, a, b)
    return np.any((denom != 0.0) & (r > 0.0) & (r < 1.0) & (s >= 0.0) & (s <= 1.0), axis=1)


def _ray_segment(origin, d, a, b):
    """Solves origin + r d = a + s (b - a) for every (ray, segment) pair; each result is (N, M)."""
    ex, ey = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    wx, wy = a[:, 0] - origin[0], a[:, 1] - origin[1]
    dx, dy = d[:, 0, None], d[:, 1, None]
    denom = dx * ey - dy * ex
    with np.errstate(divide="ignore", invalid="ignore"):
        return (wx * ey - wy * ex) / denom, (wx * dy - wy * dx) / denom, denom


def nearest_in_cone(origin: np.ndarray, axis_yaw: float, half_angle: float, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from origin to the nearest point of segments a-b (M, 2) inside a cone narrower
    than 180 deg; inf if none. The nearest point can't be hidden, so this needs no occlusion test."""
    if len(a) == 0:
        return math.inf
    pa, pb = a - origin, b - origin
    c, s, sin_h, cos_h = math.cos(axis_yaw), math.sin(axis_yaw), math.sin(half_angle), math.cos(half_angle)
    t0, t1 = np.zeros(len(a)), np.ones(len(a))
    outside = np.zeros(len(a), dtype=bool)
    for side in (-1.0, 1.0):    # inside each cone edge: x sin_h + side * y cos_h >= 0, in the axis frame
        nx, ny = c * sin_h - side * s * cos_h, s * sin_h + side * c * cos_h
        fa, fb = pa[:, 0] * nx + pa[:, 1] * ny, pb[:, 0] * nx + pb[:, 1] * ny
        df = fb - fa
        with np.errstate(divide="ignore", invalid="ignore"):
            cross = -fa / df
        t0 = np.maximum(t0, np.where(df > 0.0, cross, 0.0))
        t1 = np.minimum(t1, np.where(df < 0.0, cross, 1.0))
        outside |= (df == 0.0) & (fa < 0.0)
    d = pb - pa
    keep = (t0 <= t1) & ~outside
    if not keep.any():
        return math.inf
    pa, d, t0, t1 = pa[keep], d[keep], t0[keep], t1[keep]
    t = np.clip(-(pa[:, 0] * d[:, 0] + pa[:, 1] * d[:, 1]) / np.maximum((d * d).sum(axis=1), 1e-12), t0, t1)
    return float(np.hypot(pa[:, 0] + t * d[:, 0], pa[:, 1] + t * d[:, 1]).min())
