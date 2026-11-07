"""Convex polygon helpers: exact overlap test and minimum distance."""
from __future__ import annotations

import math

import numpy as np


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def rect(cx: float, cy: float, yaw: float, length: float, width: float) -> np.ndarray:
    """Oriented rectangle corners (CCW) centered at (cx, cy)."""
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    c, s = math.cos(yaw), math.sin(yaw)
    return local @ np.array([[c, -s], [s, c]]).T + np.array([cx, cy])


def _axes(poly: np.ndarray) -> np.ndarray:
    edges = np.roll(poly, -1, axis=0) - poly
    normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def polygons_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex polygons."""
    for axis in np.vstack([_axes(a), _axes(b)]):
        pa, pb = a @ axis, b @ axis
        if pa.max() < pb.min() or pb.max() < pa.min():
            return False
    return True


def _point_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distances from points p (N,2) to segment a-b."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=1)


def polygon_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Minimum distance between convex polygons (0 if they overlap)."""
    if polygons_overlap(a, b):
        return 0.0
    best = math.inf
    for poly_p, poly_e in ((a, b), (b, a)):
        n = len(poly_e)
        for i in range(n):
            d = _point_segment_dist(poly_p, poly_e[i], poly_e[(i + 1) % n]).min()
            best = min(best, float(d))
    return best
