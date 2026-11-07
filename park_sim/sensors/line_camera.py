"""Bird's-eye-view camera: the painted line segments visible in its ground footprint.

Output segments are in the vehicle frame. Noise and false positives are new every frame. Faded
paint and gaps are wear on the lot itself, so they stay put during a run and every camera
sharing a PaintWear sees the same wear.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .base import Measurement, MountPose, Sensor, load_sensor_params, map_to_vehicle
from .scene import SceneArrays, scene_arrays, sight_blocked

FALSE_POSITIVE = -1


@dataclass(frozen=True, eq=False)
class LineCameraMeasurement(Measurement):
    segments: np.ndarray = None     # (K, 2, 2) start and end points, vehicle frame
    truth_line: np.ndarray = None   # (K,) index into scenario.lines, FALSE_POSITIVE if none; for tests and display


class PaintWear:
    """Which lines are faded and where the gaps are. Drawn once per lot, from the first camera's stream."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._lines: Optional[np.ndarray] = None    # the paint this was drawn for; actors don't redraw it
        self.faded = np.zeros(0, dtype=bool)
        self.gaps: List[np.ndarray] = []    # per line: (G, 2) start and end, meters along the line

    def ensure(self, scene: SceneArrays, rng: np.random.Generator) -> None:
        if self._lines is scene.lines_a:
            return
        self._lines = scene.lines_a
        lengths = np.hypot(*(scene.lines_b - scene.lines_a).T)
        fade, gap = self.cfg["faded"], self.cfg["gaps"]
        self.faded = (rng.random(len(lengths)) < fade["p_faded"]) if fade["enabled"] else np.zeros(len(lengths), dtype=bool)
        self.gaps = [np.zeros((0, 2))] * len(lengths)
        if gap["enabled"]:
            for i, length in enumerate(lengths):
                n = rng.poisson(gap["rate_per_m"] * length)
                if n:
                    starts = rng.uniform(0.0, length, n)
                    self.gaps[i] = np.column_stack([starts, starts + rng.uniform(*gap["length"], n)])


class LineCamera(Sensor):
    def __init__(self, name: str, mount: MountPose, cfg: dict, wear: Optional[PaintWear] = None):
        super().__init__(name, mount, cfg["rate_hz"])
        self.cfg = cfg
        self.wear = wear if wear is not None else PaintWear(cfg)
        fp = cfg["footprint"]
        near, far, half = fp["near"], fp["near"] + fp["depth"], fp["width"] / 2.0
        self.footprint = mount.to_vehicle(np.array([[near, -half], [far, -half], [far, half], [near, half]]))
        edges = np.roll(self.footprint, -1, axis=0) - self.footprint
        self._normals = np.column_stack([-edges[:, 1], edges[:, 0]])    # counter-clockwise: pointing inward
        self._reach = np.vstack([self.footprint, [mount.x, mount.y]])
        self._origin = np.array([mount.x, mount.y])

    def measure(self, scenario, state, t: float, rng: np.random.Generator) -> LineCameraMeasurement:
        cfg, scene = self.cfg, scene_arrays(scenario)
        self.wear.ensure(scene, rng)
        kept = np.flatnonzero(~self.wear.faded)
        a = map_to_vehicle(scene.lines_a[kept], state)
        b = map_to_vehicle(scene.lines_b[kept], state)
        t0, t1 = self._clip(a, b)
        inside = t0 < t1
        kept, a, b, t0, t1 = kept[inside], a[inside], b[inside], t0[inside], t1[inside]

        points, owners, visible = self._sample(kept, a, b, t0, t1)
        if cfg["occlusion"]["ego_body"] and len(points):
            visible &= ~_under_body(points, scenario.vehicle)
        if cfg["occlusion"]["parked_cars"] and len(points):
            car_a, car_b = self._nearby_edges(scene, state)
            todo = np.flatnonzero(visible)
            visible[todo] = ~sight_blocked(self._origin, points[todo], car_a, car_b)

        segments, truth = self._visible_runs(points, owners, visible, kept)
        fp_segments = self._false_positives(rng, scenario.vehicle) if cfg["false_positives"]["enabled"] else np.zeros((0, 2, 2))
        segments = np.concatenate([segments, fp_segments])
        truth = np.concatenate([truth, np.full(len(fp_segments), FALSE_POSITIVE, dtype=int)])
        if cfg["noise"]["enabled"] and len(segments):
            segments = segments + rng.normal(0.0, cfg["noise"]["std"], segments.shape)
        return LineCameraMeasurement(self.name, t, segments, truth)

    def _clip(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Part of each segment inside the convex footprint, as parameters 0-1 along it."""
        fa = (a[:, None, :] - self.footprint[None]) * self._normals[None]
        fb = (b[:, None, :] - self.footprint[None]) * self._normals[None]
        fa, fb = fa.sum(axis=2), fb.sum(axis=2)     # (L, edges); >= 0 is inside that edge
        df = fb - fa
        with np.errstate(divide="ignore", invalid="ignore"):
            cross = -fa / df
        t0 = np.where(df > 0.0, cross, 0.0).max(axis=1).clip(min=0.0)
        t1 = np.where(df < 0.0, cross, 1.0).min(axis=1).clip(max=1.0)
        t1[np.any((df == 0.0) & (fa < 0.0), axis=1)] = -1.0
        return t0, t1

    def _sample(self, lines: np.ndarray, a: np.ndarray, b: np.ndarray, t0: np.ndarray, t1: np.ndarray):
        """Test points along each clipped line, always including both clip ends, minus paint gaps."""
        step, chunks, owners, visible = self.cfg["sample_step"], [], [], []
        lengths = np.hypot(*(b - a).T)
        for k, line in enumerate(lines):
            n = max(2, int(math.ceil((t1[k] - t0[k]) * lengths[k] / step)) + 1)
            s = np.linspace(t0[k], t1[k], n)
            chunks.append(a[k] + s[:, None] * (b[k] - a[k]))
            owners.append(np.full(n, k))
            keep = np.ones(n, dtype=bool)
            for start, end in self.wear.gaps[line]:
                keep &= ~((s * lengths[k] >= start) & (s * lengths[k] <= end))
            visible.append(keep)
        if not chunks:
            return np.zeros((0, 2)), np.zeros(0, dtype=int), np.zeros(0, dtype=bool)
        return np.concatenate(chunks), np.concatenate(owners), np.concatenate(visible)

    def _nearby_edges(self, scene: SceneArrays, state) -> Tuple[np.ndarray, np.ndarray]:
        """Parked-car edges, vehicle frame, that could cross a sight line from this camera."""
        a, b = map_to_vehicle(scene.car_a, state), map_to_vehicle(scene.car_b, state)
        lo, hi = self._reach.min(axis=0), self._reach.max(axis=0)
        keep = np.all((np.minimum(a, b) <= hi) & (np.maximum(a, b) >= lo), axis=1)
        return a[keep], b[keep]

    def _visible_runs(self, points: np.ndarray, owners: np.ndarray, visible: np.ndarray,
                      lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Each unbroken run of visible points on a line becomes one segment."""
        segments, truth = [], []
        min_length = self.cfg["min_segment_length"]
        for k in np.unique(owners):
            idx = np.flatnonzero(owners == k)
            changes = np.flatnonzero(np.diff(np.concatenate([[0], visible[idx].astype(np.int8), [0]])))
            for first, last in zip(changes[::2], changes[1::2] - 1):
                p, q = points[idx[first]], points[idx[last]]
                if np.hypot(*(q - p)) >= min_length:
                    segments.append((p, q))
                    truth.append(lines[k])
        if not segments:
            return np.zeros((0, 2, 2)), np.zeros(0, dtype=int)
        return np.array(segments), np.array(truth, dtype=int)

    def _false_positives(self, rng: np.random.Generator, vehicle) -> np.ndarray:
        fp = self.cfg["false_positives"]
        n = rng.poisson(fp["fp_rate"] / self.rate_hz)
        if n == 0:
            return np.zeros((0, 2, 2))
        u, v = rng.random(n), rng.random(n)
        f = self.footprint
        starts = f[0] + u[:, None] * (f[1] - f[0]) + v[:, None] * (f[3] - f[0])
        heading, length = rng.uniform(-math.pi, math.pi, n), rng.uniform(*fp["length"], n)
        ends = starts + length[:, None] * np.column_stack([np.cos(heading), np.sin(heading)])
        t0, t1 = self._clip(starts, ends)
        p, q = starts + t0[:, None] * (ends - starts), starts + t1[:, None] * (ends - starts)
        keep = (t0 < t1) & ~_under_body((p + q) / 2.0, vehicle)
        return np.stack([p[keep], q[keep]], axis=1)


def _under_body(points: np.ndarray, vehicle) -> np.ndarray:
    return ((points[:, 0] >= -vehicle.rear_overhang) & (points[:, 0] <= vehicle.wheelbase + vehicle.front_overhang)
            & (np.abs(points[:, 1]) <= vehicle.width / 2.0))


def make_line_cameras(cfg: Optional[dict] = None) -> List[LineCamera]:
    """The configured cameras, sharing one PaintWear. A camera entry can override `footprint`."""
    cfg = cfg if cfg is not None else load_sensor_params()["line_camera"]
    wear = PaintWear(cfg)
    return [LineCamera(cam["name"], MountPose.from_config(cam), {**cfg, **cam}, wear) for cam in cfg["cameras"]]
