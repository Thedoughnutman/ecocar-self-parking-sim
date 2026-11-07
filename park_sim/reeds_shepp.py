"""Reeds-Shepp paths (forward/reverse, minimum turning radius).

Path families follow Reeds & Shepp (1990) as arranged in PythonRobotics.
Every candidate is integrated and checked against the goal, so a formula
that does not land on the goal is simply discarded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

PI = math.pi


def _mod2pi(x: float) -> float:
    v = math.fmod(x, 2 * PI)
    if v < -PI:
        v += 2 * PI
    elif v > PI:
        v -= 2 * PI
    return v


def _polar(x, y):
    return math.hypot(x, y), math.atan2(y, x)


# Each returns (lengths, segment types) in normalized units (radius = 1) or None.
def _LSL(x, y, phi):
    u, t = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if 0.0 <= t <= PI:
        v = _mod2pi(phi - t)
        if 0.0 <= v <= PI:
            return [t, u, v], "LSL"
    return None


def _LSR(x, y, phi):
    u1, t1 = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 ** 2
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = _mod2pi(t1 + theta)
        v = _mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return [t, u, v], "LSR"
    return None


def _LxRxL(x, y, phi):
    u1, theta = _polar(x - math.sin(phi), y - 1 + math.cos(phi))
    if u1 <= 4.0:
        a = math.acos(0.25 * u1)
        t = _mod2pi(a + theta + PI / 2)
        u = _mod2pi(PI - 2 * a)
        v = _mod2pi(phi - t - u)
        return [t, -u, v], "LRL"
    return None


def _LxRL(x, y, phi):
    u1, theta = _polar(x - math.sin(phi), y - 1 + math.cos(phi))
    if u1 <= 4.0:
        a = math.acos(0.25 * u1)
        t = _mod2pi(a + theta + PI / 2)
        u = _mod2pi(PI - 2 * a)
        v = _mod2pi(-phi + t + u)
        return [t, -u, -v], "LRL"
    return None


def _LRxL(x, y, phi):
    u1, theta = _polar(x - math.sin(phi), y - 1 + math.cos(phi))
    if 1e-9 < u1 <= 4.0:
        u = math.acos(1 - u1 ** 2 * 0.125)
        a = math.asin(max(-1.0, min(1.0, 2 * math.sin(u) / u1)))
        t = _mod2pi(-a + theta + PI / 2)
        v = _mod2pi(t - u - phi)
        return [t, u, -v], "LRL"
    return None


def _LRxLR(x, y, phi):
    u1, theta = _polar(x + math.sin(phi), y - 1 - math.cos(phi))
    if u1 <= 2:
        a = math.acos((u1 + 2) * 0.25)
        t = _mod2pi(theta + a + PI / 2)
        u = _mod2pi(a)
        v = _mod2pi(phi - t + 2 * u)
        if t >= 0 and u >= 0 and v >= 0:
            return [t, u, -u, -v], "LRLR"
    return None


def _LxRLxR(x, y, phi):
    u1, theta = _polar(x + math.sin(phi), y - 1 - math.cos(phi))
    u2 = (20 - u1 ** 2) / 16
    if 0 <= u2 <= 1 and u1 > 1e-9:
        u = math.acos(u2)
        a = math.asin(max(-1.0, min(1.0, 2 * math.sin(u) / u1)))
        t = _mod2pi(theta + a + PI / 2)
        v = _mod2pi(t - phi)
        if t >= 0 and v >= 0:
            return [t, -u, -u, v], "LRLR"
    return None


def _LxR90SL(x, y, phi):
    u1, theta = _polar(x - math.sin(phi), y - 1 + math.cos(phi))
    if u1 >= 2.0:
        u = math.sqrt(u1 ** 2 - 4) - 2
        a = math.atan2(2, math.sqrt(u1 ** 2 - 4))
        t = _mod2pi(theta + a + PI / 2)
        v = _mod2pi(t - phi + PI / 2)
        if t >= 0 and v >= 0:
            return [t, -PI / 2, -u, -v], "LRSL"
    return None


def _LSR90xL(x, y, phi):
    u1, theta = _polar(x - math.sin(phi), y - 1 + math.cos(phi))
    if u1 >= 2.0:
        u = math.sqrt(u1 ** 2 - 4) - 2
        a = math.atan2(math.sqrt(u1 ** 2 - 4), 2)
        t = _mod2pi(theta - a + PI / 2)
        v = _mod2pi(t - phi - PI / 2)
        if t >= 0 and v >= 0:
            return [t, u, PI / 2, -v], "LSRL"
    return None


def _LxR90SR(x, y, phi):
    u1, theta = _polar(x + math.sin(phi), y - 1 - math.cos(phi))
    if u1 >= 2.0:
        t = _mod2pi(theta + PI / 2)
        u = u1 - 2
        v = _mod2pi(phi - t - PI / 2)
        if t >= 0 and v >= 0:
            return [t, -PI / 2, -u, -v], "LRSR"
    return None


def _LSL90xR(x, y, phi):
    u1, theta = _polar(x + math.sin(phi), y - 1 - math.cos(phi))
    if u1 >= 2.0:
        t = _mod2pi(theta)
        u = u1 - 2
        v = _mod2pi(phi - t - PI / 2)
        if t >= 0 and v >= 0:
            return [t, u, PI / 2, -v], "LSLR"
    return None


def _LxR90SL90xR(x, y, phi):
    u1, theta = _polar(x + math.sin(phi), y - 1 - math.cos(phi))
    if u1 >= 4.0:
        u = math.sqrt(u1 ** 2 - 4) - 4
        a = math.atan2(2, math.sqrt(u1 ** 2 - 4))
        t = _mod2pi(theta + a + PI / 2)
        v = _mod2pi(t - phi)
        if t >= 0 and v >= 0:
            return [t, -PI / 2, -u, -PI / 2, v], "LRSLR"
    return None


_FAMILIES = [_LSL, _LSR, _LxRxL, _LxRL, _LRxL, _LRxLR, _LxRLxR,
             _LxR90SL, _LxR90SR, _LSR90xL, _LSL90xR, _LxR90SL90xR]

_REFLECT = {"L": "R", "R": "L", "S": "S"}


@dataclass
class RSPath:
    lengths: List[float]     # signed, meters (negative = reverse)
    types: str               # e.g. "LSR"
    x: np.ndarray = None
    y: np.ndarray = None
    yaw: np.ndarray = None
    direction: np.ndarray = None   # +1 / -1 per sample

    @property
    def total_length(self) -> float:
        return float(sum(abs(l) for l in self.lengths))


def _integrate(start, lengths, types, radius, step):
    x, y, yaw = start
    xs, ys, yaws, ds_ = [x], [y], [yaw], []
    for length, t in zip(lengths, types):
        if abs(length) < 1e-10:
            continue
        direction = 1.0 if length > 0 else -1.0
        k = {"L": 1.0, "R": -1.0, "S": 0.0}[t] / radius
        n = max(1, int(math.ceil(abs(length) / step)))
        h = length / n
        for _ in range(n):
            if k == 0.0:
                x += h * math.cos(yaw)
                y += h * math.sin(yaw)
            else:
                dyaw = h * k
                x += (math.sin(yaw + dyaw) - math.sin(yaw)) / k
                y += (-math.cos(yaw + dyaw) + math.cos(yaw)) / k
                yaw += dyaw
            xs.append(x); ys.append(y); yaws.append(yaw); ds_.append(direction)
    if not ds_:
        ds_ = [1.0]
    dirs = [ds_[0]] + ds_
    return np.array(xs), np.array(ys), np.array(yaws), np.array(dirs)


def reeds_shepp_paths(start: Tuple[float, float, float], goal: Tuple[float, float, float],
                      radius: float, step: float = 0.1, tol: float = 1e-2) -> List[RSPath]:
    """All valid Reeds-Shepp paths from start to goal, shortest first."""
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    c, s = math.cos(start[2]), math.sin(start[2])
    x = (c * dx + s * dy) / radius
    y = (-s * dx + c * dy) / radius
    phi = _mod2pi(goal[2] - start[2])

    raw = []
    for f in _FAMILIES:
        for sx, sy, sphi, flip, refl in [(1, 1, 1, False, False), (-1, 1, -1, True, False),
                                         (1, -1, -1, False, True), (-1, -1, 1, True, True)]:
            r = f(sx * x, sy * y, sphi * phi)
            if r is None:
                continue
            lengths, types = r
            if flip:
                lengths = [-l for l in lengths]
            if refl:
                types = "".join(_REFLECT[t] for t in types)
            raw.append(([l * radius for l in lengths], types))

    out, seen = [], set()
    for lengths, types in raw:
        key = (types, tuple(round(l, 4) for l in lengths))
        if key in seen:
            continue
        seen.add(key)
        px, py, pyaw, pdir = _integrate(start, lengths, types, radius, step)
        err = math.hypot(px[-1] - goal[0], py[-1] - goal[1])
        if err > tol or abs(_mod2pi(pyaw[-1] - goal[2])) > tol:
            continue
        out.append(RSPath(lengths, types, px, py, pyaw, pdir))
    out.sort(key=lambda p: p.total_length)
    return out


def reeds_shepp_length(start, goal, radius) -> float:
    """Shortest RS length without generating samples (used as a heuristic)."""
    paths = reeds_shepp_paths(start, goal, radius, step=1e3, tol=1e-2)
    return paths[0].total_length if paths else math.inf
