"""Scripted moving obstacles: pedestrians (a circle) and vehicles (a rectangle down the aisle).

Sim-only. Each actor follows a polyline at a set speed once its trigger fires, can stop along the
way (for a while or for good), and either stays at the end of its path or leaves the scene. A trigger
is a run time or a supervisor state plus a delay: "2 s after MANEUVERING starts".

Paths are in one of two frames. "map" is the lot's frame. "target" follows the stall being parked in,
so one script fits every lot: u along the aisle, v out of the stall toward the aisle, origin at the
center of the stall's opening (v = -depth is the back of the stall, v = aisle_width the far side of
the aisle). An actor in the target frame waits until a target is known.

An actor with yield_gap_m > 0 stops rather than walk or drive into the car: it holds while the space
yield_gap_m ahead of it comes within yield_side_m of the car, and passes otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import polygon_distance, rect

KINDS = ("pedestrian", "vehicle")
FRAMES = ("map", "target")
ENDS = ("stay", "leave")


@dataclass(frozen=True)
class Trigger:
    """Fires at time_s, or delay_s after the supervisor first enters `state`. Empty = from the start."""
    time_s: Optional[float] = None
    state: Optional[str] = None
    delay_s: float = 0.0

    def fire_time(self, entered: Dict[str, float]) -> Optional[float]:
        if self.state is not None:
            return entered[self.state] + self.delay_s if self.state in entered else None
        return self.time_s if self.time_s is not None else 0.0


@dataclass(frozen=True)
class TargetFrame:
    origin: Tuple[float, float]
    u: Tuple[float, float]
    v: Tuple[float, float]

    @classmethod
    def from_scenario(cls, sc) -> "TargetFrame":
        """From a scenario with a target stall."""
        out = sc.goal.yaw if sc.cfg["scenario"]["maneuver"] == "reverse_in" else sc.goal.yaw + math.pi
        ox, oy = math.cos(out), math.sin(out)
        half = sc.cfg["scenario"]["stall_depth"] / 2.0
        return cls((sc.stall_center[0] + half * ox, sc.stall_center[1] + half * oy), (oy, -ox), (ox, oy))

    def to_map(self, points: np.ndarray) -> np.ndarray:
        return np.array(self.origin) + points[:, :1] * np.array(self.u) + points[:, 1:2] * np.array(self.v)


@dataclass(frozen=True)
class ActorStop:
    at_m: float                 # m along the path
    duration_s: Optional[float] # None = stays there


class Actor:
    def __init__(self, name: str, kind: str, path: Sequence[Sequence[float]], speed: float, size: Tuple[float, ...],
                 trigger: Trigger = Trigger(), frame: str = "map", stops: Sequence[ActorStop] = (),
                 hidden_until_trigger: bool = True, on_end: str = "stay", yield_gap: float = 0.0, yield_side: float = 0.0,
                 sides: int = 12):
        if kind not in KINDS:
            raise ValueError(f"unknown actor kind {kind!r}; expected one of {KINDS}")
        if frame not in FRAMES:
            raise ValueError(f"unknown actor frame {frame!r}; expected one of {FRAMES}")
        if on_end not in ENDS:
            raise ValueError(f"unknown on_end {on_end!r}; expected one of {ENDS}")
        self.name, self.kind, self.speed, self.size = name, kind, speed, size
        self.trigger, self.frame, self.stops = trigger, frame, sorted(stops, key=lambda s: s.at_m)
        self.hidden_until_trigger, self.on_end, self.sides = hidden_until_trigger, on_end, sides
        self.yield_gap, self.yield_side = yield_gap, yield_side
        self._radius = math.hypot(*size) / 2.0      # circumradius: a pedestrian's size is just its diameter
        angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
        self._circle = size[0] / 2.0 * np.column_stack([np.cos(angles), np.sin(angles)])    # a pedestrian, at the origin
        self._script = np.array(path, dtype=float).reshape(-1, 2)
        self._points: Optional[np.ndarray] = None   # the path in the map frame, once its frame is known
        self._s = np.zeros(1)
        self.progress = 0.0
        self.triggered = False
        self.left = False
        self.moving = False
        self._next_stop = 0
        self._stopped_until: Optional[float] = None
        if frame == "map":
            self._set_points(self._script)

    def _set_points(self, points: np.ndarray) -> None:
        self._points = points
        self._s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])

    def resolve(self, target: TargetFrame) -> None:
        if self._points is None:
            self._set_points(target.to_map(self._script))

    @property
    def visible(self) -> bool:
        if self._points is None or self.left:
            return False
        return self.triggered or not self.hidden_until_trigger

    @property
    def stationary(self) -> bool:
        return self.visible and not self.moving

    def pose_at(self, s: float) -> Tuple[float, float, float]:
        """x, y, heading at s meters along the path."""
        p = self._points
        if len(p) == 1:
            return float(p[0, 0]), float(p[0, 1]), 0.0
        i = min(max(int(np.searchsorted(self._s, s, side="right")) - 1, 0), len(p) - 2)
        length = self._s[i + 1] - self._s[i]
        f = (s - self._s[i]) / length if length > 0.0 else 0.0
        d = p[i + 1] - p[i]
        return float(p[i, 0] + f * d[0]), float(p[i, 1] + f * d[1]), math.atan2(d[1], d[0])

    def polygon_at(self, s: float) -> np.ndarray:
        x, y, yaw = self.pose_at(s)
        if self.kind == "vehicle":
            return rect(x, y, yaw, self.size[0], self.size[1])
        return self._circle + (x, y)

    def polygon(self) -> np.ndarray:
        return self.polygon_at(self.progress)

    def step(self, t: float, dt: float, ego: np.ndarray, entered: Dict[str, float]) -> None:
        """Advance to the end of this tick. `ego` is the car's body polygon; `entered` is when the
        supervisor first entered each state, by state name."""
        self.moving = False
        if self.left or self._points is None:
            return
        if not self.triggered:
            fire = self.trigger.fire_time(entered)
            if fire is None or t < fire:
                return
            self.triggered = True
        if self._stopped_until is not None:
            if t < self._stopped_until:
                return
            self._stopped_until = None
        total = float(self._s[-1])
        if self.progress >= total:
            self.left = self.on_end == "leave"
            return
        stop = self.stops[self._next_stop] if self._next_stop < len(self.stops) else None
        ahead = min(self.progress + self.speed * dt, total)
        if stop is not None and self.progress >= stop.at_m:
            self._next_stop += 1
            self._stopped_until = math.inf if stop.duration_s is None else t + stop.duration_s
            return
        if stop is not None:
            ahead = min(ahead, stop.at_m)
        if self.yield_gap > 0.0 and self._would_crowd(ego, ahead):
            return
        self.progress = ahead
        self.moving = True

    def _would_crowd(self, ego: np.ndarray, ahead: float) -> bool:
        """Where it is heading, yield_gap on, is too near the car, and nearer than where it is now."""
        s = min(ahead + self.yield_gap, float(self._s[-1]))
        if self._surely_clear(ego, s):
            return False
        look = polygon_distance(self.polygon_at(s), ego)
        return look < self.yield_side and look < polygon_distance(self.polygon_at(self.progress), ego)

    def _surely_clear(self, ego: np.ndarray, s: float) -> bool:
        """Cheap test that never changes the answer: the true distance can't be less than the gap from
        the actor's center to the car's bounding box, minus the actor's circumradius. Most actors are
        far from the car most of the time, and the exact distance is the slow part of a tick."""
        x, y, _ = self.pose_at(s)
        low, high = ego.min(axis=0), ego.max(axis=0)
        gap = math.hypot(max(low[0] - x, x - high[0], 0.0), max(low[1] - y, y - high[1], 0.0))
        return gap - self._radius >= self.yield_side


class ActorSet:
    """All of a run's actors. Call step() once per tick."""

    def __init__(self, actors: Sequence[Actor] = ()):
        self.actors: List[Actor] = list(actors)
        self.entered: Dict[str, float] = {}

    def __bool__(self) -> bool:
        return bool(self.actors)

    def set_target(self, target: TargetFrame) -> None:
        for actor in self.actors:
            actor.resolve(target)

    def step(self, t: float, dt: float, ego: np.ndarray, state: str) -> None:
        self.entered.setdefault(state, t)
        for actor in self.actors:
            actor.step(t, dt, ego, self.entered)

    def polygons(self) -> List[np.ndarray]:
        return [a.polygon() for a in self.actors if a.visible]

    def stationary_polygons(self) -> List[np.ndarray]:
        """Ground-truth footprints of the actors standing still, for the planner."""
        return [a.polygon() for a in self.actors if a.stationary]


def make_actor(spec: dict, cfg: dict) -> Actor:
    """One actor from its scenario entry; `cfg` is the actors: section of park_params.yaml."""
    kind = spec["kind"]
    if kind not in KINDS:
        raise ValueError(f"unknown actor kind {kind!r}; expected one of {KINDS}")
    own = cfg[kind]
    if kind == "pedestrian":
        low, high = own["speed_range_mps"]
        speed, size = spec.get("speed_mps", (low + high) / 2.0), (own["diameter_m"],)
    else:
        low, high = (v / 3.6 for v in own["speed_range_kph"])
        speed, size = spec.get("speed_kph", (low + high) / 2.0 * 3.6) / 3.6, (own["length_m"], own["width_m"])
    path = spec["path"]
    if len(path) > 1 and not low - 1e-9 <= speed <= high + 1e-9:
        raise ValueError(f"{kind} speed {speed:.2f} m/s is outside {low:.2f}-{high:.2f} m/s")
    t = spec.get("trigger", {})
    trigger = Trigger(t.get("time_s"), t.get("state"), t.get("delay_s", 0.0))
    stops = [ActorStop(s["at_m"], s.get("duration_s")) for s in spec.get("stops", ())]
    return Actor(spec.get("name", kind), kind, path, speed, size, trigger, spec.get("frame", "map"), stops,
                 spec.get("hidden_until_trigger", True), spec.get("on_end", "stay"),
                 spec.get("yield_gap_m", own["yield_gap_m"]), spec.get("yield_side_m", own["yield_side_m"]),
                 own.get("polygon_sides", 12))


def make_actors(specs: Sequence[dict], cfg: dict) -> ActorSet:
    return ActorSet([make_actor(spec, cfg) for spec in specs])
