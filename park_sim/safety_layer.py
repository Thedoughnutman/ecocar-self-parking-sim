"""What a run adds around the safety monitor: actors, injected faults, the watchdog's inputs, the brake
request and the safety metrics. Sim glue, used by both run_closed_loop and run_search.

The watchdog runs on every run, and only ever reads stream ages. Everything else is on when
safety.enabled says so: `auto` means on for runs with actors or injected faults, so a plain run never
has a sensor return change what it does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .actors import ActorSet, TargetFrame
from .controller import ParkingController
from .geometry import polygon_distance
from .hybrid_astar import PlannedPath
from .localization import Localizer
from .safety import (RangeReading, SafetyConfig, SafetyMonitor, SafetyOutput, ScanReading, SensorSpec,
                     SteeringMonitor, StreamWatchdog)
from .scenario import Scenario
from .scripted_driver import EventScript, has_faults
from .sensors import load_sensor_params, make_lidar
from .sensors.base import Measurement, SensorSuite
from .sensors.lidar2d import Lidar2D, LidarMeasurement
from .sensors.line_camera import LineCameraMeasurement
from .sensors.scene import set_actor_polygons
from .sensors.ultrasonic import UltrasonicMeasurement, UltrasonicSensor, make_ultrasonic_sensors
from .supervisor import ControllerAction, SensorHealth, State, SupervisorConfig, SupervisorOutput
from .vehicle import Vehicle

SAFETY_METRICS = ("actor_collisions", "min_actor_distance_m", "stop_distance_after_detection_m", "time_paused_s",
                  "emergency_brakes", "obstacle_stops", "blocked_replans")
_WATCHED_STATES = (State.PLANNING, State.MANEUVERING, State.PAUSED)
_KIND_OF = {UltrasonicMeasurement: "ultrasonic", LidarMeasurement: "lidar", LineCameraMeasurement: "camera"}


@dataclass(frozen=True)
class BlockedReplan:
    t: float        # s, run clock
    kind: str       # "replan", "failed" (no path), "refused" (gear limit), "capped" (max_blocked_replans used)


@dataclass(frozen=True, eq=False)
class SafetySample:
    """One tick of what the monitor saw, for /safety."""
    t: float
    obstacle_in_path: bool
    emergency_brake: bool
    ttc_s: Optional[float]
    gap_m: Optional[float]
    stop_distance_m: float
    horizon_m: float
    brake_request: float
    sweep: np.ndarray       # (K, 3)
    hazards: np.ndarray     # (H, 2)


@dataclass
class SafetyInfo:
    """What a run with the monitor on adds to its RunResult."""
    outcome: str = ""           # completed | aborted | fault | incomplete | no_path
    outcome_reason: str = ""
    samples: List[SafetySample] = field(default_factory=list)
    actor_frames: List[tuple] = field(default_factory=list)    # per tick: (t, ((name, kind, polygon), ...))
    replans: List[BlockedReplan] = field(default_factory=list)
    plans: List[Tuple[float, PlannedPath]] = field(default_factory=list)    # after a replan: the plan before it, then each new one


def is_enabled(cfg: dict, actors: Optional[ActorSet], events: Sequence) -> bool:
    setting = cfg["safety"]["enabled"]
    if setting == "auto":
        return bool(actors) or has_faults(events)
    return bool(setting)


class _StreamFaults:
    """Applies sensor_dropout and sensor_frozen to what the sensors published."""

    def __init__(self):
        self._last: Dict[str, Measurement] = {}

    @staticmethod
    def _matches(names, sensor: str, kind: str) -> bool:
        return bool(names) and (sensor in names or kind in names)

    def apply(self, batch: Sequence[Measurement], dropped: Sequence[str], frozen: Sequence[str]) -> List[Measurement]:
        out = []
        for m in batch:
            kind = _KIND_OF.get(type(m), "")
            if self._matches(dropped, m.sensor, kind):
                continue
            if self._matches(frozen, m.sensor, kind) and m.sensor in self._last:
                out.append(replace(self._last[m.sensor], t=m.t))
                continue
            self._last[m.sensor] = m
            out.append(m)
        return out


def _names(value) -> Tuple[str, ...]:
    if not value:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


class SafetyLayer:
    def __init__(self, sc: Scenario, sup_cfg: SupervisorConfig, veh: Vehicle, loc: Localizer, script: EventScript,
                 actors: Optional[ActorSet], suite: Optional[SensorSuite], enabled: bool, sensor_seed: int = 0):
        self.sc, self.veh, self.loc, self.script = sc, veh, loc, script
        self.dt = sc.cfg["controller"]["dt"]
        self.enabled = enabled
        self.actors = actors if actors is not None else ActorSet()
        self.suite = suite                  # the run's own sensors
        self.info: Optional[SafetyInfo] = SafetyInfo() if enabled else None
        self.sup_cfg = sup_cfg
        self._faults = _StreamFaults()
        self._state = State.OFF
        self._last_output: Optional[SafetyOutput] = None
        self._emergency = False
        self._planned_speed = 0.0
        self._last_steer_cmd = 0.0
        self._delay_applied: Optional[float] = None
        self.monitor: Optional[SafetyMonitor] = None
        self.extra_suite: Optional[SensorSuite] = None
        self._steering: Optional[SteeringMonitor] = None
        self._replans_accepted = 0
        self._collisions, self._touching = 0, set()
        self._min_distance = math.inf
        self._paused_s = 0.0
        self._emergencies, self._stops = 0, 0
        self._stop_travel: Optional[float] = None
        self._stop_longest = math.nan
        self._was_blocked = False
        self._perception = [s.name for s in suite.sensors] if suite is not None else []
        streams = list(sup_cfg.sensor_timeouts_s) + self._perception
        if enabled:
            streams += self._start_monitor(sc, suite, sensor_seed)
        self.watchdog = StreamWatchdog(streams)
        if self.actors and sc.goal is not None:
            self.actors.set_target(TargetFrame.from_scenario(sc))

    def _start_monitor(self, sc: Scenario, suite: Optional[SensorSuite], seed: int) -> List[str]:
        """Starts the monitor, with sensors of its own if the run has none. Returns the new streams,
        which also go into the supervisor's timeouts: that check is the only stale check."""
        cfg = sc.cfg["safety"]
        wd = cfg["watchdog"]
        sensors = list(suite.sensors) if suite is not None else []
        if not any(isinstance(s, UltrasonicSensor) for s in sensors):
            config = load_sensor_params()
            own = list(make_ultrasonic_sensors(config["ultrasonic"]))
            if config["lidar"]["enabled"] and not any(isinstance(s, Lidar2D) for s in sensors):
                own.append(make_lidar(config["lidar"]))
            self.extra_suite = SensorSuite(own, seed)
            sensors += own
        ranged = [s for s in sensors if isinstance(s, UltrasonicSensor)]
        self.monitor = SafetyMonitor(
            SafetyConfig.from_params(cfg, sc.vehicle, self.sup_cfg.standstill_speed),
            [SensorSpec(s.name, s.mount.x, s.mount.y, s.mount.yaw, math.radians(s.cone_deg) / 2.0, s.range_min, s.range_max)
             for s in ranged])
        self._steering = SteeringMonitor(math.radians(wd["steer_error_deg"]), math.radians(wd["steer_stuck_rate_deg_s"]),
                                         wd["steer_fault_time_s"])
        ages = {s.name: wd["ultrasonic_max_age_s"] for s in ranged}
        ages.update({s.name: wd["lidar_max_age_s"] for s in sensors if isinstance(s, Lidar2D)})
        self.sup_cfg = replace(self.sup_cfg, sensor_timeouts_s={**self.sup_cfg.sensor_timeouts_s, **ages})
        return list(ages)

    # ---------- every tick, before the sensors ----------
    def begin_tick(self, t: float) -> None:
        """Faults come into force, then the actors move and the sensors see them."""
        if not self.enabled:
            return
        faults = self.script.faults(t)
        limit = faults.get("eps_saturation_deg")
        self.veh.steer_limit = math.radians(limit) if limit is not None else None
        self.veh.steer_stuck = bool(faults.get("steer_not_followed", False))
        delay = faults.get("pose_delay_s")
        if delay is not None and delay != self._delay_applied:
            self.loc.set_latency(delay)
            self._delay_applied = delay
        if self.actors:
            s = self.veh.state
            ego = self.sc.vehicle.footprint(s.x, s.y, s.yaw)
            self.actors.step(t, self.dt, ego, self._state.value)
            polygons = self.actors.polygons()
            set_actor_polygons(self.sc, polygons)
            self._score_actors(t, ego)

    def _score_actors(self, t: float, ego: np.ndarray) -> None:
        frame = []
        for actor in self.actors.actors:
            if not actor.visible:
                continue
            polygon = actor.polygon()
            frame.append((actor.name, actor.kind, polygon))
            d = polygon_distance(ego, polygon)
            self._min_distance = min(self._min_distance, d)
            if d <= 0.0 and actor.name not in self._touching:
                self._touching.add(actor.name)
                self._collisions += 1
            elif d > 0.0:
                self._touching.discard(actor.name)
        self.info.actor_frames.append((t, tuple(frame)))

    # ---------- sensors ----------
    def publish(self, t: float, s, batch: Sequence[Measurement]) -> Sequence[Measurement]:
        """What the run's sensors publish this tick, after faults. Everything also goes to the watchdog and
        the monitor, along with any sensors the monitor runs for itself."""
        if not self.enabled:
            self._heard(t, s, batch)
            return batch
        faults = self.script.faults(t)
        dropped, frozen = _names(faults.get("sensor_dropout")), _names(faults.get("sensor_frozen"))
        own = self._faults.apply(batch, dropped, frozen)
        extra = self._faults.apply(self.extra_suite.step(self.sc, s, t), dropped, frozen) if self.extra_suite else []
        self._heard(t, s, own, extra)
        self._feed_monitor(own, extra)
        return own

    def _heard(self, t: float, s, batch: Sequence[Measurement], extra: Sequence[Measurement] = ()) -> None:
        moving = abs(s.v) > self.sup_cfg.standstill_speed
        for m in list(batch) + list(extra):
            self.watchdog.heard(m.sensor, m.t, _value_of(m), moving)

    def _feed_monitor(self, own: Sequence[Measurement], extra: Sequence[Measurement]) -> None:
        e = self.loc.estimate
        pose = (e.x, e.y, e.yaw)
        readings, scan = [], None
        for m in list(own) + list(extra):
            if isinstance(m, UltrasonicMeasurement):
                readings.append(RangeReading(m.sensor, m.distance))
            elif isinstance(m, LidarMeasurement):
                scan = self._to_scan(m, pose)
        if readings or scan is not None:
            self.monitor.observe(pose, readings, scan)

    def _to_scan(self, m: LidarMeasurement, pose: Tuple[float, float, float]) -> ScanReading:
        suite = self.suite if self.suite is not None and m.sensor in self.suite.by_name else self.extra_suite
        mount = suite.by_name[m.sensor].mount
        c, s = math.cos(pose[2]), math.sin(pose[2])
        ox, oy = pose[0] + c * mount.x - s * mount.y, pose[1] + s * mount.x + c * mount.y
        yaw = pose[2] + mount.yaw
        points = m.points()
        cy, sy = math.cos(yaw), math.sin(yaw)
        world = np.column_stack([ox + cy * points[:, 0] - sy * points[:, 1], oy + sy * points[:, 0] + cy * points[:, 1]])
        return ScanReading(world, (ox, oy), m.angles + yaw, m.ranges)

    # ---------- the supervisor's inputs ----------
    def assess(self, t: float, s, ctrl: Optional[ParkingController]) -> bool:
        """obstacle_in_path for this tick. Only while the system has the car and a path to check."""
        if not self.enabled or ctrl is None or self._state not in _WATCHED_STATES:
            self._last_output = None
            return False
        e = self.loc.estimate
        look = self.monitor.look_ahead(s.v, self._planned_speed)
        output = self.monitor.check((e.x, e.y, e.yaw), s.v, ctrl.remaining_path(look, (e.x, e.y)), self._planned_speed)
        self._last_output = output
        if output.emergency_brake and not self._emergency:
            self._emergency = True
            self._emergencies += 1
        if output.obstacle_in_path and not self._was_blocked and abs(s.v) > self.sup_cfg.standstill_speed:
            self._stops += 1
            self._stop_travel = 0.0
        self._was_blocked = output.obstacle_in_path
        return output.obstacle_in_path

    def health(self, t: float) -> Tuple[SensorHealth, ...]:
        """The supervisor's sensors input, from what actually arrived."""
        faults = self.script.faults(t) if self.enabled else {}
        down = _names(faults.get("sensor_dropout")) + _names(faults.get("sensor_frozen"))
        w = self.watchdog
        if "wheel_speed" not in down:
            w.last["wheel_speed"] = t
        if "slot_perception" not in down:   # the newest thing the detector had to work with
            w.last["slot_perception"] = max((w.last[n] for n in self._perception), default=t)
        w.last["localization"] = self.loc.sample_time
        steering = ()
        if self._steering is not None and not self._steering.healthy:
            steering = (SensorHealth("steering", False, t),)
        return w.health(steering)

    def pose_alive(self, t: float) -> bool:
        """Whether the localization stream still updates this tick."""
        if not self.enabled:
            return True
        faults = self.script.faults(t)
        down = _names(faults.get("sensor_dropout")) + _names(faults.get("sensor_frozen"))
        return "localization" not in down and not ("wheel_speed" in down and self.loc.frame == "odom")

    # ---------- after the supervisor ----------
    def observe(self, t: float, out: SupervisorOutput) -> None:
        self._state = out.state
        self.script.observe(t, out.state)
        if self.enabled and out.state is State.PAUSED:
            self._paused_s += self.dt

    def set_target(self, sc_with_target: Scenario) -> None:
        """The stall the maneuver is going to, for actors scripted relative to it."""
        if self.actors:
            self.actors.set_target(TargetFrame.from_scenario(sc_with_target))

    def planning_scene(self, sc: Scenario) -> Scenario:
        """The lot as the planner knows it, plus the stationary actors' ground-truth footprints. Moving actors
        and objects not known_to_planner stay unknown to it."""
        known = sc.planner_view()
        polygons = self.actors.stationary_polygons() if self.enabled else []
        if not polygons:
            return known
        scene = replace(known, obstacles=list(known.obstacles) + polygons, _dist=None, _boxes=None, _probe=None)
        scene.build_distance_field()
        return scene

    def may_replan(self, t: float) -> bool:
        """Blocked replans have their own cap, not refinement's."""
        if self._replans_accepted < self.sc.cfg["safety"]["max_blocked_replans"]:
            return True
        self.info.replans.append(BlockedReplan(t, "capped"))
        return False

    def judge_replan(self, t: float, path: Optional[PlannedPath], ctrl: ParkingController,
                     current: Optional[PlannedPath] = None) -> Optional[ParkingController]:
        """A blocked replan's result: the controller to carry on with, or None to stay put. Same gear rule as
        refinement: a plan that would break the limit would abort the maneuver, so it is refused."""
        if path is None:
            self.info.replans.append(BlockedReplan(t, "failed"))
            return None
        if ctrl.gear_changes_with(path) > self.sc.cfg["supervisor"]["max_gear_changes"]:
            self.info.replans.append(BlockedReplan(t, "refused"))
            return None
        self._replans_accepted += 1
        self.info.replans.append(BlockedReplan(t, "replan"))
        if not self.info.plans and current is not None:
            self.info.plans.append((0.0, current))
        self.info.plans.append((t, path))
        return ctrl.continued_on(path)

    def brake_request(self, out: SupervisorOutput) -> float:
        """m/s^2 for the plant. A plain run's stops stay with the speed loop."""
        if not self.enabled:
            return 0.0
        p = self.sc.vehicle
        if self._emergency:
            return p.brake_emergency_decel
        return p.brake_service_decel if out.action is ControllerAction.STOP else 0.0

    # ---------- after the plant ----------
    def end_tick(self, t: float, s, steer_cmd: float, speed_cmd: float, brake: float, maneuvering: bool) -> None:
        if not self.enabled:
            return
        standing = abs(s.v) <= self.sup_cfg.standstill_speed
        # A stopped car looks as far as it would need to if it pulled away, so an obstacle that only just
        # let it stop doesn't count as clear, and the car doesn't creep up to it in small hops
        self._planned_speed = self.sc.vehicle.max_speed if standing else speed_cmd
        if self._emergency and standing:
            self._emergency = False
        if self._stop_travel is not None:
            self._stop_travel += abs(s.v) * self.dt
            if standing:
                self._stop_longest = max(self._stop_travel, self._stop_longest) if not math.isnan(self._stop_longest) else self._stop_travel
                self._stop_travel = None
        if maneuvering and self._steering is not None:
            self._steering.update(t, self.dt, steer_cmd, s.delta)
        if self._last_output is not None:
            o = self._last_output
            self.info.samples.append(SafetySample(t, o.obstacle_in_path, o.emergency_brake, o.ttc_s, o.gap_m,
                                                  o.stop_distance_m, o.horizon_m, brake, o.sweep, o.hazards))

    # ---------- at the end ----------
    def finish(self, res, planned: bool, ended: str = "") -> None:
        """Clears the scene and adds the safety results to `res`. A plain run's result is left as it was."""
        set_actor_polygons(self.sc, ())
        if not self.enabled:
            return
        info = self.info
        state = res.final_state
        if state is State.COMPLETED:
            info.outcome = "completed"
        elif not planned:
            info.outcome = "no_path"
        elif state is State.ABORTED:
            info.outcome = "aborted"
        elif state is State.FAULT:
            info.outcome = "fault"
        else:
            info.outcome = "incomplete"
        info.outcome_reason = res.reason or ended
        res.safety = info
        res.metrics.update(self.metrics())
        if res.checks:
            res.checks["actor_collision"] = self._collisions == 0

    def metrics(self) -> Dict[str, float]:
        return {
            "actor_collisions": float(self._collisions),
            "min_actor_distance_m": self._min_distance if math.isfinite(self._min_distance) else math.nan,
            "stop_distance_after_detection_m": self._stop_longest,
            "time_paused_s": self._paused_s,
            "emergency_brakes": float(self._emergencies),
            "obstacle_stops": float(self._stops),
            "blocked_replans": float(self._replans_accepted),
        }


def _value_of(m: Measurement):
    if isinstance(m, UltrasonicMeasurement):
        return m.distance
    if isinstance(m, LidarMeasurement):
        return m.ranges
    if isinstance(m, LineCameraMeasurement):
        return m.segments
    return None

