"""Closed-loop runs: driver -> supervisor -> plan, control, plant; then score.

run_closed_loop starts next to the target stall. run_search adds the drive down the aisle,
slot detection, selection and stopping before the same maneuver.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .controller import ParkingController
from .driver import DriverConfig, SearchDriver
from .geometry import wrap_angle
from .hybrid_astar import HybridAStar, PlannedPath, PlannerObserver, observing
from .actors import ActorSet
from .localization import Localizer, MapToOdom, Measurements, PoseEstimate, make_localizer
from .safety_layer import SafetyInfo, SafetyLayer, is_enabled
from .scenario import (Pose, Scenario, add_target_obstacles, goal_from_corners, make_aisle_lot,
                       scene_from_distance_field)
from .scripted_driver import EventScript, ScriptedDriver, ScriptedEvent
from .sensed_map import BodyOutline, RangeSensor, SensedMap, SensedMapConfig, SensedMapFrame
from .sensors import make_suite
from .sensors.base import Measurement, SensorSuite
from .sensors.line_camera import LineCamera, LineCameraMeasurement
from .sensors.ultrasonic import UltrasonicMeasurement, UltrasonicSensor
from .slot_detection import SlotDetector, make_detector
from .supervisor import (AbortReason, ControllerAction, ControllerStatus, ParkingSupervisor,
                         PlannerResult, PlannerStatus, SensorHealth, SlotCandidate, State,
                         SupervisorConfig, SupervisorInput, SupervisorOutput, Transition)
from .vehicle import Vehicle, VehicleParams, VehicleState


# Why a plan found nothing, for a person to read
PLAN_STALL_BLOCKED, PLAN_STALL_UNSEEN, PLAN_START_BLOCKED, PLAN_TIMED_OUT, PLAN_NO_ROUTE = "stall_blocked", "stall_unseen", "start_blocked", "timed_out", "no_route"
PLAN_NEAR_LIMIT = 0.95      # a plan that took this share of its limit ran out of time


@dataclass(frozen=True)
class PlanAttempt:
    t: float            # s, run clock
    slot_id: int
    time_s: float       # s, wall clock
    found: bool
    map_center: Tuple[float, float] = (math.nan, math.nan)   # the slot's center, in the map frame, at that moment
    why: str = ""       # if not found: one of the PLAN_ reasons


@dataclass(frozen=True)
class Refinement:
    """The selected slot's goal moved during the maneuver, and what was done about it."""
    t: float            # s, run clock
    kind: str           # "replan", "stop_adjust", or "replan_failed" / "replan_refused" (kept the old path)
    moved_m: float
    moved_deg: float


@dataclass
class SearchInfo:
    """What a search run adds to its RunResult."""
    scenario: Scenario                  # the lot, with the true chosen stall as target if there was one
    slots: Tuple[SlotCandidate, ...]    # everything the detector reported, in the order first seen
    slot_id: Optional[int] = None       # the slot the maneuver went to
    plan_attempts: List[PlanAttempt] = field(default_factory=list)
    refinements: List["Refinement"] = field(default_factory=list)
    plans: List[Tuple[float, PlannedPath]] = field(default_factory=list)   # every plan the car followed, with its time
    sensed_frames: List[SensedMapFrame] = field(default_factory=list)      # planner.map: sensed only


@dataclass
class RunResult:
    planned: Optional[PlannedPath]
    log: Dict[str, np.ndarray] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    checks: Dict[str, bool] = field(default_factory=dict)
    reason: str = ""
    transitions: List[Transition] = field(default_factory=list)
    final_state: State = State.OFF
    search: Optional[SearchInfo] = None
    measurements: List[Measurement] = field(default_factory=list)   # in time order; empty without sensors
    sensor_suite: Optional[SensorSuite] = None
    estimator: str = ""     # which localization estimator ran
    safety: Optional[SafetyInfo] = None     # only with the safety monitor on

    @property
    def success(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    def summary(self) -> str:
        if self.planned is None:
            return f"FAIL ({self.reason})"
        m = self.metrics
        flags = ", ".join(k for k, ok in self.checks.items() if not ok)
        return (f"{'PASS' if self.success else 'FAIL'}  lat {m['lateral_err']*100:+.1f} cm  "
                f"head {m['heading_err_deg']:+.2f} deg  depth {m['depth_err']*100:+.1f} cm  "
                f"clear {m['min_clearance']:.2f} m  gears {int(m['gear_changes'])}  "
                f"t {m['time_s']:.1f} s  vmax {m['max_speed_kph']:.1f} km/h"
                + (f"  [failed: {flags}]" if flags else "")
                + (f"  ({self.reason})" if self.reason else ""))


@dataclass(frozen=True, eq=False)
class RunTick:
    """One tick of a run as it happens, for an observer. `row` is the tick's log entry, one sample per column."""
    t: float
    row: Dict[str, tuple]
    true_pose: VehicleState
    slot_frame: object              # the pose that goes with true_pose to take a slot to the map frame
    slots: Tuple[SlotCandidate, ...]
    selected_slot: Optional[int]
    transitions: Sequence[Transition]
    path: Optional[PlannedPath]     # the plan being followed; None before there is one
    final: bool = False             # the log's last row
    failed_plan: Optional[PlanAttempt] = None   # the latest plan, if it found nothing


class RunObserver:
    """Watches a run as it happens, for display. It gets the run's own objects and must only read them: nothing
    here feeds back into the run. It may take as long as it likes between ticks, since the run has no clock of its
    own there; only `planner`, if set, runs inside the planner's time limit."""
    planner: Optional[PlannerObserver] = None

    def on_run_start(self, sc: Scenario, suite: Optional[SensorSuite], measurements: Sequence[Measurement],
                     safety: Optional[SafetyInfo], map_frames: Sequence[SensedMapFrame], search_flow: bool) -> None:
        """The lists fill up during the run. suite is the one whose measurements are logged, if any."""

    def on_tick(self, tick: RunTick) -> None:
        """After the tick's log entry is made and everything it decided is known."""


def run_closed_loop(sc: Scenario, noise_rng: Optional[np.random.Generator] = None,
                    pose_noise_xy: float = 0.0, pose_noise_yaw_deg: float = 0.0,
                    planner_cfg: Optional[dict] = None,
                    events: Sequence[ScriptedEvent] = (),
                    sensors: Optional[SensorSuite] = None,
                    localization_rng: Optional[np.random.Generator] = None,
                    actors: Optional[ActorSet] = None, sensor_seed: int = 0,
                    observer: Optional[RunObserver] = None) -> RunResult:
    """Run the feature from the first button press. The car waits at the start pose until
    the maneuver begins (driving to the slot is not modeled); `events` inject driver inputs,
    system events and faults (see scripted_driver.py). Metrics cover the maneuver, not the lead-up.
    `sensors` (one suite per run) are logged in RunResult.measurements. The safety monitor acts on them,
    if it is on (safety.enabled); without a suite it runs ultrasonics of its own, seeded by sensor_seed.
    The controller follows localization.estimator's pose; pose noise arguments add white noise to it.
    `observer` watches the run without changing it."""
    cfg = sc.cfg
    plans_on_sensed_map(cfg)    # checks the name. Whichever it is, a single-stall run plans on the lot it's given: it starts next to the stall, having seen nothing
    cc = cfg["controller"]
    dt = cc["dt"]
    veh = Vehicle(sc.vehicle, VehicleState(sc.start.x, sc.start.y, sc.start.yaw), dt=dt)
    rng = noise_rng if noise_rng is not None else np.random.default_rng(0)
    loc = make_localizer(cfg, sc.start, sc.vehicle, False, localization_rng, rng, pose_noise_xy, pose_noise_yaw_deg)
    sup_cfg = SupervisorConfig(**cfg["supervisor"])
    slot_id = cfg["scenario"]["target_stall"]
    driver = ScriptedDriver(slot_id, sup_cfg.sensor_timeouts_s, events)
    layer = SafetyLayer(sc, sup_cfg, veh, loc, driver.script, actors, sensors,
                        is_enabled(cfg, actors, events), sensor_seed)
    sup_cfg = layer.sup_cfg
    sup = ParkingSupervisor(sup_cfg)

    path: Optional[PlannedPath] = None
    ctrl: Optional[ParkingController] = None
    planner = PlannerResult()

    recorder = _Recorder()

    def record(steer_cmd: float, speed_cmd: float, with_lookahead: bool = True) -> None:
        recorder.add(t, veh.state, steer_cmd, speed_cmd, ctrl, out, t_maneuver, loc.estimate, with_lookahead)

    t = 0.0
    t_maneuver: Optional[float] = None
    status = ControllerStatus.IDLE
    measurements: List[Measurement] = []
    plan_observer = observer.planner if observer is not None else None

    def show_tick(final: bool = False) -> None:
        if observer is not None:
            observer.on_tick(RunTick(t, recorder.last_row(), veh.state, veh.state, (), None, sup.transitions, path, final))

    if observer is not None:
        observer.on_run_start(sc, sensors, measurements, layer.info, (), False)
    while t < cc["timeout_s"]:
        s = veh.state
        layer.begin_tick(t)
        published = layer.publish(t, s, sensors.step(sc, s, t) if sensors is not None else ())
        if sensors is not None:
            measurements.extend(published)
        obstacle = layer.assess(t, s, ctrl)
        out = sup.step(driver.input(t, s.v, status, planner, ctrl.gear_changes if ctrl else 0,
                                    layer.health(t), obstacle))
        driver.observe(out.state)
        layer.observe(t, out)

        if out.request_plan and planner.status is PlannerStatus.NONE:
            with observing(plan_observer):
                path = _plan_single_stall(layer.planning_scene(sc), s, loc, planner_cfg)
            if path is None:
                planner = PlannerResult(PlannerStatus.NOT_FOUND, out.plan_slot_id)
            else:
                planner = PlannerResult(PlannerStatus.FOUND, out.plan_slot_id)
                ctrl = ParkingController(path, sc.vehicle, cc, initial_gear=1)
        if out.request_replan and layer.may_replan(t):
            with observing(plan_observer):
                fresh = _plan_single_stall(layer.planning_scene(sc), s, loc, planner_cfg, cfg["safety"]["replan_time_limit_s"])
            taken = layer.judge_replan(t, fresh, ctrl, path)
            if taken is not None:
                path, ctrl = fresh, taken
                status = ControllerStatus.IDLE

        if planner.status is PlannerStatus.NOT_FOUND and out.state is State.SLOTS_AVAILABLE:
            break   # the scripted driver gives up rather than picking another slot
        if out.state is State.COMPLETED:
            break
        if out.state in (State.ABORTED, State.FAULT) and abs(s.v) <= sup_cfg.standstill_speed:
            break

        if out.action is ControllerAction.DRIVE:
            if t_maneuver is None:
                t_maneuver = 0.0
            steer_cmd, speed_cmd = ctrl.update(_control_state(loc, s), dt)
            status = _controller_status(ctrl)
        else:
            steer_cmd, speed_cmd = (ctrl.last_steer if ctrl else 0.0), 0.0
        record(steer_cmd, speed_cmd)
        brake = layer.brake_request(out)
        if t_maneuver is not None:
            veh.step(steer_cmd, speed_cmd, brake)
            t_maneuver += dt
        layer.end_tick(t, veh.state, steer_cmd, speed_cmd, brake, t_maneuver is not None)
        show_tick()
        if layer.pose_alive(t):
            loc.update(Measurements.from_state(veh.state, sc.vehicle.wheelbase, sc.vehicle.track), dt)
        t += dt
    else:
        out = sup.step(driver.input(t, veh.state.v, status, planner, ctrl.gear_changes if ctrl else 0,
                                    layer.health(t)))

    if ctrl is None:
        reason = "no path found" if planner.status is PlannerStatus.NOT_FOUND else "timeout"
        res = RunResult(None, reason=_failure_reason(sup) or reason, transitions=sup.transitions,
                        final_state=sup.state, measurements=measurements, sensor_suite=sensors,
                        estimator=loc.name)
        layer.finish(res, False)
        return res

    record(0.0, 0.0, with_lookahead=False)
    show_tick(final=True)
    log = recorder.arrays()

    res = RunResult(path, log, transitions=sup.transitions, final_state=sup.state,
                    measurements=measurements, sensor_suite=sensors, estimator=loc.name)
    _, maneuver = maneuver_slice(log)
    res.metrics = score(sc, maneuver, ctrl.gear_changes, ctrl.state == ctrl.DONE)
    res.metrics.update(_estimation_metrics(maneuver))
    res.checks = check(cfg["criteria"], res.metrics)
    if ctrl.state != ctrl.DONE:
        res.reason = _failure_reason(sup) or "timeout"
    layer.finish(res, True)
    return res


def run_search(sc: Scenario, driver_cfg: Optional[DriverConfig] = None,
               detector: Optional[SlotDetector] = None,
               noise_rng: Optional[np.random.Generator] = None,
               pose_noise_xy: float = 0.0, pose_noise_yaw_deg: float = 0.0,
               planner_cfg: Optional[dict] = None,
               sensors: Optional[SensorSuite] = None,
               localization_rng: Optional[np.random.Generator] = None,
               sensor_seed: int = 0, events: Sequence[ScriptedEvent] = (),
               actors: Optional[ActorSet] = None, observer: Optional[RunObserver] = None,
               driver=None) -> RunResult:
    """Drive -> search -> select -> stop -> park, on a lot from make_aisle_lot. `driver` takes the place of the
    scripted SearchDriver (driver_cfg is then unused): anything with its top_speed, start_speed, act, observe,
    gave_up and stop_requested, such as hmi.ManualDriver. Metrics cover the
    maneuver as in run_closed_loop, except min_clearance, which covers the whole run, and the
    search metrics. The final pose is scored against the true stall nearest the chosen slot.

    Without `detector`, slot_detection.detector in the config picks one. A detector with
    uses_estimated_pose gets the localization.estimator pose instead of the true one, and the
    controller always follows that pose. `sensors` (one suite per run) are logged in
    RunResult.measurements; a detector that needs a suite and isn't given one gets its own, seeded
    by sensor_seed, unlogged. `events` and `actors` as in run_closed_loop; the safety monitor acts only
    while the system has the car. The scripted driver keeps driving the aisle whatever is in it.
    `observer` watches the run without changing it."""
    cfg, search = sc.cfg, sc.cfg["search"]
    dt = cfg["controller"]["dt"]
    driver_cfg = driver_cfg or DriverConfig.from_params(search)
    suite = sensors
    on_sensed_map = plans_on_sensed_map(cfg)
    if detector is None:
        name = cfg["slot_detection"]["detector"]
        if name != "ideal" and suite is None:
            suite = make_suite(seed=sensor_seed)
        detector = make_detector(name, sc, suite)
    if on_sensed_map and suite is None:
        suite = make_suite(seed=sensor_seed)
    if driver is None:
        driver = SearchDriver(driver_cfg, sc.vehicle.wheelbase, sc.vehicle.max_steer, sc.aisle_y, sc.aisle_end_x)
    sup_cfg = SupervisorConfig(**cfg["supervisor"])
    rng = noise_rng if noise_rng is not None else np.random.default_rng(0)
    loc = make_localizer(cfg, sc.start, sc.vehicle, detector.uses_estimated_pose, localization_rng, rng,
                         pose_noise_xy, pose_noise_yaw_deg)
    in_odometry = loc.frame == "odom"
    slots_in_odometry = in_odometry and detector.uses_estimated_pose    # the ideal detector's are always in the map frame
    sensed: Optional[_SensedMapFeed] = None
    if on_sensed_map:
        if in_odometry and not slots_in_odometry:
            raise ValueError("planner.map: sensed needs the slots and the map in one frame: with the ideal "
                             "detector, use an estimator in the map frame (truth or gnss_ins)")
        sensed = _SensedMapFeed(cfg, suite, sc.vehicle)

    # vehicle.yaml's speed and decel limits are the parking controller's, not the car's
    driver_plant = replace(sc.vehicle, max_speed=driver.top_speed, decel=search["brake_decel"])
    veh = Vehicle(driver_plant, VehicleState(sc.start.x, sc.start.y, sc.start.yaw, v=driver.start_speed), dt=dt)
    script = EventScript(events)
    layer = SafetyLayer(sc, sup_cfg, veh, loc, script, actors, suite, is_enabled(cfg, actors, events), sensor_seed)
    sup_cfg = layer.sup_cfg
    sup = ParkingSupervisor(sup_cfg)

    path: Optional[PlannedPath] = None
    ctrl: Optional[ParkingController] = None
    planner = PlannerResult()
    attempts: List[PlanAttempt] = []
    plans: List[Tuple[float, PlannedPath]] = []
    plan_observer = observer.planner if observer is not None else None

    def plan_to(slot: SlotCandidate, true_pose, est, limit_s: float, slot_in_map: bool = False) -> Tuple[Optional[PlannedPath], float]:
        """The first plan, refinement replans and blocked replans all come through here, so all use one map."""
        with observing(plan_observer):
            if sensed is not None:
                return sensed.plan(t, slot, est, planner_cfg, limit_s)
            return _plan_in_frame(layer.planning_scene(sc), slot, true_pose, est, planner_cfg, limit_s, in_odometry, slot_in_map)

    refiner = _GoalRefiner(sc, planner_cfg, search["planning_time_limit_s"], in_odometry,
                           plan_fn=plan_to) if detector.uses_estimated_pose else None
    status = ControllerStatus.IDLE
    recorder = _Recorder()
    t, steer_hold, ended = 0.0, 0.0, ""
    t_maneuver: Optional[float] = None
    slots: Tuple[SlotCandidate, ...] = ()
    measurements: List[Measurement] = []
    offers: Dict[int, Tuple[float, np.ndarray]] = {}    # slot id: (time first offered, corners in the map frame)

    def show_tick(final: bool = False) -> None:
        if observer is not None:
            frame = loc.estimate if slots_in_odometry else veh.state
            shown = plans[-1][1] if plans else None
            failed = attempts[-1] if attempts and not attempts[-1].found else None
            observer.on_tick(RunTick(t, recorder.last_row(), veh.state, frame, slots, sup.selected_slot, sup.transitions, shown, final, failed))

    if observer is not None:
        observer.on_run_start(sc, sensors, measurements, layer.info, sensed.frames if sensed is not None else (), True)
    while t < search["timeout_s"]:
        s = veh.state
        layer.begin_tick(t)
        new = layer.publish(t, s, suite.step(sc, s, t) if suite is not None else [])
        if sensors is not None:
            measurements.extend(new)
        estimate = loc.estimate
        slot_frame = estimate if slots_in_odometry else s      # the same pose twice: no frame change
        if sensed is not None:
            sensed.observe(t, new, estimate)
        detector.observe(new)
        slots = detector.update(t, estimate if detector.uses_estimated_pose else s)
        for slot in slots:
            if slot.slot_id not in offers:
                offers[slot.slot_id] = (t, _corners_in_map(slot, s, slot_frame))
        act = driver.act(t, s.x, s.y, s.yaw, s.v, slots)
        obstacle = layer.assess(t, s, ctrl)
        if sensed is not None and ctrl is not None and sup.state in (State.MANEUVERING, State.PAUSED):
            obstacle = sensed.path_blocked(t, ctrl, estimate) or obstacle
        out = sup.step(script.apply(SupervisorInput(
            t=t, feature_button=act.feature_button, selected_slot_id=act.selected_slot_id,
            confirm_button=act.confirm_button, brake=act.brake, speed=s.v, slots=slots,
            steering_torque_nm=act.steering_torque_nm,
            planner=planner, controller=status, gear_changes=ctrl.gear_changes if ctrl else 0,
            obstacle_in_path=obstacle, sensors=layer.health(t))))
        driver.observe(out.state)
        layer.observe(t, out)
        detector.set_selected(sup.selected_slot if out.state not in (State.SEARCHING, State.SLOTS_AVAILABLE) else None)

        if out.state is State.SLOTS_AVAILABLE:
            planner = PlannerResult()
        if out.request_plan and planner.status is PlannerStatus.NONE:
            slot = next(c for c in slots if c.slot_id == out.plan_slot_id)
            path, took = plan_to(slot, s, estimate, search["planning_time_limit_s"], slot_in_map=not slots_in_odometry)
            cx, cy = np.mean(_corners_in_map(slot, s, slot_frame), axis=0)
            why = ""
            if path is None:
                limit = search["planning_time_limit_s"]
                why = (sensed.last_failure if sensed is not None else "") or (PLAN_TIMED_OUT if took >= limit * PLAN_NEAR_LIMIT else PLAN_NO_ROUTE)
            attempts.append(PlanAttempt(t, slot.slot_id, took, path is not None, (float(cx), float(cy)), why))
            if path is None:
                planner = PlannerResult(PlannerStatus.NOT_FOUND, slot.slot_id)
            else:
                planner = PlannerResult(PlannerStatus.FOUND, slot.slot_id)
                ctrl = ParkingController(path, sc.vehicle, cfg["controller"], initial_gear=1)
                ctrl.last_steer = s.delta
                plans.append((t, path))
                if refiner is not None:
                    refiner.start(slot)
                if layer.enabled:
                    layer.set_target(sc.with_target(sc.nearest_stall(*np.mean(_corners_in_map(slot, s, slot_frame), axis=0))))
        if out.request_replan and layer.enabled and layer.may_replan(t):    # with the monitor off, the map check can only stop and abort
            slot = next((c for c in slots if c.slot_id == sup.selected_slot), None)
            fresh = None
            if slot is not None:
                fresh, _ = plan_to(slot, s, estimate, cfg["safety"]["replan_time_limit_s"], slot_in_map=not slots_in_odometry)
            taken = layer.judge_replan(t, fresh, ctrl)
            if taken is not None:
                path, ctrl = fresh, taken
                plans.append((t, path))
                status = ControllerStatus.IDLE

        if out.state is State.COMPLETED:
            break
        if out.state in (State.ABORTED, State.FAULT) and abs(s.v) <= sup_cfg.standstill_speed:
            break
        if driver.stop_requested():
            ended = "stopped by the driver"
            break
        if out.state in (State.SEARCHING, State.SLOTS_AVAILABLE) and driver.gave_up(s.v, sup_cfg.standstill_speed):
            ended = "reached the end of the aisle"
            break

        if out.action is ControllerAction.NONE:
            veh.params = driver_plant
            steer_cmd, speed_cmd = act.steer_cmd, act.speed_cmd
        else:
            veh.params = sc.vehicle
            if out.action is ControllerAction.DRIVE:
                if t_maneuver is None:
                    t_maneuver = 0.0
                if refiner is not None:
                    replanned = refiner.step(t, slots, ctrl, s, estimate)
                    if replanned is not None:
                        path, ctrl = replanned
                        plans.append((t, path))
                steer_cmd, speed_cmd = ctrl.update(_control_state(loc, s), dt)
                status = _controller_status(ctrl)
            else:
                steer_cmd, speed_cmd = (ctrl.last_steer if ctrl else steer_hold), 0.0
        steer_hold = steer_cmd
        recorder.add(t, veh.state, steer_cmd, speed_cmd, ctrl, out, t_maneuver, loc.estimate)
        brake = layer.brake_request(out)
        veh.step(steer_cmd, speed_cmd, brake)
        layer.end_tick(t, veh.state, steer_cmd, speed_cmd, brake, t_maneuver is not None)
        show_tick()
        if layer.pose_alive(t):
            loc.update(Measurements.from_state(veh.state, sc.vehicle.wheelbase, sc.vehicle.track), dt)
        if t_maneuver is not None:
            t_maneuver += dt
        t += dt
    else:
        ended = "timeout"

    recorder.add(t, veh.state, 0.0, 0.0, ctrl, out, t_maneuver, loc.estimate, with_lookahead=False)
    show_tick(final=True)
    log = recorder.arrays()

    slot_frame = loc.estimate if slots_in_odometry else veh.state
    chosen = next((c for c in slots if c.slot_id == sup.selected_slot), None)
    target = sc
    if chosen is not None:
        target = sc.with_target(sc.nearest_stall(*np.mean(_corners_in_map(chosen, veh.state, slot_frame), axis=0)))
    res = RunResult(path, log, transitions=sup.transitions, final_state=sup.state,
                    search=SearchInfo(target, slots, chosen.slot_id if chosen else None, attempts,
                                      refiner.events if refiner is not None else [], plans),
                    measurements=measurements, sensor_suite=suite if sensors is not None else None,
                    estimator=loc.name)
    res.metrics = _search_metrics(log, sup.transitions, attempts, len(slots))
    res.metrics.update(_detection_metrics(sc, offers, chosen, veh.state, slot_frame, log, detector, res.search))
    if sensed is not None:
        res.search.sensed_frames = sensed.frames
        res.metrics.update(sensed.metrics(attempts))
    if ctrl is not None:
        _, maneuver = maneuver_slice(log)
        res.metrics.update(score(target, maneuver, ctrl.gear_changes, ctrl.state == ctrl.DONE, clearance_log=log))
        res.metrics.update(_estimation_metrics(maneuver))
        res.checks = check(cfg["criteria"], res.metrics)
        if ctrl.state != ctrl.DONE:
            res.reason = _failure_reason(sup) or ended or "timeout"
    elif attempts:
        res.reason = f"no path to slot {attempts[-1].slot_id}"
    else:
        res.reason = _failure_reason(sup) or ended or "no slot selected"
    layer.finish(res, ctrl is not None, ended)
    return res


def search_trial(vehicle: VehicleParams, params: dict, seed: int, pose_noise_xy: float = 0.0,
                 pose_noise_yaw_deg: float = 0.0, events: Sequence[ScriptedEvent] = (),
                 actors: Optional[ActorSet] = None,
                 target_obstacles: Sequence[dict] = (), sensors: Optional[SensorSuite] = None,
                 observer: Optional[RunObserver] = None, driver=None) -> Tuple[Scenario, RunResult, Dict[str, float]]:
    """One search Monte Carlo run: random occupancy, search speed and free slot.
    Returns the lot, the result and the values drawn. target_obstacles go around the chosen free slot
    (scenario.add_target_obstacles), and keep a sensor-based run's driver looking in that slot's row.
    `driver` replaces the scripted driver, as in run_search; the lot and every other draw stay the same."""
    rng = np.random.default_rng(seed)
    mc = params["search"]["monte_carlo"]
    occupancy = float(rng.uniform(*mc["occupancy_prob"]))
    sc = make_aisle_lot(vehicle, params, rng, occupancy_prob=occupancy)
    speed = float(rng.uniform(*mc["speed_kph"]))
    slot = int(rng.choice([st.slot_id for st in sc.stalls if not st.occupied]))
    stall = next(st for st in sc.stalls if st.slot_id == slot)
    if target_obstacles:
        sc = add_target_obstacles(sc, stall, target_obstacles, np.random.default_rng(30_000 + seed))
    cfg = replace(DriverConfig.from_params(params["search"]), speed_kph=speed)
    if params["slot_detection"]["detector"] == "ideal":
        cfg = replace(cfg, policy="select_by_index", slot_index=slot)
    else:   # slot ids are track numbers: aim for the same place, take what's on offer from there on
        cfg = replace(cfg, policy="select_first_after", after_x=stall.center[0] - params["scenario"]["stall_width"] / 2.0,
                      row_y=stall.center[1] if target_obstacles else cfg.row_y)
    res = run_search(sc, cfg, noise_rng=np.random.default_rng(10_000 + seed),
                     pose_noise_xy=pose_noise_xy, pose_noise_yaw_deg=pose_noise_yaw_deg,
                     localization_rng=np.random.default_rng(20_000 + seed), sensor_seed=seed,
                     events=events, actors=actors, sensors=sensors, observer=observer, driver=driver)
    return sc, res, {"occupancy": occupancy, "speed_kph": speed, "chosen_slot": slot}


def _plan_to_slot(sc: Scenario, slot: SlotCandidate, start: Pose, planner_cfg: Optional[dict],
                  limit_s: float) -> Tuple[Optional[PlannedPath], float]:
    """Plan from the car's pose to the slot as detected. Returns (path or None, wall-clock s);
    over limit_s counts as no path, including the planner's setup time."""
    t0 = time.perf_counter()
    goal = goal_from_corners(slot.corners, sc.vehicle, sc.cfg["scenario"]["maneuver"])
    planner = HybridAStar(replace(sc, goal=goal), planner_cfg)
    path = planner.plan(start, time_limit_s=max(limit_s - (time.perf_counter() - t0), 0.0))
    took = time.perf_counter() - t0
    return (path if took < limit_s else None), took


def _corners_in_map(slot: SlotCandidate, true_pose, estimate) -> np.ndarray:
    corners = np.array(slot.corners)
    x, y = MapToOdom(true_pose, estimate).to_map(corners[:, 0], corners[:, 1])
    return np.column_stack([x, y])


def _plan_in_frame(sc: Scenario, slot: SlotCandidate, true_pose, estimate, planner_cfg: Optional[dict],
                   limit_s: float, in_odometry: bool, slot_in_map: bool = False) -> Tuple[Optional[PlannedPath], float]:
    """Plan to a slot. An estimate in the map frame plans from where it says the car is. In the odometry
    frame, the car and the slot go into the map frame for the planner, because its obstacle map is the
    true lot, and the path comes back for the controller. slot_in_map: the slot is already there."""
    if not in_odometry:
        return _plan_to_slot(sc, slot, Pose(estimate.x, estimate.y, estimate.yaw), planner_cfg, limit_s)
    start = Pose(true_pose.x, true_pose.y, true_pose.yaw)
    if not slot_in_map:
        corners = _corners_in_map(slot, true_pose, estimate)
        slot = replace(slot, corners=tuple(map(tuple, corners.tolist())))
    path, took = _plan_to_slot(sc, slot, start, planner_cfg, limit_s)
    return (None if path is None else _path_in_odometry(path, true_pose, estimate)), took


def _path_in_odometry(path: PlannedPath, true_pose, estimate) -> PlannedPath:
    frame = MapToOdom(true_pose, estimate)
    x, y = frame.to_odom(path.x, path.y)
    return replace(path, x=x, y=y, yaw=path.yaw + frame.dyaw)


def plans_on_sensed_map(cfg: dict) -> bool:
    """Whether a search run plans from what its sensors saw. The ideal detector always plans on the true lot: it knows every stall at once, whether or not
    any sensor has seen it, so a sensed map would have no way to reach them. It's the ground-truth reference, and nothing else uses the true lot unless told to."""
    name = cfg["planner"].get("map", "true_lot")
    if name not in ("true_lot", "sensed"):
        raise ValueError(f"unknown planner.map {name!r}, expected true_lot or sensed")
    return name == "sensed" and cfg["slot_detection"]["detector"] != "ideal"


def _plan_on_sensed_map(sensed_map: SensedMap, slot: SlotCandidate, estimate, vehicle: VehicleParams, cfg: dict,
                        planner_cfg: Optional[dict], limit_s: float, why: Optional[List[str]] = None) -> Tuple[Optional[PlannedPath], float]:
    """Plan from the estimated pose to the slot as detected, around what the map holds. All three are in the
    estimate's frame, so no frame change, and neither the lot nor the true pose comes in here. Returns
    (path or None, wall-clock s); over limit_s counts as no path. Unlike the search, which stops at the staging
    pose, this also checks the final straight on the map, so a stall with something seen in it fails at once."""
    t0 = time.perf_counter()
    goal = goal_from_corners(slot.corners, vehicle, cfg["scenario"]["maneuver"])
    sensed_map.assume_slot(slot.corners)
    start = Pose(estimate.x, estimate.y, estimate.yaw)
    field, bounds = sensed_map.distance_field([(start.x, start.y), (goal.x, goal.y)])
    scene = scene_from_distance_field(field, bounds, sensed_map.resolution, goal, start, cfg, vehicle)
    planner_settings = planner_cfg if planner_cfg is not None else cfg["planner"]
    straight = np.linspace(0.0, planner_settings.get("final_straight", 0.0), 8)
    approach = -1.0 if cfg["scenario"]["maneuver"] == "reverse_in" else 1.0
    xs = goal.x - approach * straight * math.cos(goal.yaw)
    ys = goal.y - approach * straight * math.sin(goal.yaw)
    margin = planner_settings["collision_margin"]
    if not scene.collision_free(xs, ys, np.full(len(xs), goal.yaw), margin):
        if why is not None:     # the map blocks both what it has seen and what it hasn't: these are very different things to tell a person
            sweep = np.column_stack([xs, ys, np.full(len(xs), goal.yaw)])
            why.append(PLAN_STALL_BLOCKED if sensed_map.occupied_in_sweep(sweep, margin, margin) else PLAN_STALL_UNSEEN)
        return None, time.perf_counter() - t0
    planner = HybridAStar(scene, planner_cfg)
    path = planner.plan(start, time_limit_s=max(limit_s - (time.perf_counter() - t0), 0.0))
    took = time.perf_counter() - t0
    if why is not None and (path is None or took >= limit_s):
        if not scene.collision_free(np.array([start.x]), np.array([start.y]), np.array([start.yaw]), margin):
            why.append(PLAN_START_BLOCKED)
        else:
            why.append(PLAN_TIMED_OUT if took >= limit_s * PLAN_NEAR_LIMIT else PLAN_NO_ROUTE)
    return (path if took < limit_s else None), took


class _SensedMapFeed:
    """Passes one run's sensor measurements and estimated pose to its SensedMap. Lidar isn't used. Reads only what
    a measurement would carry on the car, which leaves out a camera measurement's truth_line."""

    def __init__(self, cfg: dict, suite: SensorSuite, vehicle: VehicleParams):
        settings = cfg["sensed_map"]
        ranged = [RangeSensor(s.name, s.mount.x, s.mount.y, s.mount.yaw, math.radians(s.cone_deg) / 2.0, s.range_min, s.range_max)
                  for s in suite.sensors if isinstance(s, UltrasonicSensor)]
        if not ranged:
            raise ValueError("planner.map: sensed needs ultrasonic sensors in the suite")
        cameras = {s.name: (s.mount.x, s.mount.y) for s in suite.sensors if isinstance(s, LineCamera)}
        body = BodyOutline(-vehicle.rear_overhang, vehicle.wheelbase + vehicle.front_overhang, vehicle.width / 2.0)
        self.map = SensedMap(SensedMapConfig.from_params(settings), ranged, cameras, body)
        self.cfg, self.vehicle = cfg, vehicle
        self.period = settings["snapshot_period_s"]
        self.frames: List[SensedMapFrame] = []
        self._next_frame = 0.0
        self.check = settings["path_check"]
        self._next_check = 0.0
        self._blocked = False
        self._seen_in_a_row = 0
        self.path_stops = 0     # times the check went from clear to blocked
        self.last_failure = ""  # why the last plan found nothing: one of the PLAN_ reasons

    def path_blocked(self, t: float, ctrl: ParkingController, estimate) -> bool:
        """Whether the map now has something on the next stretch of the path being driven. A plan is only as good as the map
        it was made on, and the map keeps filling in. The answer goes to the supervisor as obstacle_in_path, so it pauses
        the car, asks for a replan and aborts just as it would for the monitor."""
        check = self.check
        if not check["enabled"]:
            return False
        if t + 1e-9 >= self._next_check:
            self._next_check = t + check["period_s"]
            path = ctrl.remaining_path(check["horizon_m"], (estimate.x, estimate.y))
            if len(path):
                along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path[:, :2], axis=0).T))])
                picks = np.unique(np.searchsorted(along, np.arange(0.0, along[-1] + check["sweep_step_m"], check["sweep_step_m"]))).clip(max=len(path) - 1)
                path = path[picks]
            cells = self.map.occupied_in_sweep(path, check["lateral_margin_m"], check["end_margin_m"], check["min_log_odds"])
            self._seen_in_a_row = self._seen_in_a_row + 1 if cells >= check["min_cells"] else 0
            blocked = self._seen_in_a_row >= check["confirm_checks"]
            self.path_stops += blocked and not self._blocked
            self._blocked = blocked
        return self._blocked

    def observe(self, t: float, measurements: Sequence[Measurement], estimate) -> None:
        self.map.update_body(estimate)
        for m in measurements:
            if isinstance(m, UltrasonicMeasurement):
                self.map.add_range(m.sensor, m.distance, estimate)
            elif isinstance(m, LineCameraMeasurement):
                self.map.add_lines(m.sensor, m.segments, estimate)
        if t + 1e-9 >= self._next_frame:
            self._next_frame = t + self.period
            self._keep_frame(t)

    def plan(self, t: float, slot: SlotCandidate, estimate, planner_cfg: Optional[dict],
             limit_s: float) -> Tuple[Optional[PlannedPath], float]:
        why: List[str] = []
        result = _plan_on_sensed_map(self.map, slot, estimate, self.vehicle, self.cfg, planner_cfg, limit_s, why)
        self.last_failure = why[0] if why else ""
        self._keep_frame(t)     # what this plan saw, with the slot it assumed
        return result

    def _keep_frame(self, t: float) -> None:
        frame = self.map.frame(t)
        if frame is not None:
            self.frames.append(frame)

    def metrics(self, attempts: Sequence["PlanAttempt"]) -> Dict[str, float]:
        area = self.map.resolution ** 2
        counts = self.map.counts()
        return {"map_free_m2": counts["free"] * area, "map_occupied_m2": counts["occupied"] * area,
                "map_assumed_m2": counts["assumed"] * area, "map_path_stops": float(self.path_stops),
                "plans_failed": float(sum(1 for a in attempts if not a.found)),
                "planning_time_max_s": max((a.time_s for a in attempts), default=math.nan)}


def _plan_single_stall(sc: Scenario, true_pose, loc: Localizer, planner_cfg: Optional[dict],
                       limit_s: Optional[float] = None) -> Optional[PlannedPath]:
    """The single-stall plan, with the frames handled as in _plan_in_frame."""
    planner = HybridAStar(sc, planner_cfg)
    e = loc.estimate
    if loc.frame == "odom":
        path = planner.plan(Pose(true_pose.x, true_pose.y, true_pose.yaw), time_limit_s=limit_s)
        return None if path is None else _path_in_odometry(path, true_pose, e)
    return planner.plan(Pose(e.x, e.y, e.yaw), time_limit_s=limit_s)


def _control_state(loc: Localizer, s: VehicleState) -> VehicleState:
    """What the controller sees: the estimated pose, with speed and steering angle from the car."""
    e = loc.for_control()
    return VehicleState(e.x, e.y, e.yaw, s.v, s.delta)


class _GoalRefiner:
    """Watches the selected slot during the maneuver. If its goal moves past the threshold, replans
    from the current pose; inside the final straight, only moves the stop point."""

    def __init__(self, sc: Scenario, planner_cfg: Optional[dict], limit_s: float, in_odometry: bool = True,
                 plan_fn=None):
        self.sc, self.planner_cfg, self.limit_s, self.in_odometry = sc, planner_cfg, limit_s, in_odometry
        self.plan_fn = plan_fn      # plan_fn(slot, true pose, estimate, limit_s); default: plan on the lot itself
        self.cfg = sc.cfg["slot_detection"]["refinement"]
        self.final_straight = sc.cfg["planner"].get("final_straight", 0.0)
        self.events: List[Refinement] = []
        self._slot_id: Optional[int] = None
        self._goal: Optional[Pose] = None
        self._last_replan = -math.inf
        self._replans = 0

    def _goal_of(self, slot: SlotCandidate) -> Pose:
        return goal_from_corners(slot.corners, self.sc.vehicle, self.sc.cfg["scenario"]["maneuver"])

    def start(self, slot: SlotCandidate) -> None:
        self._slot_id, self._goal = slot.slot_id, self._goal_of(slot)

    def step(self, t: float, slots: Sequence[SlotCandidate], ctrl: ParkingController, true_pose,
             estimate) -> Optional[Tuple[PlannedPath, ParkingController]]:
        slot = next((c for c in slots if c.slot_id == self._slot_id), None)
        if slot is None or ctrl.state == ctrl.DONE:
            return None
        goal = self._goal_of(slot)
        moved_m = math.hypot(goal.x - self._goal.x, goal.y - self._goal.y)
        moved_deg = abs(math.degrees(wrap_angle(goal.yaw - self._goal.yaw)))
        if moved_m <= self.cfg["replan_threshold_m"] and moved_deg <= self.cfg["replan_threshold_deg"]:
            return None
        if ctrl.distance_left_is_within(self.final_straight):
            seg = ctrl.segments[-1]
            along = (goal.x - self._goal.x) * math.cos(seg.yaw[-1]) + (goal.y - self._goal.y) * math.sin(seg.yaw[-1])
            ctrl.move_final_stop(along * seg.direction)
            self._goal = goal
            self.events.append(Refinement(t, "stop_adjust", moved_m, moved_deg))
            return None
        if self._replans >= self.cfg["max_replans"] or t - self._last_replan < self.cfg["min_replan_interval_s"]:
            return None
        self._last_replan = t
        if self.plan_fn is not None:
            path, _ = self.plan_fn(slot, true_pose, estimate, self.limit_s)
        else:
            path, _ = _plan_in_frame(self.sc, slot, true_pose, estimate, self.planner_cfg, self.limit_s, self.in_odometry)
        if path is None:
            self.events.append(Refinement(t, "replan_failed", moved_m, moved_deg))
            return None
        if ctrl.gear_changes_with(path) > self.sc.cfg["supervisor"]["max_gear_changes"]:
            self.events.append(Refinement(t, "replan_refused", moved_m, moved_deg))    # it would abort the maneuver
            return None
        self._replans += 1
        self._goal = goal
        self.events.append(Refinement(t, "replan", moved_m, moved_deg))
        return path, ctrl.continued_on(path)


def _search_metrics(log: Dict[str, np.ndarray], transitions: Sequence[Transition],
                    attempts: Sequence[PlanAttempt], n_slots: int) -> Dict[str, float]:
    """Times are nan for phases the run did not reach. After a failed plan the latest selection
    and confirm count."""
    def last_entry(state: State) -> float:
        return next((tr.t for tr in reversed(transitions) if tr.to_state is state), math.nan)

    selected, stopped = last_entry(State.SLOT_SELECTED), last_entry(State.WAIT_CONFIRM)
    confirmed, parked = last_entry(State.PLANNING), last_entry(State.COMPLETED)
    searching = last_entry(State.SEARCHING)
    t, x, y = log["t"], log["x"], log["y"]
    a = np.searchsorted(t, searching - 1e-9) if not math.isnan(searching) else len(t)
    b = np.searchsorted(t, selected + 1e-9) if not math.isnan(selected) else len(t)
    return {
        "search_distance_m": float(np.sum(np.hypot(np.diff(x[a:b]), np.diff(y[a:b])))),
        "selection_to_stop_s": stopped - selected,
        "confirm_to_parked_s": parked - confirmed,
        "total_time_s": float(t[-1]),
        "planning_time_s": attempts[-1].time_s if attempts else math.nan,
        "slots_detected": float(n_slots),
    }


def _detection_metrics(sc: Scenario, offers: Dict[int, Tuple[float, np.ndarray]], chosen: Optional[SlotCandidate],
                       true_pose, estimate, log: Dict[str, np.ndarray], detector: SlotDetector,
                       search: SearchInfo) -> Dict[str, float]:
    """How well the slots on offer match the true lot. Each slot is compared when first offered,
    with the odometry drift of that moment removed. A slot within half a stall width of the center
    of a free stall is a match; anything else on offer is a false positive."""
    match = sc.cfg["scenario"]["stall_width"] / 2.0
    free = [st for st in sc.stalls if not st.occupied]
    errors, false_positives, matched = [], 0, set()
    for _, corners in offers.values():
        stall = min(free, key=lambda st: math.dist(st.center, corners.mean(axis=0)))
        if math.dist(stall.center, corners.mean(axis=0)) > match:
            false_positives += 1
            continue
        matched.add(stall.slot_id)
        errors.append(float(np.mean(np.hypot(*(corners - np.array(stall.corners)).T))))
    far = float(log["x"].max()) - sc.cfg["slot_detection"]["pass_margin"] - sc.cfg["scenario"]["stall_width"]
    passed = [st for st in free if max(c[0] for c in st.corners) <= far]
    times = detector.confirm_times()
    selected_error = math.nan
    if chosen is not None:
        corners = _corners_in_map(chosen, true_pose, estimate)
        stall = search.scenario.nearest_stall(*corners.mean(axis=0))
        selected_error = float(np.mean(np.hypot(*(corners - np.array(stall.corners)).T)))
    return {
        "slot_corner_error_m": float(np.mean(errors)) if errors else math.nan,
        "selected_corner_error_m": selected_error,
        "time_to_confirm_s": float(np.mean(times)) if times else math.nan,
        "slots_offered": float(len(offers)),
        "slots_false_positive": float(false_positives),
        "free_stalls_passed": float(len(passed)),
        "free_stalls_missed": float(sum(1 for st in passed if st.slot_id not in matched)),
        "replans": float(sum(1 for r in search.refinements if r.kind == "replan")),
        "stop_adjustments": float(sum(1 for r in search.refinements if r.kind == "stop_adjust")),
    }


class _Recorder:
    """Per-tick log of a run, as columns."""
    NUMERIC = ("t", "x", "y", "yaw", "v", "delta", "steer_cmd", "speed_cmd", "gear",
               "lookahead_x", "lookahead_y", "seg_i", "dist_to_go", "seg_dist_to_go", "t_maneuver",
               "sup_must_shift", "sup_request_park",
               "est_x", "est_y", "est_yaw", "est_sigma_xy", "est_sigma_yaw")
    LABELS = ("ctrl_state", "sup_state", "sup_action", "sup_abort_reason", "sup_hmi")

    def __init__(self):
        self.rows, self.labels = [], []

    def add(self, t: float, s: VehicleState, steer_cmd: float, speed_cmd: float,
            ctrl: Optional[ParkingController], out: SupervisorOutput,
            t_maneuver: Optional[float], estimate: PoseEstimate, with_lookahead: bool = True) -> None:
        self.rows.append((t, s.x, s.y, s.yaw, s.v, s.delta, steer_cmd, speed_cmd,
                          ctrl.gear if ctrl else 1, *_telemetry(ctrl, with_lookahead),
                          t_maneuver if t_maneuver is not None else math.nan,
                          out.driver_must_shift, out.request_park,
                          estimate.x, estimate.y, estimate.yaw, estimate.sigma_xy, estimate.sigma_yaw))
        self.labels.append((ctrl.state if ctrl else "idle", out.state.value, out.action.value,
                            out.abort_reason.value, out.hmi_message))

    def last_row(self) -> Dict[str, tuple]:
        """The newest entry, as columns of one sample, so a reader of arrays() can read it the same way."""
        row, labels = self.rows[-1], self.labels[-1]
        return {**{k: (row[i],) for i, k in enumerate(self.NUMERIC)},
                **{k: (labels[i],) for i, k in enumerate(self.LABELS)}}

    def arrays(self) -> Dict[str, np.ndarray]:
        arr = np.array(self.rows, dtype=float)
        log = {k: arr[:, i] for i, k in enumerate(self.NUMERIC)}
        for i, k in enumerate(self.LABELS):
            log[k] = np.array([row[i] for row in self.labels])
        return log


def _telemetry(ctrl: Optional[ParkingController], with_lookahead: bool) -> Tuple[float, ...]:
    if ctrl is None:
        return (math.nan, math.nan, 0, math.nan, math.nan)
    la = ctrl.lookahead if with_lookahead and ctrl.lookahead is not None else (math.nan, math.nan)
    return (*la, min(ctrl.seg_i, len(ctrl.segments) - 1), ctrl.dist_to_go, ctrl.seg_dist_to_go)


def _estimation_metrics(maneuver: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Estimated against true pose over the maneuver. These errors count from the start of the run,
    which for odometry includes the drive down the aisle; est_drift_final_m counts only what the
    maneuver added, with the error at its start removed."""
    m = maneuver
    xy = np.hypot(m["est_x"] - m["x"], m["est_y"] - m["y"])
    yaw = np.abs(np.degrees((m["est_yaw"] - m["yaw"] + math.pi) % (2.0 * math.pi) - math.pi))
    at_start = MapToOdom(Pose(m["x"][0], m["y"][0], m["yaw"][0]), Pose(m["est_x"][0], m["est_y"][0], m["est_yaw"][0]))
    expected_x, expected_y = at_start.to_odom(m["x"][-1], m["y"][-1])
    return {"est_err_final_m": float(xy[-1]), "est_err_max_m": float(xy.max()),
            "est_yaw_err_final_deg": float(yaw[-1]), "est_yaw_err_max_deg": float(yaw.max()),
            "est_drift_final_m": float(math.hypot(expected_x - m["est_x"][-1], expected_y - m["est_y"][-1]))}


def _controller_status(ctrl: ParkingController) -> ControllerStatus:
    if ctrl.state == ctrl.DONE:
        return ControllerStatus.FINISHED
    if ctrl.state == ctrl.SHIFTING:
        return ControllerStatus.SEGMENT_DONE
    return ControllerStatus.DRIVING


def _failure_reason(sup: ParkingSupervisor) -> str:
    if sup.abort_reason is AbortReason.NONE:
        return ""
    label = "fault" if sup.state is State.FAULT else "aborted"
    return f"{label}: {sup.abort_reason.value}"


def maneuver_slice(log: Dict[str, np.ndarray]) -> Tuple[int, Dict[str, np.ndarray]]:
    """(first index, log from there on) for the maneuver, with "t" on the maneuver clock.
    A run that never started driving gives its last sample at t = 0."""
    started = np.flatnonzero(~np.isnan(log["t_maneuver"]))
    if len(started) == 0:
        start = len(log["t"]) - 1
        part = {k: v[start:] for k, v in log.items()}
        part["t"] = np.zeros(1)
        return start, part
    start = int(started[0])
    part = {k: v[start:] for k, v in log.items()}
    part["t"] = part["t_maneuver"]
    return start, part


def clearance_samples(sc: Scenario, log: Dict[str, np.ndarray]):
    """Exact body-to-obstacle clearance at every 5th sample plus the last: (indices, meters)."""
    n = len(log["x"])
    idx = np.arange(0, n, 5).tolist() + [n - 1]
    return idx, [sc.clearance_exact(log["x"][i], log["y"][i], log["yaw"][i]) for i in idx]


def score(sc: Scenario, log: Dict[str, np.ndarray], gear_changes: int, completed: bool,
          clearance_log: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
    """clearance_log: the samples to take min_clearance over, if not just `log`."""
    v = sc.vehicle
    x, y, yaw = log["x"][-1], log["y"][-1], log["yaw"][-1]
    off = v.center_offset()
    cx, cy = x + off * math.cos(yaw), y + off * math.sin(yaw)
    scx, scy = sc.stall_center

    # Minimum exact clearance along the executed trajectory
    min_clear = min(clearance_samples(sc, log if clearance_log is None else clearance_log)[1])

    return {
        "completed": float(completed),
        "lateral_err": cx - scx,
        "depth_err": cy - scy,
        "heading_err_deg": math.degrees(wrap_angle(yaw - sc.goal.yaw)),
        "min_clearance": min_clear,
        "gear_changes": float(gear_changes),
        "time_s": float(log["t"][-1]),
        "max_speed_kph": float(np.max(np.abs(log["v"])) * 3.6),
        "path_length_m": float(np.sum(np.hypot(np.diff(log["x"]), np.diff(log["y"])))),
    }


def check(crit: dict, m: Dict[str, float]) -> Dict[str, bool]:
    return {
        "completed": m["completed"] > 0.5,
        "lateral": abs(m["lateral_err"]) <= crit["lateral_tol"],
        "heading": abs(m["heading_err_deg"]) <= crit["heading_tol_deg"],
        "depth": abs(m["depth_err"]) <= crit["depth_tol"],
        "clearance": m["min_clearance"] >= crit["min_clearance"],
        "gear_changes": m["gear_changes"] <= crit["max_gear_changes"],
        "speed": m["max_speed_kph"] <= crit["max_speed_kph"],
        "time": m["time_s"] <= crit["max_time_s"],
    }
