"""Parking feature state machine: takes driver inputs and system events, decides what
the planner and controller do.

Standard library only, with plain dataclasses in and out, so this can become the
parking_supervisor ROS 2 node without knowing anything about the sim. Transitions and
rules are described in docs/supervisor.md. Speeds on the input are m/s (signed); the
speed limits in the config are km/h, like the pass criteria.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class State(Enum):
    OFF = "OFF"
    SEARCHING = "SEARCHING"
    SLOTS_AVAILABLE = "SLOTS_AVAILABLE"
    SLOT_SELECTED = "SLOT_SELECTED"
    WAIT_CONFIRM = "WAIT_CONFIRM"
    PLANNING = "PLANNING"
    MANEUVERING = "MANEUVERING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"
    FAULT = "FAULT"


class ControllerAction(Enum):
    NONE = "none"     # the driver has the car
    HOLD = "hold"     # keep the car stationary
    DRIVE = "drive"   # follow the path
    STOP = "stop"     # bring the car to a stop


class ControllerStatus(Enum):
    IDLE = "idle"                 # no path being followed
    DRIVING = "driving"
    SEGMENT_DONE = "segment_done"  # stopped at a cusp, changing gear
    FINISHED = "finished"


class PlannerStatus(Enum):
    NONE = "none"          # nothing to report
    FOUND = "found"
    NOT_FOUND = "not_found"


class AbortReason(Enum):
    NONE = "none"
    STEERING_OVERRIDE = "steering_override"
    ACCELERATOR_OVERRIDE = "accelerator_override"
    GEAR_REQUEST = "gear_request"
    DOOR_OPEN = "door_open"
    SEATBELT_OFF = "seatbelt_off"
    SENSOR_STALE = "sensor_stale"
    OVERSPEED = "overspeed"
    MANEUVER_TIMEOUT = "maneuver_timeout"
    TOO_MANY_GEAR_CHANGES = "too_many_gear_changes"
    DRIVER_CANCEL = "driver_cancel"
    OBSTACLE_TIMEOUT = "obstacle_timeout"
    SENSOR_UNHEALTHY = "sensor_unhealthy"   # reason for FAULT, not ABORTED


@dataclass(frozen=True)
class SlotCandidate:
    """A parking slot as a detector reports it. Coordinates are in the map frame."""
    slot_id: int
    confirmed: bool = True
    corners: Tuple[Tuple[float, float], ...] = ()   # the first two are the opening (aisle side)
    entry_side: str = ""                            # map direction out of the slot: "+y" or "-y"
    slot_type: str = ""                             # e.g. "perpendicular"
    occupied: bool = False
    confidence: float = 1.0                         # 0-1
    first_seen_t: float = 0.0                       # s


@dataclass(frozen=True)
class SensorHealth:
    name: str
    healthy: bool
    last_update_s: float


@dataclass(frozen=True)
class PlannerResult:
    status: PlannerStatus = PlannerStatus.NONE
    slot_id: Optional[int] = None   # a result for any other slot than the selected one is ignored


@dataclass(frozen=True)
class SupervisorInput:
    t: float                                        # s
    feature_button: bool = False                    # acts on the press, not while held
    selected_slot_id: Optional[int] = None          # acts when given; None does not deselect
    confirm_button: bool = False
    steering_torque_nm: float = 0.0                 # driver torque, either sign
    brake: float = 0.0                              # pedal, 0-1
    accelerator: float = 0.0                        # pedal, 0-1
    gear_request: Optional[int] = None              # None = driver is not asking for a gear
    speed: float = 0.0                              # m/s, signed
    door_open: bool = False
    seatbelt_on: bool = True
    slots: Tuple[SlotCandidate, ...] = ()
    planner: PlannerResult = PlannerResult()
    controller: ControllerStatus = ControllerStatus.IDLE
    gear_changes: int = 0                           # gear changes so far in this maneuver
    obstacle_in_path: bool = False
    sensors: Tuple[SensorHealth, ...] = ()


@dataclass(frozen=True)
class SupervisorOutput:
    state: State
    action: ControllerAction
    hmi_message: str
    abort_reason: AbortReason
    driver_must_shift: bool
    request_plan: bool = False                      # plan to plan_slot_id
    plan_slot_id: Optional[int] = None
    request_park: bool = False                      # engage park / EPB
    request_replan: bool = False                    # blocked for replan_after_s: try another way to the slot


@dataclass(frozen=True)
class Transition:
    t: float
    from_state: State
    to_state: State
    reason: str


@dataclass(frozen=True)
class SupervisorConfig:
    search_speed_max: float         # km/h
    maneuver_speed_max: float       # km/h
    standstill_speed: float         # m/s
    resume_delay_s: float
    driver_override_nm: float
    accelerator_override: float     # pedal, 0-1
    brake_pressed: float            # pedal, 0-1
    max_time_s: float
    max_gear_changes: int
    planning_timeout_s: float
    replan_after_s: float
    abort_after_s: float
    sensor_timeouts_s: Dict[str, float] = field(default_factory=dict)


_ACTIVE = (State.SEARCHING, State.SLOTS_AVAILABLE, State.SLOT_SELECTED, State.WAIT_CONFIRM)
_ARMED = (State.PLANNING, State.MANEUVERING, State.PAUSED)   # the system has the car


class ParkingSupervisor:
    """Call step() once per tick. At most one transition happens per tick."""

    def __init__(self, cfg: SupervisorConfig):
        self.cfg = cfg
        self.state = State.OFF
        self.transitions: List[Transition] = []
        self.abort_reason = AbortReason.NONE
        self.selected_slot: Optional[int] = None
        self._notice = ""                 # message that outlives its transition
        self._state_since = 0.0
        self._maneuver_start: Optional[float] = None
        self._clear_since: Optional[float] = None
        self._blocked_since: Optional[float] = None     # an obstacle holding the car, until it has properly cleared
        self._last_replan_request = 0.0
        self._replan_now = False
        self._had_control = False
        self._feature_was_down = False
        self._confirm_was_down = False

    # ---------- public ----------
    def step(self, u: SupervisorInput) -> SupervisorOutput:
        feature_pressed = u.feature_button and not self._feature_was_down
        confirm_pressed = u.confirm_button and not self._confirm_was_down
        self._feature_was_down = u.feature_button
        self._confirm_was_down = u.confirm_button
        self._replan_now = False

        if self.state in _ARMED:
            reason, detail = self._abort_check(u, feature_pressed)
            if reason is not AbortReason.NONE:
                self.abort_reason = reason
                self._go(u.t, State.ABORTED, detail)
                return self._output(u)
        if self.state in _ARMED or self.state in _ACTIVE:
            bad = [s.name for s in u.sensors if not s.healthy]
            if bad:
                self.abort_reason = AbortReason.SENSOR_UNHEALTHY
                self._go(u.t, State.FAULT, f"sensor reports unhealthy: {', '.join(bad)}")
                return self._output(u)

        self._advance(u, feature_pressed, confirm_pressed)
        return self._output(u)

    # ---------- transitions ----------
    def _advance(self, u: SupervisorInput, feature_pressed: bool, confirm_pressed: bool) -> None:
        cfg, st = self.cfg, self.state
        speed_kph = abs(u.speed) * 3.6
        stopped = abs(u.speed) <= cfg.standstill_speed

        if st is State.OFF:
            if feature_pressed:
                if speed_kph < cfg.search_speed_max:
                    self._go(u.t, State.SEARCHING, "feature button")
                else:
                    self._notice = f"Slow below {cfg.search_speed_max:g} km/h to search for a slot"
            return

        if st in _ACTIVE and feature_pressed:
            self._go(u.t, State.OFF, "feature button")
            return

        if st in (State.COMPLETED, State.ABORTED, State.FAULT):
            if feature_pressed and stopped:
                self._go(u.t, State.OFF, "feature button after stop")
            return

        if st is State.SEARCHING:
            if _confirmed_ids(u):
                self._go(u.t, State.SLOTS_AVAILABLE, "confirmed slot found")
        elif st is State.SLOTS_AVAILABLE:
            ids = _confirmed_ids(u)
            if not ids:
                self._go(u.t, State.SEARCHING, "no confirmed slots left")
            elif u.selected_slot_id is not None and u.selected_slot_id in ids:
                self.selected_slot = u.selected_slot_id
                self._go(u.t, State.SLOT_SELECTED, f"slot {u.selected_slot_id} selected")
        elif st is State.SLOT_SELECTED:
            if self.selected_slot not in _confirmed_ids(u):
                self._go(u.t, State.SLOTS_AVAILABLE, "selected slot no longer available")
            elif stopped:
                self._go(u.t, State.WAIT_CONFIRM, "vehicle stopped")
        elif st is State.WAIT_CONFIRM:
            if self.selected_slot not in _confirmed_ids(u):
                self._go(u.t, State.SLOTS_AVAILABLE, "selected slot no longer available")
            elif not stopped:
                self._go(u.t, State.SLOT_SELECTED, "vehicle moved before confirm")
            elif confirm_pressed:
                self._go(u.t, State.PLANNING, "driver confirmed")
        elif st is State.PLANNING:
            self._advance_planning(u)
        elif st is State.MANEUVERING:
            if u.controller is ControllerStatus.FINISHED:
                self._go(u.t, State.COMPLETED, "controller finished")
            elif self._pause_cause(u):
                self._clear_since = None
                self._note_blocked(u)
                self._go(u.t, State.PAUSED, self._pause_cause(u))
        elif st is State.PAUSED:
            self._advance_paused(u)

    def _advance_planning(self, u: SupervisorInput) -> None:
        result = u.planner
        if result.slot_id == self.selected_slot and result.status is PlannerStatus.FOUND:
            self._maneuver_start = u.t
            self._go(u.t, State.MANEUVERING, "path found")
        elif result.slot_id == self.selected_slot and result.status is PlannerStatus.NOT_FOUND:
            self._had_control = False
            self._notice = "no path to this slot"
            self._go(u.t, State.SLOTS_AVAILABLE, "no path to this slot")
        elif u.t - self._state_since > self.cfg.planning_timeout_s:
            self._had_control = False
            self._notice = "no path to this slot"
            self._go(u.t, State.SLOTS_AVAILABLE, "planning timed out")

    def _note_blocked(self, u: SupervisorInput) -> None:
        if u.obstacle_in_path and self._blocked_since is None:
            self._blocked_since = self._last_replan_request = u.t

    def _advance_paused(self, u: SupervisorInput) -> None:
        self._note_blocked(u)
        if u.obstacle_in_path:
            if u.t - self._blocked_since > self.cfg.abort_after_s:
                self.abort_reason = AbortReason.OBSTACLE_TIMEOUT
                self._go(u.t, State.ABORTED, f"blocked over {self.cfg.abort_after_s:g} s")
                return
            if u.t - self._last_replan_request > self.cfg.replan_after_s:
                self._last_replan_request = u.t
                self._replan_now = True
        if self._pause_cause(u):
            self._clear_since = None
            return
        if self._clear_since is None:
            self._clear_since = u.t
        if u.t - self._clear_since >= self.cfg.resume_delay_s:
            self._blocked_since = None      # a flicker doesn't restart the clock; a real clearing does
            if u.confirm_button:
                self._go(u.t, State.MANEUVERING, "path clear and confirm held")

    def _pause_cause(self, u: SupervisorInput) -> str:
        if u.obstacle_in_path:
            return "obstacle in path"
        if u.brake > self.cfg.brake_pressed:
            return "brake pressed"
        return ""

    def _abort_check(self, u: SupervisorInput, feature_pressed: bool) -> Tuple[AbortReason, str]:
        cfg = self.cfg
        if abs(u.steering_torque_nm) > cfg.driver_override_nm:
            return AbortReason.STEERING_OVERRIDE, f"steering torque {u.steering_torque_nm:+.1f} Nm"
        if u.accelerator > cfg.accelerator_override:
            return AbortReason.ACCELERATOR_OVERRIDE, f"accelerator {u.accelerator:.2f}"
        if u.gear_request is not None:
            return AbortReason.GEAR_REQUEST, f"driver requested gear {u.gear_request}"
        if u.door_open:
            return AbortReason.DOOR_OPEN, "door open"
        if not u.seatbelt_on:
            return AbortReason.SEATBELT_OFF, "seatbelt off"
        stale = self._stale_sensors(u)
        if stale:
            return AbortReason.SENSOR_STALE, f"sensor stale: {', '.join(stale)}"
        if abs(u.speed) * 3.6 > cfg.maneuver_speed_max:
            return AbortReason.OVERSPEED, f"speed {abs(u.speed) * 3.6:.1f} km/h"
        if self._maneuver_start is not None and u.t - self._maneuver_start > cfg.max_time_s:
            return AbortReason.MANEUVER_TIMEOUT, f"maneuver over {cfg.max_time_s:g} s"
        if u.gear_changes > cfg.max_gear_changes:
            return AbortReason.TOO_MANY_GEAR_CHANGES, f"{u.gear_changes} gear changes"
        if feature_pressed:
            return AbortReason.DRIVER_CANCEL, "feature button during maneuver"
        return AbortReason.NONE, ""

    def _stale_sensors(self, u: SupervisorInput) -> List[str]:
        """A configured sensor that is missing from the input counts as stale."""
        latest = {s.name: s.last_update_s for s in u.sensors}
        return [name for name, timeout in self.cfg.sensor_timeouts_s.items()
                if name not in latest or u.t - latest[name] > timeout]

    def _go(self, t: float, to: State, reason: str) -> None:
        self.transitions.append(Transition(t, self.state, to, reason))
        self.state = to
        self._state_since = t
        if to in _ARMED:
            self._had_control = True
        if to is not State.SLOTS_AVAILABLE:
            self._notice = ""
        if to is State.OFF:
            self.abort_reason = AbortReason.NONE
            self.selected_slot = None
            self._maneuver_start = None
            self._blocked_since = None
            self._had_control = False

    # ---------- output ----------
    def _output(self, u: SupervisorInput) -> SupervisorOutput:
        st = self.state
        stopped = abs(u.speed) <= self.cfg.standstill_speed
        ended_with_control = st in (State.ABORTED, State.FAULT) and self._had_control
        planning = st is State.PLANNING
        return SupervisorOutput(
            state=st,
            action=self._action(stopped),
            hmi_message=self._message(stopped),
            abort_reason=self.abort_reason,
            driver_must_shift=ended_with_control and stopped,
            request_plan=planning,
            plan_slot_id=self.selected_slot if planning else None,
            request_park=st is State.COMPLETED,
            request_replan=self._replan_now,
        )

    def _action(self, stopped: bool) -> ControllerAction:
        st = self.state
        if st is State.MANEUVERING:
            return ControllerAction.DRIVE
        if st is State.PLANNING or st is State.COMPLETED:
            return ControllerAction.HOLD
        if st is State.PAUSED or (st in (State.ABORTED, State.FAULT) and self._had_control):
            return ControllerAction.HOLD if stopped else ControllerAction.STOP
        return ControllerAction.NONE

    def _message(self, stopped: bool) -> str:
        st = self.state
        if self._notice:
            return self._notice
        if st is State.OFF:
            return ""
        if st is State.PAUSED:
            return "Paused. Hold confirm to continue when clear"
        if st in (State.ABORTED, State.FAULT):
            label = "Parking aborted" if st is State.ABORTED else "Parking fault"
            cause = f"{label}: {self.abort_reason.value}."
            if not self._had_control:
                return cause
            return f"{cause} Take over and shift" if stopped else f"{cause} Stopping"
        return {
            State.SEARCHING: "Searching for a slot",
            State.SLOTS_AVAILABLE: "Select a slot",
            State.SLOT_SELECTED: "Stop the vehicle",
            State.WAIT_CONFIRM: "Press confirm to start parking",
            State.PLANNING: "Planning",
            State.MANEUVERING: "Parking. Keep your hands off the wheel",
            State.COMPLETED: "Parked",
        }[st]


def _confirmed_ids(u: SupervisorInput) -> List[int]:
    return [s.slot_id for s in u.slots if s.confirmed]
