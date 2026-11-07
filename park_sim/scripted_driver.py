"""Scripted driver for the sim: press the feature button, select the slot, confirm, and
keep holding confirm. Scripted events override any supervisor input from a given time on.

    events = [(4.0, {"brake": 0.5}), (4.5, {"brake": 0.0})]               # brake tap
    events = [(3.0, {"steering_torque_nm": 5.0})]                          # torque spike
    events = [(3.0, {"sensors": (SensorHealth("localization", True, 3.0),
                                 SensorHealth("wheel_speed", True, 3.0))})]  # frozen timestamps

The same events inject faults, using these keys:

    sensor_dropout: names          the streams stop arriving
    sensor_frozen: names           the streams keep arriving, with new timestamps and the last value
    pose_delay_s: seconds          the localization output falls this far behind (odometry, gnss_ins)
    steer_not_followed: True       the road wheels stop answering steering commands
    eps_saturation_deg: degrees    the road wheels go no further than this; None lifts it

A name is a sensor ("uss_rl_inner"), a kind ("ultrasonic", "lidar", "camera"), or one of "localization",
"wheel_speed", "slot_perception". An empty list ends the fault.

An override stays in force until a later event replaces it. Times are on the run clock:
t = 0 is the first tick, before the driver has pressed anything. A time can also be
("MANEUVERING", 2.0): 2 s after the supervisor first enters that state.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .supervisor import (ControllerStatus, PlannerResult, SensorHealth, SlotCandidate, State,
                         SupervisorInput)

ScriptedEvent = Tuple[Union[float, Tuple[str, float]], Dict[str, object]]
FAULT_KEYS = ("sensor_dropout", "sensor_frozen", "pose_delay_s", "steer_not_followed", "eps_saturation_deg")


def has_faults(events: Sequence[ScriptedEvent]) -> bool:
    return any(key in FAULT_KEYS for _, changes in events for key in changes)


def events_from_config(entries: Sequence[dict]) -> List[ScriptedEvent]:
    """Events from a scenario file: {t: 3.0, set: {...}} or {after_state: MANEUVERING, delay_s: 2.0, set: {...}}."""
    events: List[ScriptedEvent] = []
    for entry in entries:
        when = (entry["after_state"], entry.get("delay_s", 0.0)) if "after_state" in entry else entry["t"]
        events.append((when, dict(entry["set"])))
    return events


class EventScript:
    """Which overrides are in force at a given time."""

    def __init__(self, events: Sequence[ScriptedEvent] = ()):
        self.events = list(events)
        self._entered: Dict[str, float] = {}

    def observe(self, t: float, state: State) -> None:
        self._entered.setdefault(state.value, t)

    def _in_force(self, t: float) -> Dict[str, object]:
        due = []
        for when, changes in self.events:
            if isinstance(when, tuple):
                if when[0] not in self._entered:
                    continue
                when = self._entered[when[0]] + when[1]
            if when <= t:
                due.append((when, changes))
        merged: Dict[str, object] = {}
        for _, changes in sorted(due, key=lambda e: e[0]):
            merged.update(changes)
        return merged

    def inputs(self, t: float) -> Dict[str, object]:
        """The supervisor input overrides in force."""
        if not self.events:
            return {}
        return {k: v for k, v in self._in_force(t).items() if k not in FAULT_KEYS}

    def faults(self, t: float) -> Dict[str, object]:
        """The injected faults in force."""
        if not self.events:
            return {}
        return {k: v for k, v in self._in_force(t).items() if k in FAULT_KEYS}

    def apply(self, base: SupervisorInput) -> SupervisorInput:
        overrides = self.inputs(base.t)
        return replace(base, **overrides) if overrides else base


class ScriptedDriver:
    def __init__(self, slot_id: int, sensor_names: Iterable[str],
                 events: Sequence[ScriptedEvent] = ()):
        self.slot_id = slot_id
        self.sensor_names = tuple(sensor_names)
        self.script = EventScript(sorted(events, key=lambda e: e[0] if isinstance(e[0], (int, float)) else 0.0))
        self._state = State.OFF
        self._pressed = False
        self._selected = False
        self._confirming = False
        self._t = 0.0

    def input(self, t: float, speed: float, controller: ControllerStatus,
              planner: PlannerResult, gear_changes: int, sensors: Optional[Tuple[SensorHealth, ...]] = None,
              obstacle_in_path: bool = False) -> SupervisorInput:
        """sensors: the watchdog's health; without one, every sensor reports fresh."""
        self._t = t
        press = not self._pressed
        self._pressed = True
        select: Optional[int] = None
        if self._state is State.SLOTS_AVAILABLE and not self._selected:
            select, self._selected = self.slot_id, True
        self._confirming = self._confirming or self._state is State.WAIT_CONFIRM
        if sensors is None:
            sensors = tuple(SensorHealth(name, True, t) for name in self.sensor_names)
        return self.script.apply(SupervisorInput(
            t=t, feature_button=press, selected_slot_id=select, confirm_button=self._confirming,
            speed=speed, slots=(SlotCandidate(self.slot_id),), planner=planner,
            controller=controller, gear_changes=gear_changes, obstacle_in_path=obstacle_in_path,
            sensors=sensors))

    def observe(self, state: State) -> None:
        self._state = state
        self.script.observe(self._t, state)
