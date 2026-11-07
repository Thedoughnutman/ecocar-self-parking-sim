"""The logic behind the interactive window, with no window in it: keys to inputs, inputs to a driver, a run to a status.
See docs/interactive.md. Standard library, numpy, and the supervisor and frames the sim already uses; no plotting.

The window talks to a run only through the two messages here: HmiInput going in, HmiStatus coming out. ManualDriver plugs
into run_search in place of the scripted driver, so the supervisor, detectors, planner and controller are the same ones a
Monte Carlo run uses. Pausing lives here, not in the supervisor: confirm going off presses the brake, which is what pauses a maneuver.
On the keyboard, Enter toggles confirm.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .driver import DriverAction, Rejections
from .localization import MapToOdom
from .supervisor import SlotCandidate, State

_SYSTEM_HAS_THE_CAR = (State.PLANNING, State.MANEUVERING, State.PAUSED)
# Slot status, for coloring
UNCONFIRMED, OFFERED, SELECTED, REJECTED = "unconfirmed", "offered", "selected", "rejected"
REVERSING_ABOVE_MPS = 0.02  # the display's "R", while a person is driving; the same as a standstill


@dataclass(frozen=True)
class HmiInput:
    """Everything the person is doing on one tick."""
    forward: bool = False
    back: bool = False
    left: bool = False
    right: bool = False
    feature: bool = False           # a press, for one tick
    confirm: bool = False           # held
    brake: bool = False             # held
    takeover: bool = False          # hands on the wheel, held
    select: Optional[int] = None    # a click on a slot, for one tick
    stop: bool = False              # end the run now


@dataclass(frozen=True)
class HmiStatus:
    """What the panel shows on one tick. Read from the run; nothing here is worked out twice."""
    t: float
    state: str
    message: str
    gear: str
    speed_kph: float
    steering_deg: float
    distance_to_go_m: Optional[float]
    last_reason: str        # why the state last changed
    abort_reason: str
    selected_slot: Optional[int]
    slots: Dict[int, str]   # slot id: status
    estimate_off_m: float   # how far the car's own estimate is from the truth


@dataclass(frozen=True)
class ManualConfig:
    max_speed_kph: float
    max_reverse_kph: float
    speed_change_mps2: float
    steer_rate_deg_s: float
    steer_return_deg_s: float
    takeover_torque_nm: float
    retry_after_driven_m: float

    @classmethod
    def from_params(cls, section: dict) -> "ManualConfig":
        return cls(**{k: section[k] for k in cls.__dataclass_fields__})


class ManualDriver:
    """A person at the controls, for run_search. `source` is asked once per tick, so a recording replays tick for tick."""

    def __init__(self, cfg: ManualConfig, max_steer: float, dt: float, source: Callable[[], HmiInput]):
        self.cfg, self.dt, self.source = cfg, dt, source
        self._max_steer = max_steer
        self._state = State.OFF
        self._speed = 0.0       # m/s, what the keys are asking for
        self._steer = 0.0       # rad
        self._stop = False
        self._rejections = Rejections(cfg.retry_after_driven_m)
        self._selected: Optional[int] = None
        self.offered: Tuple[int, ...] = ()  # confirmed slots with a chance of working, on the last tick

    @property
    def rejected(self) -> Set[int]:
        """Slots with no path from where the car was, until it has moved on."""
        return self._rejections.ids

    @property
    def top_speed(self) -> float:
        return self.cfg.max_speed_kph / 3.6

    @property
    def start_speed(self) -> float:
        return 0.0

    def stop_requested(self) -> bool:
        return self._stop

    def gave_up(self, v: float, standstill_speed: float) -> bool:
        return False        # only the person ends the run

    def observe(self, state: State) -> None:
        if state is State.SLOTS_AVAILABLE and self._state is State.PLANNING and self._selected is not None:
            self._rejections.add(self._selected)
        self._state = state

    def act(self, t: float, x: float, y: float, yaw: float, v: float, slots: Sequence[SlotCandidate]) -> DriverAction:
        u = self.source()
        self._stop = self._stop or u.stop
        self._rejections.moved_to(x, y)
        self.offered = tuple(s.slot_id for s in slots if s.confirmed and s.slot_id not in self._rejections)
        select = u.select if u.select in self.offered else None
        if select is not None:
            self._selected = select
        has_car = self._state in _SYSTEM_HAS_THE_CAR
        if has_car:         # it's handed back at a standstill, so don't lurch off with what was asked before
            self._speed = 0.0
        else:
            self._drive(u)
        self._turn(u)
        brake = u.brake or (has_car and not u.confirm)      # confirm off is the brake
        return DriverAction(u.feature, select, u.confirm, 1.0 if brake else 0.0, self._steer, self._speed,
                            self.cfg.takeover_torque_nm if u.takeover else 0.0)

    def _drive(self, u: HmiInput) -> None:
        step = self.cfg.speed_change_mps2 * self.dt
        if u.brake:
            self._speed = 0.0
        elif u.forward:
            self._speed = min(self._speed + step, self.top_speed)
        elif u.back:
            self._speed = max(self._speed - step, -self.cfg.max_reverse_kph / 3.6)

    def _turn(self, u: HmiInput) -> None:
        if u.left != u.right:
            step = math.radians(self.cfg.steer_rate_deg_s) * self.dt
            self._steer += step if u.left else -step
            self._steer = max(-self._max_steer, min(self._max_steer, self._steer))
        else:
            step = math.radians(self.cfg.steer_return_deg_s) * self.dt
            self._steer -= max(-step, min(step, self._steer))


class KeyState:
    """Turns key and mouse events into one HmiInput per tick. A press or click that comes and goes between two ticks still
    counts once. Quitting is separate from the window's own keys, which never get here. Enter is a toggle: press once to go, again to pause."""
    HELD = {"w": "forward", "up": "forward", "s": "back", "down": "back", "a": "left", "left": "left", "d": "right",
            "right": "right", "b": "brake", "t": "takeover"}

    def __init__(self):
        self._held: Set[str] = set()
        self._confirm = False
        self._feature = False
        self._select: Optional[int] = None
        self._stop = False

    def press(self, key: Optional[str]) -> None:
        key = _key_name(key)
        if key in self.HELD:
            self._held.add(self.HELD[key])
        elif key == "enter":
            self._confirm = not self._confirm
        elif key == "p":
            self._feature = True

    def clear_confirm(self) -> None:
        """The system has handed the car back, so the next stall needs its own Enter."""
        self._confirm = False

    def release(self, key: Optional[str]) -> None:
        key = _key_name(key)
        if key in self.HELD:
            self._held.discard(self.HELD[key])

    def click_slot(self, slot_id: int) -> None:
        self._select = slot_id

    def request_stop(self) -> None:
        self._stop = True

    @property
    def stop_requested(self) -> bool:
        return self._stop

    def focus_lost(self) -> None:
        """A key let go in another window never reports it, so nothing stays held. Enter is a toggle, not a held key, so it stays as it is."""
        self._held.clear()

    def read(self) -> HmiInput:
        u = HmiInput(feature=self._feature, select=self._select, stop=self._stop, confirm=self._confirm, **{name: True for name in self._held})
        self._feature, self._select = False, None
        return u


def _key_name(key: Optional[str]) -> str:
    key = (key or "").lower()
    return "enter" if key == "return" else key


class InputLog:
    """Wraps a source and keeps what it said on each tick. Keeps only the changes, so a long drive stays small."""

    def __init__(self, source: Callable[[], HmiInput]):
        self.source = source
        self.tick = 0
        self.changes: List[Tuple[int, HmiInput]] = []
        self._last: Optional[HmiInput] = None

    def __call__(self) -> HmiInput:
        u = self.source()
        if u != self._last:
            self.changes.append((self.tick, u))
            self._last = u
        self.tick += 1
        return u

    def save(self, path, setup: dict) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"setup": setup, "ticks": self.tick,
                                    "changes": [[k, asdict(u)] for k, u in self.changes]}, indent=1))
        return path


class Replay:
    """A saved session as a source. Once it runs out it asks for a stop, so a replay always ends."""

    def __init__(self, path):
        session = json.loads(Path(path).read_text())
        self.setup: dict = session["setup"]
        self.ticks: int = session["ticks"]
        self._changes = [(k, HmiInput(**u)) for k, u in session["changes"]]
        self._tick = 0
        self._now = HmiInput()

    def __call__(self) -> HmiInput:
        if self._changes and self._changes[0][0] <= self._tick:
            self._now = self._changes.pop(0)[1]
        self._tick += 1
        return replace(self._now, stop=True) if self.finished else self._now

    @property
    def finished(self) -> bool:
        return self._tick > self.ticks


def slots_in_map(tick) -> Dict[int, list]:
    """Each slot's corners in the map frame, where the lot is drawn."""
    frame = MapToOdom(tick.true_pose, tick.slot_frame)
    out = {}
    for slot in tick.slots:
        points = np.array(slot.corners, dtype=float)
        x, y = frame.to_map(points[:, 0], points[:, 1])
        out[slot.slot_id] = list(zip(x.tolist(), y.tolist()))
    return out


def slot_statuses(slots: Sequence[SlotCandidate], selected: Optional[int], rejected: Set[int]) -> Dict[int, str]:
    out = {}
    for slot in slots:
        if slot.slot_id in rejected:        # first: after "no path", the supervisor still has it as its last selection
            out[slot.slot_id] = REJECTED
        elif slot.slot_id == selected:
            out[slot.slot_id] = SELECTED
        else:
            out[slot.slot_id] = OFFERED if slot.confirmed else UNCONFIRMED
    return out


class StallNumbers:
    """The numbers on screen, 1, 2, 3...: a stall gets one the first time it's available, in the order they were found, and keeps it for the
    run. A stall that goes away and comes back has its old number, and a number is never given to another stall. The slot ids underneath
    stay as they are: they count every track, including stalls that turned out to be taken, so they have gaps."""

    def __init__(self):
        self._numbers: Dict[int, int] = {}

    def update(self, slots: Sequence[SlotCandidate], statuses: Dict[int, str]) -> Dict[int, int]:
        """The numbers to show now: only stalls available at this moment."""
        available = [s for s in slots if statuses.get(s.slot_id) in (OFFERED, SELECTED)]
        for slot in sorted(available, key=lambda s: (s.first_seen_t, s.slot_id)):
            if slot.slot_id not in self._numbers:
                self._numbers[slot.slot_id] = len(self._numbers) + 1
        return {s.slot_id: self._numbers[s.slot_id] for s in available}

    def in_words(self, text: str) -> str:
        """A supervisor message, with slot ids changed to the numbers on screen."""
        return re.sub(r"\bslot (\d+)", lambda m: f"stall {self._numbers[int(m.group(1))]}" if int(m.group(1)) in self._numbers else m.group(0), text)


    def number_of(self, slot_id: int) -> Optional[int]:
        return self._numbers.get(slot_id)


# ---------- why it didn't park, in words ----------
PLAN_FAILURES = {
    "stall_blocked": "The map has an obstacle in that stall, so the car won't reverse into it. It may be a false mark: get closer to look again.",
    "stall_unseen": "The car hasn't seen enough of that stall yet. Get closer to it and try again.",
    "start_blocked": "The map has something too close to the car to start from here. Pull forward and try again.",
    "timed_out": "No route found within the planning time limit. The car can only plan through space its sensors have seen, so drive further past the stall and try again.",
    "no_route": "No route into that stall from here. Drive further past the stall, or straighten up, and try again.",
}
ABORTS = {
    "steering_override": "You took the wheel, so the system handed the car back.",
    "accelerator_override": "The accelerator was pressed, so the system handed the car back.",
    "gear_request": "A gear was requested, so the system handed the car back.",
    "door_open": "A door opened, so the maneuver stopped.",
    "seatbelt_off": "The seatbelt came off, so the maneuver stopped.",
    "sensor_stale": "A sensor stopped updating, so the maneuver stopped.",
    "sensor_unhealthy": "A sensor or the steering reported a fault, so the maneuver stopped.",
    "overspeed": "The car went over the maneuver speed limit, so the maneuver stopped.",
    "maneuver_timeout": "The maneuver ran out of time. The route was probably long, so try from closer to the stall.",
    "too_many_gear_changes": "The maneuver needed too many gear changes.",
    "driver_cancel": "You cancelled the maneuver.",
    "obstacle_timeout": "Something stayed in the path too long, so the maneuver stopped.",
}


def plan_failure_text(why: str) -> str:
    return PLAN_FAILURES.get(why, "No route into that stall from here.")


def to_feet(meters: float) -> float:
    return meters * 3.28084


def to_inches(meters: float) -> float:
    return meters * 39.3701


def to_mph(kph: float) -> float:
    return kph * 0.621371


@dataclass(frozen=True)
class Units:
    """What the window shows. Everything underneath stays metric: the sim, config, logs and MCAPs. Metric shows speed in m/s, as Foxglove's plots do."""
    imperial: bool = True

    @classmethod
    def named(cls, name: str) -> "Units":
        if name not in ("imperial", "metric"):
            raise ValueError(f"unknown units {name!r}, expected imperial or metric")
        return cls(name == "imperial")

    def other(self) -> "Units":
        return Units(not self.imperial)

    @property
    def name(self) -> str:
        return "imperial" if self.imperial else "metric"

    @property
    def speed_unit(self) -> str:
        return "mph" if self.imperial else "m/s"

    @property
    def long_unit(self) -> str:
        return "ft" if self.imperial else "m"

    @property
    def short_unit(self) -> str:
        return "in" if self.imperial else "cm"

    def speed(self, kph: float) -> float:
        return to_mph(kph) if self.imperial else kph / 3.6

    def long(self, meters: float) -> float:
        return to_feet(meters) if self.imperial else meters

    def short(self, meters: float) -> float:
        return to_inches(meters) if self.imperial else meters * 100.0

    def message(self, text: str) -> str:
        """A supervisor message with its speeds in these units."""
        return re.sub(r"(\d+(?:\.\d+)?) km/h", lambda m: f"{self.speed(float(m.group(1))):.1f} {self.speed_unit}", text)


IMPERIAL, METRIC = Units(True), Units(False)


def summary_text(res, units: Units = IMPERIAL) -> str:
    """RunResult.summary(), in the window's units."""
    if res.planned is None:
        return f"FAIL ({res.reason})"
    m = res.metrics
    failed = ", ".join(k for k, ok in res.checks.items() if not ok)
    return (f"{'PASS' if res.success else 'FAIL'}  lateral {units.short(m['lateral_err']):+.1f} {units.short_unit}  heading {m['heading_err_deg']:+.2f} deg  depth {units.short(m['depth_err']):+.1f} {units.short_unit}  "
            f"clearance {units.long(m['min_clearance']):.2f} {units.long_unit}  gears {int(m['gear_changes'])}  time {m['time_s']:.1f} s  top speed {units.speed(m['max_speed_kph']):.1f} {units.speed_unit}"
            + (f"  [failed: {failed}]" if failed else "") + (f"  ({res.reason})" if res.reason else ""))


def explain_result(res, criteria: dict, units: Units = IMPERIAL) -> str:
    """Why a finished run didn't park, or didn't meet the criteria, in a sentence or two. Empty if it did."""
    if res.success:
        return ""
    for abort, words in ABORTS.items():
        if res.reason.endswith(f": {abort}"):
            return words
    if res.reason == "stopped by the driver":
        return "You ended the run."
    if res.reason.startswith("no path"):
        attempts = res.search.plan_attempts if res.search else ()
        return plan_failure_text(attempts[-1].why) if attempts else "No route to any stall."
    if res.reason:
        return {"no slot selected": "No stall was selected.", "reached the end of the aisle": "The car reached the end of the aisle with no stall.", "timeout": "The run ran out of time."}.get(res.reason, res.reason)
    m, failed = res.metrics, [k for k, ok in res.checks.items() if not ok]
    short, long_, speed = units.short_unit, units.long_unit, units.speed_unit
    words = {"lateral": lambda: f"It parked {units.short(abs(m['lateral_err'])):.1f} {short} off the stall center (limit {units.short(criteria['lateral_tol']):.1f} {short}).",
             "heading": lambda: f"It parked {abs(m['heading_err_deg']):.1f} deg crooked (limit {criteria['heading_tol_deg']:g} deg).",
             "depth": lambda: f"It parked {units.short(abs(m['depth_err'])):.1f} {short} from the right depth (limit {units.short(criteria['depth_tol']):.1f} {short}).",
             "clearance": lambda: f"It came within {units.long(m['min_clearance']):.2f} {long_} of something (limit {units.long(criteria['min_clearance']):.2f} {long_}).",
             "gear_changes": lambda: f"It took {int(m['gear_changes'])} gear changes (limit {criteria['max_gear_changes']}).",
             "speed": lambda: f"It reached {units.speed(m['max_speed_kph']):.1f} {speed} (limit {units.speed(criteria['max_speed_kph']):.1f} {speed}).",
             "time": lambda: f"It took {m['time_s']:.0f} s (limit {criteria['max_time_s']:g} s)."}
    return " ".join(words[k]() if k in words else f"The {k} check failed." for k in failed)


def slot_at(x: float, y: float, corners_by_slot: Dict[int, list]) -> Optional[int]:
    """The slot whose outline holds this point, or None."""
    for slot_id, corners in corners_by_slot.items():
        inside = False
        for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1]):
            if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
                inside = not inside
        if inside:
            return slot_id
    return None


def status_from_tick(tick, rejected: Set[int]) -> HmiStatus:
    row = tick.row
    reversing = float(row["gear"][0]) < 0 or float(row["v"][0]) < -REVERSING_ABOVE_MPS     # by hand, the controller's gear doesn't say
    to_go = float(row["dist_to_go"][0])
    last = tick.transitions[-1].reason if tick.transitions else ""
    off = math.hypot(float(row["est_x"][0]) - float(row["x"][0]), float(row["est_y"][0]) - float(row["y"][0]))
    selected = selected_now(tick)
    return HmiStatus(tick.t, str(row["sup_state"][0]), str(row["sup_hmi"][0]), "R" if reversing else "D",
                     abs(float(row["v"][0])) * 3.6, math.degrees(float(row["delta"][0])), None if math.isnan(to_go) else to_go,
                     last, str(row["sup_abort_reason"][0]), selected, slot_statuses(tick.slots, selected, rejected), off)


def selected_now(tick) -> Optional[int]:
    """The selected slot, if one is. Back to choosing, the supervisor still holds its last selection, and that isn't one."""
    return None if str(tick.row["sup_state"][0]) in ("OFF", "SEARCHING", "SLOTS_AVAILABLE") else tick.selected_slot
