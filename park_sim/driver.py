"""Scripted driver for search runs: drives the aisle centerline, reacts to the supervisor.

Presses the feature button on entering, holds the lane at the search speed, selects a slot
according to its policy, brakes once the selection is accepted, and presses and holds confirm
when asked. If planning fails it leaves that slot alone until the car has moved on: it selects
another one the policy allows, or else drives on, and looks at the slot again from the new place.
It brakes at the end of the aisle if nothing was selected.

Policies:
  select_first                  the slot seen earliest
  select_nearest_after_passing  the slot nearest the car when the driver decides
  select_by_index               only the slot with id slot_index
  select_first_after            the slot seen earliest with its center past after_x, in row_y's row if set; at the
                                end of the aisle, the nearest slot. For sensor-based detectors, whose ids are track numbers
  never_select                  keeps driving; for search-only runs
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .supervisor import SlotCandidate, State

POLICIES = ("select_first", "select_nearest_after_passing", "select_by_index", "select_first_after", "never_select")


class Rejections:
    """Slots with no path. "No path" is only true of where the car was when it tried, so once the car has moved far enough a slot can be tried again."""

    def __init__(self, retry_after_driven_m: float):
        self.retry_after = retry_after_driven_m
        self._driven = 0.0
        self._last: Optional[Tuple[float, float]] = None
        self._at: Dict[int, float] = {}     # slot id: distance driven when it failed

    def add(self, slot_id: int) -> None:
        self._at[slot_id] = self._driven

    def moved_to(self, x: float, y: float) -> None:
        if self._last is not None:
            self._driven += math.hypot(x - self._last[0], y - self._last[1])
        self._last = (x, y)
        self._at = {slot: at for slot, at in self._at.items() if self._driven - at < self.retry_after}

    def __contains__(self, slot_id: object) -> bool:
        return slot_id in self._at

    @property
    def ids(self) -> Set[int]:
        return set(self._at)


@dataclass(frozen=True)
class DriverConfig:
    speed_kph: float
    policy: str
    slot_index: Optional[int]
    reaction_time_s: float
    confirm_delay_s: float
    lane_lookahead_min: float
    lane_lookahead_s: float
    aisle_end_margin: float
    retry_after_driven_m: float = 6.0   # search.retry_after_driven_m
    after_x: Optional[float] = None
    row_y: Optional[float] = None       # select_first_after: a y in the row to stay in

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(f"unknown driver policy {self.policy!r}, expected one of {POLICIES}")
        if self.policy == "select_by_index" and self.slot_index is None:
            raise ValueError("select_by_index needs slot_index")
        if self.policy == "select_first_after" and self.after_x is None:
            raise ValueError("select_first_after needs after_x")

    @classmethod
    def from_params(cls, search: dict) -> "DriverConfig":
        return cls(**{k: search[k] for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class DriverAction:
    feature_button: bool
    selected_slot_id: Optional[int]
    confirm_button: bool
    brake: float            # pedal, 0-1
    steer_cmd: float        # road-wheel angle, rad; applied while the driver has the car
    speed_cmd: float        # m/s; applied while the driver has the car
    steering_torque_nm: float = 0.0     # hands on the wheel; above the override limit it aborts a maneuver


class SearchDriver:
    def __init__(self, cfg: DriverConfig, wheelbase: float, max_steer: float,
                 aisle_y: float, aisle_end_x: float):
        self.cfg = cfg
        self._wheelbase, self._max_steer = wheelbase, max_steer
        self._aisle_y = aisle_y
        self._brake_x = aisle_end_x + cfg.aisle_end_margin
        self._state = State.OFF
        self._pressed = False
        self._rejected = Rejections(cfg.retry_after_driven_m)
        self._selected: Optional[int] = None
        self._options_since: Optional[float] = None
        self._prompted_at: Optional[float] = None
        self._past_end = False
        self._has_options = False

    @property
    def top_speed(self) -> float:
        """m/s, the most the car does while the driver has it."""
        return self.cfg.speed_kph / 3.6

    @property
    def start_speed(self) -> float:
        return self.top_speed

    def stop_requested(self) -> bool:
        return False

    def act(self, t: float, x: float, y: float, yaw: float, v: float,
            slots: Sequence[SlotCandidate]) -> DriverAction:
        press = not self._pressed
        self._pressed = True
        self._past_end = x >= self._brake_x
        self._rejected.moved_to(x, y)

        options = self._options(slots)
        self._has_options = bool(options)
        if not options:
            self._options_since = None
        elif self._options_since is None:
            self._options_since = t
        select = None
        if (self._state is State.SLOTS_AVAILABLE and options
                and t - self._options_since >= self.cfg.reaction_time_s):
            select = self._selected = self._choose(options, x, y)

        if self._state is State.WAIT_CONFIRM and self._prompted_at is None:
            self._prompted_at = t
        confirm = (self._prompted_at is not None
                   and t - self._prompted_at >= self.cfg.confirm_delay_s)

        # Off the pedal once the system has the car: a pressed brake would pause the maneuver
        stopping = self._state in (State.SLOT_SELECTED, State.WAIT_CONFIRM) or (
            self._past_end and self._state in (State.OFF, State.SEARCHING, State.SLOTS_AVAILABLE))
        return DriverAction(press, select, confirm, 1.0 if stopping else 0.0,
                            self._lane_steer(x, y, yaw, v),
                            0.0 if stopping else self.cfg.speed_kph / 3.6)

    def observe(self, state: State) -> None:
        """Take in the supervisor's state for this tick."""
        if state is State.SLOTS_AVAILABLE and self._state is State.PLANNING:
            self._rejected.add(self._selected)
        if state in (State.OFF, State.SEARCHING, State.SLOTS_AVAILABLE, State.SLOT_SELECTED):
            self._prompted_at = None
        self._state = state

    def gave_up(self, v: float, standstill_speed: float) -> bool:
        """Stopped at the end of the aisle with nothing left to select."""
        return self._past_end and not self._has_options and abs(v) <= standstill_speed

    def _options(self, slots: Sequence[SlotCandidate]) -> List[SlotCandidate]:
        if self.cfg.policy == "never_select":
            return []
        options = [s for s in slots if s.slot_id not in self._rejected]
        if self.cfg.policy == "select_by_index":
            return [s for s in options if s.slot_id == self.cfg.slot_index]
        if self.cfg.policy == "select_first_after":
            ahead = [s for s in options if _center_x(s) >= self.cfg.after_x and self._in_row(s)]
            return ahead or (options if self._past_end else [])
        return options

    def _in_row(self, slot: SlotCandidate) -> bool:
        if self.cfg.row_y is None:
            return True
        center_y = sum(c[1] for c in slot.corners) / len(slot.corners)
        return (center_y - self._aisle_y) * (self.cfg.row_y - self._aisle_y) > 0.0

    def _choose(self, options: List[SlotCandidate], x: float, y: float) -> int:
        if self.cfg.policy == "select_nearest_after_passing" or (self.cfg.policy == "select_first_after" and self._past_end):
            return min(options, key=lambda s: _opening_distance(s, x, y)).slot_id
        return min(options, key=lambda s: (s.first_seen_t, s.slot_id)).slot_id

    def _lane_steer(self, x: float, y: float, yaw: float, v: float) -> float:
        """Pure pursuit to a point ahead on the aisle centerline (the aisle runs along +x)."""
        ahead = max(self.cfg.lane_lookahead_min, self.cfg.lane_lookahead_s * abs(v))
        dx, dy = ahead, self._aisle_y - y
        lx = math.cos(yaw) * dx + math.sin(yaw) * dy
        ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
        steer = math.atan(self._wheelbase * 2.0 * ly / max(lx * lx + ly * ly, 1e-6))
        return max(-self._max_steer, min(self._max_steer, steer))


def _center_x(slot: SlotCandidate) -> float:
    return sum(c[0] for c in slot.corners) / len(slot.corners)


def _opening_distance(slot: SlotCandidate, x: float, y: float) -> float:
    (ax, ay), (bx, by) = slot.corners[:2]
    return math.hypot((ax + bx) / 2.0 - x, (ay + by) / 2.0 - y)
