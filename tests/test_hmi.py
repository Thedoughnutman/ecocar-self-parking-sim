"""The interactive HMI: keys to inputs, the manual driver, and whole runs with a scripted person at the controls."""
import ast
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest

from park_sim.hmi import (OFFERED, REJECTED, SELECTED, UNCONFIRMED, HmiInput, InputLog, KeyState, ManualConfig, ManualDriver, Replay,
                          slot_at, slot_statuses)
from park_sim.scenario import load_params
from park_sim.supervisor import AbortReason, SlotCandidate, State
from park_sim.vehicle import VehicleParams

ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params()
VEHICLE = VehicleParams.from_yaml()
CFG = ManualConfig.from_params(PARAMS["interactive"])
DT = PARAMS["controller"]["dt"]
spec = importlib.util.spec_from_file_location("interactive_script", ROOT / "scripts" / "interactive.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)
SETUP = {"seed": 1003, "detector": "fused", "planner_map": None, "scenario": None, "monitor": None}


def manual(inputs):
    feed = iter(inputs)
    return ManualDriver(CFG, VEHICLE.max_steer, DT, lambda: next(feed))


def act(driver, slots=(), v=0.0):
    return driver.act(0.0, 0.0, 0.0, 0.0, v, slots)


# ---------- keys ----------
def test_keys_and_arrows_both_drive():
    keys = KeyState()
    keys.press("w")
    assert keys.read().forward
    keys.release("w")
    keys.press("up")
    assert keys.read().forward
    keys.release("up")
    assert not keys.read().forward


def test_a_held_key_stays_until_released_and_a_capital_counts():
    keys = KeyState()
    keys.press("B")
    assert keys.read().brake and keys.read().brake
    keys.release("b")
    assert not keys.read().brake


def test_enter_is_a_toggle_not_a_hold():
    keys = KeyState()
    keys.press("return")
    keys.release("return")
    assert keys.read().confirm and keys.read().confirm
    keys.press("return")
    keys.release("return")
    assert not keys.read().confirm


def test_enter_clears_when_the_car_is_handed_back_so_the_next_stall_needs_its_own_press(window):
    from types import SimpleNamespace
    window.keys.press("enter")
    row = {"sup_state": ("SLOTS_AVAILABLE",)}
    pose = SimpleNamespace(x=0.0, y=0.0, yaw=0.0)
    tick = SimpleNamespace(row=row, slots=(), selected_slot=None, true_pose=pose, slot_frame=pose)
    window._take_in(tick)
    assert not window.keys.read().confirm
    window.keys.press("enter")
    window._take_in(SimpleNamespace(**{**vars(tick), "row": {"sup_state": ("MANEUVERING",)}}))
    assert window.keys.read().confirm


def test_the_feature_button_and_a_click_last_one_tick():
    keys = KeyState()
    keys.press("p")
    keys.click_slot(4)
    first = keys.read()
    assert first.feature and first.select == 4
    second = keys.read()
    assert not second.feature and second.select is None


def test_losing_focus_lets_go_of_held_keys_but_does_not_pause_the_car():
    """A key released in another window never reports it. Enter is a toggle, so the car carries on."""
    keys = KeyState()
    keys.press("enter")
    keys.press("w")
    keys.focus_lost()
    u = keys.read()
    assert u.confirm and not u.forward


# ---------- the manual driver ----------
def test_the_car_starts_stopped_and_top_speed_comes_from_the_config():
    d = manual([])
    assert d.start_speed == 0.0
    assert d.top_speed == pytest.approx(PARAMS["interactive"]["max_speed_kph"] / 3.6)
    assert d.top_speed * 3.6 < PARAMS["supervisor"]["search_speed_max"]     # or the feature button wouldn't work at top speed


def test_the_speed_keys_ramp_hold_and_stop_at_the_limits():
    d = manual([HmiInput(forward=True)] * 2000 + [HmiInput()] + [HmiInput(brake=True)] + [HmiInput(back=True)] * 2000)
    for _ in range(2000):
        a = act(d)
    assert a.speed_cmd == pytest.approx(d.top_speed)
    assert act(d).speed_cmd == pytest.approx(d.top_speed)   # let go: it keeps the speed
    assert act(d).speed_cmd == 0.0                          # brake
    for _ in range(2000):
        a = act(d)
    assert a.speed_cmd == pytest.approx(-CFG.max_reverse_kph / 3.6)


def test_steering_ramps_stops_at_the_limit_and_returns_to_straight():
    d = manual([HmiInput(left=True)] * 500 + [HmiInput()] * 500)
    for _ in range(500):
        a = act(d)
    assert a.steer_cmd == pytest.approx(VEHICLE.max_steer)
    for _ in range(500):
        a = act(d)
    assert a.steer_cmd == 0.0


@pytest.mark.parametrize("state", [State.PLANNING, State.MANEUVERING, State.PAUSED])
def test_with_the_system_driving_letting_go_of_confirm_is_the_brake(state):
    d = manual([HmiInput(confirm=True), HmiInput()])
    d.observe(state)
    assert act(d).brake == 0.0
    assert act(d).brake == 1.0


def test_with_the_person_driving_no_confirm_is_not_a_brake():
    d = manual([HmiInput()])
    d.observe(State.SEARCHING)
    assert act(d).brake == 0.0


def test_the_car_does_not_lurch_off_with_an_old_speed_when_handed_back():
    d = manual([HmiInput(forward=True)] * 10 + [HmiInput(confirm=True), HmiInput()])
    for _ in range(10):
        assert act(d).speed_cmd > 0.0
    d.observe(State.PLANNING)
    act(d)
    d.observe(State.SLOTS_AVAILABLE)    # no path
    assert act(d).speed_cmd == 0.0


def test_takeover_puts_enough_torque_on_the_wheel_to_override():
    d = manual([HmiInput(takeover=True), HmiInput()])
    assert act(d).steering_torque_nm > PARAMS["supervisor"]["driver_override_nm"]
    assert act(d).steering_torque_nm == 0.0


def test_a_click_on_a_slot_with_no_path_is_ignored_until_the_car_moves():
    slots = (SlotCandidate(3), SlotCandidate(4))
    d = manual([HmiInput(select=3), HmiInput(select=3), HmiInput(select=4), HmiInput(select=3), HmiInput(select=3)])
    d.observe(State.SLOTS_AVAILABLE)
    assert act(d, slots).selected_slot_id == 3
    d.observe(State.PLANNING)
    d.observe(State.SLOTS_AVAILABLE)
    assert d.rejected == {3}
    assert act(d, slots).selected_slot_id is None
    assert act(d, slots).selected_slot_id == 4
    assert d.act(0.0, 0.5, 0.0, 0.0, 0.0, slots).selected_slot_id is None      # moved a little: not far enough
    assert d.act(0.0, CFG.retry_after_driven_m + 0.1, 0.0, 0.0, 0.0, slots).selected_slot_id == 3
    assert d.rejected == set()


def test_reversing_back_counts_as_moving_on():
    from park_sim.driver import Rejections
    r = Rejections(2.0)
    r.moved_to(10.0, 9.0)
    r.add(5)
    r.moved_to(9.0, 9.0)
    assert 5 in r
    r.moved_to(8.0, 9.0)
    assert 5 not in r and r.ids == set()


def test_a_stall_tried_again_after_moving_is_not_stuck_red_or_green():
    """After no path the supervisor still holds it as its last selection."""
    from types import SimpleNamespace
    from park_sim.hmi import selected_now
    tick = SimpleNamespace(row={"sup_state": ("SLOTS_AVAILABLE",)}, selected_slot=3)
    assert selected_now(tick) is None
    assert slot_statuses((SlotCandidate(3),), selected_now(tick), set()) == {3: OFFERED}
    assert selected_now(SimpleNamespace(row={"sup_state": ("WAIT_CONFIRM",)}, selected_slot=3)) == 3


def test_the_failure_message_goes_once_the_stall_can_be_tried_again(window):
    from types import SimpleNamespace
    from park_sim.simulator import PlanAttempt
    run_person(Person(quit_after=2), observer=window)
    window._frame_view(window._last_tick.true_pose)
    drawn = []
    window._text = lambda target, text, *a, **k: drawn.append(text) or pygame_rect()
    tick = SimpleNamespace(failed_plan=PlanAttempt(0.0, 7, 2.0, False, (0.0, 0.0), "timed_out"), row={"sup_state": ("SLOTS_AVAILABLE",)})
    window.rejected_of = lambda: {7}
    window._draw_plan_failure(tick)
    assert drawn
    drawn.clear()
    window.rejected_of = lambda: set()
    window._draw_plan_failure(tick)
    assert not drawn


def test_a_click_on_an_unconfirmed_slot_is_ignored():
    d = manual([HmiInput(select=3)])
    assert act(d, (SlotCandidate(3, confirmed=False),)).selected_slot_id is None


# ---------- slots on screen ----------
def test_slot_status():
    slots = (SlotCandidate(1, confirmed=False), SlotCandidate(2), SlotCandidate(3), SlotCandidate(4))
    assert slot_statuses(slots, 3, {4}) == {1: UNCONFIRMED, 2: OFFERED, 3: SELECTED, 4: REJECTED}


def test_a_selected_stall_with_no_path_turns_red():
    """The supervisor keeps it as its last selection, and it used to stay green."""
    assert slot_statuses((SlotCandidate(3),), 3, {3}) == {3: REJECTED}


def test_the_panel_shows_r_while_reversing_by_hand():
    from park_sim.hmi import status_from_tick
    from types import SimpleNamespace
    row = {"gear": (1.0,), "dist_to_go": (float("nan"),), "est_x": (0.0,), "est_y": (0.0,), "x": (0.0,), "y": (0.0,), "sup_state": ("searching",),
           "sup_hmi": ("",), "v": (-1.0,), "delta": (0.0,), "sup_abort_reason": ("none",)}
    tick = SimpleNamespace(row=row, transitions=[], t=1.0, selected_slot=None, slots=())
    assert status_from_tick(tick, set()).gear == "R"
    assert status_from_tick(SimpleNamespace(**{**vars(tick), "row": {**row, "v": (1.0,)}}), set()).gear == "D"


def test_long_messages_wrap_to_fit_the_panel():
    from park_sim.hmi_view import PANEL_WIDTH, _wrap
    lines = _wrap("Parking. Keep your hands off the wheel and hold the confirm button", PANEL_WIDTH)
    assert len(lines) > 1 and all(len(line) <= PANEL_WIDTH for line in lines)


def test_the_setup_is_sensors_only_with_the_monitor_on_by_default():
    a = script.parse_args([])
    assert a.planner_map == "sensed" and a.monitor == "on"
    params = script.build_params(script.setup_of(a))
    assert params["planner"]["map"] == "sensed" and params["safety"]["enabled"] is True


def numbered(numbers, *slots):
    """slots: (id, status). Found in the order given."""
    found = [SlotCandidate(i, first_seen_t=float(k)) for k, (i, _) in enumerate(slots)]
    return numbers.update(found, dict(slots))


def test_only_available_stalls_are_numbered_counting_from_one():
    from park_sim.hmi import StallNumbers
    # slot ids have gaps, since they count stalls that turned out to be taken
    shown = numbered(StallNumbers(), (6, OFFERED), (7, UNCONFIRMED), (10, OFFERED), (13, REJECTED), (20, SELECTED))
    assert shown == {6: 1, 10: 2, 20: 3}


def test_a_stall_keeps_its_number_as_others_come_and_go_and_none_is_reused():
    from park_sim.hmi import StallNumbers
    numbers = StallNumbers()
    assert numbered(numbers, (6, OFFERED), (10, OFFERED)) == {6: 1, 10: 2}
    assert numbered(numbers, (10, OFFERED)) == {10: 2}                      # stall 1 expired, and stall 2 stays 2
    assert numbered(numbers, (10, OFFERED), (30, OFFERED)) == {10: 2, 30: 3}    # a new stall doesn't take the free 1


def test_a_stall_that_comes_back_has_its_old_number():
    from park_sim.hmi import StallNumbers
    numbers = StallNumbers()
    numbered(numbers, (6, OFFERED), (10, OFFERED))
    numbered(numbers, (10, OFFERED))
    assert numbered(numbers, (6, OFFERED), (10, OFFERED)) == {6: 1, 10: 2}


def test_a_stall_with_no_path_shows_no_number_and_keeps_it_for_later():
    from park_sim.hmi import StallNumbers
    numbers = StallNumbers()
    numbered(numbers, (6, OFFERED), (10, OFFERED))
    assert numbered(numbers, (6, REJECTED), (10, OFFERED)) == {10: 2}
    assert numbered(numbers, (6, OFFERED), (10, OFFERED))[6] == 1


def test_messages_use_the_numbers_on_screen():
    from park_sim.hmi import StallNumbers
    numbers = StallNumbers()
    numbered(numbers, (36, OFFERED))
    assert numbers.in_words("slot 36 selected") == "stall 1 selected"
    assert numbers.in_words("no path to this slot") == "no path to this slot"


def test_a_new_run_numbers_from_one_again(window):
    run_person(Person(quit_after=800), observer=window)
    seen_first = dict(window._shown_numbers)
    run_person(Person(quit_after=2), observer=window)
    assert window._shown_numbers == {} and seen_first
    assert min(seen_first.values()) >= 1


def result_of(reason="", checks=None, metrics=None, attempts=()):
    from types import SimpleNamespace
    checks = checks if checks is not None else {"completed": False}
    return SimpleNamespace(success=all(checks.values()) and not reason, reason=reason, checks=checks, metrics=metrics or {}, search=SimpleNamespace(plan_attempts=attempts))


def test_every_abort_reason_has_words():
    from park_sim.hmi import ABORTS
    from park_sim.supervisor import AbortReason
    assert {r.value for r in AbortReason} - {AbortReason.NONE.value} <= set(ABORTS)


def test_no_path_says_why():
    from types import SimpleNamespace
    from park_sim.hmi import PLAN_FAILURES, explain_result
    from park_sim import simulator as sim
    assert {sim.PLAN_STALL_BLOCKED, sim.PLAN_STALL_UNSEEN, sim.PLAN_START_BLOCKED, sim.PLAN_TIMED_OUT, sim.PLAN_NO_ROUTE} == set(PLAN_FAILURES)
    res = result_of("no path to slot 4", attempts=(SimpleNamespace(why=sim.PLAN_STALL_BLOCKED),))
    assert explain_result(res, PARAMS["criteria"]) == PLAN_FAILURES[sim.PLAN_STALL_BLOCKED]


def test_a_missed_check_says_by_how_much():
    from park_sim.hmi import explain_result
    res = result_of(checks={"lateral": False, "heading": True}, metrics={"lateral_err": -0.21})
    assert "8.3 in off" in explain_result(res, PARAMS["criteria"])


def test_the_window_shows_imperial_units():
    from types import SimpleNamespace
    from park_sim.hmi import summary_text, to_feet, to_inches, to_mph
    assert to_mph(100.0) == pytest.approx(62.1371) and to_feet(1.0) == pytest.approx(3.28084) and to_inches(1.0) == pytest.approx(39.3701)
    m = {"lateral_err": 0.0254, "heading_err_deg": 0.5, "depth_err": -0.0254, "min_clearance": 0.3048, "gear_changes": 1.0, "time_s": 20.0, "max_speed_kph": 3.6}
    text = summary_text(SimpleNamespace(planned=object(), metrics=m, checks={"lateral": True}, success=True, reason=""))
    assert "+1.0 in" in text and "-1.0 in" in text and "1.00 ft" in text and "2.2 mph" in text
    assert "km/h" not in text and " cm" not in text


def test_the_features_own_messages_follow_the_units():
    from park_sim.hmi import IMPERIAL, METRIC
    assert IMPERIAL.message("Slow below 20 km/h to search for a slot") == "Slow below 12.4 mph to search for a slot"
    assert METRIC.message("speed 3.6 km/h") == "speed 1.0 m/s"
    assert IMPERIAL.message("Select a slot") == "Select a slot"


def test_metric_matches_foxgloves_units():
    from types import SimpleNamespace
    from park_sim.hmi import METRIC, Units, explain_result, summary_text
    assert (METRIC.speed(3.6), METRIC.long(2.5), METRIC.short(0.05)) == pytest.approx((1.0, 2.5, 5.0))
    assert (METRIC.speed_unit, METRIC.long_unit, METRIC.short_unit) == ("m/s", "m", "cm")
    m = {"lateral_err": 0.05, "heading_err_deg": 0.5, "depth_err": -0.02, "min_clearance": 0.4, "gear_changes": 1.0, "time_s": 20.0, "max_speed_kph": 3.6}
    text = summary_text(SimpleNamespace(planned=object(), metrics=m, checks={"lateral": True}, success=True, reason=""), METRIC)
    assert "+5.0 cm" in text and "0.40 m" in text and "1.0 m/s" in text
    res = result_of(checks={"lateral": False}, metrics={"lateral_err": -0.21})
    assert "21.0 cm off" in explain_result(res, PARAMS["criteria"], METRIC)
    assert Units.named("metric") == METRIC and METRIC.other().imperial
    with pytest.raises(ValueError):
        Units.named("furlongs")


def test_the_u_key_switches_units_everywhere_including_a_finished_run(window):
    start = window.units
    _, res, _ = run_person(Person(quit_after=2), observer=window)
    window.show_result(res, PARAMS["criteria"])
    window._on_key_press("u")
    assert window.units == start.other() and window._backdrop is None and window._needs_redraw
    assert window.units.short_unit in window._banner[1] or res.planned is None
    window._on_key_press("u")
    assert window.units == start


def test_the_units_stay_through_a_new_run_and_the_option_sets_them(window):
    window._on_key_press("u")
    chosen = window.units
    run_person(Person(quit_after=2), observer=window)
    assert window.units == chosen
    a = script.parse_args(["--units", "metric"])
    assert script.build_params(script.setup_of(a))["interactive"]["units"] == "metric"
    assert PARAMS["interactive"]["units"] == "imperial"


def test_no_metric_units_reach_the_window():
    source = (ROOT / "park_sim" / "hmi_view.py").read_text()
    assert "km/h" not in source and '"m")' not in source and " m away" not in source


def test_an_abort_gets_plain_words():
    from park_sim.hmi import ABORTS, explain_result
    assert explain_result(result_of("aborted: obstacle_timeout"), PARAMS["criteria"]) == ABORTS["obstacle_timeout"]


def test_every_check_has_words():
    from park_sim.hmi import explain_result
    metrics = {"lateral_err": 1.0, "heading_err_deg": 9.0, "depth_err": 1.0, "min_clearance": 0.1, "gear_changes": 9.0, "max_speed_kph": 9.0, "time_s": 99.0}
    checks = {k: False for k in ("lateral", "heading", "depth", "clearance", "gear_changes", "speed", "time")}
    assert "check failed" not in explain_result(result_of(checks=checks, metrics=metrics), PARAMS["criteria"])


def test_a_real_run_records_why_its_plan_failed():
    """With no time to plan, nothing can be found, and the run's attempts say why."""
    from park_sim.simulator import plans_on_sensed_map
    from park_sim.simulator import search_trial
    params = load_params()
    params["slot_detection"]["detector"] = "fused"
    params["search"]["planning_time_limit_s"] = 0.0     # nothing can be planned in no time
    _, res, _ = search_trial(VehicleParams.from_yaml(), params, 1003)
    assert plans_on_sensed_map(params)
    failed = [a for a in res.search.plan_attempts if not a.found]
    assert failed and all(a.why in ("stall_blocked", "stall_unseen", "start_blocked", "timed_out", "no_route") for a in failed)


def test_a_stall_blocked_only_by_what_it_has_not_seen_says_so_not_that_something_is_there():
    import numpy as np
    from park_sim.scenario import load_params as params_of
    from park_sim.sensed_map import BodyOutline, SensedMap, SensedMapConfig
    from park_sim.simulator import PLAN_STALL_BLOCKED, PLAN_STALL_UNSEEN, _plan_on_sensed_map
    from park_sim.supervisor import SlotCandidate as Slot
    from types import SimpleNamespace
    p = params_of()
    v = VehicleParams.from_yaml()
    fresh = SensedMap(SensedMapConfig.from_params(p["sensed_map"]), (), {}, BodyOutline(-v.rear_overhang, v.wheelbase + v.front_overhang, v.width / 2.0))
    car = SimpleNamespace(x=-8.0, y=9.0, yaw=0.0)
    fresh.update_body(car)
    far = Slot(0, corners=((60.0, 5.5), (62.7, 5.5), (62.7, 0.0), (60.0, 0.0)))      # the slot's own margin is too small to hold the goal plus clearance everywhere, and nothing has been seen around it
    why = []
    fresh.assume_slot = lambda corners: None    # nothing assumed: all of it unknown
    assert _plan_on_sensed_map(fresh, far, car, v, p, None, 0.5, why)[0] is None
    assert why == [PLAN_STALL_UNSEEN]
    fresh._log[fresh._window(60.0, 62.7, 2.0, 3.0)[0]:fresh._window(60.0, 62.7, 2.0, 3.0)[1], fresh._window(60.0, 62.7, 2.0, 3.0)[2]:fresh._window(60.0, 62.7, 2.0, 3.0)[3]] = 3.0
    why.clear()
    assert _plan_on_sensed_map(fresh, far, car, v, p, None, 0.5, why)[0] is None
    assert why == [PLAN_STALL_BLOCKED]


def test_the_failure_shows_on_the_lot_and_clears_when_a_stall_is_picked(window):
    from types import SimpleNamespace
    from park_sim.simulator import PlanAttempt
    run_person(Person(quit_after=2), observer=window)
    tick = window._last_tick
    window._frame_view(tick.true_pose)
    fail = PlanAttempt(0.0, 7, 2.0, False, (0.0, 0.0), "timed_out")
    window.rejected_of = lambda: {7}
    drawn = []
    window._text = lambda target, text, *a, **k: drawn.append(text) or pygame_rect()
    window._draw_plan_failure(SimpleNamespace(failed_plan=fail, row={"sup_state": ("SLOTS_AVAILABLE",)}))
    assert any("Can't park" in t for t in drawn) and any("planning time" in t for t in drawn)
    drawn.clear()
    window._draw_plan_failure(SimpleNamespace(failed_plan=fail, row={"sup_state": ("SLOT_SELECTED",)}))
    assert not drawn


def pygame_rect():
    import pygame
    return pygame.Rect(0, 0, 1, 1)


def test_the_zoomed_view_follows_the_car_every_frame(window):
    from types import SimpleNamespace
    run_person(Person(quit_after=2), observer=window)
    window._on_key_press("z")
    for x, y in ((3.0, 8.0), (11.5, 9.5)):
        window._frame_view(SimpleNamespace(x=x, y=y))
        assert window._view[:2] == (x, y)


def test_a_click_finds_the_slot_it_landed_in():
    corners = {7: [(0.0, 0.0), (2.7, 0.0), (2.7, 5.5), (0.0, 5.5)], 8: [(2.7, 0.0), (5.4, 0.0), (5.4, 5.5), (2.7, 5.5)]}
    assert slot_at(1.0, 2.0, corners) == 7
    assert slot_at(3.0, 2.0, corners) == 8
    assert slot_at(6.0, 2.0, corners) is None


# ---------- recording ----------
def test_a_saved_session_gives_back_every_tick(tmp_path):
    inputs = [HmiInput()] * 3 + [HmiInput(forward=True)] * 4 + [HmiInput(select=2)] + [HmiInput()] * 2
    feed = iter(inputs)
    log = InputLog(lambda: next(feed))
    seen = [log() for _ in inputs]
    path = log.save(tmp_path / "s" / "session.json", {"seed": 5})
    replay = Replay(path)
    assert replay.setup == {"seed": 5}
    assert [replay() for _ in inputs] == seen
    assert len(log.changes) == 4        # only the changes are kept


def test_a_replay_that_runs_out_asks_to_stop(tmp_path):
    feed = iter([HmiInput()])
    log = InputLog(lambda: next(feed))
    log()
    replay = Replay(log.save(tmp_path / "session.json", {}))
    assert not replay().stop
    assert replay().stop


# ---------- whole runs, with a scripted person ----------
class Person:
    """Drives on at a walking pace, turns the feature on, takes the first slot ahead of the car, stops, and holds Enter."""

    def __init__(self, release_after=None, takeover_after=None, quit_after=None):
        self.driver = None
        self.tick = 0
        self.release_after, self.takeover_after, self.quit_after = release_after, takeover_after, quit_after
        self.maneuver_ticks = 0
        self.paused_seen = None

    def __call__(self) -> HmiInput:
        self.tick += 1
        d = self.driver
        state = d._state
        if self.quit_after is not None and self.tick > self.quit_after:
            return HmiInput(stop=True)
        if self.tick == 2:
            return HmiInput(feature=True)
        if state in (State.OFF, State.SEARCHING):
            return HmiInput(forward=self.tick < 150)
        if state is State.SLOTS_AVAILABLE:
            return HmiInput(select=d.offered[0]) if d.offered else HmiInput()
        if state is State.SLOT_SELECTED:
            return HmiInput(brake=True)
        if state is State.WAIT_CONFIRM:
            return HmiInput(confirm=True) if self.tick % 2 else HmiInput()
        if state is State.MANEUVERING:
            self.maneuver_ticks += 1
            if self.takeover_after is not None and self.maneuver_ticks > self.takeover_after:
                return HmiInput(confirm=True, takeover=True)
        if self.release_after is not None and self.release_after < self.maneuver_ticks <= self.release_after + 100:
            self.maneuver_ticks += 1
            return HmiInput()       # let go for 2 s
        return HmiInput(confirm=True)


def run_person(person, setup=SETUP, observer=None):
    def source():
        return person()

    class Grab(ManualDriver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            person.driver = self

    original, script.ManualDriver = script.ManualDriver, Grab
    try:
        return script.run_once(setup, source, observer)
    finally:
        script.ManualDriver = original


def test_a_person_can_drive_up_pick_a_slot_and_let_it_park():
    _, res, _ = run_person(Person())
    states = [t.to_state for t in res.transitions]
    assert states[:2] == [State.SEARCHING, State.SLOTS_AVAILABLE]
    assert State.MANEUVERING in states
    assert res.final_state is State.COMPLETED
    assert res.success, res.summary()


def test_letting_go_pauses_the_car_and_holding_again_carries_on():
    person = Person(release_after=250)
    _, res, _ = run_person(person)
    moves = [(t.from_state, t.to_state) for t in res.transitions]
    assert (State.MANEUVERING, State.PAUSED) in moves
    assert (State.PAUSED, State.MANEUVERING) in moves
    paused = next(t for t in res.transitions if t.to_state is State.PAUSED)
    assert paused.reason == "brake pressed"
    assert res.final_state is State.COMPLETED


def test_the_car_really_stops_while_paused():
    _, res, _ = run_person(Person(release_after=250))
    log = res.log
    entered = next(t.t for t in res.transitions if t.to_state is State.PAUSED)
    resumed = next(t.t for t in res.transitions if t.from_state is State.PAUSED)
    settled = (log["t"] > entered + 2.0) & (log["t"] < resumed)
    if settled.any():
        assert np.abs(log["v"][settled]).max() < 0.05
    assert np.abs(log["speed_cmd"][(log["t"] > entered) & (log["t"] < resumed)]).max() == 0.0


def test_taking_over_the_wheel_aborts_the_maneuver():
    _, res, _ = run_person(Person(takeover_after=100))
    assert res.final_state is State.ABORTED
    assert next(t for t in res.transitions if t.to_state is State.ABORTED).reason.startswith("steering torque")
    assert not res.success


def test_quitting_mid_run_still_gives_a_result():
    _, res, _ = run_person(Person(quit_after=50))
    assert res.reason == "stopped by the driver"
    assert len(res.log["t"]) >= 50


def test_a_recorded_session_replays_to_the_same_end(tmp_path):
    person = Person()
    log = InputLog(person)
    _, first, _ = run_person_logged(person, log)
    path = log.save(tmp_path / "session.json", SETUP)
    replay = Replay(path)
    _, again, _ = script.run_once(replay.setup, replay)
    assert again.final_state is first.final_state
    assert again.success == first.success
    assert again.log["x"][-1] == pytest.approx(first.log["x"][-1], abs=1e-9)
    assert again.log["y"][-1] == pytest.approx(first.log["y"][-1], abs=1e-9)
    assert [t.reason for t in again.transitions] == [t.reason for t in first.transitions]


def run_person_logged(person, log):
    class Grab(ManualDriver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            person.driver = self

    original, script.ManualDriver = script.ManualDriver, Grab
    try:
        return script.run_once(SETUP, log)
    finally:
        script.ManualDriver = original


def test_the_seed_gives_the_same_lot_as_monte_carlo():
    from park_sim.simulator import search_trial
    plain, _, _ = search_trial(VEHICLE, load_params(), 1003)
    mine, _, _ = run_person(Person(quit_after=2))
    assert [s.occupied for s in plain.stalls] == [s.occupied for s in mine.stalls]
    assert all(np.array_equal(a, b) for a, b in zip(plain.parked_cars, mine.parked_cars))


def test_the_monitor_switch_gets_past_a_scenarios_own_setting():
    """cart_in_stall turns the monitor on, and the scenario loader puts that on again over the params."""
    off = {**SETUP, "scenario": "cart_in_stall", "monitor": "off"}
    on = {**SETUP, "scenario": "cart_in_stall", "monitor": None}
    assert run_person(Person(quit_after=5), off)[1].safety is None
    assert run_person(Person(quit_after=5), on)[1].safety is not None


# ---------- the window, under Agg ----------
@pytest.fixture
def window():
    pygame = pytest.importorskip("pygame")
    from park_sim.hmi_view import HmiWindow
    made = HmiWindow(PARAMS["interactive"], KeyState(), realtime=False)
    yield made
    pygame.quit()


def test_the_window_draws_a_whole_run_and_a_click_selects_a_slot(window):
    _, res, _ = run_person(Person(quit_after=1000), observer=window)
    assert window.frames_drawn > 100
    assert window.status.state in {s.value for s in State}
    window.show_result(res, PARAMS["criteria"])
    assert window._corners      # the scripted person gets as far as a slot
    slot, corners = next(iter(window._corners.items()))
    window._on_click(window._points([np.mean(corners, axis=0)])[0])
    assert window.keys.read().select == slot


def test_a_click_outside_the_lot_selects_nothing(window):
    run_person(Person(quit_after=2), observer=window)
    window._on_click((5, 790))
    assert window.keys.read().select is None


def test_meters_and_pixels_convert_both_ways(window):
    run_person(Person(quit_after=2), observer=window)
    x, y = window._to_world(window._points([(12.5, 7.5)])[0])
    assert (x, y) == pytest.approx((12.5, 7.5))


def test_the_zoomed_view_keeps_the_car_in_the_picture(window):
    from park_sim.hmi_view import MAP
    run_person(Person(quit_after=2), observer=window)
    window._on_key_press("z")
    assert window.zoomed
    run_person(Person(quit_after=320), observer=window)
    log_car = window._last_tick.true_pose
    assert MAP.collidepoint(window._points([(log_car.x, log_car.y)])[0])


def test_window_keys_quit_and_restart(window):
    window._on_key_press("r")
    assert window.next_action == "again" and window.keys.stop_requested
    window.next_action = None
    window._on_key_press("escape")
    assert window.wait_for_next() == "quit"


def test_real_key_events_reach_the_inputs_and_losing_focus_lets_go_of_held_keys(window):
    pygame = pytest.importorskip("pygame")
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w))
    window.pump()
    u = window.keys.read()
    assert u.confirm and u.forward
    pygame.event.post(pygame.event.Event(pygame.WINDOWFOCUSLOST))
    window.pump()
    u = window.keys.read()
    assert u.confirm and not u.forward


def test_closing_the_window_quits(window):
    pygame = pytest.importorskip("pygame")
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    window.pump()
    assert window.closed and window.wait_for_next() == "quit"
    assert window.keys.stop_requested


def test_the_empty_option_makes_a_lot_with_only_painted_lines():
    sc, res, _ = run_person(Person(quit_after=2), {**SETUP, "empty": True})
    assert sc.parked_cars == [] and sc.lines


def test_the_window_module_is_the_only_one_needing_pygame():
    for name in ("hmi.py", "simulator.py", "driver.py"):
        assert "pygame" not in (ROOT / "park_sim" / name).read_text().split('"""', 2)[-1]


# ---------- rules ----------
def test_the_logic_module_does_not_import_plotting():
    tree = ast.parse((ROOT / "park_sim" / "hmi.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    assert "matplotlib" not in imported


def test_the_script_has_a_headless_replay(tmp_path):
    person = Person(quit_after=30)
    log = InputLog(person)
    run_person_logged(person, log)
    path = log.save(tmp_path / "session.json", SETUP)
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "interactive.py"), "--replay", str(path), "--headless"],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 2, done.stderr
    assert "stopped by the driver" in done.stdout


def test_the_config_section_says_its_numbers_are_placeholders():
    text = (ROOT / "config" / "park_params.yaml").read_text()
    assert "every number [est]" in text.split("interactive:")[1].splitlines()[0]
