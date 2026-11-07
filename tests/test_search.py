"""Search phase: aisle lot, ideal slot detector, scripted driver, and the end-to-end flow."""
import copy
import importlib.util
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from park_sim import simulator  # noqa: E402
from park_sim.driver import POLICIES, DriverConfig, SearchDriver  # noqa: E402
from park_sim.hybrid_astar import HybridAStar  # noqa: E402
from park_sim.mcap_export import SEARCH_TOPICS, export_run  # noqa: E402
from park_sim.scenario import Pose, goal_from_corners, load_params, make_aisle_lot, make_perpendicular  # noqa: E402
from park_sim.simulator import run_search, search_trial  # noqa: E402
from park_sim.slot_detection import IdealSlotDetector, SlotDetector  # noqa: E402
from park_sim.supervisor import SlotCandidate, State  # noqa: E402
from park_sim.vehicle import VehicleParams  # noqa: E402

PARAMS = load_params()
PARAMS["slot_detection"]["detector"] = "ideal"   # these tests pin down the flow, on known stall ids
VEH = VehicleParams.from_yaml()
SEED = 1      # free stalls 1, 3, 23, 24, 29, 35
SLOT_FIELDS = {"slot_id", "corners", "entry_side", "slot_type", "occupied", "confidence", "first_seen_t"}


def driver_cfg(**changes):
    return replace(DriverConfig.from_params(PARAMS["search"]), **changes)


@pytest.fixture(scope="module")
def lot():
    return make_aisle_lot(VEH, PARAMS, np.random.default_rng(SEED))


@pytest.fixture(scope="module")
def parked(lot):
    return run_search(lot)


@pytest.fixture(scope="module")
def drive_through(lot):
    return run_search(lot, driver_cfg(policy="never_select"))


def free_ids(lot):
    return {s.slot_id for s in lot.stalls if not s.occupied}


def order_of(res):
    return [tr.to_state for tr in res.transitions]


# ---------- aisle lot ----------
def test_lot_has_both_rows_of_twenty_with_unique_ids(lot):
    assert PARAMS["aisle_lot"]["stalls_per_row"] == 20
    assert len(lot.stalls) == 40
    assert sorted(s.slot_id for s in lot.stalls) == list(range(40))
    assert {s.row for s in lot.stalls} == {0, 1}
    assert all(s.slot_type == "perpendicular" for s in lot.stalls)


def test_stall_count_is_configurable():
    params = copy.deepcopy(PARAMS)
    params["aisle_lot"]["stalls_per_row"] = 8
    lot = make_aisle_lot(VEH, params, np.random.default_rng(0))
    assert len(lot.stalls) == 16
    assert lot.aisle_end_x == pytest.approx(8 * params["scenario"]["stall_width"])


@pytest.mark.parametrize("seed", range(8))
def test_two_or_three_stalls_stay_free_in_a_full_lot(seed):
    lot = make_aisle_lot(VEH, PARAMS, np.random.default_rng(seed), occupancy_prob=1.0)
    assert len(free_ids(lot)) in (2, 3)


def test_occupancy_is_configurable():
    empty = make_aisle_lot(VEH, PARAMS, np.random.default_rng(0), occupancy_prob=0.0)
    assert len(free_ids(empty)) == 40 and not empty.parked_cars
    half = make_aisle_lot(VEH, PARAMS, np.random.default_rng(0), occupancy_prob=0.5)
    assert 10 < len(free_ids(half)) < 30


def test_every_parked_car_sits_in_an_occupied_stall(lot):
    assert len(lot.parked_cars) == 40 - len(free_ids(lot))
    for car in lot.parked_cars:
        stall = lot.nearest_stall(*car.mean(axis=0))
        assert stall.occupied
        assert math.dist(car.mean(axis=0), stall.center) < 0.5


def test_car_enters_at_one_end_on_the_centerline(lot):
    assert lot.start.x < 0 and lot.start.yaw == 0.0
    assert lot.start.y == pytest.approx((lot.stalls[0].center[1] + lot.stalls[20].center[1]) / 2)
    assert lot.start.y == lot.aisle_y
    assert lot.goal is None and lot.target_stall is None
    assert lot.collision_free(lot.start.x, lot.start.y, lot.start.yaw, 0.5)


def test_stall_opening_faces_the_aisle(lot):
    for stall in lot.stalls:
        opening_y = stall.corners[0][1]
        assert stall.corners[1][1] == opening_y
        assert abs(opening_y - lot.aisle_y) < abs(stall.center[1] - lot.aisle_y)
        assert stall.entry_side == ("+y" if stall.row == 0 else "-y")


@pytest.mark.parametrize("maneuver", ["reverse_in", "head_in"])
def test_goal_from_corners_matches_the_single_stall_scenario(maneuver):
    params = copy.deepcopy(PARAMS)
    params["scenario"]["maneuver"] = maneuver
    single = make_perpendicular(VEH, params, np.random.default_rng(0))
    w, d = params["scenario"]["stall_width"], params["scenario"]["stall_depth"]
    x0 = params["scenario"]["target_stall"] * w
    goal = goal_from_corners(((x0, d), (x0 + w, d), (x0 + w, 0.0), (x0, 0.0)), VEH, maneuver)
    assert (goal.x, goal.y, goal.yaw) == pytest.approx((single.goal.x, single.goal.y, single.goal.yaw))


def test_goal_in_the_far_row_is_the_mirror_image(lot):
    near, far = lot.with_target(lot.stalls[3]).goal, lot.with_target(lot.stalls[23]).goal
    assert far.x == pytest.approx(near.x)
    assert far.y - lot.aisle_y == pytest.approx(lot.aisle_y - near.y)
    assert far.yaw == pytest.approx(-near.yaw)


def test_with_target_shares_the_distance_field(lot):
    target = lot.with_target(lot.stalls[3])
    assert target._dist is lot._dist and lot.goal is None
    assert target.stall_center == lot.stalls[3].center


def test_exact_clearance_shortcut_matches_checking_every_obstacle(lot):
    from park_sim.geometry import polygon_distance
    for x, y, yaw in [(10.0, 9.15, 0.0), (25.0, 8.0, 0.7), (50.0, 11.0, -1.2), (-5.0, 9.0, 0.1)]:
        body = VEH.footprint(x, y, yaw)
        assert lot.clearance_exact(x, y, yaw) == min(polygon_distance(body, ob) for ob in lot.obstacles)


# ---------- ideal detector ----------
def sweep(detector, lot, x_from=None, x_to=None, y=None, step=0.1):
    """Drive straight along the aisle, returning {slot id: x when first seen}."""
    y = lot.aisle_y if y is None else y
    x, t, seen = lot.start.x if x_from is None else x_from, 0.0, {}
    end = lot.aisle_end_x + 5.0 if x_to is None else x_to
    while x <= end:
        for c in detector.update(t, Pose(x, y, 0.0)):
            seen.setdefault(c.slot_id, x)
        x += step
        t += 0.1
    return seen


def make_detector(lot, **changes):
    return IdealSlotDetector(lot.stalls, VEH, {**PARAMS["slot_detection"], **changes})


@pytest.mark.parametrize("seed", range(4))
def test_detector_finds_every_free_stall_passed_and_no_occupied_one(seed):
    lot = make_aisle_lot(VEH, PARAMS, np.random.default_rng(seed))
    assert set(sweep(make_detector(lot), lot)) == free_ids(lot)


def test_detector_defaults_match_the_request():
    assert PARAMS["slot_detection"]["detection_range"] == 6.0
    assert PARAMS["slot_detection"]["pass_margin"] == 1.0


def test_detector_reports_only_after_passing_by_the_margin(lot):
    seen = sweep(make_detector(lot), lot, step=0.01)
    for stall in lot.stalls:
        if not stall.occupied:
            assert seen[stall.slot_id] == pytest.approx(stall.corners[1][0] + 1.0, abs=0.011)


def test_pass_margin_is_configurable(lot):
    seen = sweep(make_detector(lot, pass_margin=3.0), lot, step=0.01)
    stall = lot.stalls[1]
    assert seen[1] == pytest.approx(stall.corners[1][0] + 3.0, abs=0.011)


def test_detector_ignores_stalls_not_yet_passed(lot):
    last_x = lot.stalls[3].corners[1][0] + 1.0   # stall 3 and stall 23 across from it only just count
    seen = sweep(make_detector(lot), lot, x_to=last_x + 0.05, step=0.01)
    assert set(seen) == {1, 3, 23}


def test_detector_needs_the_opening_within_range(lot):
    """On the centerline the openings are 3.65 m from the axis, 2.66 m from the car's side."""
    assert sweep(make_detector(lot, detection_range=2.5), lot) == {}
    assert set(sweep(make_detector(lot, detection_range=3.2), lot)) == free_ids(lot)


def test_detector_range_is_measured_from_the_side_nearest_the_stall(lot):
    """Far from the row 1 openings by range, near the row 0 ones."""
    near_row0 = lot.stalls[0].corners[0][1] + VEH.width / 2 + 0.5   # side 0.5 m off the row 0 openings
    seen = sweep(make_detector(lot, detection_range=1.0), lot, y=near_row0)
    assert set(seen) == {s for s in free_ids(lot) if s < 20}


def test_range_must_hold_while_near_enough_not_only_after(lot):
    """Starting past a stall, out of range of it, never reports it."""
    x_from = lot.stalls[1].corners[1][0] + 20.0
    assert 1 not in sweep(make_detector(lot), lot, x_from=x_from)


def test_candidate_has_every_requested_field_in_the_map_frame(lot):
    detector = make_detector(lot)
    sweep(detector, lot)
    stalls = {s.slot_id: s for s in lot.stalls}
    assert SLOT_FIELDS <= set(SlotCandidate.__dataclass_fields__)
    for c in detector.slots:
        assert c.corners == stalls[c.slot_id].corners and len(c.corners) == 4
        assert c.entry_side == stalls[c.slot_id].entry_side
        assert c.slot_type == "perpendicular"
        assert c.occupied is False and c.confirmed is True and c.confidence == 1.0
        assert c.first_seen_t > 0.0


def test_detector_never_forgets_and_keeps_the_first_seen_time(lot):
    detector = make_detector(lot)
    x, t, history = lot.start.x, 0.0, {}
    n_before = 0
    while x < lot.aisle_end_x + 60.0:    # far past everything
        slots = detector.update(t, Pose(x, lot.aisle_y, 0.0))
        assert len(slots) >= n_before
        n_before = len(slots)
        for c in slots:
            assert history.setdefault(c.slot_id, c.first_seen_t) == c.first_seen_t
        x, t = x + 0.2, t + 0.1
    assert set(history) == free_ids(lot)
    assert [c.first_seen_t for c in detector.slots] == sorted(history.values())


def test_detector_interface_is_abstract():
    with pytest.raises(TypeError):
        SlotDetector()


# ---------- driver ----------
def slot(sid, x=0.0, first_seen=0.0):
    return SlotCandidate(sid, corners=((x, 0.0), (x + 2.7, 0.0), (x + 2.7, -5.5), (x, -5.5)), first_seen_t=first_seen)


def make_driver(**changes):
    return SearchDriver(driver_cfg(**changes), VEH.wheelbase, VEH.max_steer, 9.15, 54.0)


def test_policies_are_the_requested_ones():
    assert set(POLICIES) == {"select_first", "select_nearest_after_passing", "select_by_index", "select_first_after",
                             "never_select"}


def test_config_rejects_bad_policies():
    with pytest.raises(ValueError, match="unknown driver policy"):
        driver_cfg(policy="select_best")
    with pytest.raises(ValueError, match="slot_index"):
        driver_cfg(policy="select_by_index", slot_index=None)


def test_search_speed_default_is_ten():
    assert driver_cfg().speed_kph == 10.0


def test_driver_presses_the_feature_button_once():
    d = make_driver()
    assert d.act(0.0, 0, 9.15, 0, 2.8, ()).feature_button is True
    assert d.act(0.02, 0, 9.15, 0, 2.8, ()).feature_button is False


@pytest.mark.parametrize("policy, kwargs, expected", [
    ("select_first", {}, 7),
    ("select_nearest_after_passing", {}, 9),
    ("select_by_index", {"slot_index": 8}, 8),
    ("never_select", {}, None),
])
def test_policy_picks(policy, kwargs, expected):
    d = make_driver(policy=policy, reaction_time_s=0.0, **kwargs)
    d.observe(State.SLOTS_AVAILABLE)
    slots = (slot(7, x=0.0, first_seen=1.0), slot(8, x=3.0, first_seen=2.0), slot(9, x=6.0, first_seen=3.0))
    assert d.act(5.0, 10.0, 9.15, 0.0, 2.8, slots).selected_slot_id == expected


def test_select_by_index_waits_for_its_slot():
    d = make_driver(policy="select_by_index", slot_index=8, reaction_time_s=0.0)
    d.observe(State.SLOTS_AVAILABLE)
    assert d.act(1.0, 5.0, 9.15, 0.0, 2.8, (slot(7),)).selected_slot_id is None
    assert d.act(2.0, 10.0, 9.15, 0.0, 2.8, (slot(7), slot(8))).selected_slot_id == 8


def test_driver_takes_its_reaction_time_before_selecting():
    d = make_driver(reaction_time_s=0.5)
    d.observe(State.SLOTS_AVAILABLE)
    assert d.act(1.0, 5.0, 9.15, 0.0, 2.8, (slot(7),)).selected_slot_id is None
    assert d.act(1.4, 6.0, 9.15, 0.0, 2.8, (slot(7),)).selected_slot_id is None
    assert d.act(1.5, 6.3, 9.15, 0.0, 2.8, (slot(7),)).selected_slot_id == 7


def test_driver_brakes_once_a_slot_is_selected():
    d = make_driver()
    cruising = d.act(0.0, 5.0, 9.15, 0.0, 2.8, ())
    assert cruising.speed_cmd == pytest.approx(10 / 3.6) and cruising.brake == 0.0
    for state in (State.SLOT_SELECTED, State.WAIT_CONFIRM):
        d.observe(state)
        act = d.act(1.0, 6.0, 9.15, 0.0, 1.0, (slot(7),))
        assert act.speed_cmd == 0.0 and act.brake > 0.0


def test_driver_presses_confirm_after_its_delay_and_holds_it():
    d = make_driver(confirm_delay_s=1.0)
    d.observe(State.WAIT_CONFIRM)
    assert d.act(5.0, 6.0, 9.15, 0.0, 0.0, (slot(7),)).confirm_button is False
    assert d.act(5.9, 6.0, 9.15, 0.0, 0.0, (slot(7),)).confirm_button is False
    assert d.act(6.0, 6.0, 9.15, 0.0, 0.0, (slot(7),)).confirm_button is True
    d.observe(State.MANEUVERING)
    assert d.act(7.0, 6.0, 9.15, 0.0, 0.0, (slot(7),)).confirm_button is True


def test_driver_lifts_off_the_brake_when_the_system_has_the_car():
    """Regression: past the end of the aisle, a held brake paused the maneuver until it timed out."""
    d = make_driver()
    assert d.act(0.0, 57.0, 9.15, 0.0, 0.0, ()).brake > 0.0
    for state in (State.PLANNING, State.MANEUVERING):
        d.observe(state)
        assert d.act(1.0, 57.0, 9.15, 0.0, 0.0, ()).brake == 0.0


def test_driver_drops_a_slot_after_failed_planning():
    d = make_driver(reaction_time_s=0.0)
    slots = (slot(7, first_seen=1.0), slot(8, x=3.0, first_seen=2.0))
    d.observe(State.SLOTS_AVAILABLE)
    assert d.act(3.0, 10.0, 9.15, 0.0, 0.0, slots).selected_slot_id == 7
    d.observe(State.PLANNING)
    d.observe(State.SLOTS_AVAILABLE)
    act = d.act(4.0, 10.0, 9.15, 0.0, 0.0, slots)
    assert act.selected_slot_id == 8 and act.confirm_button is False


def test_gave_up_only_when_stopped_past_the_end_with_nothing_to_select():
    d = make_driver(reaction_time_s=10.0)
    d.act(0.0, 30.0, 9.15, 0.0, 0.0, ())
    assert not d.gave_up(0.0, 0.02)              # not at the end yet
    d.act(1.0, 57.0, 9.15, 0.0, 1.0, ())
    assert not d.gave_up(1.0, 0.02)              # still rolling
    assert d.gave_up(0.0, 0.02)
    d.act(2.0, 57.0, 9.15, 0.0, 0.0, (slot(7),))
    assert not d.gave_up(0.0, 0.02)              # something still to select


def test_lane_keeping_steers_back_to_the_centerline():
    d = make_driver()
    assert d.act(0.0, 5.0, 8.65, 0.0, 2.8, ()).steer_cmd > 0.0   # right of center: steer left
    assert d.act(0.0, 5.0, 9.65, 0.0, 2.8, ()).steer_cmd < 0.0
    assert d.act(0.0, 5.0, 9.15, 0.0, 2.8, ()).steer_cmd == pytest.approx(0.0)


@pytest.mark.parametrize("speed", [5.0, 15.0])
def test_lane_keeping_converges_from_an_offset_start(speed):
    lot = make_aisle_lot(VEH, PARAMS, np.random.default_rng(SEED), start=Pose(-8.0, 9.15 + 0.6, math.radians(4.0)))
    res = run_search(lot, driver_cfg(policy="never_select", speed_kph=speed))
    tail = res.log["x"] > 20.0
    assert np.abs(res.log["y"][tail] - lot.aisle_y).max() < 0.02
    assert np.abs(res.log["yaw"][tail]).max() < math.radians(0.5)
    cruising = res.log["x"] < lot.aisle_end_x
    assert res.log["v"][cruising].max() == pytest.approx(speed / 3.6)
    assert res.log["v"][cruising].min() == pytest.approx(speed / 3.6)


# ---------- planner ----------
def test_planner_honors_its_time_limit(lot):
    planner = HybridAStar(lot.with_target(lot.stalls[3]))
    assert planner.plan(Pose(15.0, lot.aisle_y, 0.0), time_limit_s=0.0) is None
    assert planner.plan(Pose(15.0, lot.aisle_y, 0.0), time_limit_s=30.0) is not None


# ---------- end to end ----------
def test_search_run_parks_in_the_chosen_slot_and_meets_every_criterion(parked, lot):
    assert parked.success, parked.summary()
    assert parked.final_state is State.COMPLETED
    assert parked.search.slot_id == 1
    stall = lot.stalls[1]
    log = parked.log
    off = VEH.center_offset()
    cx = log["x"][-1] + off * math.cos(log["yaw"][-1])
    cy = log["y"][-1] + off * math.sin(log["yaw"][-1])
    assert abs(cx - stall.center[0]) <= PARAMS["criteria"]["lateral_tol"]
    assert abs(cy - stall.center[1]) <= PARAMS["criteria"]["depth_tol"]
    assert set(parked.checks) == {"completed", "lateral", "heading", "depth", "clearance",
                                  "gear_changes", "speed", "time"}


def test_search_run_goes_through_the_states_in_order(parked):
    assert order_of(parked) == [State.SEARCHING, State.SLOTS_AVAILABLE, State.SLOT_SELECTED, State.WAIT_CONFIRM,
                                State.PLANNING, State.MANEUVERING, State.COMPLETED]


def test_car_drives_at_search_speed_until_selection_then_stops(parked):
    log = parked.log
    selected = next(tr.t for tr in parked.transitions if tr.to_state is State.SLOT_SELECTED)
    before = log["t"] <= selected
    assert log["v"][before] == pytest.approx(10 / 3.6)
    stopped = next(tr.t for tr in parked.transitions if tr.to_state is State.WAIT_CONFIRM)
    assert stopped > selected
    assert abs(log["v"][np.searchsorted(log["t"], stopped)]) <= PARAMS["supervisor"]["standstill_speed"]


def test_planning_starts_from_the_real_stop_pose(parked):
    log = parked.log
    attempt = parked.search.plan_attempts[-1]
    i = int(np.searchsorted(log["t"], attempt.t))
    assert parked.planned.x[0] == log["x"][i] and parked.planned.y[0] == log["y"][i]
    assert parked.planned.yaw[0] == log["yaw"][i]
    assert abs(log["x"][i] - log["x"][0]) > 10.0   # nowhere near where the car began
    goal = parked.search.scenario.goal
    assert (parked.planned.x[-1], parked.planned.y[-1]) == pytest.approx((goal.x, goal.y))


def test_planning_time_is_logged_and_under_the_limit(parked):
    assert PARAMS["search"]["planning_time_limit_s"] == 2.0
    assert PARAMS["supervisor"]["planning_timeout_s"] == 2.0
    attempts = parked.search.plan_attempts
    assert len(attempts) == 1 and attempts[0].found
    assert 0.0 < attempts[0].time_s < 2.0
    assert parked.metrics["planning_time_s"] == attempts[0].time_s


def test_search_metrics_match_the_log_and_transitions(parked):
    m, log = parked.metrics, parked.log
    at = {tr.to_state: tr.t for tr in parked.transitions}
    assert m["selection_to_stop_s"] == pytest.approx(at[State.WAIT_CONFIRM] - at[State.SLOT_SELECTED])
    assert m["confirm_to_parked_s"] == pytest.approx(at[State.COMPLETED] - at[State.PLANNING])
    assert m["total_time_s"] == pytest.approx(log["t"][-1])
    # straight along the aisle at constant speed from the button press to the selection
    assert m["search_distance_m"] == pytest.approx(at[State.SLOT_SELECTED] * 10 / 3.6, rel=0.01)
    assert m["slots_detected"] == len(parked.search.slots)
    assert m["time_s"] < m["confirm_to_parked_s"] < m["total_time_s"]


def test_speed_criterion_covers_only_the_maneuver_but_clearance_the_whole_run(parked, lot):
    assert parked.metrics["max_speed_kph"] <= PARAMS["criteria"]["max_speed_kph"]
    assert np.abs(parked.log["v"]).max() * 3.6 == pytest.approx(10.0)
    samples = simulator.clearance_samples(lot, parked.log)
    assert parked.metrics["min_clearance"] == min(samples[1])
    assert len(samples[0]) > len(parked.log["t"]) / 5


@pytest.mark.parametrize("cfg, chosen", [
    (dict(policy="select_nearest_after_passing"), None),
    (dict(policy="select_by_index", slot_index=23), 23),    # across the aisle
    (dict(policy="select_by_index", slot_index=35, speed_kph=15.0), 35),
    (dict(policy="select_first", speed_kph=5.0), 1),
])
def test_other_policies_and_speeds_park(lot, cfg, chosen):
    res = run_search(lot, driver_cfg(**cfg))
    assert res.success, res.summary()
    if chosen is not None:
        assert res.search.slot_id == chosen


def test_last_stall_of_the_aisle_parks(lot):
    """Regression: the car stops past where the driver brakes for the aisle end."""
    params = copy.deepcopy(PARAMS)
    lot = make_aisle_lot(VEH, params, np.random.default_rng(12), occupancy_prob=0.6)
    last = next(s.slot_id for s in reversed(lot.stalls[:20]) if not s.occupied)
    res = run_search(lot, driver_cfg(policy="select_by_index", slot_index=last))
    assert res.success, res.summary()


def test_head_in_search_run_parks():
    params = copy.deepcopy(PARAMS)
    params["scenario"]["maneuver"] = "head_in"
    lot = make_aisle_lot(VEH, params, np.random.default_rng(SEED))
    res = run_search(lot, driver_cfg(policy="select_by_index", slot_index=23))
    assert res.search.scenario.goal.yaw == pytest.approx(math.pi / 2)
    assert res.search.slot_id == 23
    assert res.final_state is State.COMPLETED, res.summary()


class ShiftedDetector(SlotDetector):
    """Stand-in for a sensor-based detector: own ids, corners off by a known amount."""
    def __init__(self, stall, shift):
        super().__init__()
        self.stall, self.shift = stall, shift

    def update(self, t, pose):
        if not self._slots and pose.x > self.stall.corners[1][0] + 1.0:
            corners = tuple((x + self.shift, y) for x, y in self.stall.corners)
            self._slots[501] = SlotCandidate(501, corners=corners, first_seen_t=t)
        return self.slots


def test_another_detector_is_a_drop_in_replacement(lot):
    """The maneuver aims at the slot as detected and is scored against the true stall."""
    res = run_search(lot, detector=ShiftedDetector(lot.stalls[3], 0.05))
    assert res.search.slot_id == 501
    assert res.search.scenario.stall_center == lot.stalls[3].center
    assert res.planned.x[-1] == pytest.approx(lot.with_target(lot.stalls[3]).goal.x + 0.05)
    assert res.success, res.summary()


# ---------- never_select ----------
def test_never_select_drives_the_whole_aisle_without_a_maneuver(drive_through, lot):
    res = drive_through
    assert res.log["x"][-1] > lot.aisle_end_x
    assert set(res.log["sup_state"]) <= {"SEARCHING", "SLOTS_AVAILABLE"}
    assert res.final_state in (State.SEARCHING, State.SLOTS_AVAILABLE)
    assert np.isnan(res.log["t_maneuver"]).all()
    assert res.planned is None and not res.search.plan_attempts
    assert res.search.slot_id is None
    assert not res.checks and not res.success
    assert res.reason == "reached the end of the aisle"
    assert abs(res.log["v"][-1]) <= PARAMS["supervisor"]["standstill_speed"]


def test_never_select_detects_every_free_stall(drive_through, lot):
    assert {c.slot_id for c in drive_through.search.slots} == free_ids(lot)


def test_never_select_keeps_nan_for_phases_not_reached(drive_through, lot):
    m = drive_through.metrics
    assert math.isnan(m["selection_to_stop_s"]) and math.isnan(m["confirm_to_parked_s"])
    assert math.isnan(m["planning_time_s"])
    assert m["search_distance_m"] > lot.aisle_end_x
    assert m["total_time_s"] == pytest.approx(drive_through.log["t"][-1])


def test_selecting_an_occupied_stall_never_starts_a_maneuver(lot):
    occupied = next(s.slot_id for s in lot.stalls if s.occupied)
    res = run_search(lot, driver_cfg(policy="select_by_index", slot_index=occupied))
    assert res.planned is None and res.search.slot_id is None
    assert res.reason == "reached the end of the aisle"
    assert np.isnan(res.log["t_maneuver"]).all()


# ---------- planning failure ----------
def test_planning_over_the_limit_falls_back_to_slots_available():
    params = copy.deepcopy(PARAMS)
    params["search"]["planning_time_limit_s"] = 0.0
    lot = make_aisle_lot(VEH, params, np.random.default_rng(SEED))
    res = run_search(lot)
    order = order_of(res)
    i = order.index(State.PLANNING)
    assert order[i + 1] is State.SLOTS_AVAILABLE
    fallback = res.transitions[i + 1]
    assert fallback.reason == "no path to this slot"
    assert State.MANEUVERING not in order
    assert res.planned is None and not res.success
    assert res.search.plan_attempts
    assert all(not a.found for a in res.search.plan_attempts)
    assert np.isnan(res.log["t_maneuver"]).all()
    assert res.final_state is State.SLOTS_AVAILABLE
    assert res.reason.startswith("no path to slot")
    # a slot may be tried again, but only after the car has moved on: "no path" is only true of where it was
    log, last_try = res.log, {}
    driven = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(log["x"]), np.diff(log["y"])))])
    for a in res.search.plan_attempts:
        at = driven[min(int(np.searchsorted(log["t"], a.t)), len(driven) - 1)]
        if a.slot_id in last_try:
            assert at - last_try[a.slot_id] >= params["search"]["retry_after_driven_m"] - 0.1
        last_try[a.slot_id] = at


def test_a_plan_slower_than_the_limit_is_discarded(lot, monkeypatch):
    real = HybridAStar.plan

    class Clock:
        now = 0.0

    def slow_plan(self, start=None, time_limit_s=None):
        path = real(self, start)
        Clock.now += 3.0
        return path

    monkeypatch.setattr(HybridAStar, "plan", slow_plan)
    monkeypatch.setattr(simulator.time, "perf_counter", lambda: Clock.now)
    path, took = simulator._plan_to_slot(lot, slot_for(lot, 3), Pose(15.0, lot.aisle_y, 0.0), None, 2.0)
    assert path is None and took == pytest.approx(3.0)


def slot_for(lot, sid):
    return SlotCandidate(sid, corners=lot.stalls[sid].corners)


def test_a_slot_with_no_path_is_tried_again_once_the_car_has_moved_on(lot, monkeypatch):
    real, calls = simulator._plan_to_slot, []

    def fail_until_moved(sc, slot, start, *args):
        calls.append((slot.slot_id, start.x))
        return (None, 0.01) if len(calls) == 1 else real(sc, slot, start, *args)

    monkeypatch.setattr(simulator, "_plan_to_slot", fail_until_moved)
    only_this = DriverConfig(**{**DriverConfig.from_params(lot.cfg["search"]).__dict__, "policy": "select_by_index", "slot_index": 1})
    res = run_search(lot, only_this)
    assert [c[0] for c in calls[:2]] == [1, 1]
    assert calls[1][1] - calls[0][1] >= lot.cfg["search"]["retry_after_driven_m"] - 0.1
    assert res.success, res.summary()


def test_driver_picks_another_slot_after_a_failed_plan(lot, monkeypatch):
    real, calls = simulator._plan_to_slot, []

    def fail_first(sc, slot, *args):
        calls.append(slot.slot_id)
        return (None, 0.01) if len(calls) == 1 else real(sc, slot, *args)

    monkeypatch.setattr(simulator, "_plan_to_slot", fail_first)
    res = run_search(lot)
    assert calls[0] == 1 and calls[1] != 1
    assert res.search.slot_id == calls[1]
    assert [a.found for a in res.search.plan_attempts] == [False, True]
    assert res.success, res.summary()
    order = order_of(res)
    assert order.count(State.PLANNING) == 2 and order[-1] is State.COMPLETED


# ---------- Monte Carlo ----------
def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_search_monte_carlo_passes_at_least_95_percent():
    rows = []
    passed, fails = load_script("monte_carlo").batch(50, VEH, PARAMS, 0.0, 0.0, rows=rows, mode="search")
    assert passed / 50 >= 0.95, fails
    mc = PARAMS["search"]["monte_carlo"]
    assert mc["speed_kph"] == [5.0, 15.0]
    speeds = [r["speed_kph"] for r in rows]
    assert min(speeds) >= 5.0 and max(speeds) <= 15.0 and max(speeds) - min(speeds) > 5.0
    assert max(r["occupancy"] for r in rows) > min(r["occupancy"] for r in rows)
    assert len({r["chosen_slot"] for r in rows}) > 10
    assert max(r["planning_time_s"] for r in rows) < 2.0


def test_search_trial_is_repeatable_and_picks_a_free_slot():
    fast = copy.deepcopy(PARAMS)
    fast["search"]["monte_carlo"]["speed_kph"] = [12.0, 12.0]
    lot, res, drawn = search_trial(VEH, fast, 3)
    assert drawn["chosen_slot"] in free_ids(lot)
    assert drawn["speed_kph"] == 12.0
    assert res.search.slot_id == drawn["chosen_slot"]
    _, again, drawn2 = search_trial(VEH, fast, 3)
    assert drawn2 == drawn and np.array_equal(again.log["x"], res.log["x"])


def test_monte_carlo_cli_search_mode(tmp_path):
    out = tmp_path / "search.csv"
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "monte_carlo.py"), "-n", "1", "--mode", "search",
                           "--csv", str(out)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    header = out.read_text().splitlines()[0].split(",")
    assert {"occupancy", "speed_kph", "chosen_slot", "search_distance_m", "selection_to_stop_s",
            "confirm_to_parked_s", "total_time_s", "planning_time_s"} <= set(header)
    assert "search:" in done.stdout


# ---------- MCAP ----------
def read_mcap(path):
    pytest.importorskip("mcap")
    spec = importlib.util.spec_from_file_location("mcap_tests", ROOT / "tests" / "test_mcap_export.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_mcap(path)


@pytest.fixture(scope="module")
def exported(lot, parked, tmp_path_factory):
    pytest.importorskip("mcap")
    return read_mcap(export_run(lot, parked, tmp_path_factory.mktemp("search") / "run.mcap"))


def test_search_mcap_has_every_channel(exported):
    channels, schemas, _, messages = exported
    assert set(channels) == set(SEARCH_TOPICS)
    assert all(messages[t] for t in SEARCH_TOPICS)
    assert schemas["/slots"].name == "foxglove.SceneUpdate"


def test_slots_appear_over_time_when_first_seen(exported, parked):
    updates = exported[3]["/slots"]
    assert len(updates) == len(parked.search.slots) + 1     # each slot, then the chosen one again
    times = [ns for ns, _ in updates]
    assert times == sorted(times)
    first = {}
    for ns, msg in updates:
        first.setdefault(msg.entities[0].id, ns)
    for c in parked.search.slots:
        assert first[f"slot_{c.slot_id}"] == pytest.approx(c.first_seen_t * 1e9, abs=1)
    assert times[0] > 0


def test_selected_slot_changes_color_and_shows_the_goal(exported, parked):
    updates = exported[3]["/slots"]
    selected_t = next(tr.t for tr in parked.transitions if tr.to_state is State.SLOT_SELECTED)
    ns, msg = next((n, m) for n, m in updates if len(m.entities) == 2)
    assert ns == pytest.approx(selected_t * 1e9, abs=1)
    assert msg.entities[0].id == f"slot_{parked.search.slot_id}"
    assert msg.entities[0].texts[0].text == str(parked.search.slot_id)
    assert msg.entities[1].id == "goal_footprint"
    same = [m for _, m in updates if m.entities[0].id == msg.entities[0].id]
    assert same[0].entities[0].cubes[0].color != same[1].entities[0].cubes[0].color


def test_target_and_plan_are_not_shown_before_the_driver_chooses(exported, parked):
    lot_ids = {e.id for e in exported[3]["/lot"][0][1].entities}
    assert lot_ids == {"stall_lines", "parked_cars", "curbs"}
    plan = exported[3]["/plan"]
    assert len(plan) == 1
    assert plan[0][0] == pytest.approx(parked.search.plan_attempts[-1].t * 1e9, abs=1)
    assert plan[0][0] > 0 and plan[0][1].entities


def test_search_mcap_metrics_and_clearance(exported, parked):
    doc = exported[3]["/metrics"][0][1]
    assert doc["slot_id"] == parked.search.slot_id
    assert [s["slot_id"] for s in doc["slots"]] == [c.slot_id for c in parked.search.slots]
    assert doc["plan_attempts"][0]["found"] is True
    assert doc["metrics"]["total_time_s"] == pytest.approx(parked.metrics["total_time_s"])
    state = [d for _, d in exported[3]["/state"]]
    assert state[-1]["min_clearance"] == parked.metrics["min_clearance"]
    assert all(b["min_clearance"] <= a["min_clearance"] for a, b in zip(state, state[1:]))


def test_run_with_no_maneuver_exports_the_drive(lot, drive_through, tmp_path):
    channels, _, _, messages = read_mcap(export_run(lot, drive_through, tmp_path / "drive.mcap"))
    assert set(channels) == set(SEARCH_TOPICS)
    assert len(messages["/tf"]) == len(drive_through.log["t"])
    assert not messages["/plan"][0][1].entities
    doc = messages["/metrics"][0][1]
    assert doc["slot_id"] is None and doc["metrics"]["selection_to_stop_s"] is None


def test_search_layout_shows_slots_and_frames_the_aisle():
    panels = json.loads((ROOT / "config" / "foxglove_search_layout.json").read_text())["configById"]
    view = panels["3D!parking"]
    assert view["topics"]["/slots"]["visible"] is True
    al, sc = PARAMS["aisle_lot"], PARAMS["scenario"]
    assert view["cameraState"]["targetOffset"][0] == pytest.approx(al["stalls_per_row"] * sc["stall_width"] / 2)


# ---------- demo ----------
def demo(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "run_search_demo.py"), "--seed", str(SEED), *args],
                          capture_output=True, text=True)


def test_demo_runs_the_whole_flow_and_saves_png_and_mcap(tmp_path):
    png, mcap = tmp_path / "s.png", tmp_path / "s.mcap"
    done = demo("--png", str(png), "--mcap", str(mcap))
    assert done.returncode == 0, done.stderr
    assert png.read_bytes().startswith(b"\x89PNG")
    assert "SLOT_SELECTED -> WAIT_CONFIRM" in done.stdout
    pytest.importorskip("mcap")
    assert set(read_mcap(mcap)[0]) == set(SEARCH_TOPICS) | {"/sensed_map"}     # the default detector is sensor-based, so it plans on the sensed map


@pytest.mark.parametrize("detector", ["fused", "lines"])
def test_a_car_parks_in_a_lot_with_only_painted_lines(detector):
    """No cars anywhere, so nothing but paint says where a stall is. On the sensed map, the default."""
    from park_sim.vehicle import VehicleParams
    params = load_params()
    params["slot_detection"]["detector"] = detector
    params["search"]["monte_carlo"]["occupancy_prob"] = [0.0, 0.0]
    sc, res, _ = search_trial(VehicleParams.from_yaml(), params, 1001)
    assert sc.parked_cars == []
    assert res.success, res.summary()
    assert res.search.sensed_frames


def test_demo_never_select(tmp_path):
    done = demo("--policy", "never_select", "--png", str(tmp_path / "n.png"), "--mcap", str(tmp_path / "n.mcap"))
    assert done.returncode == 0, done.stderr
    assert "PLANNING" not in done.stdout
