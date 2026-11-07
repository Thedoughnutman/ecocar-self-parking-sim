"""Harder straight-stall cases: obstacles around the target stall, who can see them, and what the car does."""
import csv
import functools
import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from park_sim.driver import DriverConfig, SearchDriver  # noqa: E402
from park_sim.geometry import polygons_overlap  # noqa: E402
from park_sim.mcap_export import export_run  # noqa: E402
from park_sim.scenario import add_target_obstacles, load_params, make_aisle_lot  # noqa: E402
from park_sim.scenario_files import (STRAIGHT_STALL_CASES, case_outcome, final_stall_id, list_scenarios,  # noqa: E402
                                     load_scenario_file, lowest_clearance, meets_expect, merged, run_scenario_file,
                                     tried_target)
from park_sim.sensors import make_suite  # noqa: E402
from park_sim.sensors.scene import scene_arrays, sight_blocked  # noqa: E402
from park_sim.sensors.ultrasonic import UltrasonicMeasurement  # noqa: E402
from park_sim.simulator import RunResult  # noqa: E402
from park_sim.supervisor import SlotCandidate  # noqa: E402
from park_sim.vehicle import VehicleParams, VehicleState  # noqa: E402

PARAMS = load_params()
VEH = VehicleParams.from_yaml()
SEEDS = (0, 1, 2)
DETECTORS = ("ideal", "fused")
CRIT = PARAMS["criteria"]


def empty_lot(params=PARAMS):
    return make_aisle_lot(VEH, params, np.random.default_rng(0), occupancy_prob=0.0)


def stall_at(lot, row, index):
    return next(st for st in lot.stalls if st.row == row and st.index == index)


def with_obstacles(lot, stall, *specs, seed=0):
    return add_target_obstacles(lot, stall, specs, np.random.default_rng(seed))


@functools.lru_cache(maxsize=None)
def case_run(name, detector, seed):
    params = merged(PARAMS, {"slot_detection": {"detector": detector}})
    return run_scenario_file(load_scenario_file(name), seed, "search", VEH, params)


# ---------- placing obstacles ----------
@pytest.mark.parametrize("side,toward", [("far", 1.0), ("near", -1.0)])
@pytest.mark.parametrize("row", [0, 1])
def test_neighbor_is_exactly_that_far_over_the_line(side, toward, row):
    lot = empty_lot()
    stall = stall_at(lot, row, 5)
    out = with_obstacles(lot, stall, {"kind": "neighbor_over_line", "side": side, "over_line_m": 0.3})
    assert len(out.parked_cars) == 1
    xs = out.parked_cars[0][:, 0]
    line = stall.corners[1][0] if toward > 0 else stall.corners[0][0]
    edge = xs.min() if toward > 0 else xs.max()
    assert (line - edge) * toward == pytest.approx(0.3)
    assert stall_at(out, row, 5 + int(toward)).occupied and not stall_at(out, row, 5).occupied


def test_neighbor_replaces_a_car_already_there_and_leaves_the_rest():
    lot = make_aisle_lot(VEH, PARAMS, np.random.default_rng(3), occupancy_prob=0.8)
    stall = next(st for st in lot.stalls if not st.occupied and 0 < st.index < 19
                 and stall_at(lot, st.row, st.index + 1).occupied)
    out = with_obstacles(lot, stall, {"kind": "neighbor_over_line", "side": "far", "over_line_m": 0.3})
    assert len(out.parked_cars) == len(lot.parked_cars)
    assert len(out.obstacles) == len(lot.obstacles)
    assert not out.lot_objects


def test_neighbor_goes_on_the_other_side_at_the_end_of_a_row():
    lot = empty_lot()
    stall = stall_at(lot, 0, 19)
    out = with_obstacles(lot, stall, {"kind": "neighbor_over_line", "side": "far", "over_line_m": 0.3})
    assert stall_at(out, 0, 18).occupied
    assert out.parked_cars[0][:, 0].max() == pytest.approx(stall.corners[0][0] + 0.3)


@pytest.mark.parametrize("row", [0, 1])
def test_pillar_is_at_the_aisle_side_corner_outside_the_stall(row):
    lot = empty_lot()
    stall = stall_at(lot, row, 5)
    out = with_obstacles(lot, stall, {"kind": "pillar", "side": "far"})
    pillar = out.lot_objects[0].polygon
    corner_x, corner_y = stall.corners[1]
    assert pillar[:, 0].min() == pytest.approx(corner_x)        # its face is on the line, nothing inside the stall
    assert pillar[:, 1].min() < corner_y < pillar[:, 1].max()


@pytest.mark.parametrize("row", [0, 1])
def test_cart_is_always_where_the_parked_body_would_go(row):
    lot = empty_lot()
    stall = stall_at(lot, row, 5)
    target = lot.with_target(stall)
    body = VEH.footprint(target.goal.x, target.goal.y, target.goal.yaw)
    for seed in range(30):
        out = with_obstacles(lot, stall, {"kind": "cart"}, seed=seed)
        assert polygons_overlap(out.lot_objects[0].polygon, body)


def test_cart_position_is_drawn_from_the_given_rng_only():
    lot = empty_lot()
    stall = stall_at(lot, 0, 5)
    a = with_obstacles(lot, stall, {"kind": "cart"}, seed=7).lot_objects[0].polygon
    b = with_obstacles(lot, stall, {"kind": "cart"}, seed=7).lot_objects[0].polygon
    c = with_obstacles(lot, stall, {"kind": "cart"}, seed=8).lot_objects[0].polygon
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_unknown_obstacle_kind_is_refused():
    lot = empty_lot()
    with pytest.raises(ValueError, match="unknown obstacle kind"):
        with_obstacles(lot, stall_at(lot, 0, 5), {"kind": "boulder"})


def test_adding_obstacles_leaves_the_original_lot_alone():
    lot = empty_lot()
    count, dist = len(lot.obstacles), lot._dist.copy()
    with_obstacles(lot, stall_at(lot, 0, 5), {"kind": "cart"}, {"kind": "pillar"})
    assert len(lot.obstacles) == count and np.array_equal(lot._dist, dist) and not lot.lot_objects


# ---------- clearance checks ----------
@pytest.mark.parametrize("kind", ["pillar", "cart"])
def test_clearance_checks_see_the_object_even_if_the_planner_does_not(kind):
    lot = empty_lot()
    stall = stall_at(lot, 0, 5)
    out = with_obstacles(lot, stall, {"kind": kind, "known_to_planner": False})
    center = out.lot_objects[0].polygon.mean(axis=0)
    x, y, yaw = center[0], center[1], math.pi / 2.0
    assert out.clearance_exact(x, y, yaw) == 0.0
    assert out.clearance_fast(x, y, yaw)[0] < 0.1
    assert lot.clearance_exact(x, y, yaw) > 0.0


# ---------- the planner's knowledge ----------
def test_a_lot_with_nothing_hidden_is_its_own_planner_view():
    lot = empty_lot()
    assert lot.planner_view() is lot
    known = with_obstacles(lot, stall_at(lot, 0, 5), {"kind": "pillar", "known_to_planner": True})
    assert known.planner_view() is known


@pytest.mark.parametrize("spec", [{"kind": "cart"}, {"kind": "pillar"}, {"kind": "cart", "known_to_planner": False}])
def test_objects_stay_out_of_the_planners_map_unless_a_file_says_otherwise(spec):
    lot = empty_lot()
    out = with_obstacles(lot, stall_at(lot, 0, 5), spec)
    polygon = out.lot_objects[0].polygon
    view = out.planner_view()
    assert not any(ob is polygon for ob in view.obstacles) and any(ob is polygon for ob in out.obstacles)
    assert not view.lot_objects
    center = polygon.mean(axis=0)
    assert view.clearance_exact(center[0], center[1], math.pi / 2.0) > 0.0


def test_a_known_object_is_on_the_planners_map():
    lot = empty_lot()
    out = with_obstacles(lot, stall_at(lot, 0, 5), {"kind": "pillar", "known_to_planner": True},
                         {"kind": "cart", "known_to_planner": False})
    view = out.planner_view()
    pillar, cart = out.lot_objects
    assert any(ob is pillar.polygon for ob in view.obstacles)
    assert not any(ob is cart.polygon for ob in view.obstacles)


def test_planner_view_follows_the_target_and_shares_one_distance_field():
    lot = empty_lot()
    stall = stall_at(lot, 0, 5)
    out = with_obstacles(lot, stall, {"kind": "cart"})
    a, b = out.with_target(stall), out.with_target(stall_at(lot, 0, 9))
    assert a.planner_view().goal == a.goal and b.planner_view().goal == b.goal
    assert a.planner_view()._dist is b.planner_view()._dist


def test_the_plan_goes_right_through_a_cart_it_was_not_told_about():
    """The proof nothing added the cart: the ideal run's plan ends on top of it."""
    sc, res, _ = case_run("cart_in_stall", "ideal", 0)
    path = res.search.plans[0][1]
    x, y, yaw = path.x[-1], path.y[-1], path.yaw[-1]
    assert sc.clearance_exact(x, y, yaw) == 0.0
    assert sc.planner_view().clearance_exact(x, y, yaw) >= PARAMS["planner"]["collision_margin"] - 0.05


def test_told_about_the_cart_the_planner_refuses_the_stall():
    file = replace(load_scenario_file("cart_in_stall"), obstacles=({"kind": "cart", "known_to_planner": True},))
    params = merged(PARAMS, {"slot_detection": {"detector": "ideal"}})
    sc, res, drawn = run_scenario_file(file, 0, "search", VEH, params)
    assert case_outcome(sc, res, drawn["chosen_slot"]) == "no_path"
    assert lowest_clearance(sc, res) > 0.0


def test_pillar_case_tells_the_planner_and_cart_case_does_not():
    assert load_scenario_file("pillar_at_corner").obstacles[0]["known_to_planner"] is True
    assert load_scenario_file("cart_in_stall").obstacles[0]["known_to_planner"] is False


# ---------- the sensors ----------
def lowest_ultrasonic(lot, state, ticks=20):
    suite = make_suite(seed=0)
    lowest = math.inf
    for i in range(ticks):
        for m in suite.step(lot, state, i * PARAMS["controller"]["dt"] * 5):
            if isinstance(m, UltrasonicMeasurement) and m.distance is not None:
                lowest = min(lowest, m.distance)
    return lowest


@pytest.mark.parametrize("kind", ["cart", "pillar"])
def test_ultrasonics_hear_the_object_whether_or_not_the_planner_knows_it(kind):
    lot = empty_lot()
    stall = stall_at(lot, 0, 5)
    out = with_obstacles(lot, stall, {"kind": kind, "known_to_planner": False})
    center = out.lot_objects[0].polygon.mean(axis=0)
    # Backing toward the object, tail 1 m away from its center
    state = VehicleState(center[0], center[1] + 1.0 + VEH.rear_overhang, math.pi / 2.0)
    with_object, without = lowest_ultrasonic(out, state), lowest_ultrasonic(lot, state)
    assert with_object < 1.2 and with_object < without


@pytest.mark.parametrize("kind", ["cart", "pillar"])
def test_objects_hide_paint_from_the_cameras(kind):
    lot = empty_lot()
    out = with_obstacles(lot, stall_at(lot, 0, 5), {"kind": kind})
    arrays = scene_arrays(out)
    assert len(arrays.car_a) == 4 and len(arrays.obstacle_a) == 4 * len(out.obstacles)
    center = out.lot_objects[0].polygon.mean(axis=0)
    behind = center + np.array([0.0, 3.0])
    front = center - np.array([0.0, 3.0])
    assert sight_blocked(front, behind[None, :], arrays.car_a, arrays.car_b)[0]


def test_lidar_sees_the_object():
    from park_sim.sensors import load_sensor_params
    cfg = load_sensor_params()
    cfg["lidar"]["enabled"] = True
    lot = empty_lot()
    stall = stall_at(lot, 0, 5)
    out = with_obstacles(lot, stall, {"kind": "pillar", "known_to_planner": False})
    center = out.lot_objects[0].polygon.mean(axis=0)
    state = VehicleState(center[0], center[1] + 2.0, math.pi / 2.0)
    suite_with, suite_without = make_suite(cfg, 0), make_suite(cfg, 0)

    def nearest(suite, scene):
        best = math.inf
        for i in range(10):
            for m in suite.step(scene, state, i * 0.1):
                if hasattr(m, "ranges"):
                    finite = np.asarray(m.ranges)[np.isfinite(m.ranges)]
                    best = min(best, float(finite.min())) if len(finite) else best
        return best

    assert nearest(suite_with, out) < nearest(suite_without, lot)


# ---------- the MCAP ----------
def test_lot_objects_are_in_the_mcap(tmp_path):
    pytest.importorskip("mcap")
    from test_mcap_export import read_mcap
    sc, res, _ = case_run("cart_in_stall", "ideal", 0)
    path = tmp_path / "cart.mcap"
    export_run(sc, res, path)
    _, _, _, messages = read_mcap(path)
    by_id = {e.id: e for e in messages["/lot"][0][1].entities}
    assert len(by_id["lot_objects"].cubes) == 1
    cube = by_id["lot_objects"].cubes[0]
    center = sc.lot_objects[0].polygon.mean(axis=0)
    assert cube.pose.position.x == pytest.approx(center[0]) and cube.pose.position.y == pytest.approx(center[1])
    assert cube.size.z == pytest.approx(PARAMS["foxglove"]["cart_height"])


def test_a_lot_without_objects_keeps_its_entities(tmp_path):
    """The two existing lot-entity tests pin this for plain runs; narrow aisle has no objects either."""
    pytest.importorskip("mcap")
    from test_mcap_export import read_mcap
    sc, res, _ = case_run("narrow_aisle_6m", "ideal", 0)
    path = tmp_path / "plain.mcap"
    export_run(sc, res, path)
    assert "lot_objects" not in {e.id for e in read_mcap(path)[3]["/lot"][0][1].entities}


# ---------- the driver's row ----------
def slot(slot_id, x, y, t=0.0):
    corners = ((x - 1.35, y + 2.75), (x + 1.35, y + 2.75), (x + 1.35, y - 2.75), (x - 1.35, y - 2.75))
    return SlotCandidate(slot_id=slot_id, confirmed=True, corners=corners, entry_side="+y", slot_type="perpendicular",
                         occupied=False, confidence=1.0, first_seen_t=t)


def row_driver(row_y):
    cfg = replace(DriverConfig.from_params(PARAMS["search"]), policy="select_first_after", after_x=10.0,
                  row_y=row_y, reaction_time_s=0.0)
    driver = SearchDriver(cfg, VEH.wheelbase, VEH.max_steer, aisle_y=9.15, aisle_end_x=54.0)
    return driver


def test_driver_stays_in_the_row_it_was_given():
    near_row, far_row = slot(1, 12.0, 2.75), slot(2, 11.0, 15.55, t=-1.0)
    assert row_driver(2.75)._options((near_row, far_row)) == [near_row]
    assert row_driver(15.55)._options((near_row, far_row)) == [far_row]


def test_driver_without_a_row_takes_both():
    near_row, far_row = slot(1, 12.0, 2.75), slot(2, 11.0, 15.55)
    assert row_driver(None)._options((near_row, far_row)) == [near_row, far_row]


def test_driver_takes_any_row_at_the_end_of_the_aisle():
    driver = row_driver(2.75)
    far_row = slot(2, 11.0, 15.55)
    assert driver._options((far_row,)) == []
    driver._past_end = True
    assert driver._options((far_row,)) == [far_row]


def test_plain_config_has_no_row():
    assert PARAMS["search"]["row_y"] is None


# ---------- files ----------
@pytest.mark.parametrize("name", STRAIGHT_STALL_CASES)
def test_case_file_is_search_flow_and_says_what_a_safe_car_does(name):
    assert name in list_scenarios()
    file = load_scenario_file(name)
    assert file.flows == ("search",)
    assert file.expect["contacts"] == 0 and file.expect["completed_meets_criteria"] is True
    assert file.overrides["safety"]["enabled"] is True


@pytest.mark.parametrize("name", ["cart_in_stall", "neighbor_over_line"])
def test_blocked_cases_do_not_expect_parking_in_the_target(name):
    assert "parked_in_target" not in load_scenario_file(name).expect["outcome_any_of"]


def test_file_overrides_give_the_named_sizes():
    assert load_scenario_file("narrow_aisle_6m").overrides["scenario"]["aisle_width"] == 6.0
    assert load_scenario_file("narrow_stall_2p4m").overrides["scenario"]["stall_width"] == 2.4
    assert load_scenario_file("neighbor_over_line").obstacles[0]["over_line_m"] == 0.3


def test_obstacles_outside_the_search_flow_are_refused(tmp_path, monkeypatch):
    from park_sim import scenario_files
    (tmp_path / "bad.yaml").write_text("name: bad\ndescription: x\nflows: [single_stall]\nobstacles:\n  - {kind: cart}\n")
    monkeypatch.setattr(scenario_files, "SCENARIO_DIR", tmp_path)
    with pytest.raises(ValueError, match="only work in the search flow"):
        scenario_files.load_scenario_file("bad")


def test_narrow_cases_really_build_narrow_lots():
    aisle = case_run("narrow_aisle_6m", "ideal", 0)[0]
    stall = case_run("narrow_stall_2p4m", "ideal", 0)[0]
    assert aisle.stalls[20].corners[0][1] - aisle.stalls[0].corners[0][1] == pytest.approx(6.0)
    assert stall.stalls[0].corners[1][0] - stall.stalls[0].corners[0][0] == pytest.approx(2.4)


# ---------- judging a run ----------
def judged(success, reason="", completed=None):
    res = RunResult(None, reason=reason)
    res.checks = {"all": success}
    if completed is not None:
        res.metrics["completed"] = completed
    return res


EXPECT = {"outcome_any_of": ["parked_elsewhere", "no_path", "aborted"], "contacts": 0, "completed_meets_criteria": True}


def test_expect_refuses_an_outcome_not_listed():
    assert not meets_expect(EXPECT, "parked_in_target", 1.0, judged(True))


def test_expect_refuses_contact_whatever_the_outcome():
    assert not meets_expect(EXPECT, "aborted", 0.0, judged(False))
    assert meets_expect(EXPECT, "aborted", 0.01, judged(False))


def test_expect_wants_a_parked_car_to_meet_criteria_but_not_a_refused_one():
    assert not meets_expect(EXPECT, "parked_elsewhere", 0.5, judged(False))
    assert meets_expect(EXPECT, "parked_elsewhere", 0.5, judged(True))
    assert meets_expect(EXPECT, "no_path", 5.0, judged(False))


# ---------- every case, end to end ----------
@pytest.mark.parametrize("name", STRAIGHT_STALL_CASES)
@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("seed", SEEDS)
def test_case_runs_and_ends_cleanly(name, detector, seed):
    sc, res, drawn = case_run(name, detector, seed)
    assert res.safety is not None           # the monitor is on for these cases
    outcome = case_outcome(sc, res, drawn["chosen_slot"])
    assert outcome in ("parked_in_target", "parked_elsewhere", "no_path", "no_slot", "aborted", "fault", "incomplete")
    assert abs(res.log["v"][-1]) <= 0.1     # the run ended stopped, not by running out of time on the move
    assert isinstance(tried_target(sc, res, drawn["chosen_slot"]), bool)
    assert lowest_clearance(sc, res) > 0.0, f"contact: {res.summary()}"


def finished_body(sc, res):
    x, y, yaw = res.log["x"][-1], res.log["y"][-1], res.log["yaw"][-1]
    return VEH.footprint(x, y, yaw)


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("seed", SEEDS)
def test_cart_is_never_parked_on(detector, seed):
    sc, res, drawn = case_run("cart_in_stall", detector, seed)
    assert lowest_clearance(sc, res) > 0.0
    assert final_stall_id(sc, res) != drawn["chosen_slot"]
    assert not polygons_overlap(finished_body(sc, res), sc.lot_objects[0].polygon)
    assert case_outcome(sc, res, drawn["chosen_slot"]) in ("parked_elsewhere", "no_path", "no_slot", "aborted")


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("seed", SEEDS)
def test_neighbor_over_the_line_is_never_touched_nor_its_space_entered(detector, seed):
    sc, res, drawn = case_run("neighbor_over_line", detector, seed)
    assert lowest_clearance(sc, res) > 0.0
    stall = next(st for st in sc.stalls if st.slot_id == drawn["chosen_slot"])
    x_near, x_far = stall.corners[0][0], stall.corners[1][0]
    over = 0.3
    # The strip of the stall the neighbor has taken, on whichever side it is
    neighbor_far = stall.index < 19
    strip_x = (x_far - over, x_far) if neighbor_far else (x_near, x_near + over)
    y0, y1 = sorted((stall.corners[0][1], stall.corners[2][1]))
    strip = np.array([[strip_x[0], y0], [strip_x[1], y0], [strip_x[1], y1], [strip_x[0], y1]])
    assert not polygons_overlap(finished_body(sc, res), strip)


def test_ideal_detector_offers_the_blocked_stalls_so_something_else_must_refuse():
    """Stall.occupied says nothing about a cart or a neighbor's overhang, so the ideal run is a real test."""
    for name in ("cart_in_stall", "neighbor_over_line"):
        sc, res, drawn = case_run(name, "ideal", 0)
        assert not next(st for st in sc.stalls if st.slot_id == drawn["chosen_slot"]).occupied
        assert tried_target(sc, res, drawn["chosen_slot"])


def test_monitor_is_what_stops_the_car_for_the_cart():
    sc, res, drawn = case_run("cart_in_stall", "ideal", 0)
    assert res.metrics["obstacle_stops"] >= 1.0 and res.metrics["time_paused_s"] > 0.0
    assert case_outcome(sc, res, drawn["chosen_slot"]) == "aborted"
    assert "obstacle_timeout" in res.reason


# ---------- the script ----------
def test_script_writes_the_prompts_columns(tmp_path):
    spec = importlib.util.spec_from_file_location("straight_stall_cases", ROOT / "scripts" / "straight_stall_cases.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    out = tmp_path / "cases.csv"
    assert script.main(["-n", "1", "--cases", "narrow_aisle_6m", "--detectors", "ideal", "--csv", str(out)]) == 0
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    for column in ("case", "detector", "pass_rate", "outcomes", "lowest_clearance_m", "contacts"):
        assert row[column] != ""
    assert row["case"] == "narrow_aisle_6m" and row["detector"] == "ideal" and row["runs"] == "1"
    assert (tmp_path / "cases_runs.csv").is_file()
