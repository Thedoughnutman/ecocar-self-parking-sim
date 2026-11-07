"""The sensed map: what it believes from each kind of evidence, what it may read, and planning on it."""
import ast
import csv
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from park_sim import simulator
from park_sim.geometry import rect
from park_sim.hybrid_astar import PlannedPath
from park_sim.localization import PoseEstimate
from park_sim.scenario import Pose, Scenario, load_params, make_aisle_lot, make_perpendicular, scene_from_distance_field
from park_sim.sensed_map import (ASSUMED, FREE, OCCUPIED, UNKNOWN, BodyOutline, RangeSensor, SensedMap, SensedMapConfig,
                                 SensedMapFrame)
from park_sim.sensors import load_sensor_params, make_suite
from park_sim.sensors.scene import scene_arrays
from park_sim.simulator import run_closed_loop, run_search, search_trial
from park_sim.supervisor import SlotCandidate
from park_sim.vehicle import VehicleParams

ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params()
VEHICLE = VehicleParams.from_yaml()
FORWARD = RangeSensor("front", 0.0, 0.0, 0.0, math.radians(15.0), 0.2, 4.5)
SIDE = RangeSensor("side", 0.0, 0.0, math.pi / 2.0, math.radians(15.0), 0.2, 4.5)
BODY = BodyOutline(-1.0, 4.0, 1.0)


def sensed_params(detector="fused", estimator=None, **sections):
    params = dict(PARAMS)
    params["planner"] = {**PARAMS["planner"], "map": "sensed"}
    params["slot_detection"] = {**PARAMS["slot_detection"], "detector": detector}
    if estimator:
        params["localization"] = {**PARAMS["localization"], "estimator": estimator}
    params.update(sections)
    return params


def new_map(sensors=(FORWARD, SIDE), cameras=None):
    return SensedMap(SensedMapConfig.from_params(PARAMS["sensed_map"]), sensors, cameras or {}, BODY)


def at(x=0.0, y=0.0, yaw=0.0):
    return Pose(x, y, yaw)


def drive_past(m, xs, y_obstacle, sensor="side"):
    """A sideways sensor driving along y = 0, reading an obstacle wall at y_obstacle for x in [lo, hi] if given."""
    lo, hi = xs
    for x in np.arange(-10.0, 10.0, 0.2):
        m.add_range(sensor, y_obstacle if lo <= x <= hi else None, at(x))


# ---------- unknown, free, occupied ----------
def test_a_map_with_no_data_knows_nothing():
    m = new_map()
    assert m.cell_at(0.0, 0.0) == UNKNOWN
    assert m.frame(0.0) is None
    assert m.counts() == {"free": 0, "occupied": 0, "assumed": 0}


def test_a_clear_reading_frees_the_cone_and_nothing_past_it():
    m = new_map()
    m.add_range("front", None, at())
    m.add_range("front", None, at())
    assert m.cell_at(2.0, 0.0) == FREE
    assert m.cell_at(4.3, 0.0) == FREE
    assert m.cell_at(5.0, 0.0) == UNKNOWN       # out of range
    assert m.cell_at(2.0, 2.0) == UNKNOWN       # out of the cone
    assert m.cell_at(-2.0, 0.0) == UNKNOWN      # behind it


def test_the_blind_zone_is_not_called_free():
    m = new_map()
    m.add_range("front", None, at())
    m.add_range("front", None, at())
    assert m.cell_at(0.1, 0.0) == UNKNOWN


def test_one_empty_reading_frees_nothing():
    """It may be a dropout. It used to clear the whole cone, out to 4.5 m, and a third of the inside of parked cars read as free."""
    m = new_map()
    m.add_range("front", None, at())
    assert m.cell_at(2.0, 0.0) == UNKNOWN
    assert m.cell_at(4.3, 0.0) == UNKNOWN


def test_a_dropout_between_returns_does_not_count():
    m = new_map()
    for x in (0.0, 0.2, 0.4):
        m.add_range("front", 1.0, at(x=x))
        m.add_range("front", None, at(x=x + 0.1))
    assert m.cell_at(3.0, 0.0) == UNKNOWN       # past every return: only the empty readings could have called it free


def test_empty_readings_count_per_sensor():
    m = new_map()
    m.add_range("front", None, at())
    m.add_range("side", None, at())
    assert m.cell_at(2.0, 0.0) == UNKNOWN
    assert m.cell_at(0.0, 2.0) == UNKNOWN


def test_a_return_starts_the_count_again():
    m = new_map()
    m.add_range("front", None, at())
    m.add_range("front", 1.0, at())
    m.add_range("front", None, at(x=0.2))
    assert m.cell_at(3.0, 0.0) == UNKNOWN
    m.add_range("front", None, at(x=0.2))
    assert m.cell_at(3.0, 0.0) == FREE


def test_a_return_frees_short_of_it_and_marks_it_after_two_looks():
    m = new_map()
    m.add_range("front", 3.0, at())
    assert m.cell_at(3.0, 0.0) == UNKNOWN       # one return isn't enough
    assert m.cell_at(2.0, 0.0) == FREE
    m.add_range("front", 3.0, at(x=0.02, y=0.0, yaw=math.radians(3.0)))
    assert m.cell_at(3.0, 0.0) == OCCUPIED
    assert m.cell_at(3.6, 0.0) == UNKNOWN       # nothing is known behind the return


def test_a_sensor_standing_still_does_not_pile_up_a_reading():
    m = new_map()
    for _ in range(50):
        m.add_range("front", 3.0, at())
    assert m.cell_at(3.0, 0.0) == UNKNOWN


def test_a_new_object_stops_being_free_while_the_sensor_stands_still():
    m = new_map()
    for k in range(8):      # well past one reading's worth of free evidence
        m.add_range("front", None, at(yaw=math.radians(k * 3.0 - 10.0)))
    m.add_range("front", None, at())
    assert m.cell_at(2.0, 0.0) == FREE
    for _ in range(5):
        m.add_range("front", 2.0, at())
    assert m.cell_at(2.0, 0.0) == UNKNOWN       # the planner keeps out: unknown is blocked
    m.add_range("front", 2.0, at(yaw=math.radians(3.0)))    # one more look, after turning a little
    assert m.cell_at(2.0, 0.0) == OCCUPIED


def test_a_return_arc_leaves_known_free_cells_alone_if_something_else_on_it_could_explain_it():
    m = new_map(sensors=(FORWARD, RangeSensor("wide", 0.0, 0.0, 0.0, math.radians(30.0), 0.2, 4.5)))
    for _ in range(3):      # the narrow cone says the middle is free out to its range
        m.add_range("front", None, at(yaw=math.radians(_ * 3.0)))
    wide_edge = (3.0 * math.cos(math.radians(28.0)), 3.0 * math.sin(math.radians(28.0)))
    for k in range(4):      # the wide cone sees something on its arc: it isn't in the known-free middle
        m.add_range("wide", 3.0, at(x=0.0, y=0.0, yaw=math.radians(0.5 * (k % 2) * 4.0)))
    assert m.cell_at(3.0, 0.0) != OCCUPIED
    assert m.cell_at(*wide_edge) == OCCUPIED


def test_an_arc_over_only_free_cells_marks_a_new_object():
    m = new_map()
    for k in range(4):
        m.add_range("front", None, at(yaw=math.radians(k * 3.0 - 4.5)))
    assert m.cell_at(3.0, 0.0) == FREE
    for k in range(6):
        m.add_range("front", 3.0, at(yaw=math.radians(k * 2.0 - 5.0)))
    assert m.cell_at(3.0, 0.0) == OCCUPIED


def test_a_later_return_further_out_wipes_a_wrong_mark():
    m = new_map()
    m.add_range("front", 2.0, at())
    m.add_range("front", 2.0, at(yaw=math.radians(3.0)))
    assert m.cell_at(2.0, 0.0) == OCCUPIED
    m.add_range("front", 4.0, at(yaw=math.radians(6.0)))
    assert m.cell_at(2.0, 0.0) != OCCUPIED
    m.add_range("front", 4.0, at())
    assert m.cell_at(2.0, 0.0) == FREE


def test_a_dropout_does_not_wipe_an_obstacle():
    m = new_map()
    for k in range(6):
        m.add_range("front", 3.0, at(yaw=math.radians(k * 2.0)))
    m.add_range("front", None, at(yaw=math.radians(0.0)))
    assert m.cell_at(3.0, 0.0) == OCCUPIED


def test_the_ground_under_the_car_is_free():
    m = new_map()
    m.update_body(at())
    assert m.cell_at(1.5, 0.5) == FREE
    assert m.cell_at(-0.9, -0.9) == FREE
    assert m.cell_at(4.3, 0.0) == UNKNOWN
    assert m.cell_at(1.5, 1.3) == UNKNOWN


def test_an_obstacle_gets_placed_by_the_pose_it_is_given():
    m = new_map()
    pose = at(x=10.0, y=5.0, yaw=math.pi / 2.0)
    m.add_range("front", 3.0, pose)
    m.add_range("front", 3.0, Pose(10.0, 5.0, math.pi / 2.0 + math.radians(3.0)))
    assert m.cell_at(10.0, 8.0) == OCCUPIED


def test_the_mount_offset_counts():
    m = new_map(sensors=(RangeSensor("rear", -1.0, 0.0, math.pi, math.radians(15.0), 0.2, 4.5),))
    m.add_range("rear", 2.0, at())
    m.add_range("rear", 2.0, at(yaw=math.radians(3.0)))
    assert m.cell_at(-3.0, 0.0) == OCCUPIED


# ---------- cameras ----------
def camera_map():
    return new_map(cameras={"cam": (0.0, 0.0)})


def test_seeing_paint_means_the_ground_up_to_it_is_visible():
    m = camera_map()
    segments = np.array([[[3.0, -1.0], [3.0, 1.0]]])
    m.add_lines("cam", segments, at())
    m.add_lines("cam", segments, at(x=0.3))
    assert m.cell_at(2.0, 0.0) == FREE
    assert m.cell_at(3.8, 0.0) == UNKNOWN


def test_one_frame_of_a_false_segment_is_not_enough():
    m = camera_map()
    m.add_lines("cam", np.array([[[3.0, -1.0], [3.0, 1.0]]]), at())
    assert m.cell_at(2.0, 0.0) == UNKNOWN


def test_a_camera_can_not_clear_an_obstacle():
    m = new_map(cameras={"cam": (0.0, 0.0)})
    for k in range(4):
        m.add_range("front", 2.0, at(yaw=math.radians(k * 3.0)))
    assert m.cell_at(2.0, 0.0) == OCCUPIED
    for k in range(20):
        m.add_lines("cam", np.array([[[3.0, -1.0], [3.0, 1.0]]]), at(x=0.3 * (k % 2)))
    assert m.cell_at(2.0, 0.0) == OCCUPIED


def test_camera_evidence_alone_can_still_be_overturned_by_a_return():
    m = new_map(sensors=(FORWARD,), cameras={"cam": (0.0, 0.0)})
    for k in range(20):
        m.add_lines("cam", np.array([[[3.0, -1.0], [3.0, 1.0]]]), at(x=0.3 * (k % 2)))
    assert m.cell_at(2.0, 0.0) == FREE
    for k in range(4):
        m.add_range("front", 2.0, at(yaw=math.radians(k * 3.0)))
    assert m.cell_at(2.0, 0.0) == OCCUPIED


def test_paint_is_kept_for_display_in_the_map_frame():
    m = camera_map()
    m.add_lines("cam", np.array([[[3.0, -1.0], [3.0, 1.0]]]), at(x=5.0))
    frame = m.frame(1.0)
    assert frame.paint.shape == (1, 2, 2)
    assert frame.paint[0].tolist() == [[8.0, -1.0], [8.0, 1.0]]


# ---------- the assumed slot and what the planner gets ----------
SLOT = ((0.0, 0.0), (2.7, 0.0), (2.7, -5.5), (0.0, -5.5))    # opening first


def seeded_map():
    m = new_map()
    m.update_body(at(x=0.0, y=3.0))
    return m


def test_the_selected_slot_is_assumed_not_free():
    m = seeded_map()
    m.assume_slot(SLOT)
    assert m.cell_at(1.35, -3.0) == ASSUMED
    assert m.cell_at(-0.2, -3.0) == ASSUMED     # the margin
    assert m.cell_at(-0.5, -3.0) == UNKNOWN
    assert m.cell_at(1.35, -5.7) == ASSUMED
    assert m.cell_at(1.35, -6.0) == UNKNOWN


def test_a_weak_mark_in_the_selected_slot_does_not_block_it_but_a_strong_one_does():
    """On first plans, every mark traced in a refused stall was false: two long-range arcs crossing."""
    m = seeded_map()
    m.assume_slot(SLOT)
    i, j = int((1.35 - m._x0) // m.resolution), int((-3.0 - m._y0) // m.resolution)
    log_odds = PARAMS["sensed_map"]
    m._log[i, j] = log_odds["occupied_above"] + 0.1
    assert m.cell_at(1.35, -3.0) == ASSUMED
    m._log[i, j] = log_odds["slot_mark_min_log_odds"]
    assert m.cell_at(1.35, -3.0) == OCCUPIED


def test_a_weak_mark_outside_the_selected_slot_still_blocks():
    m = seeded_map()
    m.assume_slot(SLOT)
    i, j = int((6.0 - m._x0) // m.resolution), int((-3.0 - m._y0) // m.resolution)
    m._log[i, j] = PARAMS["sensed_map"]["occupied_above"] + 0.1
    assert m.cell_at(6.0, -3.0) == OCCUPIED


def test_a_new_assumed_slot_replaces_the_last():
    m = seeded_map()
    m.assume_slot(SLOT)
    m.assume_slot(tuple((x + 20.0, y) for x, y in SLOT))
    assert m.cell_at(1.35, -3.0) == UNKNOWN
    assert m.cell_at(21.35, -3.0) == ASSUMED


def test_a_return_inside_the_assumed_slot_counts():
    m = new_map(sensors=(RangeSensor("down", 0.0, 0.0, -math.pi / 2.0, math.radians(15.0), 0.2, 4.5),))
    for k in range(4):
        m.add_range("down", 3.0, at(x=1.35, y=0.0, yaw=math.radians(k * 3.0)))
    m.assume_slot(SLOT)
    assert m.cell_at(1.35, -3.0) == OCCUPIED


# A wide rear cone looking into the stall from 1 m in, as a car reversing in has. Without the rule these returns draw a wall
# across the empty stall, about 1.3 m in front of the curb, and 41 of 44 failed replans came from cells like that.
WIDE = RangeSensor("rear", 0.0, 0.0, -math.pi / 2.0, math.radians(30.0), 0.2, 4.5)


def reversing_in(reading, m=None, times=4):
    m = m if m is not None else new_map(sensors=(WIDE,))
    m.update_body(at(x=1.35, y=3.0))
    m.assume_slot(SLOT)
    for k in range(times):
        m.add_range("rear", reading, at(x=1.35, y=-1.0, yaw=math.radians(k * 3.0)))
    return m


def test_a_return_whose_arc_reaches_past_the_lines_draws_no_wall_across_the_slot():
    m = reversing_in(3.0)       # half a cone at 3 m is 1.5 m, past the lines 1.35 m away
    between_the_lines = [m.cell_at(x, y) for x in np.arange(0.05, 2.7, 0.1) for y in np.arange(-5.45, -1.0, 0.1)]
    assert OCCUPIED not in between_the_lines
    assert m.cell_at(1.35, -4.0) == ASSUMED


def test_an_ambiguous_return_adds_nothing_in_the_margin_either():
    """The goal sits 0.36 m from a line and plans need 0.35 m, so one stray cell here refuses every replan."""
    m = reversing_in(3.0)
    assert m.cell_at(-0.1, -3.6) == ASSUMED
    assert m.cell_at(2.8, -3.6) == ASSUMED


def test_an_ambiguous_return_still_marks_what_is_past_the_margin():
    m = reversing_in(4.0)
    assert m.cell_at(3.23, -4.53) == OCCUPIED   # a neighbor, off the side of the stall
    assert m.cell_at(1.35, -5.0) == ASSUMED


def test_something_real_in_the_stall_is_marked_once_its_arc_fits_between_the_lines():
    m = reversing_in(1.5)       # half a cone at 1.5 m is 0.75 m
    assert m.cell_at(1.35, -2.5) == OCCUPIED


def test_an_ambiguous_return_does_not_forget_what_is_already_marked():
    m = reversing_in(1.5)
    m.add_range("rear", 3.5, at(x=1.35, y=1.0))     # now it reaches past the lines, and its arc runs through the mark
    assert m.cell_at(1.35, -2.5) == OCCUPIED


def test_a_return_is_not_ambiguous_if_ultrasonics_have_cleared_everything_past_the_lines():
    """Nowhere else for it to be, so it is in the slot."""
    m = new_map(sensors=(WIDE,))
    m.update_body(at(x=1.35, y=3.0))
    xs, ys = np.meshgrid(m._xc, m._yc, indexing="ij")
    m._log[(xs < 0.0) | (xs > 2.7) | (ys > 0.0) | (ys < -5.5)] = PARAMS["sensed_map"]["log_odds"]["min"]
    reversing_in(3.0, m)
    assert m.cell_at(1.35, -4.0) == OCCUPIED


def test_camera_evidence_past_the_lines_does_not_rule_out_a_neighbor():
    """A camera seeing paint past a cell says little about a car standing on it."""
    m = new_map(sensors=(WIDE,))
    m.update_body(at(x=1.35, y=3.0))
    xs, ys = np.meshgrid(m._xc, m._yc, indexing="ij")
    m._log[(xs < 0.0) | (xs > 2.7)] = PARAMS["sensed_map"]["log_odds"]["camera_floor"]
    reversing_in(3.0, m)
    assert m.cell_at(1.35, -4.0) == ASSUMED


def test_a_return_nowhere_near_the_slot_counts_as_it_always_did():
    """Seed 153 found no path with this wrong: a slot refused earlier stayed selected while the car drove on, and every return
    down the aisle took the slot's rule, putting hits on ground the cameras had seen as free until the way was blocked."""
    far = tuple((x + 40.0, y) for x, y in SLOT)
    log_odds = PARAMS["sensed_map"]["log_odds"]
    results = []
    for slot in (None, far):
        m = new_map(sensors=(WIDE,))
        m.update_body(at(x=1.35, y=3.0))
        if slot:
            m.assume_slot(slot)
        xs, ys = np.meshgrid(m._xc, m._yc, indexing="ij")
        m._log[np.hypot(xs - 1.35, ys + 4.0) < 1.0] = log_odds["camera_floor"]
        m.add_range("rear", 3.0, at(x=1.35, y=-1.0))
        results.append(m._log.copy())
    assert np.array_equal(results[0], results[1])


@pytest.mark.parametrize("seed", [153])
def test_a_refused_slot_does_not_block_the_next_one(seed):
    from park_sim.simulator import search_trial as trial
    _, res, _ = trial(VEHICLE, sensed_params(), seed)
    assert [a.found for a in res.search.plan_attempts][-1]
    assert res.success


def test_before_any_slot_is_selected_a_return_counts_as_it_always_did():
    m = new_map(sensors=(WIDE,))
    for k in range(4):
        m.add_range("rear", 3.0, at(x=1.35, y=-1.0, yaw=math.radians(k * 3.0)))
    assert m.cell_at(1.35, -4.0) == OCCUPIED


def test_the_map_does_not_call_the_inside_of_parked_cars_free():
    from matplotlib.path import Path as ShapePath
    from park_sim.localization import MapToOdom
    sc, res, _ = search_trial(VEHICLE, sensed_params(), 1003)
    log, frame = res.log, res.search.sensed_frames[-1]
    i = min(int(np.searchsorted(log["t"], frame.t)), len(log["t"]) - 1)
    to_odom = MapToOdom(Pose(log["x"][i], log["y"][i], log["yaw"][i]), Pose(log["est_x"][i], log["est_y"][i], log["est_yaw"][i]))
    x = frame.x0 + (np.arange(frame.cells.shape[0]) + 0.5) * frame.resolution
    y = frame.y0 + (np.arange(frame.cells.shape[1]) + 0.5) * frame.resolution
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    free, total = 0, 0
    for car in sc.parked_cars:
        cx, cy = to_odom.to_odom(car[:, 0], car[:, 1])
        cells = frame.cells[ShapePath(np.column_stack([cx, cy])).contains_points(points).reshape(xx.shape)]
        free += int((cells == FREE).sum())
        total += cells.size
    assert total > 0 and free / total < 0.05


@pytest.mark.parametrize("seed", [1003, 1004, 1010])
def test_replans_get_made_on_a_run_with_goal_corrections(seed):
    _, res, _ = search_trial(VEHICLE, sensed_params(), seed)
    kinds = [r.kind for r in res.search.refinements]
    assert "replan" in kinds
    assert "replan_failed" not in kinds
    assert res.success


@pytest.mark.parametrize("seed", [1003, 1004])
def test_a_cart_in_the_stall_still_gets_marked_and_the_car_stops_short(seed):
    from park_sim.localization import MapToOdom
    from park_sim.scenario_files import case_outcome, load_scenario_file, lowest_clearance, run_scenario_file
    sc, res, drawn = run_scenario_file(load_scenario_file("cart_in_stall"), seed, "search", VEHICLE, sensed_params())
    assert case_outcome(sc, res, drawn["chosen_slot"]) == "aborted"
    assert lowest_clearance(sc, res) > 0.3
    log, cart = res.log, sc.lot_objects[0].polygon
    frame = res.search.sensed_frames[-1]
    i = int(np.searchsorted(log["t"], frame.t))
    to_odom = MapToOdom(Pose(log["x"][i], log["y"][i], log["yaw"][i]), Pose(log["est_x"][i], log["est_y"][i], log["est_yaw"][i]))
    x, y = to_odom.to_odom(cart[:, 0], cart[:, 1])
    a, b = np.nonzero(frame.cells == OCCUPIED)
    px, py = frame.x0 + (a + 0.5) * frame.resolution, frame.y0 + (b + 0.5) * frame.resolution
    assert np.any((px >= x.min() - 0.2) & (px <= x.max() + 0.2) & (py >= y.min() - 0.2) & (py <= y.max() + 0.2))


def test_the_distance_field_blocks_unknown_and_occupied_and_opens_the_slot():
    m = new_map()
    pose = at(x=0.0, y=3.0)
    m.update_body(pose)
    field, (xmin, xmax, ymin, ymax) = m.distance_field([(pose.x, pose.y)])
    res = m.resolution

    def distance(x, y):
        return field[int((x - xmin) / res), int((y - ymin) / res)]

    assert distance(1.5, 3.0) > 0.0     # under the car
    assert distance(1.5, -3.0) <= 0.0   # unknown
    m.assume_slot(((0.0, 0.0), (2.7, 0.0), (2.7, -5.5), (0.0, -5.5)))
    field, (xmin, xmax, ymin, ymax) = m.distance_field([(pose.x, pose.y)])
    assert distance(1.35, -3.0) > 1.0   # opened, away from the unknown around it


def test_the_distance_field_uses_the_planning_window():
    m = new_map()
    m.update_body(at())
    _, (xmin, xmax, ymin, ymax) = m.distance_field([(0.0, 0.0), (10.0, 5.0)])
    w = PARAMS["sensed_map"]["planning_window_m"]
    assert xmin == pytest.approx(-w, abs=0.11) and xmax == pytest.approx(10.0 + w, abs=0.11)
    assert ymin == pytest.approx(-w, abs=0.11) and ymax == pytest.approx(5.0 + w, abs=0.11)


def test_the_window_edge_is_a_wall():
    m = new_map()
    m.update_body(at())
    field, _ = m.distance_field([(0.0, 0.0)])
    assert np.all(field[0, :] <= 0.0) and np.all(field[:, -1] <= 0.0)


def test_evidence_off_the_grid_is_dropped_not_a_crash():
    params = {**PARAMS["sensed_map"], "half_extent_m": [5.0, 5.0]}
    m = SensedMap(SensedMapConfig.from_params(params), (FORWARD,), {}, BODY)
    m.add_range("front", None, at())
    m.add_range("front", None, at())
    m.add_range("front", None, at(x=100.0))
    assert m.cell_at(100.0, 0.0) == UNKNOWN
    assert m.cell_at(2.0, 0.0) == FREE


def test_a_frame_is_a_copy():
    m = new_map()
    m.add_range("front", None, at())
    m.add_range("front", None, at())
    frame = m.frame(0.0)
    before = frame.cells.copy()
    m.add_range("front", 1.0, at(yaw=0.1))
    m.add_range("front", 1.0, at(yaw=0.2))
    assert np.array_equal(frame.cells, before)


def test_frame_runs_join_cells_along_x():
    cells = np.zeros((6, 2), dtype=np.int8)
    cells[1:4, 0] = FREE
    cells[5, 0] = FREE
    frame = SensedMapFrame(0.0, 10.0, 20.0, 0.5, cells, np.zeros((0, 2, 2)))
    assert frame.runs(FREE) == [(10.5, 12.0, 20.25), (12.5, 13.0, 20.25)]
    assert frame.runs(OCCUPIED) == []


def test_the_config_section_says_its_numbers_are_placeholders():
    text = (ROOT / "config" / "park_params.yaml").read_text()
    section = text[text.index("\nsensed_map:"):].splitlines()[1]
    assert "[est]" in section


# ---------- isolation ----------
def test_the_map_module_imports_nothing_from_the_sim():
    tree = ast.parse((ROOT / "park_sim" / "sensed_map.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    assert imported <= {"__future__", "math", "collections", "dataclasses", "typing", "numpy", "scipy.ndimage"}, imported


FORBIDDEN = {"obstacles", "parked_cars", "curbs", "lot_objects", "stalls", "lines", "_dist", "_boxes", "goal", "start",
             "target_stall", "stall_center", "planner_view", "collision_free", "clearance_fast", "clearance_exact",
             "nearest_stall", "aisle_y", "aisle_end_x", "bounds", "with_target", "build_distance_field"}


class _Tripwire:
    armed = False
    reads = []


class TrippedLot(Scenario):
    """A lot whose true data raises if read while the tripwire is armed. Sensors and scoring run with it off."""

    def __getattribute__(self, name):
        if _Tripwire.armed and name in FORBIDDEN:
            _Tripwire.reads.append(name)
            raise AssertionError(f"read the true lot's {name} while mapping or planning")
        return object.__getattribute__(self, name)


def tripped_lot(seed=1):
    sc = make_aisle_lot(VEHICLE, sensed_params(), np.random.default_rng(seed), 0.7)
    lot = TrippedLot.__new__(TrippedLot)
    for f in dataclasses.fields(Scenario):
        object.__setattr__(lot, f.name, getattr(sc, f.name))
    return lot


@pytest.fixture
def armed_mapping(monkeypatch):
    """Arms the tripwire whenever the run is inside the sensed map or the planner running on it."""
    _Tripwire.reads = []
    planned = []

    def armed(fn, record=False):
        def wrapper(*args, **kwargs):
            before = _Tripwire.armed
            _Tripwire.armed = True
            try:
                out = fn(*args, **kwargs)
            finally:
                _Tripwire.armed = before
            if record:
                planned.append(out)
            return out
        return wrapper

    monkeypatch.setattr(simulator._SensedMapFeed, "observe", armed(simulator._SensedMapFeed.observe))
    monkeypatch.setattr(simulator, "_plan_on_sensed_map", armed(simulator._plan_on_sensed_map, record=True))

    def no_true_lot_planning(*args, **kwargs):
        raise AssertionError("planned on the true lot with planner.map: sensed")

    monkeypatch.setattr(simulator, "_plan_in_frame", no_true_lot_planning)
    monkeypatch.setattr(simulator, "_plan_to_slot", no_true_lot_planning)
    yield planned
    _Tripwire.armed = False


def test_the_tripped_lot_does_trip():
    lot = tripped_lot()
    _Tripwire.armed = True
    try:
        with pytest.raises(AssertionError):
            lot.obstacles
        with pytest.raises(AssertionError):
            lot.collision_free(0.0, 0.0, 0.0, 0.1)
        assert lot.vehicle is not None and lot.cfg is not None
    finally:
        _Tripwire.armed = False


def test_a_full_run_maps_and_plans_without_touching_the_true_lot(armed_mapping):
    lot = tripped_lot(seed=1)
    driver = simulator.DriverConfig.from_params(lot.cfg["search"])
    res = run_search(lot, driver, sensor_seed=1)
    assert armed_mapping, "no plan was asked for"
    assert any(path is not None for path, _ in armed_mapping)
    assert res.planned is not None
    assert _Tripwire.reads == []


def test_planning_takes_no_lot_and_no_true_pose():
    """What the sensed plan is given, by signature, and with a pose the true one can't be swapped into."""
    m = new_map()
    pose = at(x=0.0, y=3.0)
    m.update_body(pose)
    slot = SlotCandidate(slot_id=1, confirmed=True, corners=SLOT, entry_side="+y", slot_type="perpendicular",
                         occupied=False, confidence=1.0, first_seen_t=0.0)
    path, took = simulator._plan_on_sensed_map(m, slot, pose, VEHICLE, PARAMS, None, 0.5)
    assert path is None     # nothing around the car is known: it has nowhere to go
    assert took < 0.5
    names = list(__import__("inspect").signature(simulator._plan_on_sensed_map).parameters)
    assert names == ["sensed_map", "slot", "estimate", "vehicle", "cfg", "planner_cfg", "limit_s", "why"]    # why: a list to put the reason in


def test_the_map_and_the_plan_only_ever_get_the_estimated_pose(monkeypatch):
    """In a run with odometry errors, where the estimate and the true pose really differ."""
    seen = []
    real_observe, real_plan = simulator._SensedMapFeed.observe, simulator._plan_on_sensed_map

    def observe(self, t, measurements, estimate):
        seen.append(estimate)
        return real_observe(self, t, measurements, estimate)

    def plan(sensed_map, slot, estimate, *rest):
        seen.append(estimate)
        return real_plan(sensed_map, slot, estimate, *rest)

    monkeypatch.setattr(simulator._SensedMapFeed, "observe", observe)
    monkeypatch.setattr(simulator, "_plan_on_sensed_map", plan)
    _, res, _ = search_trial(VEHICLE, sensed_params(), 1000)
    assert res.estimator == "odometry"
    assert seen and all(isinstance(pose, PoseEstimate) for pose in seen)
    assert res.metrics["est_err_max_m"] > 0.0


def test_the_feed_reads_only_what_the_car_would_have():
    """A measurement's truth_line stays unread: a camera message with no such field goes through."""
    suite = make_suite(load_sensor_params(), seed=1)
    feed = simulator._SensedMapFeed(PARAMS, suite, VEHICLE)
    sc = make_aisle_lot(VEHICLE, PARAMS, np.random.default_rng(1), 0.7)
    from park_sim.sensors.line_camera import LineCameraMeasurement
    state = simulator.VehicleState(sc.start.x, sc.start.y, sc.start.yaw)
    measured = suite.step(sc, state, 0.0)
    stripped = [dataclasses.replace(m, truth_line=None) if isinstance(m, LineCameraMeasurement) else m for m in measured]
    feed.observe(0.0, stripped, at(sc.start.x, sc.start.y, sc.start.yaw))
    assert feed.map.counts()["free"] > 0


def test_the_body_is_placed_at_the_estimate_not_where_the_car_really_is():
    suite = make_suite(load_sensor_params(), seed=1)
    sc = make_aisle_lot(VEHICLE, PARAMS, np.random.default_rng(1), 0.7)
    state = simulator.VehicleState(sc.start.x, sc.start.y, sc.start.yaw)
    estimate = at(sc.start.x + 20.0, sc.start.y - 3.0, 0.0)
    feed = simulator._SensedMapFeed(PARAMS, suite, VEHICLE)
    feed.observe(0.0, [], estimate)
    assert feed.map.cell_at(estimate.x + 1.5, estimate.y) == FREE
    assert feed.map.cell_at(state.x + 1.5, state.y) == UNKNOWN


# ---------- it only knows what it has seen ----------
def run_recording_map(seed=1, policy="never_select", **kwargs):
    """A search drive past a lot, keeping map frames. With a perfect estimate, so map and lot can be compared directly."""
    params = sensed_params(estimator="truth")
    sc = make_aisle_lot(VEHICLE, params, np.random.default_rng(seed), 0.8)
    cfg = dataclasses.replace(simulator.DriverConfig.from_params(params["search"]), policy=policy, **kwargs)
    return sc, run_search(sc, cfg, sensor_seed=seed)


def cell_in(frame, x, y):
    i, j = int((x - frame.x0) // frame.resolution), int((y - frame.y0) // frame.resolution)
    if 0 <= i < frame.cells.shape[0] and 0 <= j < frame.cells.shape[1]:
        return int(frame.cells[i, j])
    return UNKNOWN


def bumper_points(sc):
    """A point on the aisle-side face of each parked car, just inside the body."""
    points = []
    for car in sc.parked_cars:
        cx = float(car[:, 0].mean())
        toward = sc.aisle_y - float(car[:, 1].mean())
        edge = car[:, 1].max() if toward > 0 else car[:, 1].min()
        points.append((cx, float(edge), float(np.sign(toward))))
    return points


def test_an_obstacle_is_not_on_the_map_before_the_car_passes_and_is_after():
    sc, res = run_recording_map()
    frames = res.search.sensed_frames
    log = res.log
    checked = 0
    for cx, edge_y, toward in bumper_points(sc):
        if not 10.0 <= cx <= 40.0:
            continue
        t_abeam = float(log["t"][np.argmax(log["x"] >= cx)])
        before = [f for f in frames if f.t <= t_abeam - 3.0]
        after = [f for f in frames if f.t >= t_abeam + 2.0]
        assert before and after
        near = [(cx + dx, edge_y + toward * dy) for dx in np.arange(-0.6, 0.61, 0.1) for dy in np.arange(-0.15, 0.16, 0.05)]
        assert not any(cell_in(before[-1], *p) == OCCUPIED for p in near), (cx, before[-1].t)
        assert any(cell_in(after[0], *p) == OCCUPIED for p in near), (cx, after[0].t)
        checked += 1
    assert checked >= 5


def test_the_space_behind_a_parked_car_stays_unknown():
    sc, res = run_recording_map()
    frame = res.search.sensed_frames[-1]
    centers = [(float(c[:, 0].mean()), float(c[:, 1].mean())) for c in sc.parked_cars if 10.0 <= c[:, 0].mean() <= 40.0]
    assert centers
    assert all(cell_in(frame, cx, cy) == UNKNOWN or cell_in(frame, cx, cy) == OCCUPIED for cx, cy in centers)


def test_most_of_the_aisle_it_drove_is_free():
    sc, res = run_recording_map()
    frame = res.search.sensed_frames[-1]
    points = [(x, sc.aisle_y + dy) for x in np.arange(5.0, 45.0, 1.0) for dy in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    free = sum(cell_in(frame, *p) == FREE for p in points)
    assert free / len(points) > 0.95


def test_the_map_is_in_the_estimates_frame():
    """Shift the estimate and everything on the map shifts with it: a car at x seen from a pose off by dx."""
    m = new_map()
    for k in range(4):
        m.add_range("front", 3.0, at(x=1.0, yaw=math.radians(k * 3.0)))
    assert m.cell_at(4.0, 0.0) == OCCUPIED
    assert m.cell_at(3.0, 0.0) != OCCUPIED


# ---------- runs ----------
def test_with_sensed_on_a_run_plans_and_parks():
    params = sensed_params()
    sc, res, _ = search_trial(VEHICLE, params, 1000)
    assert res.search.plans
    assert res.metrics["completed"] == 1.0
    assert res.success, res.summary()
    assert res.search.sensed_frames
    assert res.metrics["map_occupied_m2"] > 0.0
    assert res.metrics["map_free_m2"] > 0.0


def test_scoring_uses_the_true_lot_on_a_sensed_run():
    params = sensed_params()
    sc, res, _ = search_trial(VEHICLE, params, 1000)
    log = res.log
    idx, values = simulator.clearance_samples(sc, log)
    assert res.metrics["min_clearance"] == pytest.approx(min(values))


def test_replans_use_the_sensed_map(monkeypatch):
    calls = []
    real = simulator._plan_on_sensed_map

    def spy(*args, **kwargs):
        calls.append(args[1].slot_id)
        return real(*args, **kwargs)

    monkeypatch.setattr(simulator, "_plan_on_sensed_map", spy)
    monkeypatch.setattr(simulator, "_plan_in_frame", lambda *a, **k: pytest.fail("planned on the true lot"))
    _, res, _ = search_trial(VEHICLE, sensed_params(), 1002)
    replans = [r for r in res.search.refinements if r.kind in ("replan", "replan_failed", "replan_refused")]
    assert replans
    assert len(calls) == len(res.search.plan_attempts) + len(replans)


def test_a_replan_sees_what_was_seen_after_the_first_plan(monkeypatch):
    """The first plan works; then the sensors see something in the stall; the same request now fails."""
    real = simulator._plan_on_sensed_map
    results = []

    def spy(sensed_map, slot, estimate, vehicle, cfg, planner_cfg, limit_s, why=None):
        first = real(sensed_map, slot, estimate, vehicle, cfg, planner_cfg, limit_s, why)
        if first[0] is not None and not results:
            center = np.mean(slot.corners, axis=0)
            sensed_map.sensors["test_probe"] = RangeSensor("test_probe", 0.0, 0.0, 0.0, math.radians(15.0), 0.2, 4.5)
            for k in range(4):     # a sensor 3 m off the stall's center, looking at it
                sensed_map.add_range("test_probe", 3.0, Pose(center[0] - 3.0, center[1], math.radians(k * 2.0)))
            results.append(first[0] is not None)
            results.append(real(sensed_map, slot, estimate, vehicle, cfg, planner_cfg, limit_s)[0] is not None)
        return first

    monkeypatch.setattr(simulator, "_plan_on_sensed_map", spy)
    search_trial(VEHICLE, sensed_params(), 1000)
    assert results == [True, False]


def test_something_seen_in_the_slot_fails_the_plan_at_once():
    m = new_map(sensors=(RangeSensor("down", 0.0, 0.0, -math.pi / 2.0, math.radians(15.0), 0.2, 4.5),))
    for x in np.arange(-5.0, 8.0, 0.2):
        m.update_body(Pose(x, 3.0, 0.0))
    for k in range(6):
        m.add_range("down", 3.0, at(x=1.35, y=0.0, yaw=math.radians(k * 2.0)))
    slot = SlotCandidate(slot_id=1, confirmed=True, corners=SLOT, entry_side="+y", slot_type="perpendicular",
                         occupied=False, confidence=1.0, first_seen_t=0.0)
    path, took = simulator._plan_on_sensed_map(m, slot, Pose(5.0, 3.0, 0.0), VEHICLE, PARAMS, None, 2.0)
    assert path is None
    assert took < 0.5


def test_a_slow_plan_counts_as_none(monkeypatch):
    m = new_map()
    pose = at(x=0.0, y=3.0)
    m.update_body(pose)
    slot = SlotCandidate(slot_id=1, confirmed=True, corners=SLOT, entry_side="+y", slot_type="perpendicular",
                         occupied=False, confidence=1.0, first_seen_t=0.0)
    path, _ = simulator._plan_on_sensed_map(m, slot, pose, VEHICLE, PARAMS, None, 0.0)
    assert path is None


def test_a_single_stall_run_plans_on_the_lot_it_is_given():
    """It starts next to the stall having seen nothing, so there's no map to use. The sensed default doesn't change it."""
    params = sensed_params()
    sc = make_perpendicular(VEHICLE, params, np.random.default_rng(0))
    res = run_closed_loop(sc)
    assert res.planned is not None and res.success
    plain = make_perpendicular(VEHICLE, {**params, "planner": {**params["planner"], "map": "true_lot"}}, np.random.default_rng(0))
    assert np.array_equal(res.log["x"], run_closed_loop(plain).log["x"])


def test_an_unknown_map_name_is_refused():
    params = dict(PARAMS)
    params["planner"] = {**PARAMS["planner"], "map": "mystery"}
    sc = make_perpendicular(VEHICLE, params, np.random.default_rng(0))
    with pytest.raises(ValueError, match="mystery"):
        run_closed_loop(sc)


@pytest.mark.parametrize("estimator", [None, "odometry"])
def test_the_ideal_detector_stays_on_the_true_lot_as_the_reference(estimator):
    """It knows every stall at once, seen or not, so a sensed map couldn't reach them."""
    params = sensed_params(detector="ideal", estimator=estimator)
    assert not simulator.plans_on_sensed_map(params)
    sc = make_aisle_lot(VEHICLE, params, np.random.default_rng(3), 0.6)
    res = run_search(sc, dataclasses.replace(simulator.DriverConfig.from_params(params["search"]), policy="select_first"),
                     sensor_seed=3)
    assert res.search.sensed_frames == []
    assert res.search.plans


@pytest.mark.parametrize("detector", ["fused", "lines", "space"])
def test_every_sensor_based_detector_plans_on_the_sensed_map_by_default(detector):
    assert simulator.plans_on_sensed_map({**PARAMS, "slot_detection": {**PARAMS["slot_detection"], "detector": detector}})


def test_a_suite_with_no_ultrasonics_is_refused():
    with pytest.raises(ValueError, match="ultrasonic"):
        simulator._SensedMapFeed(PARAMS, simulator.SensorSuite([], 0), VEHICLE)


# ---------- the default, and the true lot on request ----------
def test_the_default_map_is_the_sensed_one():
    assert PARAMS["planner"]["map"] == "sensed"
    assert PARAMS["slot_detection"]["detector"] != "ideal"


def test_a_true_lot_run_has_no_sensed_map_or_extra_metrics():
    params = {**PARAMS, "planner": {**PARAMS["planner"], "map": "true_lot"}}
    sc = make_aisle_lot(VEHICLE, params, np.random.default_rng(3), 0.6)
    res = run_search(sc, sensor_seed=3)
    assert res.search.sensed_frames == []
    assert not any(key.startswith("map_") or key == "plans_failed" for key in res.metrics)


def test_the_first_baseline_seeds_are_still_the_reference():
    """test_park_sim checks the numbers too; this one covers scene_from_distance_field's refactor of the probe."""
    sc = make_perpendicular(VEHICLE, PARAMS, np.random.default_rng(0))
    field = np.full((20, 20), 5.0)
    scene = scene_from_distance_field(field, (0.0, 2.0, 0.0, 2.0), 0.1, Pose(1.0, 1.0, 0.0), Pose(0.0, 0.0, 0.0), PARAMS, VEHICLE)
    assert np.array_equal(scene._probe, sc._probe)
    assert scene.obstacles == []


# ---------- display ----------
@pytest.fixture(scope="module")
def sensed_export(tmp_path_factory):
    pytest.importorskip("mcap")
    pytest.importorskip("foxglove_schemas_protobuf")
    from mcap.reader import make_reader
    from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
    from park_sim.mcap_export import export_run
    params = sensed_params()
    sc, res, _ = search_trial(VEHICLE, params, 1000)
    path = export_run(sc, res, tmp_path_factory.mktemp("sensed") / "run.mcap")
    messages = []
    with open(path, "rb") as f:
        reader = make_reader(f)
        topics = {c.topic for c in reader.get_summary().channels.values()}
        for _, channel, message in reader.iter_messages(topics=["/sensed_map"]):
            update = SceneUpdate()
            update.ParseFromString(message.data)
            messages.append((message.log_time, update))
    return res, topics, messages


def test_the_sensed_map_is_on_its_topic(sensed_export):
    res, topics, messages = sensed_export
    assert "/sensed_map" in topics
    assert len(messages) == len(res.search.sensed_frames)
    assert [t for t, _ in messages] == sorted(t for t, _ in messages)


def test_the_topic_draws_free_occupied_and_the_assumed_slot(sensed_export):
    res, _, messages = sensed_export
    colors = PARAMS["foxglove"]["colors"]
    _, last = messages[-1]
    entity = last.entities[0]
    assert entity.id == "sensed_map" and entity.frame_id == "map"
    drawn = {tuple(round(v, 2) for v in (l.color.r, l.color.g, l.color.b, l.color.a)) for l in entity.lines}
    assert tuple(colors["map_free"]) in drawn
    assert tuple(colors["map_occupied"]) in drawn
    assert tuple(colors["map_paint"]) in drawn
    at_plan = next(u for t, u in messages if t >= int(res.search.plan_attempts[0].t * 1e9))
    drawn = {tuple(round(v, 2) for v in (l.color.r, l.color.g, l.color.b, l.color.a)) for l in at_plan.entities[0].lines}
    assert tuple(colors["map_assumed"]) in drawn


def test_a_true_lot_run_has_no_sensed_map_topic(tmp_path):
    pytest.importorskip("mcap")
    pytest.importorskip("foxglove_schemas_protobuf")
    from mcap.reader import make_reader
    from park_sim.mcap_export import export_run
    params = {**PARAMS, "planner": {**PARAMS["planner"], "map": "true_lot"}}
    sc = make_aisle_lot(VEHICLE, params, np.random.default_rng(3), 0.6)
    path = export_run(sc, run_search(sc, sensor_seed=3), tmp_path / "run.mcap")
    with open(path, "rb") as f:
        topics = {c.topic for c in make_reader(f).get_summary().channels.values()}
    assert "/sensed_map" not in topics


@pytest.mark.parametrize("layout", ["foxglove_layout.json", "foxglove_search_layout.json"])
def test_both_layouts_show_the_sensed_map(layout):
    panels = json.loads((ROOT / "config" / layout).read_text())["configById"]
    views = [p for pid, p in panels.items() if pid.startswith("3D")]
    assert views and all(v["topics"]["/sensed_map"]["visible"] for v in views)


# ---------- the comparison script ----------
def test_the_comparison_script_tabulates_both_maps(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("compare_planner_maps", ROOT / "scripts" / "compare_planner_maps.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    rows = [script.one_run(("true_lot", "fused", 1000)), script.one_run(("sensed", "fused", 1000))]
    text = script.table(rows, 1)
    assert "true_lot" in text and "sensed" in text
    assert "contacts" in text and "no path" in text
    assert all(r["plan_attempts"] >= 1 for r in rows)
    with open(tmp_path / "c.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=list(rows[0])).writeheader()


def test_planned_path_type_is_what_the_controller_expects():
    """A plan on the sensed map is the ordinary planner's output, in the estimate's frame."""
    params = sensed_params()
    _, res, _ = search_trial(VEHICLE, params, 1000)
    assert isinstance(res.planned, PlannedPath)
    assert res.planned.direction[-1] == -1      # reverse in


def test_scene_arrays_still_work_for_the_sensors_with_the_tripped_lot():
    lot = tripped_lot()
    assert len(scene_arrays(lot).obstacle_a) > 0
    assert isinstance(rect(0.0, 0.0, 0.0, 1.0, 1.0), np.ndarray)
