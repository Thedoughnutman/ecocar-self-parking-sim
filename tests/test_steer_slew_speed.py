"""The controller slows where a plan asks the steering to change faster than the actuator can follow at speed.

Without this, a reverse with an S-bend at top speed runs about 0.27 m wide of the path, with too little steering left
to come back (plans use 33.5 deg of the 36.4 deg limit), and comes within 0.08-0.20 m of parked cars. These seeds
failed on clearance, and seed 151 touched a car."""
import copy
import math

import numpy as np
import pytest

from park_sim.controller import ParkingController
from park_sim.hybrid_astar import PlannedPath
from park_sim.scenario import load_params
from park_sim.simulator import search_trial
from park_sim.vehicle import VehicleParams, VehicleState

VEH = VehicleParams.from_yaml()
PARAMS = load_params()
CFG = PARAMS["controller"]
WINDOW = CFG["steer_slew_window_m"]
STEP = 0.05


def path_of(curvatures, direction=-1):
    """A one-direction path from (curvature, length) pieces. Curvature is signed the way steering is."""
    x, y, yaw = [0.0], [0.0], [0.0]
    for k, length in curvatures:
        for _ in range(int(round(length / STEP))):
            yaw.append(yaw[-1] + k * direction * STEP)
            x.append(x[-1] + direction * STEP * math.cos(yaw[-1]))
            y.append(y[-1] + direction * STEP * math.sin(yaw[-1]))
    n = len(x)
    return PlannedPath(np.array(x), np.array(y), np.array(yaw), np.full(n, direction))


K_PLAN = math.tan(VEH.planner_steer) / VEH.wheelbase
S_BEND = [(-K_PLAN, 6.0), (K_PLAN, 6.0)]


def caps(path):
    ctrl = ParkingController(path, VEH, CFG)
    assert len(ctrl.segments) == 1
    return ctrl.segments[0], ctrl._speed_caps[0]


def test_the_window_is_a_positive_config_value():
    assert WINDOW > 0.0


def test_steady_curves_and_straights_stay_at_top_speed():
    for pieces in ([(0.0, 8.0)], [(K_PLAN, 8.0)], [(-K_PLAN, 8.0)]):
        _, cap = caps(path_of(pieces))
        assert cap[1:-1] == pytest.approx(VEH.max_speed)


def test_a_full_flip_slows_the_car_enough_for_the_steering_to_keep_up():
    seg, cap = caps(path_of(S_BEND))
    flip = int(np.searchsorted(seg.s, 6.0))
    needed = VEH.steer_rate * WINDOW / (2.0 * VEH.planner_steer)
    assert cap[flip] == pytest.approx(needed, abs=0.03)
    assert needed < 0.6 * VEH.max_speed


def test_a_small_steering_change_does_not_slow_the_car():
    small = math.tan(VEH.planner_steer / 4.0) / VEH.wheelbase
    _, cap = caps(path_of([(0.0, 5.0), (small, 5.0)]))
    assert cap[1:-1] == pytest.approx(VEH.max_speed)


def test_a_longer_window_means_a_gentler_slowdown():
    wide = copy.deepcopy(CFG)
    wide["steer_slew_window_m"] = 2.0 * WINDOW
    path = path_of(S_BEND)
    narrow_cap = ParkingController(path, VEH, CFG)._speed_caps[0]
    wide_cap = ParkingController(path, VEH, wide)._speed_caps[0]
    assert wide_cap.min() > narrow_cap.min()


def test_slowing_starts_early_enough_for_the_brakes_the_profile_uses():
    seg, cap = caps(path_of(S_BEND))
    decel = 0.5 * VEH.decel
    drop = cap[:-1] ** 2 - cap[1:] ** 2
    assert np.all(drop <= 2.0 * decel * np.diff(seg.s) + 1e-9)
    assert cap[0] == pytest.approx(VEH.max_speed)   # 6 m is plenty to get there from the start


def test_the_speed_command_obeys_the_cap_at_the_flip():
    path = path_of(S_BEND)
    ctrl = ParkingController(path, VEH, CFG, initial_gear=-1)
    seg, cap = ctrl.segments[0], ctrl._speed_caps[0]
    flip = int(np.searchsorted(seg.s, 6.0))
    ctrl.update(VehicleState(float(seg.x[0]), float(seg.y[0]), float(seg.yaw[0]), 0.0, 0.0), CFG["dt"])
    ctrl.near_i = flip
    _, v = ctrl.update(VehicleState(float(seg.x[flip]), float(seg.y[flip]), float(seg.yaw[flip]), -0.4, 0.0), CFG["dt"])
    assert v < 0.0
    assert abs(v) <= cap[flip:flip + 40].max() + 1e-9
    assert abs(v) < VEH.max_speed


def test_a_moved_final_stop_keeps_the_caps_lined_up_with_the_path():
    ctrl = ParkingController(path_of(S_BEND), VEH, CFG, initial_gear=-1)
    for along in (0.5, -3.0):
        ctrl.move_final_stop(along)
        assert len(ctrl._speed_caps[-1]) == len(ctrl.segments[-1].x)


def test_cusps_get_caps_for_each_segment_and_a_short_segment_does_not_break():
    x = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    path = PlannedPath(x, np.zeros(5), np.zeros(5), np.array([1, 1, 1, -1, -1]))
    ctrl = ParkingController(path, VEH, CFG)
    assert [len(c) for c in ctrl._speed_caps] == [len(s.x) for s in ctrl.segments]
    two_points = PlannedPath(np.array([0.0, 0.1]), np.zeros(2), np.zeros(2), np.array([-1, -1]))
    assert ParkingController(two_points, VEH, CFG)._speed_caps[0] == pytest.approx(VEH.max_speed)


@pytest.mark.parametrize("seed, detector", [(42, "ideal"), (78, "ideal"), (182, "ideal"), (182, "fused"), (151, "fused")])
def test_the_seeds_that_ran_wide_now_keep_their_clearance(seed, detector):
    params = copy.deepcopy(PARAMS)
    params["slot_detection"]["detector"] = detector
    _, res, _ = search_trial(VEH, params, seed)
    assert res.metrics["min_clearance"] >= params["criteria"]["min_clearance"]
    assert res.success
