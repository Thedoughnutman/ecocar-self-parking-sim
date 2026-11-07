"""Sensor-based slot detection and tracking: odometry, line and space detectors, tracker, refinement."""
import copy
import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from park_sim import simulator  # noqa: E402
from park_sim.controller import ParkingController  # noqa: E402
from park_sim.driver import DriverConfig, SearchDriver  # noqa: E402
from park_sim.geometry import rect  # noqa: E402
from park_sim.hybrid_astar import PlannedPath  # noqa: E402
from park_sim.mcap_export import export_run  # noqa: E402
from park_sim.localization import MapToOdom, Measurements, WheelOdometry  # noqa: E402
from park_sim.scenario import Pose, load_params, make_aisle_lot  # noqa: E402
from park_sim.sensors import load_sensor_params, make_suite  # noqa: E402
from park_sim.simulator import run_search, search_trial  # noqa: E402
from park_sim.slot_detection import (FusedSlotDetector, IdealSlotDetector, LineSlotDetector, SlotDetector,  # noqa: E402
                                     SpaceSlotDetector, make_detector, merge_collinear)
from park_sim.slot_tracker import SlotDetection, SlotTracker, slot_heading, corner_covariance  # noqa: E402
from park_sim.supervisor import SlotCandidate, State  # noqa: E402
from park_sim.vehicle import VehicleParams, VehicleState  # noqa: E402

VEH = VehicleParams.from_yaml()
PARAMS = load_params()
SEED = 1      # free stalls 1, 3, 23, 24, 29, 35


def rng(seed=0):
    return np.random.default_rng(seed)


def clean_sensors(faded=None):
    """Camera and ultrasonic models with every random effect off (optionally faded paint)."""
    cfg = copy.deepcopy(load_sensor_params())
    cam = cfg["line_camera"]
    cam["noise"]["enabled"] = cam["gaps"]["enabled"] = cam["false_positives"]["enabled"] = False
    cam["faded"] = {"enabled": faded is not None, "p_faded": faded or 0.0}
    cfg["ultrasonic"]["noise_std"], cfg["ultrasonic"]["dropout_prob"] = 0.0, 0.0
    return cfg


def lot_params(detector="lines", **section):
    params = copy.deepcopy(PARAMS)
    params["slot_detection"]["detector"] = detector
    params["slot_detection"]["offer_max_distance_m"] = params["slot_detection"]["offer_max_driven_m"] = 1e6     # these tests look back along the whole aisle
    params["aisle_lot"].update(section)
    return params


def drive_past(lot, detector, suite, speed=10 / 3.6, y=None, dt=0.02, x_end=None):
    """Drive straight down the aisle on true pose, feeding a detector. Returns its slots and the lots history."""
    y = lot.aisle_y if y is None else y
    x, t = lot.start.x, 0.0
    x_end = lot.aisle_end_x + 3.0 if x_end is None else x_end
    while x < x_end:
        state = VehicleState(x, y, 0.0, speed)
        detector.observe(suite.step(lot, state, t))
        detector.update(t, state)
        x, t = x + speed * dt, t + dt
    return detector


def slot_vs_truth(slot, stall):
    """(max corner error in meters, heading error in degrees) of a slot against its true stall."""
    corners, truth = np.array(slot.corners), np.array(stall.corners)
    error = float(np.hypot(*(corners - truth).T).max())
    turn = slot_heading(corners) - slot_heading(truth)
    return error, abs(math.degrees((turn + math.pi) % (2 * math.pi) - math.pi))


def stall_for(lot, slot):
    return lot.nearest_stall(*np.mean(slot.corners, axis=0))


def assert_only_free_stalls(lot, slots, tolerance=0.3):
    for slot in slots:
        stall = stall_for(lot, slot)
        assert not stall.occupied, f"slot {slot.slot_id} offered over occupied stall {stall.slot_id}"
        assert math.dist(np.mean(slot.corners, axis=0), stall.center) <= tolerance


# ---------- the prompt's tests ----------
class _At:
    def __init__(self, x, y=9.0, yaw=0.0):
        self.x, self.y, self.yaw = x, y, yaw


def aging_detector():
    """A detector with one confirmed, passed, free slot at x = 0-2.7, whose tracker has stopped seeing it."""
    from park_sim.slot_tracker import SlotTrack
    lot = make_aisle_lot(VEH, PARAMS, rng(1), occupancy_prob=0.0)
    detector = make_detector("lines", lot, clean_sensors())
    corners = np.array([[0.0, 5.5], [2.7, 5.5], [2.7, 0.0], [0.0, 0.0]])
    detector.tracker.tracks.append(SlotTrack(0, corners, np.stack([np.eye(2)] * 4), "lines", 0.0, 0.0, hits=9, slot_id=0, confirmed_t=0.0, passed=True))
    detector._occupancy = lambda *args: (99, 0)
    return detector


def offered(detector, x):
    detector._refresh_offers(1.0, _At(x))
    return next((s for s in detector.slots if s.slot_id == 0), None)


def test_a_slot_is_offered_when_near_and_not_from_the_far_end_of_the_lot():
    limit = PARAMS["slot_detection"]["offer_max_distance_m"]
    detector = aging_detector()
    assert offered(detector, 5.0) is not None
    assert offered(detector, limit + 10.0) is None


def test_a_slot_comes_back_with_its_old_number_when_the_car_is_near_again():
    limit = PARAMS["slot_detection"]["offer_max_distance_m"]
    detector = aging_detector()
    assert offered(detector, limit + 10.0) is None
    back = offered(detector, 4.0)
    assert back is not None and back.slot_id == 0


def test_a_slot_stays_whole_until_it_goes_out_of_range_then_is_gone():
    limit = PARAMS["slot_detection"]["offer_max_distance_m"]
    detector = aging_detector()
    across = 9.0 - 2.75     # the car drives at y = 9, the slot's center is at y = 2.75: range is measured to the center
    def x_at(distance):
        return 1.35 + math.sqrt(distance ** 2 - across ** 2)
    first = offered(detector, 2.0).confidence
    assert offered(detector, x_at(limit - 0.1)).confidence == first         # no fading on the way
    assert offered(detector, x_at(limit + 0.1)) is None


def test_a_stopped_car_does_not_lose_a_slot_it_has_in_front_of_it():
    detector = aging_detector()
    for _ in range(2000):
        assert offered(detector, 4.0) is not None


def test_a_slot_expires_after_driving_far_enough_without_seeing_it_even_if_near():
    """Position error grows with distance driven. Back and forth nearby still counts."""
    limit = PARAMS["slot_detection"]["offer_max_driven_m"]
    detector = aging_detector()
    x, step = 4.0, 4.0
    for _ in range(int(limit / step) + 2):
        x = 8.0 if x == 4.0 else 4.0
        latest = offered(detector, x)
    assert latest is None


def test_seeing_a_slot_again_restores_it():
    detector = aging_detector()
    limit = PARAMS["slot_detection"]["offer_max_driven_m"]
    x = 4.0
    for _ in range(int(limit / 4.0) + 2):
        x = 8.0 if x == 4.0 else 4.0
        offered(detector, x)
    detector.tracker.tracks[0].last_update_t = 5.0      # the cameras have it again
    assert offered(detector, 4.0) is not None


def test_a_parked_car_does_not_forget_a_stall_when_its_readings_age_out():
    """The readings across a stall only go back 30 s, and a car sitting still takes no new ones of a stall beside it."""
    detector = aging_detector()
    assert offered(detector, 4.0) is not None
    detector._occupancy = lambda *args: (0, 0)
    for _ in range(50):
        assert offered(detector, 4.0) is not None


def test_a_stall_with_no_readings_yet_is_not_offered():
    detector = aging_detector()
    detector._occupancy = lambda *args: (0, 0)
    assert offered(detector, 4.0) is None


def test_fresh_readings_that_find_something_still_take_a_stall_away():
    detector = aging_detector()
    assert offered(detector, 4.0) is not None
    detector._occupancy = lambda *args: (9, 5)
    assert offered(detector, 4.0) is None
    detector._occupancy = lambda *args: (0, 0)      # and it stays taken once they age out
    assert offered(detector, 4.0) is None


def test_the_ideal_detector_still_knows_everything():
    lot = make_aisle_lot(VEH, lot_params("ideal"), rng(1), occupancy_prob=0.0)
    detector = make_detector("ideal", lot)
    detector.update(0.0, _At(0.0))
    first = len(detector.slots)
    detector.update(1.0, _At(60.0))
    assert len(detector.slots) >= first


def test_clean_lines_give_a_slot_within_2_cm_and_half_a_degree():
    params = lot_params()
    lot = make_aisle_lot(VEH, params, rng(SEED))
    suite = make_suite(clean_sensors(), 0)
    detector = drive_past(lot, make_detector("lines", lot, suite), suite)
    free = {st.slot_id for st in lot.stalls if not st.occupied}
    assert {stall_for(lot, s).slot_id for s in detector.slots} == free
    for slot in detector.slots:
        error, turn = slot_vs_truth(slot, stall_for(lot, slot))
        assert error <= 0.02, slot
        assert turn <= 0.5, slot


@pytest.mark.parametrize("seed", range(6))
def test_thirty_percent_faded_lines_never_give_a_wrong_slot(seed):
    params = lot_params()
    lot = make_aisle_lot(VEH, params, rng(seed), occupancy_prob=0.5)
    suite = make_suite(clean_sensors(faded=0.3), seed)
    detector = drive_past(lot, make_detector("lines", lot, suite), suite)
    assert_only_free_stalls(lot, detector.slots, tolerance=0.1)


def test_faded_lines_still_confirm_some_slots_and_skip_those_without_a_divider():
    """Over a sweep of lots: slots whose two dividers both survive can be found, others aren't offered."""
    found = 0
    for seed in range(6):
        lot = make_aisle_lot(VEH, lot_params(), rng(seed), occupancy_prob=0.5)
        suite = make_suite(clean_sensors(faded=0.3), seed)
        detector = drive_past(lot, make_detector("lines", lot, suite), suite)
        found += len(detector.slots)
    assert found >= 10


def divider_scene(car_x, detector_name="lines"):
    """A lot with no cars but one parked straight across the stall line at car_x, near the aisle."""
    params = lot_params(detector_name, occupancy_prob=0.0)
    lot = make_aisle_lot(VEH, params, rng(0))
    d = params["scenario"]["stall_depth"]
    car = rect(car_x, d / 2.0, math.pi / 2, PARAMS["scenario"]["parked_car"]["length"], PARAMS["scenario"]["parked_car"]["width"])
    obstacles = [car] + lot.obstacles
    return replace(lot, obstacles=obstacles, parked_cars=[car], _boxes=None, _sensor_cache={})


def test_car_parked_over_a_line_makes_both_slots_occupied():
    w = PARAMS["scenario"]["stall_width"]
    lot = divider_scene(5 * w)
    suite = make_suite(clean_sensors(), 0)
    detector = drive_past(lot, make_detector("lines", lot, suite), suite)
    offered = {stall_for(lot, s).slot_id for s in detector.slots}
    assert 4 not in offered and 5 not in offered
    assert {3, 6} <= offered


def test_a_car_in_a_stall_with_visible_lines_is_reported_occupied_even_with_noise():
    """The dividers around it are visible, so the track is confirmed; the ultrasonics rule it out."""
    params = lot_params(occupancy_prob=1.0)
    for seed in range(3):
        lot = make_aisle_lot(VEH, params, rng(seed), occupancy_prob=1.0)
        suite = make_suite(seed=seed)
        detector = drive_past(lot, make_detector("lines", lot, suite), suite)
        assert_only_free_stalls(lot, detector.slots)
        confirmed = [tr for tr in detector.tracker.tracks if tr.confirmed]
        assert len(confirmed) > len(detector.slots)      # occupied ones were found, then rejected


def test_fused_monte_carlo_with_default_noise_passes_at_least_90_percent(capsys):
    params = lot_params("fused")
    assert params["search"]["monte_carlo"]["speed_kph"] == [5.0, 15.0]
    spec = importlib.util.spec_from_file_location("monte_carlo", ROOT / "scripts" / "monte_carlo.py")
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    rows, n = [], 50
    passed, fails = mc.batch(n, VEH, params, 0.0, 0.0, rows=rows, tag="fused", mode="search")
    with capsys.disabled():
        print(f"\nfused search Monte Carlo: {passed}/{n} pass ({100 * passed / n:.0f}%), main failure reasons: {fails or 'none'}")
        mc.report(rows, "fused")
    assert passed / n >= 0.90, fails
    assert sum(r["slots_false_positive"] for r in rows) == 0
    limit = params["search"]["planning_time_limit_s"]
    assert max(r["planning_time_s"] for r in rows) < limit + 0.1    # a plan that fails on the sensed map uses the whole limit, and stops within a moment of it


# ---------- odometry ----------
def odometry_tick(odom, speed, yaw_rate, dt):
    return odom.update(Measurements(0.0, 0.0, 0.0, speed, yaw_rate, (speed, speed, speed, speed)), dt)


def test_odometry_is_exact_with_no_errors():
    cfg = dict(rate_hz=100.0, wheel_source="rear_average", wheel_scale_error=0.0, wheel_noise_std=0.0, wheel_quantization=0.0,
               yaw_rate_bias_deg_s=0.0, yaw_rate_noise_std_deg_s=0.0, yaw_rate_quantization_deg_s=0.0,
               standstill_calibration=False, standstill_min_time_s=0.5)
    odom = WheelOdometry(Pose(1.0, 2.0, 0.0), cfg, rng())
    for _ in range(100):
        pose = odometry_tick(odom, 2.0, 0.0, 0.02)
    assert (pose.x, pose.y, pose.yaw) == pytest.approx((5.0, 2.0, 0.0))
    for _ in range(100):
        pose = odometry_tick(odom, 0.0, math.pi / 2, 0.01)
    assert pose.yaw == pytest.approx(math.pi / 2)


def test_odometry_drifts_with_the_configured_errors_and_repeats_by_seed():
    cfg = PARAMS["localization"]["odometry"]
    def run(seed):
        odom = WheelOdometry(Pose(0.0, 0.0, 0.0), cfg, rng(seed))
        for _ in range(2500):
            pose = odometry_tick(odom, 2.0, 0.0, 0.02)
        return pose
    a, b, c = run(1), run(1), run(2)
    assert (a.x, a.y, a.yaw) == (b.x, b.y, b.yaw)
    assert (a.x, a.y) != (c.x, c.y)
    assert abs(a.x - 100.0) < 2.0 and a.x != 100.0
    assert a.y != 0.0


def test_map_to_odom_round_trip():
    frame = MapToOdom(Pose(1.0, 2.0, 0.3), Pose(1.1, 1.9, 0.33))
    assert frame.to_odom(1.0, 2.0) == pytest.approx((1.1, 1.9))
    x, y = frame.to_odom(5.0, -3.0)
    assert frame.to_map(x, y) == pytest.approx((5.0, -3.0))


# ---------- merging ----------
def merge(segments, usable=None):
    arr = np.array(segments, dtype=float)
    usable = np.ones((len(arr), 2), dtype=bool) if usable is None else usable
    return merge_collinear(arr[:, 0], arr[:, 1], usable, PARAMS["slot_detection"]["lines"])


def test_collinear_pieces_merge_and_parallel_lines_do_not():
    lines = merge([((0, 0), (0, 1)), ((0, 1.3), (0, 2)), ((2.7, 0), (2.7, 2))])
    assert sorted(ln.n_segments for ln in lines) == [1, 2]
    pair = next(ln for ln in lines if ln.n_segments == 2)
    assert pair.direction @ np.array([0, 1]) == pytest.approx(1.0, abs=1e-6) or pair.direction @ np.array([0, -1]) == pytest.approx(1.0, abs=1e-6)


def test_pieces_too_far_apart_along_a_line_stay_separate():
    assert len(merge([((0, 0), (0, 1)), ((0, 3), (0, 4))])) == 2


def test_real_end_ignores_ends_cut_by_the_view():
    usable = np.array([[True, False], [True, False]])
    line = merge([((0, 0.0), (0, 1.0)), ((0.0, 0.0), (0.0, 2.0))], usable)[0]
    unit, end = line.real_end(np.array([0.0, -5.0]))       # toward the car, below
    assert unit @ np.array([0.0, -1.0]) == pytest.approx(1.0)
    assert line.center[1] + end * unit[1] == pytest.approx(0.0, abs=1e-6)
    unit, end = line.real_end(np.array([0.0, 5.0]))         # those ends were cut, so nothing is usable
    assert end is None


# ---------- the line detector ----------
def test_detector_uses_estimated_pose_flag():
    assert LineSlotDetector.uses_estimated_pose and SpaceSlotDetector.uses_estimated_pose
    assert FusedSlotDetector.uses_estimated_pose and not IdealSlotDetector.uses_estimated_pose


def test_config_switch_picks_the_detector():
    lot = make_aisle_lot(VEH, PARAMS, rng(0))
    suite = make_suite(seed=0)
    assert isinstance(make_detector("ideal", lot), IdealSlotDetector)
    assert isinstance(make_detector("lines", lot, suite), LineSlotDetector)
    assert isinstance(make_detector("space", lot, suite), SpaceSlotDetector)
    assert isinstance(make_detector("fused", lot, suite), FusedSlotDetector)
    with pytest.raises(ValueError, match="unknown slot detector"):
        make_detector("psychic", lot, suite)
    assert PARAMS["slot_detection"]["detector"] == "fused"


def test_segments_go_through_the_estimated_pose():
    """Feed an estimate shifted from the truth: slots appear shifted by the same amount."""
    lot = make_aisle_lot(VEH, lot_params(), rng(SEED))
    suite = make_suite(clean_sensors(), 0)
    detector = make_detector("lines", lot, suite)
    x, t, shift = lot.start.x, 0.0, 0.2
    while x < 12.0:
        truth = VehicleState(x, lot.aisle_y, 0.0, 2.8)
        detector.observe(suite.step(lot, truth, t))
        detector.update(t, Pose(x + shift, lot.aisle_y, 0.0))
        x, t = x + 2.8 * 0.02, t + 0.02
    assert detector.slots
    for slot in detector.slots:
        stall = lot.nearest_stall(np.mean(slot.corners, axis=0)[0] - shift, np.mean(slot.corners, axis=0)[1])
        offset = np.array(slot.corners) - np.array(stall.corners)
        assert offset[:, 0] == pytest.approx(shift, abs=0.02)
        assert offset[:, 1] == pytest.approx(0.0, abs=0.02)


def test_no_slot_until_the_car_has_passed_it():
    lot = make_aisle_lot(VEH, lot_params(), rng(SEED))
    suite = make_suite(clean_sensors(), 0)
    detector = make_detector("lines", lot, suite)
    stall = lot.stalls[1]
    x, t = lot.start.x, 0.0
    while x < stall.corners[1][0] + 0.9:
        state = VehicleState(x, lot.aisle_y, 0.0, 2.8)
        detector.observe(suite.step(lot, state, t))
        assert not detector.update(t, state)
        x, t = x + 2.8 * 0.02, t + 0.02


def test_candidate_has_the_expected_fields_and_opening_first():
    lot = make_aisle_lot(VEH, lot_params(), rng(SEED))
    suite = make_suite(clean_sensors(), 0)
    detector = drive_past(lot, make_detector("lines", lot, suite), suite)
    for slot in detector.slots:
        stall = stall_for(lot, slot)
        assert slot.confirmed and not slot.occupied
        assert slot.confidence == PARAMS["slot_detection"]["confidence_lines"]
        assert slot.entry_side == stall.entry_side and slot.slot_type == "perpendicular"
        assert slot.first_seen_t > 0.0
        assert abs(slot.corners[0][1] - stall.corners[0][1]) < 0.05 and abs(slot.corners[1][1] - stall.corners[0][1]) < 0.05


def test_stall_width_outside_the_range_is_not_a_slot():
    params = lot_params()
    params["scenario"]["stall_width"] = 3.5
    lot = make_aisle_lot(VEH, params, rng(SEED), occupancy_prob=0.0)
    suite = make_suite(clean_sensors(), 0)
    detector = drive_past(lot, make_detector("lines", lot, suite), suite)
    assert not detector.slots


def test_rear_line_measures_the_depth_when_a_camera_can_see_it():
    """From the aisle a 4 m footprint can't reach it, so these cameras' are stretched to 10 m."""
    params = lot_params()
    params["scenario"]["stall_depth"] = 6.0
    lot = make_aisle_lot(VEH, params, rng(SEED), occupancy_prob=0.0)
    cfg = clean_sensors()
    cfg["line_camera"]["footprint"]["depth"] = 10.0
    cfg["line_camera"]["occlusion"]["parked_cars"] = False
    suite = make_suite(cfg, 0)
    detector = make_detector("lines", lot, suite)
    drive_past(lot, detector, suite, x_end=20.0)
    assert detector.slots
    for slot in detector.slots:
        assert slot_vs_truth(slot, stall_for(lot, slot))[0] <= 0.05


def test_rear_line_coming_into_view_during_the_maneuver_corrects_the_depth():
    """A 6.0 m stall, first assumed to be 5.5 m: the goal starts 25 cm off, beyond the depth limit,
    until the rear line comes into view while reversing in."""
    params = lot_params()
    params["scenario"]["stall_depth"] = 6.0
    lot = make_aisle_lot(VEH, params, rng(SEED))
    res = run_search(lot, sensor_seed=SEED)
    assert res.search.refinements, "the goal should have moved"
    assert abs(res.metrics["depth_err"]) <= PARAMS["criteria"]["depth_tol"]
    assert res.success, res.summary()


def test_standard_depth_is_assumed_otherwise():
    lot = make_aisle_lot(VEH, lot_params(), rng(SEED))
    suite = make_suite(clean_sensors(), 0)
    detector = drive_past(lot, make_detector("lines", lot, suite), suite)
    for tr in detector.tracker.tracks:
        inward = tr.corners[2:].mean(axis=0) - tr.corners[:2].mean(axis=0)
        assert np.hypot(*inward) == pytest.approx(PARAMS["slot_detection"]["lines"]["standard_depth"], abs=0.01)


def test_lidar_rules_out_an_occupied_slot_too():
    cfg = clean_sensors()
    cfg["lidar"]["enabled"] = True
    lot = make_aisle_lot(VEH, lot_params(), rng(SEED), occupancy_prob=0.8)
    suite = make_suite(cfg, 0)
    detector = drive_past(lot, make_detector("lines", lot, suite), suite)
    assert detector.slots
    assert_only_free_stalls(lot, detector.slots, tolerance=0.1)


def test_a_slot_with_no_ultrasonic_coverage_is_not_offered():
    cfg = clean_sensors()
    cfg["ultrasonic"]["enabled"] = False
    lot = make_aisle_lot(VEH, lot_params(), rng(SEED))
    suite = make_suite(cfg, 0)
    detector = drive_past(lot, make_detector("lines", lot, suite), suite)
    assert not detector.slots
    assert any(tr.confirmed for tr in detector.tracker.tracks)


# ---------- the space detector ----------
def no_lines_lot(seed=SEED, detector="space", occupancy=None):
    params = lot_params(detector)
    params["aisle_lot"]["markings"].update(stall_lines=False, rear_line=False, aisle_ticks=False)
    return make_aisle_lot(VEH, params, rng(seed), occupancy_prob=occupancy)


def inner_free_stalls(lot):
    """Free stalls with a parked car somewhere on each side of them in the row: the gaps it can close."""
    out = []
    for row in (0, 1):
        stalls = [st for st in lot.stalls if st.row == row]
        taken = [st.index for st in stalls if st.occupied]
        out.extend(st for st in stalls if not st.occupied and taken and min(taken) < st.index < max(taken))
    return out


def test_space_detector_finds_gaps_between_cars_with_no_lines():
    lot = no_lines_lot()
    assert not lot.lines
    suite = make_suite(clean_sensors(), 0)
    detector = drive_past(lot, make_detector("space", lot, suite), suite)
    offered = {stall_for(lot, s).slot_id for s in detector.slots}
    assert {st.slot_id for st in inner_free_stalls(lot)} <= offered
    assert_only_free_stalls(lot, detector.slots, tolerance=0.5)


def test_space_slots_have_lower_confidence_and_the_same_type():
    lot = no_lines_lot()
    suite = make_suite(clean_sensors(), 0)
    detector = drive_past(lot, make_detector("space", lot, suite), suite)
    cfg = PARAMS["slot_detection"]
    assert cfg["confidence_space"] < cfg["confidence_lines"]
    assert all(isinstance(s, SlotCandidate) and s.confidence == cfg["confidence_space"] for s in detector.slots)
    assert all(s.slot_type == "perpendicular" and len(s.corners) == 4 for s in detector.slots)


def test_space_detector_gap_length_and_depth_estimate():
    lot = no_lines_lot()
    suite = make_suite(clean_sensors(), 0)
    detector = drive_past(lot, make_detector("space", lot, suite), suite)
    cfg = PARAMS["slot_detection"]["space"]
    for slot in detector.slots:
        corners = np.array(slot.corners)
        assert cfg["min_width"] - 0.3 <= np.hypot(*(corners[1] - corners[0])) <= cfg["stall_width"] + 0.05
        assert np.hypot(*(corners[3] - corners[0])) == pytest.approx(cfg["depth"], abs=0.3)


def test_full_row_gives_no_gaps():
    lot = no_lines_lot(occupancy=1.0)
    suite = make_suite(clean_sensors(), 0)
    detector = drive_past(lot, make_detector("space", lot, suite), suite)
    assert_only_free_stalls(lot, detector.slots, tolerance=0.5)


@pytest.mark.parametrize("seed", range(3))
def test_space_detector_with_noise_never_offers_an_occupied_stall(seed):
    lot = no_lines_lot(seed)
    suite = make_suite(seed=seed)
    detector = drive_past(lot, make_detector("space", lot, suite), suite)
    assert_only_free_stalls(lot, detector.slots, tolerance=0.6)


def test_space_detector_parks_end_to_end_on_a_lot_with_no_lines():
    parked = 0
    for seed in range(3):
        lot = no_lines_lot(seed + 3, occupancy=0.7)
        res = run_search(lot, sensor_seed=seed)
        parked += res.final_state is State.COMPLETED
        for slot in res.search.slots:
            assert slot.confidence == PARAMS["slot_detection"]["confidence_space"]
    assert parked >= 2


# ---------- fused ----------
def test_fused_prefers_lines_and_falls_back_to_space():
    """A fully painted lot gives line-confidence slots; with paint gone, the same lot still gives space slots."""
    painted = make_aisle_lot(VEH, lot_params("fused"), rng(SEED))
    suite = make_suite(clean_sensors(), 0)
    detector = drive_past(painted, make_detector("fused", painted, suite), suite)
    inner = {st.slot_id for st in inner_free_stalls(painted)}
    by_stall = {stall_for(painted, s).slot_id: s for s in detector.slots}
    assert inner <= set(by_stall)
    cfg = PARAMS["slot_detection"]
    assert all(by_stall[i].confidence == cfg["confidence_lines"] for i in inner)

    bare = no_lines_lot(detector="fused")
    suite = make_suite(clean_sensors(), 0)
    detector = drive_past(bare, make_detector("fused", bare, suite), suite)
    assert detector.slots and all(s.confidence == cfg["confidence_space"] for s in detector.slots)


def test_fused_does_not_duplicate_a_slot():
    lot = make_aisle_lot(VEH, lot_params("fused"), rng(SEED))
    suite = make_suite(clean_sensors(), 0)
    detector = drive_past(lot, make_detector("fused", lot, suite), suite)
    stalls = [stall_for(lot, s).slot_id for s in detector.slots]
    assert len(stalls) == len(set(stalls))


# ---------- tracker ----------
def detection(x=0.0, y=0.0, source="lines", sigma=0.02):
    corners = np.array([[x, y], [x + 2.7, y], [x + 2.7, y - 5.5], [x, y - 5.5]])
    return SlotDetection(corners, np.array([np.eye(2) * sigma ** 2] * 4), source)


def tracker(**changes):
    return SlotTracker({**PARAMS["slot_detection"]["tracker"], **changes})


def test_track_confirms_after_n_consistent_detections():
    assert PARAMS["slot_detection"]["tracker"]["confirm_detections"] == 3
    tk = tracker()
    tk.step(0.0, [detection()])
    tk.step(0.2, [detection()])
    assert not tk.tracks[0].confirmed
    tk.step(0.4, [detection()])
    assert tk.tracks[0].confirmed and tk.tracks[0].confirmed_t == 0.4


def test_detection_outside_the_gate_starts_another_track():
    tk = tracker()
    tk.step(0.0, [detection()])
    tk.step(0.2, [detection(x=3.0)])
    assert len(tk.tracks) == 2 and all(tr.hits == 1 for tr in tk.tracks)


def test_detection_with_another_heading_starts_another_track():
    tk = tracker()
    tk.step(0.0, [detection()])
    turned = detection()
    c, s = math.cos(0.5), math.sin(0.5)
    turned.corners = turned.corners @ np.array([[c, s], [-s, c]])
    tk.step(0.2, [turned])
    assert len(tk.tracks) == 2


def test_unconfirmed_track_is_dropped_after_the_timeout_and_a_confirmed_one_is_kept():
    timeout = PARAMS["slot_detection"]["tracker"]["unconfirmed_timeout_s"]
    tk = tracker()
    for t in (0.0, 0.2, 0.4):
        tk.step(t, [detection()])
    tk.step(0.6, [detection(x=20.0)])
    assert len(tk.tracks) == 2
    tk.step(0.6 + timeout + 0.1, [])
    assert len(tk.tracks) == 1 and tk.tracks[0].confirmed
    tk.step(1000.0, [])
    assert len(tk.tracks) == 1


def test_confirmed_slots_never_get_renumbered():
    tk = tracker()
    numbers = {}
    for t, x in enumerate((0.0, 10.0, 20.0)):
        for k in range(3):
            tk.step(t + k * 0.1, [detection(x=x)])
        numbers[x] = next(tr.slot_id for tr in tk.tracks if abs(tr.center[0] - x - 1.35) < 1.0)
    assert sorted(numbers.values()) == [0, 1, 2]
    for t in (10.0, 11.0):
        for x in numbers:
            tk.step(t, [detection(x=x + 0.05)])
    for x, number in numbers.items():
        assert next(tr for tr in tk.tracks if abs(tr.center[0] - x - 1.35) < 1.0).slot_id == number


def test_an_unconfirmed_track_dropping_does_not_use_up_a_number():
    tk = tracker()
    tk.step(0.0, [detection(x=50.0)])
    tk.step(100.0, [])
    for k in range(3):
        tk.step(100.0 + k * 0.1, [detection()])
    assert tk.tracks[0].slot_id == 0


def test_kalman_update_averages_and_shrinks_the_covariance():
    tk = tracker(process_noise=0.0)
    tk.step(0.0, [detection(x=0.0)])
    before = tk.tracks[0].cov[0].copy()
    tk.step(0.2, [detection(x=0.1)])
    tr = tk.tracks[0]
    assert tr.corners[0, 0] == pytest.approx(0.05)
    assert np.trace(tr.cov[0]) == pytest.approx(np.trace(before) / 2.0)


def test_track_weighs_a_detection_by_its_covariance():
    tk = tracker(process_noise=0.0)
    tk.step(0.0, [detection(x=0.0, sigma=0.01)])
    tk.step(0.2, [detection(x=0.3, sigma=0.1)])
    assert tk.tracks[0].corners[0, 0] < 0.01


def test_process_noise_grows_the_covariance_with_time():
    tk = tracker(process_noise=0.1)
    tk.step(0.0, [detection()])
    track = tk.tracks[0]
    before = track.cov[0, 0, 0]
    tk.predict(track, 4.0)
    assert track.cov[0, 0, 0] == pytest.approx(before + 0.04)


def test_line_detection_upgrades_a_space_track_and_space_never_moves_a_line_track():
    tk = tracker()
    tk.step(0.0, [detection(source="space", sigma=0.15)])
    tk.step(0.2, [detection(x=0.2, source="lines")])
    track = tk.tracks[0]
    assert track.source == "lines" and track.corners[0, 0] == pytest.approx(0.2)
    tk.step(0.4, [detection(x=0.4, source="space", sigma=0.15)])
    assert track.corners[0, 0] == pytest.approx(0.2) and track.hits == 2


def test_update_along_one_axis_leaves_the_other_alone():
    """A tilted measurement used to turn sideways error into a large move along the slot."""
    tk = tracker()
    tk.step(0.0, [SlotDetection(detection().corners, np.array([corner_covariance(np.array([0.0, 1.0]), 0.3, 0.02)] * 4), "lines")])
    track = tk.tracks[0]
    y = track.corners[2, 1]
    tk.update_along(track, 2, np.array([1.0, 0.0]), track.corners[2, 0] + 0.1, 0.02)
    assert track.corners[2, 1] == pytest.approx(y)
    assert track.corners[2, 0] > 0.0


# ---------- refinement ----------
def test_refinement_corrects_a_slot_offset_when_lines_come_back_into_view():
    """A slot believed 15 cm off sideways, with a car beside it: the divider in view pulls it back."""
    lot = make_aisle_lot(VEH, lot_params(), rng(SEED))
    suite = make_suite(clean_sensors(), 0)
    detector = make_detector("lines", lot, suite)
    drive_past(lot, detector, suite, x_end=lot.stalls[1].corners[1][0] + 2.0)
    slot = detector.slots[0]
    track = next(tr for tr in detector.tracker.tracks if tr.slot_id == slot.slot_id)
    truth = np.array(stall_for(lot, slot).corners)
    track.corners = truth + np.array([0.15, 0.0])
    detector.set_selected(slot.slot_id)
    stall = stall_for(lot, slot)
    for k in range(60):
        t = 100.0 + k * 0.05
        state = VehicleState(stall.center[0], lot.aisle_y - 3.5, -math.pi / 2)
        detector.observe(suite.step(lot, state, t))
        detector.update(t, state)
    moved = np.array(detector.slots[0].corners)
    assert np.hypot(*(moved[:2] - truth[:2]).T).max() < 0.04


def test_refinement_does_not_use_a_divider_from_the_row_across_the_aisle():
    """It lies on the same x as this slot's divider, but nowhere near the slot."""
    lot = make_aisle_lot(VEH, lot_params(), rng(SEED))
    suite = make_suite(clean_sensors(), 0)
    detector = make_detector("lines", lot, suite)
    drive_past(lot, detector, suite, x_end=lot.stalls[1].corners[1][0] + 2.0)
    slot = detector.slots[0]
    track = next(tr for tr in detector.tracker.tracks if tr.slot_id == slot.slot_id)
    stall = stall_for(lot, slot)
    track.corners = np.array(stall.corners) + np.array([0.1, 0.0])
    before = track.corners.copy()
    detector.set_selected(slot.slot_id)
    across = next(st for st in lot.stalls if st.row == 1 and st.index == stall.index)
    for k in range(30):
        t = 100.0 + k * 0.05
        state = VehicleState(across.center[0], across.corners[0][1] + 1.0, math.pi / 2)  # far side of the aisle
        detector.observe(suite.step(lot, state, t))
        detector.update(t, state)
    assert np.allclose(track.corners, before, atol=1e-6)


def test_selected_slot_stays_on_offer_and_keeps_its_id_during_the_maneuver():
    lot = make_aisle_lot(VEH, lot_params(), rng(SEED))
    res = run_search(lot, sensor_seed=1)
    assert res.search.slot_id == res.search.slots[0].slot_id
    assert res.final_state is State.COMPLETED


def ctrl_with_final_straight():
    x = np.arange(0.0, 4.01, 0.1)
    path = PlannedPath(x, np.zeros(len(x)), np.zeros(len(x)), np.ones(len(x), dtype=int))
    return ParkingController(path, VEH, PARAMS["controller"])


@pytest.mark.parametrize("along", [0.3, -0.3, 0.05])
def test_move_final_stop_keeps_the_line_and_moves_the_end(along):
    ctrl = ctrl_with_final_straight()
    ctrl.move_final_stop(along)
    seg = ctrl.segments[-1]
    assert seg.x[-1] == pytest.approx(4.0 + along)
    assert seg.s[-1] == pytest.approx(4.0 + along)
    assert np.all(np.diff(seg.s) > 0.0)
    assert np.all(seg.y == 0.0)


def test_move_final_stop_is_along_travel_when_reversing():
    x = np.arange(4.0, -0.01, -0.1)
    path = PlannedPath(x, np.zeros(len(x)), np.zeros(len(x)), -np.ones(len(x), dtype=int))
    ctrl = ParkingController(path, VEH, PARAMS["controller"], initial_gear=-1)
    ctrl.move_final_stop(0.2)     # 20 cm further along the direction of travel
    assert ctrl.segments[-1].x[-1] == pytest.approx(-0.2)


def refiner_setup():
    lot = make_aisle_lot(VEH, lot_params(), rng(SEED))
    refiner = simulator._GoalRefiner(lot, None, PARAMS["search"]["planning_time_limit_s"])
    corners = lot.stalls[1].corners
    slot = SlotCandidate(5, corners=corners)
    refiner.start(slot)
    return lot, refiner, slot, corners


def moved(slot, corners, dx=0.0, dy=0.0):
    return replace(slot, corners=tuple((x + dx, y + dy) for x, y in corners))


def test_small_goal_moves_are_ignored():
    assert PARAMS["slot_detection"]["refinement"]["replan_threshold_m"] == 0.05
    assert PARAMS["slot_detection"]["refinement"]["replan_threshold_deg"] == 1.0
    lot, refiner, slot, corners = refiner_setup()
    ctrl = ctrl_with_final_straight()
    state = VehicleState(9.0, lot.aisle_y, 0.0)
    assert refiner.step(1.0, (moved(slot, corners, dx=0.04),), ctrl, state, state) is None
    assert refiner.events == []


def test_goal_move_past_the_threshold_replans_from_the_current_pose():
    lot, refiner, slot, corners = refiner_setup()
    ctrl = ctrl_with_final_straight()
    ctrl.seg_dist_to_go = 3.0
    state = VehicleState(9.0, lot.aisle_y, 0.0)
    result = refiner.step(1.0, (moved(slot, corners, dx=0.08),), ctrl, state, state)
    assert result is not None
    path, fresh = result
    assert (path.x[0], path.y[0]) == pytest.approx((9.0, lot.aisle_y))
    assert fresh is not ctrl
    assert [e.kind for e in refiner.events] == ["replan"]
    assert refiner.events[0].moved_m == pytest.approx(0.08)
    # the new path ends at the moved goal
    assert path.x[-1] == pytest.approx(lot.with_target(lot.stalls[1]).goal.x + 0.08, abs=1e-6)


def test_turning_past_the_threshold_replans_too():
    lot, refiner, slot, corners = refiner_setup()
    ctrl = ctrl_with_final_straight()
    ctrl.seg_dist_to_go = 3.0
    c, s = math.cos(math.radians(1.5)), math.sin(math.radians(1.5))
    arr = np.array(corners)
    center = arr.mean(axis=0)
    turned = (arr - center) @ np.array([[c, s], [-s, c]]) + center
    state = VehicleState(9.0, lot.aisle_y, 0.0)
    result = refiner.step(1.0, (replace(slot, corners=tuple(map(tuple, turned.tolist()))),), ctrl, state, state)
    assert result is not None and refiner.events[0].moved_deg == pytest.approx(1.5)


def test_replan_carries_over_the_gear_and_gear_count():
    lot, refiner, slot, corners = refiner_setup()
    ctrl = ctrl_with_final_straight()
    ctrl.seg_dist_to_go, ctrl.gear, ctrl.gear_changes = 3.0, -1, 1
    state = VehicleState(9.0, lot.aisle_y, 0.0)
    _, fresh = refiner.step(1.0, (moved(slot, corners, dx=0.08),), ctrl, state, state)
    assert fresh.gear == -1 and fresh.gear_changes == 1


def test_inside_the_final_straight_only_the_stop_moves():
    lot, refiner, slot, corners = refiner_setup()
    ctrl = ctrl_with_final_straight()
    ctrl.state, ctrl.seg_dist_to_go = ctrl.DRIVING, 1.0
    state = VehicleState(9.0, lot.aisle_y, 0.0)
    result = refiner.step(1.0, (moved(slot, corners, dx=0.2, dy=0.1),), ctrl, state, state)
    assert result is None
    assert [e.kind for e in refiner.events] == ["stop_adjust"]
    seg = ctrl.segments[-1]
    # the segment runs along yaw 0 here, so of the goal's move only dx applies, along the segment
    assert seg.x[-1] == pytest.approx(4.2)
    assert seg.y[-1] == pytest.approx(0.0)


def test_replans_are_limited_in_number_and_spacing():
    cfg = PARAMS["slot_detection"]["refinement"]
    lot, refiner, slot, corners = refiner_setup()
    ctrl = ctrl_with_final_straight()
    ctrl.seg_dist_to_go = 3.0
    state = VehicleState(9.0, lot.aisle_y, 0.0)
    assert refiner.step(0.0, (moved(slot, corners, dx=0.08),), ctrl, state, state) is not None
    assert refiner.step(cfg["min_replan_interval_s"] / 2, (moved(slot, corners, dx=0.16),), ctrl, state, state) is None
    t = cfg["min_replan_interval_s"] + 0.1
    for k in range(cfg["max_replans"] + 2):
        refiner.step(t * (k + 1), (moved(slot, corners, dx=0.08 * (k + 2)),), ctrl, state, state)
    assert sum(e.kind == "replan" for e in refiner.events) == cfg["max_replans"]


def test_a_replan_that_would_break_the_gear_limit_is_refused():
    """Regression: replans pushed some maneuvers to a 4th gear change, which aborts them."""
    limit = PARAMS["supervisor"]["max_gear_changes"]
    lot, refiner, slot, corners = refiner_setup()
    ctrl = ctrl_with_final_straight()
    ctrl.seg_dist_to_go, ctrl.gear_changes = 3.0, limit     # no shift left to spend
    state = VehicleState(9.0, lot.aisle_y, 0.0)
    assert refiner.step(1.0, (moved(slot, corners, dx=0.08),), ctrl, state, state) is None
    assert [e.kind for e in refiner.events] == ["replan_refused"]


def test_a_failed_replan_keeps_the_old_path_and_is_logged(monkeypatch):
    lot, refiner, slot, corners = refiner_setup()
    monkeypatch.setattr(simulator, "_plan_in_frame", lambda *a, **k: (None, 0.1))
    ctrl = ctrl_with_final_straight()
    ctrl.seg_dist_to_go = 3.0
    state = VehicleState(9.0, lot.aisle_y, 0.0)
    assert refiner.step(1.0, (moved(slot, corners, dx=0.08),), ctrl, state, state) is None
    assert [e.kind for e in refiner.events] == ["replan_failed"]


# ---------- end to end ----------
@pytest.fixture(scope="module")
def fused_run():
    params = lot_params("fused")
    lot = make_aisle_lot(VEH, params, rng(SEED))
    return lot, run_search(lot, localization_rng=rng(2), sensor_seed=3)


def test_fused_run_parks_in_a_free_stall_and_meets_every_criterion(fused_run):
    lot, res = fused_run
    assert res.success, res.summary()
    assert res.final_state is State.COMPLETED
    assert not res.search.scenario.nearest_stall(*res.search.scenario.stall_center).occupied


def test_metrics_cover_detection_quality(fused_run):
    _, res = fused_run
    m = res.metrics
    for key in ("slot_corner_error_m", "selected_corner_error_m", "time_to_confirm_s", "slots_offered",
                "slots_false_positive", "free_stalls_passed", "free_stalls_missed", "replans", "stop_adjustments"):
        assert key in m and not math.isnan(m[key]), key
    assert m["slot_corner_error_m"] < 0.15
    assert m["selected_corner_error_m"] < 0.15
    assert 0.0 < m["time_to_confirm_s"] < 2.0
    assert m["slots_false_positive"] == 0
    assert m["replans"] == sum(r.kind == "replan" for r in res.search.refinements)


def test_controller_follows_the_estimated_pose_so_drift_shows_in_the_true_path():
    """With a heavy gyro bias, the plan starts on the estimate, away from where the car truly is."""
    params = lot_params("lines")
    params["localization"]["odometry"].update(wheel_scale_error=0.0, wheel_noise_std=0.0, yaw_rate_bias_deg_s=0.5,
                                              yaw_rate_noise_std_deg_s=0.0)
    lot = make_aisle_lot(VEH, params, rng(SEED))
    res = run_search(lot, localization_rng=rng(0))
    assert res.search.plans
    t, path = res.search.plans[0]
    i = int(np.searchsorted(res.log["t"], t))
    assert math.hypot(path.x[0] - res.log["x"][i], path.y[0] - res.log["y"][i]) > 0.005


def test_planner_gets_the_map_frame_and_the_controller_the_odometry_frame(monkeypatch):
    """With drift, the plan handed to the controller starts on the estimated pose."""
    lot = make_aisle_lot(VEH, lot_params(), rng(SEED))
    truth = VehicleState(9.0, lot.aisle_y, 0.0)
    estimate = Pose(9.3, lot.aisle_y + 0.2, 0.02)
    frame = MapToOdom(truth, estimate)
    corners = np.array(lot.stalls[1].corners)
    in_odom = np.column_stack(frame.to_odom(corners[:, 0], corners[:, 1]))
    slot = SlotCandidate(9, corners=tuple(map(tuple, in_odom.tolist())))
    seen = {}
    real = simulator._plan_to_slot

    def spy(sc, map_slot, start, *args):
        seen["start"], seen["corners"] = start, map_slot.corners
        return real(sc, map_slot, start, *args)

    monkeypatch.setattr(simulator, "_plan_to_slot", spy)
    path, _ = simulator._plan_in_frame(lot, slot, truth, estimate, None, 2.0, True)
    assert (seen["start"].x, seen["start"].y) == (truth.x, truth.y)
    assert np.array(seen["corners"]) == pytest.approx(corners)      # the slot's true place, drift removed
    assert path.x[0] == pytest.approx(estimate.x) and path.y[0] == pytest.approx(estimate.y)
    assert path.yaw[0] == pytest.approx(estimate.yaw)


def test_ideal_detector_still_follows_the_true_pose_and_keeps_stall_ids():
    params = lot_params("ideal")
    lot = make_aisle_lot(VEH, params, rng(SEED))
    res = run_search(lot)
    assert res.success and res.search.slot_id == 1
    assert res.metrics["slot_corner_error_m"] == 0.0 and res.metrics["replans"] == 0.0


def test_a_run_with_nothing_to_offer_reports_nan_error_and_ends_cleanly():
    params = lot_params("lines")
    params["aisle_lot"]["markings"].update(stall_lines=False, rear_line=False, aisle_ticks=False)
    lot = make_aisle_lot(VEH, params, rng(SEED))
    res = run_search(lot, sensor_seed=0)
    assert res.search.slot_id is None and res.reason == "reached the end of the aisle"
    assert math.isnan(res.metrics["slot_corner_error_m"]) and res.metrics["slots_offered"] == 0
    assert res.metrics["free_stalls_missed"] == res.metrics["free_stalls_passed"] > 0


def test_driver_falls_back_to_the_nearest_slot_at_the_end_of_the_aisle():
    """Regression: the earliest-seen slot could be 60 m back, too far to reverse in the time limit."""
    cfg = replace(DriverConfig.from_params(PARAMS["search"]), policy="select_first_after", after_x=50.0, reaction_time_s=0.0)
    driver = SearchDriver(cfg, VEH.wheelbase, VEH.max_steer, 9.15, 54.0)
    near = SlotCandidate(1, corners=((44.0, 5.5), (46.7, 5.5), (46.7, 0.0), (44.0, 0.0)), first_seen_t=9.0)
    far = SlotCandidate(0, corners=((2.0, 5.5), (4.7, 5.5), (4.7, 0.0), (2.0, 0.0)), first_seen_t=2.0)
    driver.observe(State.SLOTS_AVAILABLE)
    assert driver.act(10.0, 40.0, 9.15, 0.0, 2.8, (far, near)).selected_slot_id is None      # none past after_x yet
    assert driver.act(20.0, 58.0, 9.15, 0.0, 0.1, (far, near)).selected_slot_id == 1


def test_driver_after_x_picks_the_first_slot_past_it():
    cfg = replace(DriverConfig.from_params(PARAMS["search"]), policy="select_first_after", after_x=10.0, reaction_time_s=0.0)
    driver = SearchDriver(cfg, VEH.wheelbase, VEH.max_steer, 9.15, 54.0)
    before = SlotCandidate(0, corners=((2.0, 5.5), (4.7, 5.5), (4.7, 0.0), (2.0, 0.0)), first_seen_t=2.0)
    after = SlotCandidate(1, corners=((14.0, 5.5), (16.7, 5.5), (16.7, 0.0), (14.0, 0.0)), first_seen_t=9.0)
    driver.observe(State.SLOTS_AVAILABLE)
    assert driver.act(12.0, 20.0, 9.15, 0.0, 2.8, (before, after)).selected_slot_id == 1
    with pytest.raises(ValueError, match="after_x"):
        replace(cfg, after_x=None)


def test_search_trial_with_a_sensor_detector_repeats_by_seed():
    params = lot_params("fused")
    params["search"]["monte_carlo"]["speed_kph"] = [14.0, 14.0]
    _, a, drawn = search_trial(VEH, params, 2)
    _, b, drawn2 = search_trial(VEH, params, 2)
    assert drawn == drawn2
    assert np.array_equal(a.log["x"], b.log["x"])
    assert a.success == b.success


def test_export_shows_replanned_paths_and_refinements(tmp_path):
    pytest.importorskip("mcap")
    spec = importlib.util.spec_from_file_location("mcap_tests", ROOT / "tests" / "test_mcap_export.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lot = make_aisle_lot(VEH, lot_params("lines"), rng(SEED))
    res = run_search(lot)
    assert res.search.plans
    messages = module.read_mcap(export_run(lot, res, tmp_path / "r.mcap"))[3]
    assert len(messages["/plan"]) == len(res.search.plans)
    doc = messages["/metrics"][0][1]
    assert len(doc["refinements"]) == len(res.search.refinements)
    assert doc["metrics"]["time_to_confirm_s"] == pytest.approx(res.metrics["time_to_confirm_s"])
    assert messages["/slots"]


def test_detector_interface_defaults_are_harmless():
    class Plain(SlotDetector):
        def update(self, t, pose):
            return self.slots
    plain = Plain()
    plain.observe([])
    plain.set_selected(3)
    assert plain.confirm_times() == [] and not plain.uses_estimated_pose


def test_monte_carlo_cli_reports_detection_metrics(tmp_path):
    import subprocess
    out = tmp_path / "s.csv"
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "monte_carlo.py"), "-n", "1", "--mode", "search",
                           "--detector", "fused", "--csv", str(out)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    header = out.read_text().splitlines()[0].split(",")
    assert {"slot_corner_error_m", "time_to_confirm_s", "slots_false_positive", "free_stalls_missed", "replans"} <= set(header)
    assert "detection:" in done.stdout
