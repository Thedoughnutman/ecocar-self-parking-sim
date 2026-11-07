"""Synthetic sensors: geometry against hand-built scenes, effects, determinism, speed, export."""
import copy
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from park_sim.driver import DriverConfig, SearchDriver  # noqa: E402
from park_sim.geometry import rect  # noqa: E402
from park_sim.mcap_export import TOPICS, export_run, sensor_topics  # noqa: E402
from park_sim.scenario import Pose, Scenario, load_params, make_aisle_lot, make_perpendicular  # noqa: E402
from park_sim.sensors import (FALSE_POSITIVE, Lidar2D, LidarMeasurement, LineCamera, LineCameraMeasurement,  # noqa: E402
                              Measurement, MountPose, Sensor, UltrasonicMeasurement,
                              UltrasonicSensor, load_sensor_params, make_lidar, make_line_cameras, make_sensors,
                              make_suite, make_ultrasonic_sensors)
from park_sim.sensors.scene import cast_rays, nearest_in_cone, scene_arrays, sight_blocked  # noqa: E402
from park_sim.simulator import run_closed_loop, run_search  # noqa: E402
from park_sim.vehicle import Vehicle, VehicleParams, VehicleState  # noqa: E402

VEH = VehicleParams.from_yaml()
PARAMS = load_params()
PARAMS["slot_detection"]["detector"] = "ideal"   # nothing acts on the sensors with the ideal detector
CFG = load_sensor_params()
HERE = VehicleState(0.0, 0.0, 0.0)
RIGHT_CAM = MountPose(1.55, -0.95, -math.pi / 2)
SEED = 1


def rng(seed=0):
    return np.random.default_rng(seed)


def scene(lines=(), cars=(), walls=()):
    """A hand-built lot. The sensors don't need a distance field."""
    return Scenario(obstacles=list(cars) + list(walls), lines=list(lines), target_stall=None, stall_center=None,
                    goal=None, start=Pose(0.0, 0.0, 0.0), bounds=(-50, 50, -50, 50), cfg=PARAMS, vehicle=VEH,
                    parked_cars=list(cars))


def camera_cfg(**effects):
    """Line camera settings with every effect off, except those given."""
    cfg = copy.deepcopy(CFG["line_camera"])
    cfg["noise"]["enabled"] = cfg["faded"]["enabled"] = cfg["gaps"]["enabled"] = False
    cfg["false_positives"]["enabled"] = False
    cfg["occlusion"] = {"parked_cars": False, "ego_body": False}
    for key, value in effects.items():
        if isinstance(cfg[key], dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


def right_camera(**effects):
    return LineCamera("cam", RIGHT_CAM, camera_cfg(**effects))


def wall_ahead(distance, mount_x=0.0):
    """A wall across the +x direction, `distance` beyond a sensor at mount_x looking along +x."""
    return rect(mount_x + distance + 0.5, 0.0, 0.0, 1.0, 40.0)


def sonar(**changes):
    args = dict(name="u", mount=MountPose(0.0, 0.0, 0.0), rate_hz=15.0, cone_deg=60.0, range_min=0.2, range_max=4.5,
                noise_std=0.0, dropout_prob=0.0)
    args.update(changes)
    return UltrasonicSensor(**args)


def same_as_line(segment, a, b, tol):
    """The segment's endpoints match a-b within tol, in either order."""
    forward = max(math.dist(segment[0], a), math.dist(segment[1], b))
    backward = max(math.dist(segment[0], b), math.dist(segment[1], a))
    return min(forward, backward) <= tol


# ---------- interface ----------
def test_every_sensor_shares_the_interface():
    cfg = copy.deepcopy(CFG)
    cfg["lidar"]["enabled"] = True
    for sensor in make_sensors(cfg):
        assert isinstance(sensor, Sensor)
        assert isinstance(sensor.mount, MountPose) and sensor.rate_hz > 0
        m = sensor.measure(scene(), HERE, 1.5, rng())
        assert isinstance(m, Measurement)
        assert m.sensor == sensor.name and m.t == 1.5


def test_config_marks_its_values_as_estimates():
    text = (ROOT / "config" / "sensors.yaml").read_text()
    assert "Every value here is a placeholder [est]" in text
    for key in ("rate_hz:", "std:", "noise_std:", "dropout_prob:", "p_faded:", "fp_rate:", "cameras:", "sensors:",
                "side_sensors:", "x: 1.5", "beams:"):
        for line in text.splitlines():
            if line.strip().startswith(key):
                assert "[est]" in line, line


def test_mount_pose_to_vehicle_and_map():
    mount = MountPose(1.0, 2.0, math.pi / 2)
    assert mount.to_vehicle(np.array([[1.0, 0.0]])) == pytest.approx(np.array([[1.0, 3.0]]))
    x, y, yaw = mount.in_map(VehicleState(10.0, 20.0, math.pi / 2))
    assert (x, y, yaw) == pytest.approx((8.0, 21.0, math.pi))


def test_suite_runs_each_sensor_at_its_rate():
    suite = make_suite(seed=0)
    counts = {s.name: 0 for s in suite.sensors}
    lot = scene()
    for i in range(500):       # 10 s at 50 Hz
        for m in suite.step(lot, HERE, i * 0.02):
            assert m.t == i * 0.02
            counts[m.sensor] += 1
    assert counts["cam_rear"] == 300
    assert counts["uss_fl_inner"] == 150
    assert counts["uss_side_left"] == 150


def test_scene_arrays_are_cached_and_shared_with_a_target_copy():
    lot = make_aisle_lot(VEH, PARAMS, rng(SEED))
    arrays = scene_arrays(lot)
    assert scene_arrays(lot) is arrays
    assert scene_arrays(lot.with_target(lot.stalls[1])) is arrays
    assert len(arrays.lines_a) == len(lot.lines)
    assert len(arrays.obstacle_a) == 4 * len(lot.obstacles)
    assert len(arrays.car_a) == 4 * len(lot.parked_cars)


# ---------- lot markings ----------
@pytest.mark.parametrize("make, section", [(make_perpendicular, "scenario"), (make_aisle_lot, "aisle_lot")])
def test_markings_are_configurable_per_lot(make, section):
    def lot_with(**marks):
        params = copy.deepcopy(PARAMS)
        params[section]["markings"].update(marks)
        return make(VEH, params, rng(0))

    both = lot_with(rear_line=True, aisle_ticks=True)
    n_rows = 2
    stalls = PARAMS["aisle_lot"]["stalls_per_row"] if section == "aisle_lot" else PARAMS["scenario"]["n_stalls"]
    bare = lot_with(rear_line=False, aisle_ticks=False)
    assert len(bare.lines) == n_rows * (stalls + 1)      # only the lines between stalls
    assert len(lot_with(rear_line=True, aisle_ticks=False).lines) == len(bare.lines) + n_rows * stalls
    assert len(lot_with(rear_line=False, aisle_ticks=True).lines) == len(bare.lines) + n_rows * (stalls + 1)
    assert len(both.lines) == len(bare.lines) + n_rows * (2 * stalls + 1)
    assert all(np.array_equal(a, b) for a, b in zip(both.obstacles, bare.obstacles))


def test_ticks_cross_the_aisle_end_of_each_stall_line():
    lot = make_aisle_lot(VEH, PARAMS, rng(0))
    w, d = PARAMS["scenario"]["stall_width"], PARAMS["scenario"]["stall_depth"]
    tick = PARAMS["aisle_lot"]["markings"]["tick_length"]
    assert ((w - tick / 2, d), (w + tick / 2, d)) in lot.lines
    assert ((w, 0.0), (2 * w, 0.0)) in lot.lines          # rear line of stall 1


# ---------- line camera: geometry ----------
def test_line_inside_the_footprint_is_returned_within_noise():
    """An L-shaped pair, away from the body, under a turned and moved car."""
    state = VehicleState(5.0, -3.0, 0.4)
    c, s = math.cos(state.yaw), math.sin(state.yaw)
    local = [((0.5, -3.0), (2.5, -3.0)), ((1.0, -2.0), (1.0, -3.5))]
    world = [tuple((state.x + c * x - s * y, state.y + s * x + c * y) for x, y in line) for line in local]
    std = CFG["line_camera"]["noise"]["std"]
    assert std == 0.03
    camera = right_camera(noise={"enabled": True, "std": std})
    m = camera.measure(scene(lines=world), state, 0.0, rng())
    assert isinstance(m, LineCameraMeasurement)
    assert m.segments.shape == (2, 2, 2)
    assert list(m.truth_line) == [0, 1]
    for seg, line in zip(m.segments, local):
        assert same_as_line(seg, *line, tol=5 * math.sqrt(2) * std)


def test_without_noise_a_visible_line_comes_back_exactly():
    line = ((0.5, -3.0), (2.5, -3.0))
    m = right_camera().measure(scene(lines=[line]), HERE, 0.0, rng())
    assert m.segments[0] == pytest.approx(np.array(line))


def test_default_camera_set_is_three_surround_views_with_the_requested_footprint():
    cameras = make_line_cameras()
    assert len(cameras) == 3
    fp = CFG["line_camera"]["footprint"]
    assert (fp["width"], fp["depth"]) == (5.0, 4.0)
    assert {c.name for c in cameras} == {"cam_rear", "cam_left", "cam_right"}
    for cam in cameras:
        edges = np.hypot(*(np.roll(cam.footprint, -1, axis=0) - cam.footprint).T)
        assert sorted(edges) == pytest.approx([4.0, 4.0, 5.0, 5.0])
    by_name = {c.name: c.footprint.mean(axis=0) for c in cameras}
    assert by_name["cam_rear"][0] < 0.0
    assert by_name["cam_left"][1] > 1.0 and by_name["cam_right"][1] < -1.0


def test_vehicle_has_no_front_camera_but_keeps_every_other_sensor():
    """The front camera is permanently out on this LYRIQ; the rest of the surround view and all ultrasonics work."""
    names = {s.name for s in make_suite(seed=0).sensors}
    assert "cam_front" not in names
    assert {"cam_rear", "cam_left", "cam_right"} <= names
    # No camera in the default set looks forward, whatever it is named
    for cam in make_line_cameras():
        assert abs(math.atan2(math.sin(cam.mount.yaw), math.cos(cam.mount.yaw))) > math.radians(45.0)
    uss = CFG["ultrasonic"]
    assert {e["name"] for e in uss["sensors"] + uss["side_sensors"]} <= names
    assert len(uss["sensors"]) + len(uss["side_sensors"]) == 14


def test_camera_footprint_can_be_overridden_per_camera():
    cfg = copy.deepcopy(CFG["line_camera"])
    cfg["cameras"][0]["footprint"] = {"width": 2.0, "depth": 1.0, "near": 0.0}
    cameras = make_line_cameras(cfg)
    edges = np.hypot(*(np.roll(cameras[0].footprint, -1, axis=0) - cameras[0].footprint).T)
    assert sorted(edges) == pytest.approx([1.0, 1.0, 2.0, 2.0])
    assert sorted(np.hypot(*(np.roll(cameras[1].footprint, -1, axis=0) - cameras[1].footprint).T)) == pytest.approx([4, 4, 5, 5])


def test_line_is_clipped_to_the_footprint():
    # Footprint of the right camera: x 1.55 +/- 2.5, y from -0.45 down to -4.45
    m = right_camera().measure(scene(lines=[((-20.0, -3.0), (20.0, -3.0))]), HERE, 0.0, rng())
    xs = sorted(m.segments[0][:, 0])
    assert xs == pytest.approx([-0.95, 4.05])
    assert m.segments[0][:, 1] == pytest.approx([-3.0, -3.0])


def test_diagonal_line_is_clipped_on_two_edges():
    m = right_camera().measure(scene(lines=[((-3.0, -8.0), (3.0, -2.0))]), HERE, 0.0, rng())
    # y = x - 5: enters at the far edge y = -4.45 (x = 0.55), leaves at x = 3
    ends = m.segments[0][np.argsort(m.segments[0][:, 0])]
    assert ends == pytest.approx(np.array([[0.55, -4.45], [3.0, -2.0]]))


@pytest.mark.parametrize("line", [
    ((0.5, 3.0), (2.5, 3.0)),       # the other side of the car
    ((0.5, -6.0), (2.5, -6.0)),     # past the far edge
    ((5.0, -2.0), (6.0, -3.0)),     # off the front
    ((-5.0, -6.0), (6.0, -6.0)),    # passes the footprint, never enters
])
def test_line_outside_the_footprint_is_not_returned(line):
    assert len(right_camera().measure(scene(lines=[line]), HERE, 0.0, rng()).segments) == 0


def test_output_is_in_the_vehicle_frame_after_moving():
    """The front camera sees a map line at x = 10 as 6 m ahead of the axle of a car at x = 4."""
    camera = LineCamera("c", MountPose(4.0, 0.0, 0.0), camera_cfg())
    m = camera.measure(scene(lines=[((10.0, 2.0), (10.0, 4.0))]), VehicleState(4.0, 3.0, 0.0), 0.0, rng())
    assert m.segments[0] == pytest.approx(np.array([[6.0, -1.0], [6.0, 1.0]]))


def test_vehicle_frame_after_a_quarter_turn():
    """A car at the origin heading +y: map (3, 1) is vehicle (1, -3)."""
    state = VehicleState(0.0, 0.0, math.pi / 2)
    m = right_camera().measure(scene(lines=[((3.0, 1.0), (3.0, 2.0))]), state, 0.0, rng())
    assert m.segments[0] == pytest.approx(np.array([[1.0, -3.0], [2.0, -3.0]]))


# ---------- line camera: occlusion ----------
def behind_car_scene():
    """Right camera at y = -0.95. A car across y -2.0 to -3.0, a line behind it at y = -3.8."""
    car = rect(1.55, -2.5, 0.0, 3.0, 1.0)
    return car, ((1.0, -3.8), (2.1, -3.8))


def test_line_behind_a_parked_car_is_not_returned_by_a_side_camera_on_the_other_side():
    car, line = behind_car_scene()
    camera = right_camera(occlusion={"parked_cars": True, "ego_body": False})
    assert len(camera.measure(scene(lines=[line], cars=[car]), HERE, 0.0, rng()).segments) == 0
    # Control: the same camera and line without the car, and with the effect off
    assert len(camera.measure(scene(lines=[line]), HERE, 0.0, rng()).segments) == 1
    assert len(right_camera().measure(scene(lines=[line], cars=[car]), HERE, 0.0, rng()).segments) == 1


def test_line_under_a_parked_car_is_hidden():
    car = rect(1.55, -2.5, 0.0, 3.0, 1.0)
    camera = right_camera(occlusion={"parked_cars": True, "ego_body": False})
    m = camera.measure(scene(lines=[((1.0, -2.5), (2.0, -2.5))], cars=[car]), HERE, 0.0, rng())
    assert len(m.segments) == 0


def test_a_car_beyond_the_line_does_not_hide_it():
    car = rect(1.55, -3.8, 0.0, 3.0, 1.0)
    camera = right_camera(occlusion={"parked_cars": True, "ego_body": False})
    line = ((1.0, -2.0), (2.0, -2.0))
    m = camera.measure(scene(lines=[line], cars=[car]), HERE, 0.0, rng())
    assert m.segments[0] == pytest.approx(np.array(line))


def test_a_line_part_hidden_by_a_car_is_split_around_its_shadow():
    """A short car straight ahead of the camera shades the middle of a line behind it."""
    car = rect(1.55, -2.0, 0.0, 0.6, 0.4)
    camera = right_camera(occlusion={"parked_cars": True, "ego_body": False}, sample_step=0.02)
    m = camera.measure(scene(lines=[((-0.5, -4.0), (3.6, -4.0))], cars=[car]), HERE, 0.0, rng())
    assert len(m.segments) == 2
    left, right = sorted(m.segments, key=lambda seg: seg[:, 0].mean())
    # Shadow edges: the rays past the car's front corners, at y = -1.8, half-width 0.3, seen from y = -0.95
    half_shadow = 0.3 * (4.0 - 0.95) / (1.8 - 0.95)
    assert left[:, 0].max() == pytest.approx(1.55 - half_shadow, abs=0.03)
    assert right[:, 0].min() == pytest.approx(1.55 + half_shadow, abs=0.03)
    assert left[:, 0].min() == pytest.approx(-0.5) and right[:, 0].max() == pytest.approx(3.6)


def test_ego_body_hides_what_is_under_it():
    """The footprint reaches 0.5 m under the body, whose side is at y = -0.9885."""
    assert CFG["line_camera"]["footprint"]["near"] < 0.0
    line = ((1.0, -0.6), (1.0, -2.0))
    on = right_camera(occlusion={"parked_cars": False, "ego_body": True})
    m = on.measure(scene(lines=[line]), HERE, 0.0, rng())
    ys = sorted(m.segments[0][:, 1])
    assert ys[0] == pytest.approx(-2.0)
    assert ys[1] == pytest.approx(-VEH.width / 2, abs=CFG["line_camera"]["sample_step"])
    assert ys[1] <= -VEH.width / 2
    off = right_camera().measure(scene(lines=[line]), HERE, 0.0, rng())
    assert sorted(off.segments[0][:, 1]) == pytest.approx([-2.0, -0.6])


def test_line_wholly_under_the_ego_body_is_not_returned():
    line = ((1.0, -0.5), (2.0, -0.5))
    on = right_camera(occlusion={"parked_cars": False, "ego_body": True})
    assert len(on.measure(scene(lines=[line]), HERE, 0.0, rng()).segments) == 0
    assert len(right_camera().measure(scene(lines=[line]), HERE, 0.0, rng()).segments) == 1


def test_parked_cars_hide_paint_in_the_real_lot():
    """Passing a full row closer than the centerline: the side lines reach in between the cars,
    and those off to the side of the camera are partly hidden by the car in front of them."""
    lot = make_aisle_lot(VEH, PARAMS, rng(SEED), occupancy_prob=1.0)
    blocking = LineCamera("c", RIGHT_CAM, camera_cfg(occlusion={"parked_cars": True, "ego_body": False}))
    clear = LineCamera("c", RIGHT_CAM, camera_cfg())

    def seen_length(camera):
        total = 0.0
        for x in np.arange(10.0, 40.0, 0.5):
            m = camera.measure(lot, VehicleState(x, lot.aisle_y - 1.5, 0.0), 0.0, rng())
            total += np.hypot(*(m.segments[:, 1] - m.segments[:, 0]).T).sum()
        return total

    hidden, seen = seen_length(blocking), seen_length(clear)
    assert 0.0 < hidden < 0.95 * seen


# ---------- line camera: other effects ----------
def test_noise_has_the_configured_size_and_changes_every_frame():
    line = ((0.5, -3.0), (2.5, -3.0))
    camera = right_camera(noise={"enabled": True, "std": 0.03})
    lot, r = scene(lines=[line]), rng(5)
    errors = np.array([camera.measure(lot, HERE, 0.0, r).segments[0] - np.array(line) for _ in range(500)])
    assert errors.std() == pytest.approx(0.03, rel=0.1)
    assert abs(errors.mean()) < 0.005
    assert not np.array_equal(errors[0], errors[1])


def test_faded_lines_are_missing_and_stay_missing():
    lines = [((1.0 + 0.1 * i, -2.0), (1.0 + 0.1 * i, -3.0)) for i in range(20)]
    lot = scene(lines=lines)
    camera = right_camera(faded={"enabled": True, "p_faded": 0.5})
    r = rng(1)
    first = set(camera.measure(lot, HERE, 0.0, r).truth_line)
    assert 3 < len(first) < 17
    for _ in range(5):
        assert set(camera.measure(lot, HERE, 0.0, r).truth_line) == first


def test_faded_probability_extremes_and_off_switch():
    line = ((0.5, -3.0), (2.5, -3.0))
    lot = scene(lines=[line])
    assert not len(right_camera(faded={"enabled": True, "p_faded": 1.0}).measure(lot, HERE, 0.0, rng()).segments)
    assert len(right_camera(faded={"enabled": False, "p_faded": 1.0}).measure(lot, HERE, 0.0, rng()).segments)


def test_cameras_share_the_same_wear_and_a_new_lot_redraws_it():
    cfg = camera_cfg(faded={"enabled": True, "p_faded": 0.5}, gaps={"enabled": True, "rate_per_m": 1.0})
    a, b = make_line_cameras(cfg)[:2]
    assert a.wear is b.wear
    lot = make_aisle_lot(VEH, PARAMS, rng(0))
    a.measure(lot, HERE, 0.0, rng(1))
    faded = a.wear.faded.copy()
    gaps = list(a.wear.gaps)
    b.measure(lot.with_target(lot.stalls[1]), HERE, 0.0, rng(2))      # same paint
    assert np.array_equal(b.wear.faded, faded) and all(x is y for x, y in zip(b.wear.gaps, gaps))
    assert 0 < faded.sum() < len(faded)
    b.measure(make_aisle_lot(VEH, PARAMS, rng(0)), HERE, 0.0, rng(3))
    assert not np.array_equal(b.wear.faded, faded)


def test_gaps_split_a_line_and_stay_in_place():
    line = ((-0.5, -3.0), (3.5, -3.0))
    camera = right_camera(gaps={"enabled": True, "rate_per_m": 1.5, "length": [0.3, 0.3]}, sample_step=0.02)
    lot, r = scene(lines=[line]), rng(2)
    m = camera.measure(lot, HERE, 0.0, r)
    assert len(m.segments) >= 2
    assert set(m.truth_line) == {0}
    assert np.array_equal(camera.measure(lot, HERE, 0.0, r).segments, m.segments)
    spans = sorted(tuple(sorted(s[:, 0])) for s in m.segments)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start > end     # no overlaps
    assert np.all(camera.wear.gaps[0][:, 1] - camera.wear.gaps[0][:, 0] == pytest.approx(0.3))


def test_gaps_off_leave_the_line_whole():
    line = ((-0.5, -3.0), (3.5, -3.0))
    cam = right_camera(gaps={"enabled": False, "rate_per_m": 5.0})
    assert len(cam.measure(scene(lines=[line]), HERE, 0.0, rng()).segments) == 1


def test_false_positives_arrive_at_the_configured_rate_as_short_marked_segments():
    camera = right_camera(false_positives={"enabled": True, "fp_rate": 15.0, "length": [0.2, 0.6]})
    lot, r = scene(), rng(3)
    frames = [camera.measure(lot, HERE, 0.0, r) for _ in range(600)]
    counts = [len(m.segments) for m in frames]
    # 15 per second at 30 Hz is 0.5 per frame; clipping removes a few
    assert np.mean(counts) == pytest.approx(0.5, rel=0.2)
    segments = np.concatenate([m.segments for m in frames])
    assert np.all(np.concatenate([m.truth_line for m in frames]) == FALSE_POSITIVE)
    lengths = np.hypot(*(segments[:, 1] - segments[:, 0]).T)
    assert lengths.max() <= 0.6 + 1e-9
    x, y = segments[:, :, 0], segments[:, :, 1]
    assert x.min() >= 1.55 - 2.5 - 1e-9 and x.max() <= 1.55 + 2.5 + 1e-9
    assert y.max() <= -0.45 + 1e-9 and y.min() >= -4.45 - 1e-9
    assert len(np.unique(segments[:, 0, 0])) == len(segments)   # new every frame


def test_false_positives_are_off_when_disabled():
    camera = right_camera(false_positives={"enabled": False, "fp_rate": 1000.0})
    assert not len(camera.measure(scene(), HERE, 0.0, rng()).segments)


def test_false_positives_are_not_under_the_ego_body():
    camera = right_camera(false_positives={"enabled": True, "fp_rate": 300.0, "length": [0.2, 0.2]})
    r, lot = rng(4), scene()
    for _ in range(50):
        seg = camera.measure(lot, HERE, 0.0, r).segments
        if len(seg):
            mid = seg.mean(axis=1)
            assert not np.any((np.abs(mid[:, 1]) <= VEH.width / 2)
                              & (mid[:, 0] >= -VEH.rear_overhang) & (mid[:, 0] <= VEH.wheelbase + VEH.front_overhang))


def test_short_visible_pieces_are_dropped():
    line = ((1.0, -3.0), (1.05, -3.0))
    assert not len(right_camera().measure(scene(lines=[line]), HERE, 0.0, rng()).segments)
    assert len(right_camera(min_segment_length=0.01).measure(scene(lines=[line]), HERE, 0.0, rng()).segments) == 1


def test_a_parked_car_on_the_camera_line_of_sight_with_no_lines_is_fine():
    camera = right_camera(occlusion={"parked_cars": True, "ego_body": True})
    assert camera.measure(scene(cars=[rect(1.5, -2.5, 0.0, 3.0, 1.0)]), HERE, 0.0, rng()).segments.shape == (0, 2, 2)


# ---------- ultrasonic ----------
def test_sensor_facing_a_wall_reports_it_within_noise():
    std = CFG["ultrasonic"]["noise_std"]
    assert std == 0.02
    sensor = sonar(noise_std=std)
    m = sensor.measure(scene(walls=[wall_ahead(2.0)]), HERE, 0.0, rng())
    assert isinstance(m, UltrasonicMeasurement)
    assert m.distance == pytest.approx(2.0, abs=5 * std)


def test_without_noise_the_reading_is_exact():
    assert sonar().measure(scene(walls=[wall_ahead(2.0)]), HERE, 0.0, rng()).distance == pytest.approx(2.0)


def test_sensor_facing_open_space_reports_none():
    assert sonar().measure(scene(), HERE, 0.0, rng()).distance is None
    behind = rect(-3.0, 0.0, 0.0, 1.0, 40.0)
    assert sonar().measure(scene(walls=[behind]), HERE, 0.0, rng()).distance is None


@pytest.mark.parametrize("distance, expected", [(0.1, None), (0.25, 0.25), (4.4, 4.4), (4.6, None)])
def test_range_limits(distance, expected):
    m = sonar().measure(scene(walls=[wall_ahead(distance)]), HERE, 0.0, rng())
    assert m.distance == (pytest.approx(expected) if expected else None)


def test_cone_width_decides_what_is_seen():
    """A small post 25 deg off axis at 2 m."""
    post = rect(2 * math.cos(math.radians(25)), 2 * math.sin(math.radians(25)), 0.0, 0.02, 0.02)
    lot = scene(walls=[post])
    assert sonar(cone_deg=60.0).measure(lot, HERE, 0.0, rng()).distance == pytest.approx(2.0, abs=0.02)
    assert sonar(cone_deg=40.0).measure(lot, HERE, 0.0, rng()).distance is None


def test_nearest_point_can_be_off_axis_and_on_an_edge():
    """A wall at x = 2 reads 2 m on the axis; turned 45 deg, its nearest point in the cone is at the cone edge."""
    lot = scene(walls=[wall_ahead(2.0)])
    turned = sonar(mount=MountPose(0.0, 0.0, math.radians(45)))
    assert turned.measure(lot, HERE, 0.0, rng()).distance == pytest.approx(2.0 / math.cos(math.radians(15)))


def test_reading_follows_the_mount_and_the_vehicle():
    """Mounted 1 m forward of the axle, with the car 1 m back from where the wall was measured."""
    sensor = sonar(mount=MountPose(1.0, 0.0, 0.0))
    lot = scene(walls=[wall_ahead(3.0)])
    assert sensor.measure(lot, VehicleState(-1.0, 0.0, 0.0), 0.0, rng()).distance == pytest.approx(3.0)
    # Turned to face away from the wall
    assert sensor.measure(lot, VehicleState(0.0, 0.0, math.pi), 0.0, rng()).distance is None


def test_noise_has_the_configured_size():
    sensor, lot, r = sonar(noise_std=0.02), scene(walls=[wall_ahead(2.0)]), rng(1)
    readings = np.array([sensor.measure(lot, HERE, 0.0, r).distance for _ in range(800)])
    assert readings.std() == pytest.approx(0.02, rel=0.1)
    assert readings.mean() == pytest.approx(2.0, abs=0.005)


def test_dropout_returns_none_at_the_configured_probability():
    lot, r = scene(walls=[wall_ahead(2.0)]), rng(2)
    sensor = sonar(dropout_prob=0.25)
    none_rate = np.mean([sensor.measure(lot, HERE, 0.0, r).distance is None for _ in range(1000)])
    assert none_rate == pytest.approx(0.25, abs=0.05)
    assert sonar(dropout_prob=1.0).measure(lot, HERE, 0.0, r).distance is None


def test_default_layout_has_six_front_six_rear_with_side_corners_and_a_side_pair():
    us = CFG["ultrasonic"]
    assert (us["cone_deg"], us["range"], us["noise_std"], us["rate_hz"]) == (60.0, [0.2, 4.5], 0.02, 15.0)
    sensors = make_ultrasonic_sensors()
    assert len(sensors) == 14
    ring, side = sensors[:12], sensors[12:]
    center = VEH.center_offset()
    front = [s for s in ring if s.mount.x > center]
    rear = [s for s in ring if s.mount.x < center]
    assert len(front) == 6 and len(rear) == 6
    for group in (front, rear):
        corners = [s for s in group if "corner" in s.name]
        assert len(corners) == 2
        assert all(abs(math.sin(s.mount.yaw)) > 0.9 for s in corners)     # look out sideways
        assert corners[0].mount.y == pytest.approx(-corners[1].mount.y)
    assert [s.cone_deg for s in ring] == [60.0] * 12
    assert {s.name for s in side} == {"uss_side_left", "uss_side_right"}
    assert all(abs(s.mount.yaw) == pytest.approx(math.pi / 2) for s in side)
    assert all(s.rate_hz == 15.0 for s in sensors)


def test_all_mounts_sit_on_or_inside_the_body():
    half, front, rear = VEH.width / 2, VEH.wheelbase + VEH.front_overhang, -VEH.rear_overhang
    for s in make_ultrasonic_sensors():
        assert rear <= s.mount.x <= front and abs(s.mount.y) <= half


def test_side_pair_finds_the_space_between_parked_cars():
    """Driving the aisle, the side sensor reads the car bumpers and nothing across a free stall."""
    lot = make_aisle_lot(VEH, PARAMS, rng(SEED))
    cfg = copy.deepcopy(CFG["ultrasonic"])
    cfg["dropout_prob"], cfg["noise_std"] = 0.0, 0.0
    right = next(s for s in make_ultrasonic_sensors(cfg) if s.name == "uss_side_right")
    # Along the row on the right of a car heading +x. Stall 3 is free, its neighbors are occupied
    x_in_stall = lambda stall: lot.stalls[stall].center[0] - right.mount.x  # noqa: E731
    reading = lambda stall: right.measure(lot, VehicleState(x_in_stall(stall), lot.aisle_y, 0.0), 0.0, rng()).distance  # noqa: E731
    assert lot.stalls[3].occupied is False and lot.stalls[4].occupied and lot.stalls[2].occupied
    assert reading(3) is None
    assert reading(4) == pytest.approx(lot.aisle_y - 0.98 - 5.2, abs=0.4)
    assert reading(2) is not None


def test_nearest_in_cone_matches_brute_force():
    r = rng(9)
    for _ in range(40):
        pts = r.uniform(-5, 5, (6, 2))
        a, b = pts[:3], pts[3:]
        origin, yaw, half = r.uniform(-1, 1, 2), r.uniform(-3, 3), math.radians(r.uniform(5, 60))
        best = math.inf
        for p, q in zip(a, b):
            samples = p + np.linspace(0, 1, 20001)[:, None] * (q - p) - origin
            angles = np.arctan2(samples[:, 1], samples[:, 0]) - yaw
            angles = (angles + math.pi) % (2 * math.pi) - math.pi
            inside = np.abs(angles) <= half
            if inside.any():
                best = min(best, np.hypot(*samples[inside].T).min())
        got = nearest_in_cone(origin, yaw, half, a, b)
        if math.isinf(best):
            # A segment can clip the cone between samples only by a hair
            assert math.isinf(got) or got > 0.0
        else:
            assert got == pytest.approx(best, abs=2e-3)


# ---------- ray casting ----------
def test_cast_rays_known_geometry():
    a, b = np.array([[2.0, -1.0], [-1.0, 3.0]]), np.array([[2.0, 1.0], [1.0, 3.0]])
    dirs = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, 0.0]])
    assert cast_rays(np.zeros(2), dirs, a, b, 100.0) == pytest.approx([2.0, 3.0, math.inf, math.inf])
    assert cast_rays(np.zeros(2), dirs, a, b, 2.5) == pytest.approx([2.0, math.inf, math.inf, math.inf])
    assert np.all(np.isinf(cast_rays(np.zeros(2), dirs, np.zeros((0, 2)), np.zeros((0, 2)), 100.0)))


def test_cast_rays_matches_a_loop():
    r = rng(4)
    a, b = r.uniform(-10, 10, (30, 2)), r.uniform(-10, 10, (30, 2))
    angles = r.uniform(-math.pi, math.pi, 50)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    origin = np.array([0.5, -0.5])
    got = cast_rays(origin, dirs, a, b, 1000.0)
    for i, d in enumerate(dirs):
        best = math.inf
        for p, q in zip(a, b):
            e = q - p
            denom = d[0] * e[1] - d[1] * e[0]
            w = p - origin
            ray, seg = (w[0] * e[1] - w[1] * e[0]) / denom, (w[0] * d[1] - w[1] * d[0]) / denom
            if ray >= 0 and 0 <= seg <= 1:
                best = min(best, ray)
        assert got[i] == pytest.approx(best)


def test_sight_blocked():
    a, b = np.array([[1.0, -1.0]]), np.array([[1.0, 1.0]])
    points = np.array([[2.0, 0.0], [0.5, 0.0], [2.0, 5.0]])
    assert list(sight_blocked(np.zeros(2), points, a, b)) == [True, False, False]
    assert list(sight_blocked(np.zeros(2), points, np.zeros((0, 2)), np.zeros((0, 2)))) == [False] * 3


# ---------- lidar ----------
def lidar(**changes):
    args = dict(name="l", mount=MountPose(0.0, 0.0, 0.0), rate_hz=10.0, beams=360, max_range=30.0, noise_std=0.0)
    args.update(changes)
    return Lidar2D(**args)


def test_lidar_is_off_by_default():
    assert CFG["lidar"]["enabled"] is False
    assert not [s for s in make_sensors() if isinstance(s, Lidar2D)]
    cfg = copy.deepcopy(CFG)
    cfg["lidar"]["enabled"] = True
    assert [s for s in make_sensors(cfg) if isinstance(s, Lidar2D)]


def test_lidar_config_sets_beams_range_and_noise():
    sensor = make_lidar({**CFG["lidar"], "beams": 90, "range": 12.0, "noise_std": 0.5})
    assert len(sensor.angles) == 90 and sensor.max_range == 12.0 and sensor.noise_std == 0.5


def test_lidar_measures_a_wall_and_returns_nan_elsewhere():
    m = lidar().measure(scene(walls=[wall_ahead(5.0)]), HERE, 0.0, rng())
    assert isinstance(m, LidarMeasurement) and len(m.ranges) == 360
    at = lambda deg: m.ranges[np.argmin(np.abs(m.angles - math.radians(deg)))]  # noqa: E731
    assert at(0) == pytest.approx(5.0)
    assert at(60) == pytest.approx(10.0)
    assert np.isnan(at(90.5)) and np.isnan(at(-135))


def test_lidar_covers_360_degrees_and_sees_a_ring_of_walls():
    walls = [rect(10.5, 0, 0, 1, 40), rect(-10.5, 0, 0, 1, 40), rect(0, 10.5, 0, 40, 1), rect(0, -10.5, 0, 40, 1)]
    m = lidar(beams=720).measure(scene(walls=walls), HERE, 0.0, rng())
    assert np.isfinite(m.ranges).all()
    assert m.ranges.min() == pytest.approx(10.0)
    assert m.ranges.max() == pytest.approx(10.0 * math.sqrt(2), abs=0.01)
    assert np.ptp(m.angles) == pytest.approx(2 * math.pi - 2 * math.pi / 720)


def test_lidar_respects_range_and_noise():
    lot = scene(walls=[wall_ahead(5.0)])
    assert np.isnan(lidar(max_range=4.0).measure(lot, HERE, 0.0, rng()).ranges).all()
    m = lidar(noise_std=0.02).measure(lot, HERE, 0.0, rng())
    true = lidar().measure(lot, HERE, 0.0, rng())
    hit = np.isfinite(true.ranges)
    assert (m.ranges[hit] - true.ranges[hit]).std() == pytest.approx(0.02, rel=0.3)
    assert np.array_equal(np.isfinite(m.ranges), hit)


def test_lidar_points_are_in_its_frame_and_the_mount_turns_it():
    lot = scene(walls=[wall_ahead(5.0)])
    # Mount turned 90 deg left: the wall ahead of the car is at -90 deg in the lidar's frame
    m = lidar(mount=MountPose(0.0, 0.0, math.pi / 2), beams=4).measure(lot, HERE, 0.0, rng())
    assert m.points() == pytest.approx(np.array([[0.0, -5.0]]), abs=1e-9)


def test_lidar_sees_only_near_faces():
    car = rect(5.0, 0.0, 0.0, 2.0, 2.0)
    m = lidar(beams=720).measure(scene(cars=[car]), HERE, 0.0, rng())
    assert np.nanmin(m.ranges) == pytest.approx(4.0)
    assert np.nanmax(m.points()[:, 0]) <= 4.0 + 1e-9


# ---------- determinism ----------
def run_suite(seed, lidar_on=True, ticks=60):
    cfg = copy.deepcopy(CFG)
    cfg["lidar"]["enabled"] = lidar_on
    suite = make_suite(cfg, seed)
    lot = make_aisle_lot(VEH, PARAMS, rng(SEED))
    out = []
    for i in range(ticks):
        out.extend(suite.step(lot, VehicleState(10.0 + i * 0.05, lot.aisle_y, 0.0), i * 0.02))
    return out


def same(m, n):
    if type(m) is not type(n) or m.sensor != n.sensor or m.t != n.t:
        return False
    if isinstance(m, LineCameraMeasurement):
        return np.array_equal(m.segments, n.segments) and np.array_equal(m.truth_line, n.truth_line)
    if isinstance(m, UltrasonicMeasurement):
        return m.distance == n.distance
    return np.array_equal(m.ranges, n.ranges, equal_nan=True)


def test_same_seed_gives_identical_measurements():
    a, b = run_suite(7), run_suite(7)
    assert len(a) == len(b) and len(a) > 100
    assert {type(m) for m in a} == {LineCameraMeasurement, UltrasonicMeasurement, LidarMeasurement}
    assert all(same(m, n) for m, n in zip(a, b))


def test_a_different_seed_gives_different_noise():
    a, b = run_suite(7), run_suite(8)
    assert not all(same(m, n) for m, n in zip(a, b))


def test_each_sensor_family_can_differ_independently_of_the_others():
    """Turning the lidar on doesn't change the other sensors' random streams."""
    with_lidar = [m for m in run_suite(7) if not isinstance(m, LidarMeasurement)]
    without = run_suite(7, lidar_on=False)
    assert len(with_lidar) == len(without)
    assert all(same(m, n) for m, n in zip(with_lidar, without))


def test_two_runs_through_the_simulator_match():
    lot = make_aisle_lot(VEH, PARAMS, rng(SEED))
    cfg = DriverConfig.from_params(PARAMS["search"])
    a = run_search(lot, cfg, sensors=make_suite(seed=3)).measurements
    b = run_search(make_aisle_lot(VEH, PARAMS, rng(SEED)), cfg, sensors=make_suite(seed=3)).measurements
    assert len(a) == len(b) and all(same(m, n) for m, n in zip(a, b))


# ---------- speed ----------
def sim_speed(ticks=500):
    """Sim seconds per wall second: 50 Hz plant, lane keeping, and every sensor, down a full lot."""
    params = copy.deepcopy(PARAMS)
    lot = make_aisle_lot(VEH, params, rng(SEED), occupancy_prob=0.85)
    cfg = copy.deepcopy(CFG)
    cfg["lidar"]["enabled"] = True
    suite = make_suite(cfg, 0)
    assert len(suite.sensors) == 18     # 3 cameras, 14 ultrasonics, lidar
    dt = params["controller"]["dt"]
    driver = SearchDriver(DriverConfig.from_params(params["search"]), VEH.wheelbase, VEH.max_steer,
                          lot.aisle_y, lot.aisle_end_x)
    cruise = 10 / 3.6
    plant = Vehicle(VehicleParams(**{**VEH.__dict__, "max_speed": cruise}), VehicleState(-8.0, lot.aisle_y, 0.0, cruise), dt=dt)
    count = 0
    start = time.perf_counter()
    for i in range(ticks):
        s = plant.state
        count += len(suite.step(lot, s, i * dt))
        act = driver.act(i * dt, s.x, s.y, s.yaw, s.v, ())
        plant.step(act.steer_cmd, act.speed_cmd)
    wall = time.perf_counter() - start
    assert count > 3000     # the sensors did run
    return ticks * dt / wall


def test_fifty_hz_sim_with_all_sensors_runs_at_least_five_times_real_time():
    # Best of three, so another process hogging the machine doesn't fail the test
    speed = max(sim_speed() for _ in range(3))
    print(f"sim speed {speed:.1f}x real time")
    assert speed >= 5.0


# ---------- simulator ----------
@pytest.fixture(scope="module")
def lot():
    return make_aisle_lot(VEH, PARAMS, rng(SEED))


@pytest.fixture(scope="module")
def sensed(lot):
    cfg = copy.deepcopy(CFG)
    cfg["lidar"]["enabled"] = True
    return run_search(lot, sensors=make_suite(cfg, SEED))


def test_sensors_change_nothing_about_the_run(lot, sensed):
    plain = run_search(lot)
    assert plain.measurements == [] and plain.sensor_suite is None
    assert np.array_equal(plain.log["x"], sensed.log["x"])
    assert np.array_equal(plain.log["steer_cmd"], sensed.log["steer_cmd"])
    assert sensed.success


def test_run_closed_loop_logs_sensors_too():
    sc = make_perpendicular(VEH, PARAMS, rng(SEED))
    plain = run_closed_loop(sc)
    res = run_closed_loop(sc, sensors=make_suite(seed=SEED))
    assert np.array_equal(plain.log["x"], res.log["x"])
    assert len(res.measurements) > 100
    assert res.sensor_suite is not None


def test_measurements_are_in_time_order_and_cover_the_run(sensed):
    times = [m.t for m in sensed.measurements]
    assert times == sorted(times)
    assert times[0] == 0.0 and times[-1] >= sensed.log["t"][-1] - 0.1
    assert {type(m) for m in sensed.measurements} == {LineCameraMeasurement, UltrasonicMeasurement, LidarMeasurement}


def test_cameras_see_painted_lines_in_the_real_lot(sensed):
    seen = {}
    for m in sensed.measurements:
        if isinstance(m, LineCameraMeasurement):
            seen.setdefault(m.sensor, 0)
            seen[m.sensor] += int(np.count_nonzero(m.truth_line != FALSE_POSITIVE))
    assert seen["cam_left"] > 0 and seen["cam_right"] > 0 and seen["cam_rear"] > 0


def test_ultrasonics_see_the_parked_neighbors_in_the_stall(sensed):
    final_t = max(m.t for m in sensed.measurements if isinstance(m, UltrasonicMeasurement))
    last = {m.sensor: m.distance for m in sensed.measurements
            if isinstance(m, UltrasonicMeasurement) and m.t == final_t}
    assert last["uss_side_left"] is not None and last["uss_side_right"] is not None
    assert last["uss_side_left"] < 1.5 and last["uss_side_right"] < 1.5


# ---------- MCAP ----------
def read_mcap(path):
    pytest.importorskip("mcap")
    spec = importlib.util.spec_from_file_location("mcap_tests", ROOT / "tests" / "test_mcap_export.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_mcap(path)


@pytest.fixture(scope="module")
def exported(lot, sensed, tmp_path_factory):
    pytest.importorskip("mcap")
    return read_mcap(export_run(lot, sensed, tmp_path_factory.mktemp("sensors") / "run.mcap"))


def test_sensor_channels_exist_with_the_right_schemas(exported, sensed):
    channels, schemas, _, messages = exported
    topics = sensor_topics(sensed)
    assert set(topics) == {"/sensors/camera_footprints", "/sensors/lines", "/sensors/ultrasonic",
                           "/sensors/ultrasonic_ranges", "/sensors/lidar"}
    assert set(topics) <= set(channels) and all(messages[t] for t in topics)
    assert schemas["/sensors/lidar"].name == "foxglove.PointCloud"
    assert schemas["/sensors/lines"].name == "foxglove.SceneUpdate"
    assert channels["/sensors/ultrasonic_ranges"].message_encoding == "json"


def test_runs_without_sensors_have_no_sensor_channels(tmp_path):
    sc = make_perpendicular(VEH, PARAMS, rng(SEED))
    res = run_closed_loop(sc)
    assert sensor_topics(res) == ()
    assert set(read_mcap(export_run(sc, res, tmp_path / "plain.mcap"))[0]) == set(TOPICS)


def test_only_the_sensors_that_ran_get_channels(tmp_path):
    sc = make_perpendicular(VEH, PARAMS, rng(SEED))
    cfg = copy.deepcopy(CFG)
    cfg["line_camera"]["enabled"] = False
    res = run_closed_loop(sc, sensors=make_suite(cfg, 0))
    channels = read_mcap(export_run(sc, res, tmp_path / "uss.mcap"))[0]
    assert set(channels) == set(TOPICS) | {"/sensors/ultrasonic", "/sensors/ultrasonic_ranges"}


def test_footprints_are_published_once_on_the_car(exported, sensed):
    updates = exported[3]["/sensors/camera_footprints"]
    assert len(updates) == 1 and updates[0][0] == 0
    entities = {e.id: e for e in updates[0][1].entities}
    assert set(entities) == {"footprint_cam_rear", "footprint_cam_left", "footprint_cam_right"}
    entity = entities["footprint_cam_left"]
    assert entity.frame_id == "base_link" and entity.frame_locked
    points = [(p.x, p.y) for p in entity.lines[0].points]
    assert np.allclose(points, sensed.sensor_suite.by_name["cam_left"].footprint)


def test_line_messages_carry_the_segments(exported, sensed):
    messages = exported[3]["/sensors/lines"]
    for ns, update in messages[:50]:
        measured = {m.sensor: m for m in sensed.measurements
                    if isinstance(m, LineCameraMeasurement) and int(round(m.t * 1e9)) == ns}
        assert {e.id for e in update.entities} == {f"lines_{name}" for name in measured}
        for entity in update.entities:
            m = measured[entity.id.replace("lines_", "")]
            assert entity.frame_id == "base_link"
            points = [(p.x, p.y) for line in entity.lines for p in line.points]
            assert len(points) == 2 * len(m.segments)


def test_false_positives_have_their_own_color(lot, tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["line_camera"]["false_positives"]["fp_rate"] = 200.0
    res = run_search(lot, sensors=make_suite(cfg, 0))
    updates = read_mcap(export_run(lot, res, tmp_path / "fp.mcap"))[3]["/sensors/lines"]
    colors = {(round(line.color.r, 2), round(line.color.g, 2)) for _, u in updates for e in u.entities for line in e.lines}
    col = PARAMS["foxglove"]["colors"]
    assert colors == {(round(col["camera_segment"][0], 2), round(col["camera_segment"][1], 2)),
                      (round(col["false_segment"][0], 2), round(col["false_segment"][1], 2))}


def test_ultrasonic_cones_show_the_current_range(exported, sensed):
    updates = exported[3]["/sensors/ultrasonic"]
    rows = exported[3]["/sensors/ultrasonic_ranges"]
    assert len(updates) == len(rows)
    measured = {}
    for m in sensed.measurements:
        if isinstance(m, UltrasonicMeasurement):
            measured.setdefault(int(round(m.t * 1e9)), {})[m.sensor] = m
    both = 0
    for (ns, update), (ns2, row) in zip(updates, rows):
        assert ns == ns2 and len(update.entities) == 14
        for entity in update.entities:
            name = entity.id.replace("cone_", "")
            m = measured[ns][name]
            assert row[name] == m.distance
            assert len(entity.lines) == (2 if m.distance is not None else 1)
            if m.distance is not None:
                mount = sensed.sensor_suite.by_name[name].mount
                p = entity.lines[1].points[0]
                assert math.hypot(p.x - mount.x, p.y - mount.y) == pytest.approx(m.distance)
                both += 1
    assert both > 100


def test_lidar_cloud_decodes_to_the_measured_points(exported, sensed):
    clouds = exported[3]["/sensors/lidar"]
    lidar_measurements = [m for m in sensed.measurements if isinstance(m, LidarMeasurement)]
    assert len(clouds) == len(lidar_measurements)
    ns, cloud = clouds[10]
    m = lidar_measurements[10]
    assert ns == int(round(m.t * 1e9))
    assert cloud.frame_id == "base_link" and cloud.point_stride == 8
    assert [f.name for f in cloud.fields] == ["x", "y"]
    points = np.frombuffer(cloud.data, dtype="<f4").reshape(-1, 2)
    assert np.allclose(points, m.points(), atol=1e-4)
    assert len(points) > 0


def test_search_layout_shows_the_sensors():
    panels = json.loads((ROOT / "config" / "foxglove_search_layout.json").read_text())["configById"]
    topics = panels["3D!parking"]["topics"]
    for topic in ("/sensors/camera_footprints", "/sensors/lines", "/sensors/ultrasonic", "/sensors/lidar"):
        assert topics[topic]["visible"] is True


# ---------- scripts ----------
@pytest.mark.parametrize("script, extra", [("run_search_demo.py", ["--png", "PNG"]), ("run_demo.py", [])])
def test_demo_flags_add_sensors(tmp_path, script, extra):
    pytest.importorskip("mcap")
    mcap = tmp_path / "run.mcap"
    args = [a if a != "PNG" else str(tmp_path / "s.png") for a in extra]
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--seed", str(SEED), "--lidar",
                           "--mcap", str(mcap), *args], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "/sensors/lidar" in read_mcap(mcap)[0]
    assert "/sensors/lines" in read_mcap(mcap)[0]


def test_package_needs_no_ros_or_optional_imports():
    for path in (ROOT / "park_sim" / "sensors").glob("*.py"):
        text = path.read_text()
        assert "rclpy" not in text and "import mcap" not in text and "foxglove" not in text
