"""Actors, safety monitor, watchdog, brakes, supervisor reactions, faults, metrics, MCAP: units first,
then the named cases end to end in the single-stall and search flows."""
import ast
import copy
import csv
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
from park_sim.actors import Actor, ActorSet, ActorStop, TargetFrame, Trigger, make_actor, make_actors  # noqa: E402
from park_sim.controller import ParkingController  # noqa: E402
from park_sim.geometry import polygon_distance, rect  # noqa: E402
from park_sim.localization import WheelOdometry  # noqa: E402
from park_sim.safety import (RangeReading, SafetyConfig, SafetyMonitor, ScanReading, SensorSpec, SteeringMonitor,  # noqa: E402
                             StreamWatchdog, stop_distance)
from park_sim.scenario import Pose, load_params, make_aisle_lot, make_perpendicular  # noqa: E402
from park_sim.scenario_files import list_scenarios, load_scenario_file, merged, run_scenario_file  # noqa: E402
from park_sim.scripted_driver import EventScript, ScriptedDriver, events_from_config, has_faults  # noqa: E402
from park_sim.sensors import load_sensor_params, make_suite, make_ultrasonic_sensors  # noqa: E402
from park_sim.sensors.scene import scene_arrays, set_actor_polygons  # noqa: E402
from park_sim.simulator import run_closed_loop, run_search, search_trial  # noqa: E402
from park_sim.supervisor import (AbortReason, ParkingSupervisor, PlannerResult, PlannerStatus, SensorHealth,  # noqa: E402
                                 SlotCandidate, State, SupervisorConfig, SupervisorInput)
from park_sim.vehicle import Vehicle, VehicleParams, VehicleState  # noqa: E402

PARAMS = load_params()
VEH = VehicleParams.from_yaml()
DT = PARAMS["controller"]["dt"]
SAFETY = PARAMS["safety"]
SUP = PARAMS["supervisor"]
SEEDS = (1, 6)      # the car is still on its way in when the pedestrian arrives
STOPPED = SUP["standstill_speed"]


def scenario_run(name, seed=1, flow="single_stall", **changes):
    return run_scenario_file(load_scenario_file(name), seed, flow, changes=changes or None)


def monitor(sensors=()):
    return SafetyMonitor(SafetyConfig.from_params(SAFETY, VEH, STOPPED), sensors)


def straight_back(length=6.0):
    """Reversing straight along -x from the origin, heading +x."""
    xs = np.arange(0.0, -length, -0.05)
    return np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])


REAR = SensorSpec("rear", -0.95, 0.0, math.pi, math.radians(30.0), 0.2, 4.5)
POSE = (0.0, 0.0, 0.0)


def see(m, distance, sensor="rear", times=2):
    for _ in range(times):
        m.observe(POSE, [RangeReading(sensor, distance)])


# ---------- actors ----------
def actor_cfg():
    return copy.deepcopy(PARAMS["actors"])


FAR_EGO = rect(1000.0, 1000.0, 0.0, 5.0, 2.0)


def test_pedestrian_is_a_half_meter_circle_walking_its_polyline():
    a = make_actor({"kind": "pedestrian", "path": [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]], "speed_mps": 1.0}, actor_cfg())
    t = 0.0
    while t < 5.0 - 1e-9:
        a.step(t, DT, FAR_EGO, {})
        t += DT
    x, y, _ = a.pose_at(a.progress)
    assert (x, y) == pytest.approx((3.0, 2.0), abs=0.03)
    poly = a.polygon()
    assert np.hypot(*(poly - poly.mean(axis=0)).T) == pytest.approx(0.25)


def test_vehicle_is_a_rectangle_heading_along_its_path():
    a = make_actor({"kind": "vehicle", "path": [[0.0, 5.0], [0.0, 25.0]], "speed_kph": 10.0}, actor_cfg())
    a.step(0.0, DT, FAR_EGO, {})
    poly = a.polygon()
    assert np.ptp(poly[:, 1]) == pytest.approx(PARAMS["actors"]["vehicle"]["length_m"])
    assert np.ptp(poly[:, 0]) == pytest.approx(PARAMS["actors"]["vehicle"]["width_m"])


@pytest.mark.parametrize("spec", [{"kind": "pedestrian", "speed_mps": 2.0}, {"kind": "pedestrian", "speed_mps": 0.2},
                                  {"kind": "vehicle", "speed_kph": 20.0}, {"kind": "vehicle", "speed_kph": 3.0}])
def test_speeds_outside_the_allowed_range_are_refused(spec):
    with pytest.raises(ValueError, match="outside"):
        make_actor({**spec, "path": [[0, 0], [5, 0]]}, actor_cfg())


@pytest.mark.parametrize("bad", [{"kind": "dog", "path": [[0, 0]]},
                                 {"kind": "pedestrian", "path": [[0, 0]], "frame": "moon"},
                                 {"kind": "pedestrian", "path": [[0, 0]], "on_end": "explode"}])
def test_unknown_settings_are_refused(bad):
    with pytest.raises(ValueError, match="unknown"):
        make_actor(bad, actor_cfg())


def test_actor_stops_for_a_while_and_starts_again():
    a = make_actor({"kind": "pedestrian", "path": [[0, 0], [10, 0]], "speed_mps": 1.0,
                    "stops": [{"at_m": 1.0, "duration_s": 2.0}]}, actor_cfg())
    seen = {}
    for i in range(int(5.0 / DT)):
        a.step(i * DT, DT, FAR_EGO, {})
        seen[round(i * DT, 2)] = (a.progress, a.stationary)
    assert seen[2.0] == (pytest.approx(1.0), True)
    assert seen[3.5][0] > 1.0 and not seen[3.5][1]


def test_actor_can_stop_for_good_and_can_leave_at_the_end():
    stays = make_actor({"kind": "pedestrian", "path": [[0, 0], [10, 0]], "speed_mps": 1.0,
                        "stops": [{"at_m": 1.0, "duration_s": None}]}, actor_cfg())
    leaves = make_actor({"kind": "pedestrian", "path": [[0, 0], [1, 0]], "speed_mps": 1.0, "on_end": "leave"}, actor_cfg())
    for i in range(int(20.0 / DT)):
        stays.step(i * DT, DT, FAR_EGO, {})
        leaves.step(i * DT, DT, FAR_EGO, {})
    assert stays.progress == pytest.approx(1.0) and stays.stationary
    assert leaves.left and not leaves.visible


def test_trigger_by_time_and_by_state_entry():
    assert Trigger(time_s=3.0).fire_time({}) == 3.0
    assert Trigger().fire_time({}) == 0.0
    trigger = Trigger(state="MANEUVERING", delay_s=2.0)
    assert trigger.fire_time({}) is None
    assert trigger.fire_time({"MANEUVERING": 5.0}) == 7.0


def test_actor_hidden_until_trigger_then_appears():
    a = make_actor({"kind": "pedestrian", "path": [[0, 0], [5, 0]], "speed_mps": 1.0,
                    "trigger": {"state": "MANEUVERING", "delay_s": 1.0}}, actor_cfg())
    group = ActorSet([a])
    group.step(0.0, DT, FAR_EGO, "SEARCHING")
    assert group.polygons() == []
    group.step(5.0, DT, FAR_EGO, "MANEUVERING")
    group.step(5.5, DT, FAR_EGO, "MANEUVERING")
    assert group.polygons() == []
    group.step(6.1, DT, FAR_EGO, "MANEUVERING")
    assert len(group.polygons()) == 1


def test_target_frame_points_out_of_the_stall_in_both_rows():
    lot = make_aisle_lot(VEH, PARAMS, np.random.default_rng(0))
    for row in (0, 1):
        stall = next(s for s in lot.stalls if s.row == row)
        frame = TargetFrame.from_scenario(lot.with_target(stall))
        opening = np.mean(stall.corners[:2], axis=0)
        back = np.mean(stall.corners[2:], axis=0)
        assert frame.origin == pytest.approx(tuple(opening))
        inside = frame.to_map(np.array([[0.0, -PARAMS["scenario"]["stall_depth"]]]))[0]
        assert inside == pytest.approx(back)


def test_actor_in_the_target_frame_waits_for_a_target():
    a = make_actor({"kind": "pedestrian", "frame": "target", "path": [[0, 0], [1, 0]], "speed_mps": 1.0}, actor_cfg())
    a.step(0.0, DT, FAR_EGO, {})
    assert not a.visible
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    a.resolve(TargetFrame.from_scenario(sc))
    a.step(0.02, DT, FAR_EGO, {})
    assert a.visible


def test_yielding_actor_stops_short_of_the_car_but_a_non_yielding_one_does_not():
    ego = rect(5.0, 0.0, 0.0, 5.0, 2.0)
    walkers = [make_actor({"kind": "pedestrian", "path": [[0, 0], [10, 0]], "speed_mps": 1.0, "yield_gap_m": gap}, actor_cfg())
               for gap in (0.6, 0.0)]
    for i in range(int(8.0 / DT)):
        for a in walkers:
            a.step(i * DT, DT, ego, {})
    assert walkers[0].progress < 2.5 - 0.6 and walkers[1].progress == pytest.approx(8.0, abs=0.05)


def test_yielding_actor_still_passes_a_car_beside_its_path():
    ego = rect(5.0, 3.0, 0.0, 5.0, 2.0)     # 2 m to the side
    a = make_actor({"kind": "vehicle", "path": [[0, 0], [30, 0]], "speed_kph": 10.0}, actor_cfg())
    for i in range(int(8.0 / DT)):
        a.step(i * DT, DT, ego, {})
    assert a.progress > 20.0


@pytest.mark.parametrize("kind, size", [("pedestrian", (0.5,)), ("vehicle", (4.9, 1.95))])
def test_quick_yield_check_never_disagrees_with_the_exact_distance(kind, size):
    """The cull may only skip work: wherever it says clear, the exact distance has to be past yield_side."""
    actor = Actor("a", kind, [[-20.0, 0.0], [20.0, 7.0]], 1.0, size, yield_gap=1.0, yield_side=0.6)
    rng = np.random.default_rng(0)
    culled = 0
    for _ in range(400):
        ego = VEH.footprint(rng.uniform(-22.0, 22.0), rng.uniform(-6.0, 12.0), rng.uniform(-3.2, 3.2))
        s = rng.uniform(0.0, 40.0)
        if actor._surely_clear(ego, s):
            culled += 1
            assert polygon_distance(actor.polygon_at(s), ego) >= 0.6
    assert culled > 50      # and it does get to skip work


def test_only_stationary_actors_are_offered_to_the_planner():
    stands = make_actor({"kind": "pedestrian", "name": "a", "path": [[0, 0]], "hidden_until_trigger": False}, actor_cfg())
    walks = make_actor({"kind": "pedestrian", "name": "b", "path": [[5, 0], [9, 0]], "speed_mps": 1.0}, actor_cfg())
    group = ActorSet([stands, walks])
    group.step(0.0, DT, FAR_EGO, "OFF")
    assert len(group.polygons()) == 2
    assert len(group.stationary_polygons()) == 1


# ---------- actors in the sensors ----------
@pytest.fixture()
def lot():
    sc = make_aisle_lot(VEH, PARAMS, np.random.default_rng(3))
    yield sc
    set_actor_polygons(sc, ())


def test_scene_is_unchanged_without_actors_and_grows_with_them(lot):
    static = scene_arrays(lot)
    set_actor_polygons(lot, [rect(0.0, lot.aisle_y, 0.0, 1.0, 1.0)])
    with_actor = scene_arrays(lot)
    assert len(with_actor.obstacle_a) == len(static.obstacle_a) + 4
    assert len(with_actor.car_a) == len(static.car_a) + 4
    assert with_actor.lines_a is static.lines_a
    set_actor_polygons(lot, ())
    assert scene_arrays(lot) is static


def test_ultrasonic_and_lidar_see_an_actor(lot):
    state = VehicleState(0.0, lot.aisle_y, 0.0)
    cfg = load_sensor_params()
    cfg["lidar"]["enabled"] = True
    cfg["line_camera"]["enabled"] = False
    cfg["ultrasonic"]["noise_std"] = 0.0
    cfg["ultrasonic"]["dropout_prob"] = 0.0
    cfg["lidar"]["noise_std"] = 0.0
    suite = make_suite(cfg, 0)
    ahead = VEH.wheelbase + VEH.front_overhang + 1.0
    rng = np.random.default_rng(0)
    front = suite.by_name["uss_fl_inner"]
    lidar = suite.by_name["lidar"]
    empty = front.measure(lot, state, 0.0, rng).distance
    assert empty is None or empty > 1.5
    set_actor_polygons(lot, [rect(ahead + 0.5, lot.aisle_y, 0.0, 1.0, 1.0)])
    assert front.measure(lot, state, 0.0, rng).distance == pytest.approx(ahead - front.mount.x, abs=0.1)
    after = np.nanmin(lidar.measure(lot, state, 0.0, rng).ranges[abs(lidar.angles) < 0.05])
    assert after == pytest.approx(ahead - lidar.mount.x, abs=0.1)


def test_line_camera_cannot_see_paint_behind_an_actor():
    params = copy.deepcopy(PARAMS)
    sc = make_aisle_lot(VEH, params, np.random.default_rng(3), occupancy_prob=0.0)
    cfg = load_sensor_params()
    for key in ("noise", "faded", "gaps", "false_positives"):
        cfg["line_camera"][key]["enabled"] = False
    camera = next(s for s in make_suite(cfg, 0).sensors if s.name == "cam_right")
    state = VehicleState(10.0, sc.aisle_y - 1.5, 0.0)
    rng = np.random.default_rng(0)
    def visible():
        return float(np.sum(np.hypot(*(camera.measure(sc, state, 0.0, rng).segments[:, 1]
                                       - camera.measure(sc, state, 0.0, rng).segments[:, 0]).T)))
    clear = visible()
    mx, my, _ = camera.mount.in_map(state)
    set_actor_polygons(sc, [rect(mx, my - 1.0, 0.0, 3.0, 0.6)])
    try:
        assert visible() < clear
    finally:
        set_actor_polygons(sc, ())


def test_sensors_with_actors_still_run_five_times_real_time(lot):
    """All 18 sensors plus the lidar, plus five actors moved every tick."""
    cfg = load_sensor_params()
    cfg["lidar"]["enabled"] = True
    suite = make_suite(cfg, 0)
    actors = make_actors([{"kind": "pedestrian", "name": f"p{i}", "path": [[i * 3.0, lot.aisle_y + 2.0], [i * 3.0 + 40.0, lot.aisle_y + 2.0]],
                           "speed_mps": 1.0} for i in range(4)]
                         + [{"kind": "vehicle", "name": "v", "path": [[0.0, lot.aisle_y - 2.0], [60.0, lot.aisle_y - 2.0]]}], actor_cfg())
    plant = Vehicle(VEH, VehicleState(-8.0, lot.aisle_y, 0.0, 1.0), dt=DT)
    ticks = int(4.0 / DT)

    def speed():
        start = time.perf_counter()
        for i in range(ticks):
            s = plant.state
            ego = VEH.footprint(s.x, s.y, s.yaw)
            actors.step(i * DT, DT, ego, "SEARCHING")
            set_actor_polygons(lot, actors.polygons())
            suite.step(lot, s, i * DT)
            plant.step(0.0, 1.0)
        return ticks * DT / (time.perf_counter() - start)
    assert max(speed() for _ in range(3)) >= 5.0


# ---------- brakes ----------
def stopping_distance(brake, speed=1.0):
    plant = Vehicle(VEH, VehicleState(0.0, 0.0, 0.0, speed), dt=DT)
    x0, t = plant.state.x, 0.0
    while abs(plant.state.v) > 1e-3 and t < 10.0:
        plant.step(0.0, 0.0, brake)
        t += DT
    return abs(plant.state.x - x0), t


def test_a_run_that_never_brakes_sees_no_change_from_the_brake_model():
    a, b = Vehicle(VEH, VehicleState(0, 0, 0), dt=DT), Vehicle(VEH, VehicleState(0, 0, 0), dt=DT)
    for i in range(500):
        cmd = 1.0 if i < 200 else (-1.0 if i < 400 else 0.0)
        a.step(0.3, cmd)
        b.step(0.3, cmd, 0.0)
    assert (a.state.x, a.state.y, a.state.yaw, a.state.v, a.state.delta) == (b.state.x, b.state.y, b.state.yaw, b.state.v, b.state.delta)
    assert not a._braking


def test_harder_braking_stops_shorter_and_the_car_does_not_reverse():
    no_brake, _ = stopping_distance(0.0)
    service, _ = stopping_distance(VEH.brake_service_decel)
    emergency, _ = stopping_distance(VEH.brake_emergency_decel)
    assert emergency < service < no_brake
    plant = Vehicle(VEH, VehicleState(0.0, 0.0, 0.0, -1.0), dt=DT)
    for _ in range(200):
        plant.step(0.0, 0.0, VEH.brake_emergency_decel)
        assert plant.state.v <= 1e-12
    assert plant.state.v == 0.0


def test_brakes_have_a_real_stopping_distance_with_delay():
    """Dead time and the jerk limit show: longer than v^2/2a alone, shorter than the monitor's estimate."""
    measured, _ = stopping_distance(VEH.brake_emergency_decel)
    assert measured > 1.0 / (2.0 * VEH.brake_emergency_decel)
    assert measured <= stop_distance(1.0, VEH.brake_emergency_decel, VEH.brake_dead_time, VEH.brake_jerk) + 0.01


def test_monitors_stop_distance_covers_what_the_plant_does():
    for speed in (0.3, 0.6, 1.0):
        measured, _ = stopping_distance(VEH.brake_service_decel, speed)
        m = monitor()
        assert m.horizon(speed)[0] >= measured


def test_brake_config_is_marked_estimated():
    text = (ROOT / "config" / "lyriq_vehicle.yaml").read_text()
    block = text.split("brake:")[1].split("realism:")[0]
    values = [line for line in block.splitlines() if line.strip() and not line.strip().startswith("#")]
    assert len(values) == 4 and all("[est]" in line for line in values)


# ---------- safety monitor ----------
def test_stop_distance_grows_with_speed_and_delay():
    assert stop_distance(1.0, 2.0, 0.2, 15.0) == pytest.approx(0.2 + 2.0 / 30.0 + 0.25)
    assert stop_distance(0.0, 2.0, 0.2, 15.0) == 0.0
    assert stop_distance(-1.0, 2.0, 0.2, 15.0) == stop_distance(1.0, 2.0, 0.2, 15.0)
    assert stop_distance(1.0, 2.0, 0.4, 15.0) > stop_distance(1.0, 2.0, 0.2, 15.0)


def test_sweep_follows_the_path_up_to_the_horizon():
    m = monitor()
    poses, s = m.sweep(POSE, straight_back(), 1.0)
    assert poses[0] == pytest.approx(POSE)
    assert s[-1] <= 1.0 and s[-1] >= 1.0 - SAFETY["sweep_step_m"]
    assert poses[-1][0] == pytest.approx(-s[-1], abs=0.06)


def test_sweep_ends_where_the_path_ends():
    m = monitor()
    poses, _ = m.sweep(POSE, straight_back(0.5), 5.0)
    assert poses[-1][0] == pytest.approx(-0.45, abs=0.06)


def test_sweep_length_depends_on_speed():
    m = monitor()
    assert m.look_ahead(1.0) > m.look_ahead(0.2) > m.look_ahead(0.0) == pytest.approx(SAFETY["standoff_m"])


def test_a_return_in_the_sweep_is_an_obstacle():
    m = monitor([REAR])
    see(m, 0.8)
    out = m.check(POSE, -1.0, straight_back())
    assert out.obstacle_in_path and out.gap_m is not None
    assert out.ttc_s == pytest.approx(out.gap_m / 1.0)


def test_a_return_beyond_the_sweep_is_not():
    m = monitor([REAR])
    see(m, 3.5)
    assert not m.check(POSE, -0.2, straight_back()).obstacle_in_path


def test_a_single_reading_is_not_enough():
    m = monitor([REAR])
    see(m, 0.8, times=1)
    assert not m.check(POSE, -1.0, straight_back()).obstacle_in_path
    see(m, 0.8, times=1)
    assert m.check(POSE, -1.0, straight_back()).obstacle_in_path


def test_a_parked_car_at_the_edge_of_the_cone_is_not_in_the_path():
    """1.7 m is too far for the whole arc to be in the sweep, and the car may well be off to one side."""
    side = SensorSpec("rear", -0.95, 0.27, math.radians(172.0), math.radians(30.0), 0.2, 4.5)
    m = monitor([side])
    see(m, 1.74)
    assert not m.check(POSE, -1.0, straight_back()).obstacle_in_path


def test_close_up_any_part_of_the_arc_counts():
    corner = SensorSpec("corner", 3.6, 0.98, math.radians(80.0), math.radians(30.0), 0.2, 4.5)
    m = monitor([corner])
    see(m, 0.5, "corner")
    turning = np.column_stack([np.zeros(40), 0.05 * np.arange(40), np.full(40, 0.0)])   # sliding left, for simplicity
    assert m.check(POSE, 1.0, turning).obstacle_in_path
    far = monitor([corner])
    see(far, 1.5, "corner")
    assert not far.check(POSE, 1.0, straight_back()).obstacle_in_path


def test_none_clears_an_obstacle_that_is_in_view():
    m = monitor([REAR])
    see(m, 0.8)
    assert m.check(POSE, -0.5, straight_back()).obstacle_in_path
    for _ in range(SAFETY["clear_readings"] - 1):
        see(m, None, times=1)
        assert m.check(POSE, -0.5, straight_back()).obstacle_in_path
    see(m, None, times=1)
    assert not m.check(POSE, -0.5, straight_back()).obstacle_in_path


def test_none_does_not_clear_an_obstacle_tracked_into_the_blind_zone():
    m = monitor([REAR])
    see(m, 0.3)
    see(m, 0.15)    # a real sensor can't report this; it gets the return into the blind zone
    for _ in range(50):
        see(m, None, times=1)
    out = m.check(POSE, 0.0, straight_back())
    assert out.obstacle_in_path
    assert m.returns


def test_blind_zone_obstacle_is_cleared_once_the_sensor_reports_something_far():
    m = monitor([REAR])
    see(m, 0.15)
    see(m, 4.0, times=3)
    assert not m.check(POSE, 0.0, straight_back()).obstacle_in_path


def test_another_sensors_clear_view_releases_a_blind_zone_obstacle():
    other = SensorSpec("other", -3.0, 0.0, 0.0, math.radians(30.0), 0.2, 4.5)     # looking back at the point
    m = monitor([REAR, other])
    see(m, 0.15)
    for _ in range(SAFETY["clear_readings"]):
        m.observe(POSE, [RangeReading("other", None)])
    assert "rear" not in m.returns


def test_blind_zone_track_follows_the_car_moving_away():
    m = monitor([REAR])
    see(m, 0.15)
    moved = (2.0, 0.0, 0.0)     # the point is 2 m further from the sensor now, in view
    for _ in range(SAFETY["clear_readings"]):
        m.observe(moved, [RangeReading("rear", None)])
    assert not m.returns


def test_emergency_brake_when_time_to_collision_is_short():
    m = monitor([REAR])
    see(m, 0.5)
    urgent = m.check(POSE, -1.0, straight_back())
    assert urgent.ttc_s < SAFETY["ttc_emergency_s"] and urgent.emergency_brake
    relaxed = m.check(POSE, -0.1, straight_back())
    assert relaxed.ttc_s >= SAFETY["ttc_emergency_s"] and not relaxed.emergency_brake


def test_a_stopped_car_has_no_time_to_collision():
    m = monitor([REAR])
    see(m, 0.5)
    out = m.check(POSE, 0.0, straight_back())
    assert out.ttc_s is None and not out.emergency_brake and out.obstacle_in_path


def test_lidar_points_count_one_scan_at_a_time():
    m = monitor()
    angles = np.array([math.pi - 0.01])
    m.observe(POSE, [], ScanReading(np.array([[-2.0, 0.0]]), (0.0, 0.0), angles, np.array([2.0])))
    assert m.check(POSE, -0.5, straight_back()).obstacle_in_path
    m.observe(POSE, [], ScanReading(np.zeros((0, 2)), (0.0, 0.0), angles, np.array([math.nan])))
    assert not m.check(POSE, -0.5, straight_back()).obstacle_in_path


def test_lidar_sees_off_to_the_side_as_not_in_the_path():
    m = monitor()
    m.observe(POSE, [], ScanReading(np.array([[-1.0, 1.5]]), (0.0, 0.0), np.array([2.0]), np.array([1.8])))
    assert not m.check(POSE, -0.5, straight_back()).obstacle_in_path


def test_safety_uses_only_the_standard_library_numpy_and_the_supervisor():
    tree = ast.parse((ROOT / "park_sim" / "safety.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    allowed = {"__future__", "math", "dataclasses", "typing", "numpy", ".supervisor"}
    assert imported <= allowed, imported - allowed


def test_remaining_path_runs_from_the_nearest_point_across_cusps():
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    path = run_closed_loop(sc).planned
    ctrl = ParkingController(path, VEH, PARAMS["controller"])
    whole = ctrl.remaining_path(1e6)
    assert len(whole) >= sum(len(s.x) for s in ctrl.segments)
    assert whole[0, :2] == pytest.approx([path.x[0], path.y[0]])
    short = ctrl.remaining_path(1.0)
    assert 0.5 < np.sum(np.hypot(*np.diff(short[:, :2], axis=0).T)) <= 1.01
    ctrl.seg_i = len(ctrl.segments)
    assert len(ctrl.remaining_path(1.0)) == 0


def test_remaining_path_starts_where_the_car_is_after_braking_without_touching_control():
    """update() doesn't run while the car is stopped, so near_i is where the brakes came on. A sweep from
    there goes back along the path and uses up its horizon on ground already driven: a false clear."""
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    ctrl = ParkingController(run_closed_loop(sc).planned, VEH, PARAMS["controller"])
    seg = ctrl.segments[0]
    i = int(np.searchsorted(seg.s, 0.5))
    ahead = ctrl.remaining_path(1.0, (float(seg.x[i]), float(seg.y[i])))
    assert ahead[0, :2] == pytest.approx([seg.x[i], seg.y[i]])
    assert ctrl.near_i == 0
    assert ctrl.remaining_path(1.0)[0, :2] == pytest.approx([seg.x[0], seg.y[0]])


@pytest.mark.parametrize("seed", [13, 14])
def test_a_pedestrian_who_stays_does_not_look_clear_once_the_car_has_stopped(seed):
    """These seeds used to resume for one tick after the false clear, restarting the abort clock."""
    _, res, _ = scenario_run("pedestrian_stays", seed)
    assert res.final_state is State.ABORTED and "obstacle_timeout" in res.reason
    paused = next(tr.t for tr in res.transitions if tr.to_state is State.PAUSED)
    aborted = next(tr.t for tr in res.transitions if tr.to_state is State.ABORTED)
    assert aborted - paused == pytest.approx(SUP["abort_after_s"], abs=0.5)
    assert sum(tr.to_state is State.PAUSED for tr in res.transitions) == 1
    assert res.metrics["actor_collisions"] == 0


# ---------- watchdog ----------
def test_watchdog_ages_a_stream_that_stopped():
    w = StreamWatchdog(["uss"])
    w.heard("uss", 0.5, 1.0)
    assert w.health()[0] == SensorHealth("uss", True, 0.5)


def test_watchdog_does_not_count_an_unchanged_value_while_moving_as_new():
    w = StreamWatchdog(["uss"])
    w.heard("uss", 0.1, 1.5, moving=True)
    w.heard("uss", 0.2, 1.5, moving=True)
    assert w.last["uss"] == 0.1
    w.heard("uss", 0.3, 1.6, moving=True)
    assert w.last["uss"] == 0.3


def test_watchdog_counts_repeats_at_a_stop_and_repeated_nothing_as_new():
    w = StreamWatchdog(["uss"])
    w.heard("uss", 0.1, 1.5)
    w.heard("uss", 0.2, 1.5, moving=False)
    assert w.last["uss"] == 0.2
    w.heard("uss", 0.3, None, moving=True)
    w.heard("uss", 0.4, None, moving=True)
    assert w.last["uss"] == 0.4


def test_watchdog_compares_arrays_too():
    w = StreamWatchdog(["lidar"])
    scan = np.array([1.0, math.nan])
    w.heard("lidar", 0.1, scan, moving=True)
    w.heard("lidar", 0.2, scan.copy(), moving=True)
    assert w.last["lidar"] == 0.1


def test_there_is_only_one_stale_check():
    """The watchdog reports ages; it never compares them with a limit."""
    source = (ROOT / "park_sim" / "safety.py").read_text()
    assert "timeout" not in source and "max_age" not in source
    w = StreamWatchdog(["uss"])
    assert all(h.healthy for h in w.health())


def test_steering_monitor_ignores_a_big_slew_and_catches_a_stuck_wheel():
    wd = SAFETY["watchdog"]
    def make():
        return SteeringMonitor(math.radians(wd["steer_error_deg"]), math.radians(wd["steer_stuck_rate_deg_s"]), wd["steer_fault_time_s"])
    plant = Vehicle(VEH, VehicleState(0, 0, 0), dt=DT)
    m = make()
    for i in range(300):
        command = 0.6 if (i // 100) % 2 == 0 else -0.6
        plant.step(command, 0.0)
        assert m.update(i * DT, DT, command, plant.state.delta)
    stuck = make()
    healthy = True
    for i in range(100):
        healthy = stuck.update(i * DT, DT, 0.6, 0.3)
    assert not healthy


def test_real_stream_ages_reach_the_supervisor_in_a_plain_run():
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    assert run_closed_loop(sc).success


def test_a_stale_input_from_an_event_still_works_on_top():
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    frozen = tuple(SensorHealth(name, True, 3.0) for name in SUP["sensor_timeouts_s"])
    res = run_closed_loop(sc, events=[(3.0, {"sensors": frozen})])
    assert res.final_state is State.ABORTED and "sensor_stale" in res.reason


# ---------- supervisor ----------
class Bench:
    def __init__(self, **changes):
        self.sup = ParkingSupervisor(SupervisorConfig(**{**SUP, **changes}))
        self.t = 0.0
        self.out = None

    def tick(self, **kw):
        kw.setdefault("slots", (SlotCandidate(3),))
        kw.setdefault("sensors", tuple(SensorHealth(n, True, self.t) for n in SUP["sensor_timeouts_s"]))
        self.out = self.sup.step(SupervisorInput(t=self.t, **kw))
        self.t += DT
        return self.out

    def to_maneuver(self):
        self.tick(feature_button=True, slots=())
        self.tick()
        self.tick(selected_slot_id=3)
        self.tick()
        self.tick(confirm_button=True)
        self.tick(planner=PlannerResult(PlannerStatus.FOUND, 3))
        assert self.sup.state is State.MANEUVERING
        return self

    def hold(self, seconds, **kw):
        requests = 0
        end = self.t + seconds
        while self.t < end:
            requests += self.tick(**kw).request_replan
        return requests


def test_replan_asked_after_replan_after_s_and_again_each_time_that_long():
    b = Bench().to_maneuver()
    assert b.hold(SUP["replan_after_s"] - 0.1, obstacle_in_path=True) == 0
    assert b.hold(0.3, obstacle_in_path=True) == 1
    assert b.hold(SUP["replan_after_s"] + 0.1, obstacle_in_path=True) == 1


def test_abort_after_abort_after_s():
    b = Bench().to_maneuver()
    b.hold(SUP["abort_after_s"] - 0.1, obstacle_in_path=True)
    assert b.sup.state is State.PAUSED
    b.hold(0.3, obstacle_in_path=True)
    assert b.sup.state is State.ABORTED
    assert b.sup.abort_reason is AbortReason.OBSTACLE_TIMEOUT
    assert b.sup.transitions[-1].reason.startswith("blocked over")


def test_a_flicker_does_not_restart_the_blocked_clock():
    b = Bench().to_maneuver()
    for _ in range(int(SUP["abort_after_s"] / 2.0)):
        b.hold(1.8, obstacle_in_path=True, confirm_button=True)
        b.hold(0.1, confirm_button=True)
    b.hold(SUP["abort_after_s"] * 0.15, obstacle_in_path=True)
    assert b.sup.state is State.ABORTED


def test_a_real_clearing_restarts_the_blocked_clock():
    b = Bench().to_maneuver()
    b.hold(SUP["abort_after_s"] - 1.0, obstacle_in_path=True)
    b.hold(SUP["resume_delay_s"] + 0.1, confirm_button=True)
    assert b.sup.state is State.MANEUVERING
    b.hold(SUP["abort_after_s"] - 1.0, obstacle_in_path=True)
    assert b.sup.state is State.PAUSED


def test_brake_alone_never_times_out():
    b = Bench().to_maneuver()
    assert b.hold(SUP["abort_after_s"] + 5.0, brake=0.5) == 0
    assert b.sup.state is State.PAUSED


def test_resume_needs_the_delay_and_confirm_as_before():
    b = Bench().to_maneuver()
    b.tick(obstacle_in_path=True)
    b.hold(SUP["resume_delay_s"] - 0.1, confirm_button=True)
    assert b.sup.state is State.PAUSED
    b.hold(0.2, confirm_button=True)
    assert b.sup.state is State.MANEUVERING


# ---------- faults through the event mechanism ----------
def test_event_script_splits_inputs_from_faults():
    script = EventScript([(1.0, {"brake": 0.5, "sensor_dropout": ["ultrasonic"]})])
    assert script.inputs(0.5) == {} and script.faults(0.5) == {}
    assert script.inputs(1.0) == {"brake": 0.5}
    assert script.faults(1.0) == {"sensor_dropout": ["ultrasonic"]}


def test_event_can_be_timed_from_a_state():
    script = EventScript([(("MANEUVERING", 2.0), {"steer_not_followed": True})])
    assert script.faults(100.0) == {}
    script.observe(5.0, State.MANEUVERING)
    assert script.faults(6.9) == {} and script.faults(7.0) == {"steer_not_followed": True}


def test_later_event_replaces_an_earlier_one():
    script = EventScript([(1.0, {"sensor_frozen": ["a"]}), (2.0, {"sensor_frozen": []})])
    assert script.faults(1.5)["sensor_frozen"] == ["a"]
    assert script.faults(2.5)["sensor_frozen"] == []


def test_only_fault_keys_turn_the_safety_layer_on():
    assert not has_faults([(1.0, {"brake": 0.5})])
    assert has_faults([(1.0, {"eps_saturation_deg": 20.0})])


def test_events_load_from_scenario_config():
    events = events_from_config([{"t": 3.0, "set": {"brake": 1.0}},
                                 {"after_state": "MANEUVERING", "delay_s": 2.0, "set": {"sensor_frozen": ["ultrasonic"]}}])
    assert events == [(3.0, {"brake": 1.0}), (("MANEUVERING", 2.0), {"sensor_frozen": ["ultrasonic"]})]


def test_scripted_driver_does_not_pass_faults_to_the_supervisor():
    driver = ScriptedDriver(3, ["localization"], [(0.0, {"eps_saturation_deg": 20.0, "brake": 0.5})])
    from park_sim.supervisor import ControllerStatus
    u = driver.input(1.0, 0.0, ControllerStatus.IDLE, PlannerResult(), 0)
    assert u.brake == 0.5


def test_eps_saturation_limits_the_road_wheels():
    plant = Vehicle(VEH, VehicleState(0, 0, 0, 0.0), dt=DT)
    plant.steer_limit = math.radians(20.0)
    for _ in range(200):
        plant.step(math.radians(35.0), 0.0)
    assert plant.state.delta == pytest.approx(math.radians(20.0), abs=1e-3)


def test_stuck_steering_ignores_commands():
    plant = Vehicle(VEH, VehicleState(0, 0, 0, 0.0, 0.1), dt=DT)
    plant.steer_stuck = True
    for _ in range(100):
        plant.step(0.5, 0.0)
    assert plant.state.delta == 0.1


def test_delayed_pose_uses_the_estimators_latency():
    odom = WheelOdometry(Pose(0.0, 0.0, 0.0), {**PARAMS["localization"]["odometry"], "standstill_calibration": False},
                         np.random.default_rng(0), VEH.wheelbase, VEH.track)
    from park_sim.localization import Measurements
    for i in range(100):
        odom.update(Measurements(0.0, 0.0, 0.0, 1.0, 0.0, (1.0,) * 4), DT)
    assert 2.0 - odom.sample_time == pytest.approx(0.0, abs=DT + 1e-9)
    odom.set_latency(0.5)
    for i in range(100):
        odom.update(Measurements(0.0, 0.0, 0.0, 1.0, 0.0, (1.0,) * 4), DT)
    assert 4.0 - odom.sample_time == pytest.approx(0.5, abs=DT + 1e-9)


def test_truth_has_no_latency_to_change():
    from park_sim.localization import TruthLocalizer
    with pytest.raises(ValueError, match="no latency"):
        TruthLocalizer(Pose(0.0, 0.0, 0.0)).set_latency(0.1)


def test_a_named_sensor_dropout_hits_just_that_sensor():
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    res = run_closed_loop(sc, events=[(("MANEUVERING", 1.0), {"sensor_dropout": ["uss_rl_inner"]})])
    assert "sensor_stale" in res.reason and "uss_rl_inner" in res.transitions[-1].reason
    assert res.transitions[-1].reason.count("uss_") == 1


def test_a_frozen_sensor_repeats_its_last_value_with_new_timestamps():
    from park_sim.safety_layer import _StreamFaults
    from park_sim.sensors.ultrasonic import UltrasonicMeasurement
    faults = _StreamFaults()
    faults.apply([UltrasonicMeasurement("a", 0.1, 1.0), UltrasonicMeasurement("b", 0.1, 2.0)], (), ())
    out = faults.apply([UltrasonicMeasurement("a", 0.2, 1.5), UltrasonicMeasurement("b", 0.2, 2.5)], (), ("a",))
    assert [(m.sensor, m.t, m.distance) for m in out] == [("a", 0.2, 1.0), ("b", 0.2, 2.5)]
    assert faults.apply([UltrasonicMeasurement("c", 0.3, 1.0)], ("ultrasonic",), ()) == []


@pytest.mark.parametrize("stream", ["localization", "wheel_speed", "slot_perception"])
def test_dropped_system_streams_go_stale(stream):
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    res = run_closed_loop(sc, events=[(("MANEUVERING", 1.0), {"sensor_dropout": [stream]})])
    assert res.final_state is State.ABORTED and stream in res.transitions[-1].reason


def test_logged_measurements_are_what_was_published():
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    cfg = load_sensor_params()
    cfg["line_camera"]["enabled"] = False
    res = run_closed_loop(sc, events=[(("MANEUVERING", 0.5), {"sensor_dropout": ["ultrasonic"]})], sensors=make_suite(cfg, 0))
    entered = next(tr.t for tr in res.transitions if tr.to_state is State.MANEUVERING)
    assert res.measurements and max(m.t for m in res.measurements) <= entered + 0.5 + 0.1


# ---------- named scenarios ----------
def test_every_named_case_has_a_scenario_file():
    names = list_scenarios()
    for needed in ("pedestrian_walks_into_slot", "pedestrian_stays", "vehicle_passes_in_aisle", "ultrasonic_frozen",
                   "ultrasonic_dropout", "steering_saturates_20deg", "steering_not_followed", "delayed_pose", "no_actors_no_faults"):
        assert needed in names


@pytest.mark.parametrize("name", list_scenarios())
def test_scenario_files_are_well_formed(name):
    s = load_scenario_file(name)
    assert s.name == name and s.description and s.flows and s.expect
    params = merged(PARAMS, s.overrides)
    make_actors(s.actors, params["actors"])
    events_from_config(s.events)


def test_unknown_scenario_is_refused():
    with pytest.raises(ValueError, match="no scenario"):
        load_scenario_file("nope")


def test_merge_keeps_what_it_does_not_change():
    out = merged({"a": {"b": 1, "c": 2}, "d": 3}, {"a": {"b": 9}})
    assert out == {"a": {"b": 9, "c": 2}, "d": 3}


def test_wrong_flow_is_refused():
    with pytest.raises(ValueError, match="no search flow"):
        run_scenario_file(load_scenario_file("delayed_pose"), 1, "search")


# ---------- the prompt's cases, end to end ----------
def stopped_with_distance(res):
    """Time paused, and that the monitor really did stop the car for something."""
    return res.metrics["time_paused_s"] > 0.0 and res.metrics["obstacle_stops"] >= 1.0


@pytest.mark.parametrize("seed", SEEDS)
def test_pedestrian_walks_into_the_slot_car_stops_resumes_and_parks(seed):
    _, res, _ = scenario_run("pedestrian_walks_into_slot", seed)
    assert stopped_with_distance(res)
    assert res.metrics["min_actor_distance_m"] >= 0.5
    assert res.metrics["actor_collisions"] == 0
    assert res.safety.outcome == "completed"
    assert res.success, res.summary()
    states = [tr.to_state for tr in res.transitions]
    assert State.PAUSED in states
    assert states[states.index(State.PAUSED) + 1] is State.MANEUVERING


def test_pedestrian_walks_into_the_slot_search_flow():
    _, res, _ = scenario_run("pedestrian_walks_into_slot", 1, "search")
    assert stopped_with_distance(res)
    assert res.metrics["min_actor_distance_m"] >= 0.5 and res.metrics["actor_collisions"] == 0
    assert res.success, res.summary()


def test_pedestrian_walks_into_the_slot_sensor_detector_and_odometry():
    params = copy.deepcopy(PARAMS)
    params["slot_detection"]["detector"] = "fused"
    f = load_scenario_file("pedestrian_walks_into_slot")
    _, res, _ = run_scenario_file(f, 1, "search", params=params)
    assert res.estimator == "odometry"
    assert res.metrics["actor_collisions"] == 0 and res.metrics["min_actor_distance_m"] >= 0.5
    assert res.safety.outcome in ("completed", "aborted")


@pytest.mark.parametrize("flow", ["single_stall", "search"])
def test_pedestrian_stays_car_aborts_after_the_limit_without_contact(flow):
    _, res, _ = scenario_run("pedestrian_stays", 1, flow)
    assert res.final_state is State.ABORTED and "obstacle_timeout" in res.reason
    assert res.metrics["actor_collisions"] == 0 and res.metrics["min_actor_distance_m"] > 0.5
    paused = next(tr.t for tr in res.transitions if tr.to_state is State.PAUSED)
    aborted = next(tr.t for tr in res.transitions if tr.to_state is State.ABORTED)
    assert aborted - paused == pytest.approx(SUP["abort_after_s"], abs=2.5)
    assert abs(res.log["v"][-1]) <= STOPPED
    assert res.safety.outcome == "aborted"


@pytest.mark.parametrize("flow, seed", [("single_stall", 1), ("single_stall", 3), ("search", 1)])
def test_vehicle_passes_in_the_aisle_no_collision(flow, seed):
    _, res, _ = scenario_run("vehicle_passes_in_aisle", seed, flow)
    assert res.metrics["actor_collisions"] == 0 and res.metrics["min_actor_distance_m"] > 0.0
    assert res.safety.outcome in ("completed", "aborted")


@pytest.mark.parametrize("name", ["ultrasonic_frozen", "ultrasonic_dropout"])
@pytest.mark.parametrize("flow", ["single_stall", "search"])
def test_ultrasonic_stream_fails_car_stops_within_its_max_age(name, flow):
    _, res, _ = scenario_run(name, 1, flow)
    assert res.final_state is State.ABORTED and "sensor_stale" in res.reason
    maneuvering = next(tr.t for tr in res.transitions if tr.to_state is State.MANEUVERING)
    aborted = next(tr.t for tr in res.transitions if tr.to_state is State.ABORTED)
    fault_at = maneuvering + 3.0
    assert fault_at < aborted <= fault_at + SAFETY["watchdog"]["ultrasonic_max_age_s"] + 0.1
    assert abs(res.log["v"][-1]) <= STOPPED


def test_stuck_ultrasonics_are_not_caught_when_only_they_read_nothing():
    """A documented limit: all-None repeats look healthy. Nothing to catch here at t=0 without returns."""
    w = StreamWatchdog(["uss"])
    for t in (0.1, 0.2, 0.3):
        w.heard("uss", t, None, moving=True)
    assert w.last["uss"] == 0.3


def test_delayed_pose_past_its_limit_stops_the_car():
    _, res, _ = scenario_run("delayed_pose", 1)
    assert res.estimator == "gnss_ins"
    assert res.final_state is State.ABORTED and "localization" in res.transitions[-1].reason
    assert abs(res.log["v"][-1]) <= STOPPED


def test_small_pose_delay_does_not_stop_the_car():
    sc = make_perpendicular(VEH, merged(PARAMS, {"localization": {"estimator": "gnss_ins"}}), np.random.default_rng(1))
    res = run_closed_loop(sc, events=[(("MANEUVERING", 3.0), {"pose_delay_s": 0.05})],
                          localization_rng=np.random.default_rng(1))
    assert res.safety.outcome == "completed" or "localization" not in res.reason


@pytest.mark.parametrize("flow, seed", [("single_stall", 1), ("single_stall", 2), ("single_stall", 6), ("search", 1)])
def test_steering_saturates_completes_or_aborts_cleanly_no_contact(flow, seed):
    sc, res, _ = scenario_run("steering_saturates_20deg", seed, flow)
    assert res.safety.outcome in ("completed", "fault", "aborted")
    if res.safety.outcome == "completed":
        assert res.success
    else:
        assert res.final_state in (State.FAULT, State.ABORTED)
        assert abs(res.log["v"][-1]) <= STOPPED
    assert res.metrics["min_clearance"] > 0.0


def test_saturation_only_after_the_car_no_longer_needs_the_angle_changes_nothing():
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    res = run_closed_loop(sc, events=[(("MANEUVERING", 12.0), {"eps_saturation_deg": 20.0})])
    assert res.safety.outcome == "completed" and res.success


@pytest.mark.parametrize("flow, seed", [("single_stall", 2), ("search", 1)])
def test_steering_not_followed_faults_and_stops(flow, seed):
    _, res, _ = scenario_run("steering_not_followed", seed, flow)
    assert res.final_state is State.FAULT
    assert "steering" in res.transitions[-1].reason
    assert abs(res.log["v"][-1]) <= STOPPED
    assert res.metrics["min_clearance"] > 0.0


# ---------- replans for an obstacle ----------
def blocker_on_path(sc, how_far=3.0):
    """A pedestrian who appears and stays on the plain run's path, a few meters ahead of the car at that moment."""
    plain = run_closed_loop(sc)
    maneuver = ~np.isnan(plain.log["t_maneuver"])
    x, y = plain.log["x"][maneuver], plain.log["y"][maneuver]
    start_t = float(plain.log["t"][maneuver][0])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    i = int(np.searchsorted(s, s[int(2.0 / DT)] + how_far))
    return make_actors([{"kind": "pedestrian", "name": "blocker", "path": [[float(x[i]), float(y[i])]],
                         "yield_gap_m": 0.0, "trigger": {"time_s": start_t + 2.0}}], PARAMS["actors"]), plain


def test_blocked_replan_is_tried_with_the_stationary_actor_on_the_planners_map():
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    actors, _ = blocker_on_path(sc)
    res = run_closed_loop(sc, actors=actors)
    assert res.metrics["obstacle_stops"] >= 1
    assert res.safety.replans, "the car should have asked for another way"
    assert res.metrics["actor_collisions"] == 0
    assert {r.kind for r in res.safety.replans} <= {"replan", "failed", "refused", "capped"}


def test_accepted_blocked_replan_swaps_the_path_and_the_export_shows_both(monkeypatch):
    """The real planner rarely finds a way past a pedestrian in time (docs/safety.md), so it is stubbed
    here: this checks what the simulator does with a path once it has one."""
    import park_sim.simulator as simulator
    from park_sim.mcap_export import _plan_messages
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    actors, _ = blocker_on_path(sc)
    real = simulator._plan_single_stall
    replans = []

    def only_replans_get_the_plain_lot(scene, true_pose, loc, planner_cfg, limit_s=None):
        if limit_s is None:
            return real(scene, true_pose, loc, planner_cfg)
        replans.append(limit_s)
        return real(sc, true_pose, loc, planner_cfg, 10.0)
    monkeypatch.setattr(simulator, "_plan_single_stall", only_replans_get_the_plain_lot)
    res = run_closed_loop(sc, actors=actors)
    assert replans and replans[0] == SAFETY["replan_time_limit_s"]
    kinds = [r.kind for r in res.safety.replans]
    assert "replan" in kinds
    assert kinds.count("replan") <= SAFETY["max_blocked_replans"]
    assert res.metrics["blocked_replans"] == kinds.count("replan")
    assert len(res.safety.plans) == kinds.count("replan") + 1
    assert res.planned is res.safety.plans[-1][1]
    assert [ns for ns, _ in _plan_messages(res, 0)][0] == 0
    assert res.metrics["gear_changes"] <= SUP["max_gear_changes"]
    assert res.metrics["actor_collisions"] == 0


def test_planner_routes_around_a_stationary_actor_it_can_see():
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    from park_sim.hybrid_astar import HybridAStar
    from park_sim.safety_layer import SafetyLayer
    from park_sim.localization import make_localizer
    start = Pose(13.71, 8.59, math.radians(5.0))
    plain = HybridAStar(sc, None).plan(start, time_limit_s=10.0)
    actors = make_actors([{"kind": "pedestrian", "name": "p", "path": [[14.0, 11.0]], "hidden_until_trigger": False}], PARAMS["actors"])
    veh = Vehicle(VEH, VehicleState(start.x, start.y, start.yaw), dt=DT)
    layer = SafetyLayer(sc, SupervisorConfig(**SUP), veh, make_localizer(PARAMS, sc.start, VEH, False), EventScript(), actors, None, True)
    layer.begin_tick(0.0)
    set_actor_polygons(sc, ())
    around = HybridAStar(layer.planning_scene(sc), None).plan(start, time_limit_s=10.0)
    assert plain.n_direction_changes == 0
    assert around is not None and around.n_direction_changes >= 1      # it has to back away first
    ped = actors.polygons()[0]
    from park_sim.geometry import polygon_distance
    assert min(polygon_distance(VEH.footprint(x, y, yaw), ped)
               for x, y, yaw in zip(around.x, around.y, around.yaw)) >= PARAMS["planner"]["collision_margin"] - 0.06


def test_planning_scene_adds_only_stationary_actors():
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    from park_sim.safety_layer import SafetyLayer
    from park_sim.localization import make_localizer
    actors = make_actors([{"kind": "pedestrian", "name": "p", "path": [[9.0, 5.0]], "hidden_until_trigger": False},
                          {"kind": "pedestrian", "name": "q", "path": [[1.0, 8.0], [5.0, 8.0]], "speed_mps": 1.0}], PARAMS["actors"])
    veh = Vehicle(VEH, VehicleState(sc.start.x, sc.start.y, sc.start.yaw), dt=DT)
    loc = make_localizer(PARAMS, sc.start, VEH, False)
    layer = SafetyLayer(sc, SupervisorConfig(**SUP), veh, loc, EventScript(), actors, None, True)
    layer.begin_tick(0.0)
    scene = layer.planning_scene(sc)
    assert len(scene.obstacles) == len(sc.obstacles) + 1
    assert scene.clearance_fast(np.array([9.0 - VEH.center_offset()]), np.array([5.0]), np.array([0.0]))[0] <= 0.0
    assert len(sc.obstacles) == len(scene.obstacles) - 1 and sc._dist is not scene._dist
    set_actor_polygons(sc, ())


class _FakePath:
    n_direction_changes = 2
    direction = [-1]


class _Layerish:
    """A safety layer with only what judging a replan needs."""

    def __init__(self):
        sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
        from park_sim.safety_layer import SafetyLayer
        from park_sim.localization import make_localizer
        veh = Vehicle(VEH, VehicleState(sc.start.x, sc.start.y, sc.start.yaw), dt=DT)
        self.layer = SafetyLayer(sc, SupervisorConfig(**SUP), veh, make_localizer(PARAMS, sc.start, VEH, False),
                                 EventScript(), None, None, True)


def test_blocked_replan_that_would_break_the_gear_limit_is_refused():
    layer = _Layerish().layer

    class Ctrl:
        gear = 1

        @staticmethod
        def gear_changes_with(path):
            return SUP["max_gear_changes"] + 1
    assert layer.judge_replan(1.0, _FakePath(), Ctrl()) is None
    assert layer.info.replans[-1].kind == "refused"
    assert layer.metrics()["blocked_replans"] == 0


def test_blocked_replan_with_no_path_keeps_the_old_plan():
    layer = _Layerish().layer
    assert layer.judge_replan(1.0, None, object()) is None
    assert layer.info.replans[-1].kind == "failed"


def test_blocked_replans_have_their_own_cap_and_do_not_use_refinements():
    layer = _Layerish().layer

    class Ctrl:
        @staticmethod
        def gear_changes_with(path):
            return 0

        @staticmethod
        def continued_on(path):
            return "fresh"
    for i in range(SAFETY["max_blocked_replans"]):
        assert layer.may_replan(float(i))
        assert layer.judge_replan(float(i), _FakePath(), Ctrl(), _FakePath()) == "fresh"
    assert not layer.may_replan(9.0)
    assert layer.info.replans[-1].kind == "capped"
    assert SAFETY["max_blocked_replans"] == 2 and PARAMS["slot_detection"]["refinement"]["max_replans"] == 2


def test_a_failed_replan_does_not_use_up_the_cap():
    layer = _Layerish().layer
    for i in range(5):
        assert layer.may_replan(float(i))
        layer.judge_replan(float(i), None, object())
    assert layer.metrics()["blocked_replans"] == 0


def test_taking_over_a_path_keeps_gear_and_gear_count():
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    path = run_closed_loop(sc).planned
    ctrl = ParkingController(path, VEH, PARAMS["controller"], initial_gear=-1)
    ctrl.gear_changes, ctrl.last_steer = 2, 0.3
    fresh = ctrl.continued_on(path)
    assert (fresh.gear, fresh.gear_changes, fresh.last_steer) == (-1, 2, 0.3)
    first_shift = 1 if path.direction[0] != -1 else 0
    assert ctrl.gear_changes_with(path) == 2 + first_shift + path.n_direction_changes


# ---------- metrics ----------
def test_collisions_are_counted_once_per_contact():
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    start = sc.vehicle.footprint(sc.start.x, sc.start.y, sc.start.yaw).mean(axis=0)
    actors = make_actors([{"kind": "pedestrian", "name": "hit", "path": [[float(start[0]), float(start[1])]],
                           "hidden_until_trigger": False}], PARAMS["actors"])
    res = run_closed_loop(sc, actors=actors)
    assert res.metrics["actor_collisions"] == 1
    assert res.metrics["min_actor_distance_m"] == 0.0
    if res.checks:
        assert res.checks["actor_collision"] is False


def test_safety_metrics_appear_only_with_the_monitor_on():
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    plain = run_closed_loop(sc)
    assert plain.safety is None and "actor_collisions" not in plain.metrics and "actor_collision" not in plain.checks
    _, watched, _ = scenario_run("pedestrian_walks_into_slot", 1)
    for key in ("actor_collisions", "min_actor_distance_m", "stop_distance_after_detection_m", "time_paused_s"):
        assert key in watched.metrics
    assert watched.safety.outcome == "completed" and watched.checks["actor_collision"] is True


def test_stop_distance_after_detection_is_the_distance_really_driven():
    _, res, _ = scenario_run("pedestrian_stays", 1)
    d = res.metrics["stop_distance_after_detection_m"]
    assert 0.0 < d <= stop_distance(1.0, VEH.brake_service_decel, SAFETY["sensor_latency_s"] + VEH.brake_dead_time, VEH.brake_jerk)
    assert res.safety.samples[0].horizon_m > 0.0


def test_safety_off_setting_disables_the_layer_even_with_actors():
    params = merged(PARAMS, {"safety": {"enabled": False}})
    sc = make_perpendicular(VEH, params, np.random.default_rng(1))
    actors = make_actors(load_scenario_file("pedestrian_stays").actors, params["actors"])
    assert run_closed_loop(sc, actors=actors).safety is None


def test_monitor_forced_on_with_nothing_there_leaves_the_car_parking():
    params = merged(PARAMS, {"safety": {"enabled": True}})
    sc = make_perpendicular(VEH, params, np.random.default_rng(1))
    res = run_closed_loop(sc)
    assert res.safety is not None and res.success and res.metrics["time_paused_s"] == 0.0


def test_plain_scenario_is_the_plain_run():
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    plain = run_closed_loop(sc, np.random.default_rng(10_001), localization_rng=np.random.default_rng(20_001))
    _, named, _ = run_scenario_file(load_scenario_file("no_actors_no_faults"), 1)
    assert named.safety is None
    assert named.metrics == plain.metrics


def test_actor_polygons_do_not_outlive_the_run():
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    actors = make_actors(load_scenario_file("pedestrian_stays").actors, PARAMS["actors"])
    run_closed_loop(sc, actors=actors)
    assert "with_actors" not in sc._sensor_cache


# ---------- no actors, no faults: unchanged ----------
def csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_plain_baseline_csv_matches_the_reference_for_ten_seeds(tmp_path):
    out = tmp_path / "base.csv"
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "monte_carlo.py"), "-n", "10", "--csv", str(out)],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    fresh = out.read_text().splitlines()
    reference = (ROOT / "tests" / "data" / "baseline_monte_carlo_200.csv").read_text().splitlines()
    assert fresh == reference[:len(fresh)]


def test_monte_carlo_csv_has_the_safety_columns_for_a_scenario(tmp_path):
    out = tmp_path / "ped.csv"
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "monte_carlo.py"), "-n", "2", "--scenario",
                           "pedestrian_walks_into_slot", "--csv", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    row = csv_rows(out)[0]
    for key in ("actor_collisions", "min_actor_distance_m", "stop_distance_after_detection_m", "time_paused_s",
                "outcome", "outcome_reason"):
        assert key in row
    assert "safety:" in done.stdout


@pytest.mark.parametrize("script, extra", [("run_demo.py", ["--mcap"]), ("run_search_demo.py", ["--png", "PNG", "--mcap"])])
def test_demo_scripts_run_a_named_scenario(tmp_path, script, extra):
    pytest.importorskip("mcap")
    mcap = tmp_path / "s.mcap"
    args = [str(tmp_path / "s.png") if a == "PNG" else a for a in extra] + [str(mcap)]
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--seed", "1", "--scenario",
                           "ultrasonic_frozen", *args], capture_output=True, text=True)
    assert done.returncode == 2, done.stderr    # exits 2: it didn't park, which is the point
    assert "outcome: aborted" in done.stdout
    assert "/safety" in read_channels(mcap)


# ---------- MCAP ----------
def read_channels(path):
    from mcap.reader import make_reader
    with open(path, "rb") as f:
        reader = make_reader(f)
        out = {}
        for _, channel, message in reader.iter_messages():
            out.setdefault(channel.topic, []).append(message)
        return out


def test_mcap_has_actors_safety_and_sweep(tmp_path):
    pytest.importorskip("mcap")
    from park_sim.mcap_export import export_run
    from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
    sc, res, _ = scenario_run("pedestrian_walks_into_slot", 1)
    path = export_run(sc, res, tmp_path / "ped.mcap")
    channels = read_channels(path)
    assert {"/actors", "/safety", "/safety_sweep"} <= set(channels)
    rows = [json.loads(m.data) for m in channels["/safety"]]
    assert any(r["obstacle_in_path"] for r in rows)
    assert rows[0]["sweep"] and len(rows[0]["sweep"][0]) == 3
    assert any(r["ttc_s"] is not None for r in rows)
    actor_updates = [SceneUpdate.FromString(m.data) for m in channels["/actors"]]
    assert any(u.entities for u in actor_updates)
    assert any(d.id == "actor_walker" for u in actor_updates for d in u.deletions), "the pedestrian leaves"
    sweep = SceneUpdate.FromString(channels["/safety_sweep"][0].data)
    assert sweep.entities[0].id == "sweep" and sweep.entities[0].lines
    metrics = json.loads(channels["/metrics"][0].data)
    assert metrics["outcome"] == "completed" and metrics["metrics"]["actor_collisions"] == 0


def test_mcap_of_a_fault_only_run_has_safety_but_no_actors(tmp_path):
    pytest.importorskip("mcap")
    from park_sim.mcap_export import export_run
    sc, res, _ = scenario_run("ultrasonic_frozen", 1)
    channels = read_channels(export_run(sc, res, tmp_path / "f.mcap"))
    assert "/safety" in channels and "/actors" not in channels


def test_mcap_of_a_plain_run_has_no_new_channels(tmp_path):
    pytest.importorskip("mcap")
    from park_sim.mcap_export import TOPICS, export_run
    sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(1))
    channels = read_channels(export_run(sc, run_closed_loop(sc), tmp_path / "p.mcap"))
    assert set(channels) == set(TOPICS)


def test_mcap_after_a_blocked_replan_shows_every_plan(tmp_path):
    pytest.importorskip("mcap")
    from park_sim.mcap_export import _plan_messages
    from types import SimpleNamespace
    res = SimpleNamespace(search=None, planned="latest", safety=SimpleNamespace(plans=[(0.0, "first"), (9.0, "latest")]))
    assert _plan_messages(res, 0) == [(0, "first"), (9_000_000_000, "latest")]


@pytest.mark.parametrize("layout", ["foxglove_layout.json", "foxglove_search_layout.json"])
def test_both_layouts_show_actors_and_safety(layout):
    panels = json.loads((ROOT / "config" / layout).read_text())["configById"]
    topics = panels["3D!parking"]["topics"]
    assert topics["/actors"]["visible"] and topics["/safety_sweep"]["visible"]
    assert panels["RawMessages!safety"]["topicPath"] == "/safety"


def test_all_new_config_values_are_in_config():
    assert "safety" in PARAMS and "actors" in PARAMS
    source = (ROOT / "park_sim" / "actors.py").read_text()
    assert "0.5" not in source.split("def make_actor")[1]
