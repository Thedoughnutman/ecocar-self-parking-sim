"""pytest checks for the 2D parking sandbox.  Run: python3 -m pytest -q"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from park_sim.geometry import polygon_distance, polygons_overlap, rect  # noqa: E402
from park_sim.hybrid_astar import HybridAStar  # noqa: E402
from park_sim.reeds_shepp import reeds_shepp_paths  # noqa: E402
from park_sim.scenario import load_params, make_perpendicular  # noqa: E402
from park_sim.simulator import run_closed_loop  # noqa: E402
from park_sim.vehicle import Vehicle, VehicleParams, VehicleState, max_steer_from_turning_circle  # noqa: E402


@pytest.fixture(scope="module")
def veh():
    return VehicleParams.from_yaml()


@pytest.fixture(scope="module")
def params():
    return load_params()


# ---------- vehicle ----------
def test_lyriq_lock_angle_matches_turning_circle(veh):
    assert 33.0 < math.degrees(veh.max_steer) < 39.0
    # Rebuild the curb-to-curb diameter from the derived angle
    r_rear = veh.wheelbase / math.tan(veh.max_steer)
    r_outer_edge = math.hypot(veh.wheelbase, r_rear + 1.674 / 2) + 0.275 / 2
    assert abs(2 * r_outer_edge - 12.1) < 1e-6


def test_steady_circle_radius(veh):
    car = Vehicle(veh, VehicleState(0, 0, 0, v=0.5, delta=veh.max_steer * 0.5), dt=0.01)
    xs, ys = [], []
    for _ in range(4000):
        car.step(veh.max_steer * 0.5, 0.5)
        xs.append(car.state.x); ys.append(car.state.y)
    r_expected = veh.wheelbase / math.tan(veh.max_steer * 0.5)
    cx, cy = 0.0, r_expected
    r = np.hypot(np.array(xs) - cx, np.array(ys) - cy)
    assert np.allclose(r, r_expected, atol=0.02)


def test_steering_rate_limit(veh):
    car = Vehicle(veh, VehicleState(0, 0, 0), dt=0.01)
    prev = 0.0
    for _ in range(200):
        car.step(veh.max_steer, 0.0)
        assert car.state.delta - prev <= veh.steer_rate * 0.01 + 1e-9
        prev = car.state.delta


def test_reverse_kinematics(veh):
    car = Vehicle(veh, VehicleState(0, 0, 0, v=-0.5), dt=0.01)
    for _ in range(200):
        car.step(0.0, -0.5)
    assert car.state.x < -0.8 and abs(car.state.y) < 1e-6


# ---------- geometry ----------
def test_polygon_distance():
    a = rect(0, 0, 0, 2, 2)
    b = rect(3, 0, 0, 2, 2)
    assert abs(polygon_distance(a, b) - 1.0) < 1e-9
    assert polygons_overlap(a, rect(1.5, 0, 0.3, 2, 2))
    assert polygon_distance(a, rect(1.5, 0, 0.3, 2, 2)) == 0.0


# ---------- Reeds-Shepp ----------
def test_reeds_shepp_always_connects():
    rng = np.random.default_rng(3)
    for _ in range(300):
        g = (rng.uniform(-12, 12), rng.uniform(-12, 12), rng.uniform(-math.pi, math.pi))
        paths = reeds_shepp_paths((0.0, 0.0, 0.0), g, 4.7, step=0.1)
        assert paths, f"no RS path to {g}"
        p = paths[0]
        assert math.hypot(p.x[-1] - g[0], p.y[-1] - g[1]) < 0.01


def test_reeds_shepp_respects_radius():
    p = reeds_shepp_paths((0, 0, 0), (2.0, 6.0, math.pi / 2), 4.7, step=0.05)[0]
    ds = np.hypot(np.diff(p.x), np.diff(p.y))
    dyaw = np.abs(np.angle(np.exp(1j * np.diff(p.yaw))))
    k = dyaw[ds > 1e-6] / ds[ds > 1e-6]
    assert k.max() <= 1 / 4.7 + 1e-6


# ---------- planner ----------
def test_planner_path_is_collision_free(veh, params):
    sc = make_perpendicular(veh, params, np.random.default_rng(5))
    path = HybridAStar(sc).plan()
    assert path is not None
    margin = params["planner"]["collision_margin"]
    assert sc.collision_free(path.x, path.y, path.yaw, margin - 0.06)
    assert abs(path.x[-1] - sc.goal.x) < 1e-6 and abs(path.y[-1] - sc.goal.y) < 1e-6


# ---------- closed loop against pass criteria ----------
@pytest.mark.parametrize("seed", range(12))
def test_closed_loop_meets_criteria(veh, params, seed):
    sc = make_perpendicular(veh, params, np.random.default_rng(seed))
    r = run_closed_loop(sc)
    assert r.success, r.summary()


def test_head_in(veh, params):
    p = dict(params)
    p["scenario"] = dict(params["scenario"], maneuver="head_in")
    sc = make_perpendicular(veh, p, np.random.default_rng(4))
    r = run_closed_loop(sc)
    assert r.success, r.summary()


def test_monte_carlo_pass_rate(veh, params):
    n, ok = 30, 0
    for seed in range(100, 100 + n):
        sc = make_perpendicular(veh, params, np.random.default_rng(seed))
        ok += run_closed_loop(sc).success
    assert ok / n >= 0.95
