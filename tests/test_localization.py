"""Localization: estimators, frames, plant realism, metrics, export and the sweep script."""
import copy
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from park_sim.localization import (ESTIMATION_METRICS, GnssInsLocalizer, Measurements, TruthLocalizer,  # noqa: E402
                                   WheelOdometry, WhiteNoiseLocalizer, make_localizer, wheel_speeds)
from park_sim.mcap_export import export_run  # noqa: E402
from park_sim.scenario import Pose, load_params, make_aisle_lot, make_perpendicular  # noqa: E402
from park_sim.simulator import run_closed_loop, run_search, search_trial  # noqa: E402
from park_sim.vehicle import Vehicle, VehicleParams, VehicleState  # noqa: E402

VEH = VehicleParams.from_yaml()
PARAMS = load_params()
DT = PARAMS["controller"]["dt"]
START = Pose(1.0, 2.0, 0.3)
REFERENCE_CSV = ROOT / "tests" / "data" / "baseline_monte_carlo_200.csv"
ODOMETRY_OFF = dict(wheel_scale_error=0.0, wheel_noise_std=0.0, wheel_quantization=0.0, yaw_rate_bias_deg_s=0.0,
                    yaw_rate_noise_std_deg_s=0.0, yaw_rate_quantization_deg_s=0.0)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def params_with(estimator=None, latency=None, **odometry):
    params = copy.deepcopy(PARAMS)
    if estimator is not None:
        params["localization"]["estimator"] = estimator
    if latency is not None:
        params["localization"]["latency_s"] = latency
    params["localization"]["odometry"].update(odometry)
    return params


def gnss_cfg(**degraded):
    cfg = copy.deepcopy(PARAMS["localization"]["gnss_ins"])
    cfg["degraded"].update(rate_per_s=0.0, windows=[])
    cfg["degraded"].update(degraded)
    return cfg


def straight(t, speed=1.0):
    """Measurements for a car driving along +x, and the true x."""
    x = speed * t
    return Measurements(x, 0.0, 0.0, speed, 0.0, wheel_speeds(speed, 0.0, VEH.wheelbase, VEH.track)), x


def replay(log, loc, rest_s=0.0):
    """Feed a recorded true trajectory to an estimator, after rest_s at rest. Returns the last estimate."""
    first = VehicleState(log["x"][0], log["y"][0], log["yaw"][0])
    at_rest = Measurements.from_state(first, VEH.wheelbase, VEH.track)
    e = loc.estimate
    for _ in range(int(round(rest_s / DT))):
        e = loc.update(at_rest, DT)
    for i in range(len(log["x"])):
        state = VehicleState(log["x"][i], log["y"][i], log["yaw"][i], log["v"][i], log["delta"][i])
        e = loc.update(Measurements.from_state(state, VEH.wheelbase, VEH.track), DT)
    return e


@pytest.fixture(scope="module")
def baseline_runs():
    """Maneuvers followed on true pose: (scenario, result)."""
    runs = []
    for seed in range(12):
        sc = make_perpendicular(VEH, PARAMS, np.random.default_rng(seed))
        runs.append((sc, run_closed_loop(sc)))
    return runs


# ---------- choosing an estimator ----------
@pytest.mark.parametrize("name,sensor_based,cls,frame", [
    ("auto", False, TruthLocalizer, "map"),
    ("auto", True, WheelOdometry, "odom"),
    ("truth", True, TruthLocalizer, "map"),
    ("odometry", False, WheelOdometry, "odom"),
    ("gnss_ins", False, GnssInsLocalizer, "map"),
    ("white", False, WhiteNoiseLocalizer, "map"),
    ("white", True, WhiteNoiseLocalizer, "odom"),
])
def test_config_chooses_the_estimator(name, sensor_based, cls, frame):
    loc = make_localizer(params_with(name), START, VEH, sensor_based)
    assert isinstance(loc, cls) and loc.frame == frame


def test_unknown_estimator_is_refused():
    with pytest.raises(ValueError, match="estimator"):
        make_localizer(params_with("kalman"), START, VEH, False)


def test_noise_arguments_add_white_noise_to_any_estimator():
    loc = make_localizer(params_with("gnss_ins"), START, VEH, False, noise_xy=0.01)
    assert isinstance(loc, WhiteNoiseLocalizer) and isinstance(loc.inner, GnssInsLocalizer) and loc.frame == "map"
    assert loc.name == "white"


def test_every_tunable_value_is_in_the_config_and_truth_is_the_baseline_default():
    cfg = PARAMS["localization"]
    assert cfg["estimator"] == "auto"
    assert cfg["odometry"]["wheel_scale_error"] == 0.005 and cfg["odometry"]["yaw_rate_bias_deg_s"] == 0.1
    assert cfg["odometry"]["rate_hz"] == 50.0


# ---------- truth and white ----------
def test_truth_returns_the_true_pose_with_no_covariance():
    loc = TruthLocalizer(START)
    e = loc.update(Measurements(3.0, 4.0, 0.5, 1.0, 0.0), DT)
    assert (e.x, e.y, e.yaw) == (3.0, 4.0, 0.5) and e.sigma_xy == 0.0 and e.sigma_yaw == 0.0


def test_truth_estimator_reproduces_the_baseline_csv_byte_for_byte(tmp_path):
    """The reference holds the first 200 seeds from before localization existed; 10 keeps this quick.
    The full 200 are compared by hand against tests/data/baseline_monte_carlo_200.csv."""
    mc = load_script("monte_carlo")
    assert PARAMS["localization"]["estimator"] == "auto"
    rows, n = [], 10
    mc.batch(n, VEH, PARAMS, 0.0, 0.0, rows=rows, tag=f"lock_{math.degrees(VEH.max_steer):.0f}deg")
    out = tmp_path / "baseline.csv"
    mc.write_csv(out, rows)
    reference = REFERENCE_CSV.read_bytes().split(b"\r\n")
    assert out.read_bytes() == b"\r\n".join(reference[:n + 1]) + b"\r\n"


def test_explicit_truth_and_auto_give_the_same_run():
    a = run_closed_loop(make_perpendicular(VEH, params_with("truth"), np.random.default_rng(3)))
    b = run_closed_loop(make_perpendicular(VEH, params_with("auto"), np.random.default_rng(3)))
    assert a.metrics == b.metrics


def test_white_estimator_from_config_matches_the_noise_arguments():
    params = params_with("white")
    noise = params["localization"]["white"]
    sc = make_perpendicular(VEH, params, np.random.default_rng(2))
    a = run_closed_loop(sc, np.random.default_rng(5))
    b = run_closed_loop(make_perpendicular(VEH, PARAMS, np.random.default_rng(2)), np.random.default_rng(5),
                        noise["noise_xy"], noise["noise_yaw_deg"])
    assert a.metrics == b.metrics
    assert a.estimator == b.estimator == "white"
    assert a.metrics["est_err_max_m"] > 0.0


def test_white_noise_is_drawn_when_the_controller_reads_not_when_the_estimate_updates():
    loc = make_localizer(params_with("auto"), START, VEH, False, noise_rng=np.random.default_rng(0), noise_xy=0.1)
    truth = Measurements(3.0, 4.0, 0.5, 0.0, 0.0)
    assert loc.update(truth, DT).x == 3.0
    assert loc.for_control().x != 3.0
    assert loc.update(truth, DT).x == 3.0


# ---------- odometry ----------
@pytest.mark.parametrize("source", ["rear_average", "four_wheel"])
def test_odometry_with_all_errors_off_tracks_truth_within_1_mm_over_a_run(source):
    params = params_with("odometry", latency=0.0, wheel_source=source, **ODOMETRY_OFF)
    res = run_closed_loop(make_perpendicular(VEH, params, np.random.default_rng(1)))
    assert res.success, res.summary()
    log = res.log
    assert np.max(np.hypot(log["est_x"] - log["x"], log["est_y"] - log["y"])) < 1e-3
    assert res.metrics["est_err_max_m"] < 1e-3


def test_four_wheel_speeds_reduce_to_the_rear_axle_speed_in_a_tight_turn():
    delta = VEH.max_steer
    state = VehicleState(0.0, 0.0, 0.0, -0.8, delta)
    m = Measurements.from_state(state, VEH.wheelbase, VEH.track)
    fl, fr, rl, rr = m.wheels
    assert (rl + rr) / 2.0 == pytest.approx(-0.8)
    assert fl < 0 and fr < 0 and abs(fr) > abs(rr)      # negative yaw rate: the right side is on the inside
    cfg = {**PARAMS["localization"]["odometry"], **ODOMETRY_OFF, "wheel_source": "four_wheel", "standstill_calibration": False}
    odom = WheelOdometry(Pose(0.0, 0.0, 0.0), cfg, np.random.default_rng(0), VEH.wheelbase, VEH.track)
    e = odom.update(m, DT)
    assert math.hypot(e.x, e.y) == pytest.approx(0.8 * DT, rel=1e-3)


def test_unknown_wheel_source_is_refused():
    with pytest.raises(ValueError, match="wheel_source"):
        make_localizer(params_with("odometry", wheel_source="front"), START, VEH, False)


def test_odometry_with_default_errors_stays_under_5_cm_on_average_over_a_full_maneuver(baseline_runs):
    """From a stop, so the gyro has been zeroed. Scale error alone is 0.5% of the net move, about 4 cm,
    so single runs can pass 5 cm: the spread across seeds and error signs reaches about 6 cm."""
    cfg = PARAMS["localization"]["odometry"]
    errors = []
    for k, (sc, res) in enumerate(baseline_runs):
        for draw in range(4):
            odom = WheelOdometry(sc.start, cfg, np.random.default_rng(100 * k + draw), VEH.wheelbase, VEH.track)
            e = replay(res.log, odom, rest_s=2.0)
            errors.append(math.hypot(e.x - res.log["x"][-1], e.y - res.log["y"][-1]))
    assert np.mean(errors) < 0.05
    assert np.max(errors) < 0.07


def test_each_odometry_error_source_can_be_turned_on_alone(baseline_runs):
    sc, res = baseline_runs[0]
    final = {}
    for key, value in [("wheel_scale_error", 0.005), ("yaw_rate_bias_deg_s", 0.1), ("wheel_noise_std", 0.01),
                       ("yaw_rate_noise_std_deg_s", 0.05), ("wheel_quantization", 0.05), ("yaw_rate_quantization_deg_s", 0.5)]:
        cfg = {**PARAMS["localization"]["odometry"], **ODOMETRY_OFF, key: value, "standstill_calibration": False}
        e = replay(res.log, WheelOdometry(sc.start, cfg, np.random.default_rng(0), VEH.wheelbase, VEH.track))
        final[key] = math.hypot(e.x - res.log["x"][-1], e.y - res.log["y"][-1])
    cfg = {**PARAMS["localization"]["odometry"], **ODOMETRY_OFF}
    e = replay(res.log, WheelOdometry(sc.start, cfg, np.random.default_rng(0), VEH.wheelbase, VEH.track))
    clean = math.hypot(e.x - res.log["x"][-1], e.y - res.log["y"][-1])
    assert clean < 1e-3
    assert all(value > clean for value in final.values()), final


def test_standstill_calibration_removes_the_gyro_bias_and_can_be_turned_off(baseline_runs):
    sc, res = baseline_runs[0]
    errors = {}
    for calibrate in (True, False):
        cfg = {**PARAMS["localization"]["odometry"], **ODOMETRY_OFF, "yaw_rate_bias_deg_s": 0.1, "yaw_rate_noise_std_deg_s": 0.05,
               "standstill_calibration": calibrate}
        e = replay(res.log, WheelOdometry(sc.start, cfg, np.random.default_rng(0), VEH.wheelbase, VEH.track), rest_s=2.0)
        errors[calibrate] = abs(e.yaw - res.log["yaw"][-1])
    assert errors[False] > math.radians(0.3)        # 2 s at rest plus the maneuver, at 0.1 deg/s
    assert errors[True] < errors[False] / 3.0


def test_odometry_heading_holds_at_standstill_when_calibrating():
    cfg = PARAMS["localization"]["odometry"]
    odom = WheelOdometry(START, cfg, np.random.default_rng(0), VEH.wheelbase, VEH.track)
    at_rest = Measurements(START.x, START.y, START.yaw, 0.0, 0.0)
    for _ in range(500):
        e = odom.update(at_rest, DT)
    assert (e.x, e.y, e.yaw) == pytest.approx((START.x, START.y, START.yaw))


def test_odometry_sensor_rate_holds_between_samples():
    cfg = {**PARAMS["localization"]["odometry"], **ODOMETRY_OFF, "rate_hz": 10.0}
    odom = WheelOdometry(Pose(0.0, 0.0, 0.0), cfg, np.random.default_rng(0), VEH.wheelbase, VEH.track)
    xs = [odom.update(straight(i * DT)[0], DT).x for i in range(1, 51)]
    assert len(set(xs)) == 11           # 1 s at 10 Hz, ticking at 50 Hz, plus the start
    assert xs[-1] == pytest.approx(1.0)


def test_odometry_covariance_grows_with_distance():
    cfg = PARAMS["localization"]["odometry"]
    odom = WheelOdometry(Pose(0.0, 0.0, 0.0), cfg, np.random.default_rng(0), VEH.wheelbase, VEH.track)
    sigmas = [odom.update(straight(i * DT)[0], DT).sigma_xy for i in range(1, 500)]
    assert sigmas[0] < sigmas[100] < sigmas[-1]
    assert sigmas[-1] > 0.005 * 9.0     # at least the scale error over 10 m


# ---------- gnss_ins ----------
def test_gnss_error_stays_bounded_over_a_long_run():
    cfg = gnss_cfg()
    loc = GnssInsLocalizer(Pose(0.0, 0.0, 0.0), cfg, np.random.default_rng(0))
    err_xy, err_yaw = [], []
    for i in range(1, 60_000):      # 20 min
        m, x = straight(i * DT)
        e = loc.update(m, DT)
        err_xy.append(max(abs(e.x - x), abs(e.y)))
        err_yaw.append(abs(math.degrees(e.yaw)))
    assert np.max(err_xy) < 5.0 * cfg["position_sigma_m"]
    assert np.max(err_yaw) < 0.2
    assert np.std(err_xy) > 0.002       # it does move
    late = np.max(err_xy[30_000:])
    assert late < 5.0 * cfg["position_sigma_m"]     # no growth with time


def test_gnss_error_grows_during_a_degraded_period_and_recovers():
    cfg = gnss_cfg(windows=[[10.0, 20.0]])
    d = cfg["degraded"]
    loc = GnssInsLocalizer(Pose(0.0, 0.0, 0.0), cfg, np.random.default_rng(1))
    error = {}
    sigma = {}
    for i in range(1, 2500):
        t = i * DT
        m, x = straight(t)
        e = loc.update(m, DT)
        error[round(t, 2)] = math.hypot(e.x - x, e.y)
        sigma[round(t, 2)] = e.sigma_xy
    assert error[19.98] > 0.5 * d["position_growth_m_s"] * 10.0
    assert sigma[19.98] > sigma[9.98] * 2.0
    assert error[35.0] < 3.0 * math.sqrt(2.0) * cfg["position_sigma_m"]
    assert sigma[35.0] == pytest.approx(cfg["position_sigma_m"], rel=0.05)


def test_random_degraded_periods_happen_at_the_configured_rate():
    cfg = gnss_cfg(rate_per_s=0.05, duration_s=2.0)
    loc = GnssInsLocalizer(Pose(0.0, 0.0, 0.0), cfg, np.random.default_rng(2))
    starts, was = 0, False
    for i in range(1, 100_000):     # 2000 s
        loc.update(straight(i * DT)[0], DT)
        starts += loc.degraded and not was
        was = loc.degraded
    assert 50 < starts < 130        # 0.05/s while not degraded: about 90


def test_gnss_is_in_the_map_frame_and_needs_no_integration():
    loc = GnssInsLocalizer(Pose(0.0, 0.0, 0.0), gnss_cfg(), np.random.default_rng(3))
    e = loc.update(Measurements(100.0, 50.0, 1.0, 0.0, 0.0), DT)
    assert loc.frame == "map"
    assert math.hypot(e.x - 100.0, e.y - 50.0) < 0.2


# ---------- latency ----------
@pytest.mark.parametrize("latency", [0.02, 0.1, 0.5])
def test_delayed_odometry_lags_ground_truth_by_the_configured_time(latency):
    cfg = {**PARAMS["localization"]["odometry"], **ODOMETRY_OFF}
    odom = WheelOdometry(Pose(0.0, 0.0, 0.0), cfg, np.random.default_rng(0), VEH.wheelbase, VEH.track, latency_s=latency)
    speed, t = 2.0, 0.0
    for _ in range(200):
        t += DT
        e = odom.update(straight(t, speed)[0], DT)
    assert speed * t - e.x == pytest.approx(speed * latency, abs=1e-9)


@pytest.mark.parametrize("latency", [0.02, 0.1])
def test_delayed_gnss_lags_ground_truth_by_the_same_setting(latency):
    params = params_with("gnss_ins", latency=latency)
    params["localization"]["gnss_ins"].update(position_sigma_m=1e-9, heading_sigma_deg=1e-9)
    params["localization"]["gnss_ins"]["degraded"]["rate_per_s"] = 0.0
    loc = make_localizer(params, Pose(0.0, 0.0, 0.0), VEH, False, np.random.default_rng(0))
    t = 0.0
    for _ in range(200):
        t += DT
        e = loc.update(straight(t)[0], DT)
    assert t - e.x == pytest.approx(latency, abs=1e-6)


def test_zero_latency_means_no_delay():
    cfg = {**PARAMS["localization"]["odometry"], **ODOMETRY_OFF}
    odom = WheelOdometry(Pose(0.0, 0.0, 0.0), cfg, np.random.default_rng(0), VEH.wheelbase, VEH.track, latency_s=0.0)
    assert odom.update(straight(DT)[0], DT).x == pytest.approx(DT)


# ---------- frames: the whole flow with each estimator ----------
SEED = 1


@pytest.fixture(scope="module")
def fused_runs():
    """A search run per estimator with the fused detector."""
    runs = {}
    for name in ("odometry", "gnss_ins", "truth"):
        params = params_with(name)
        params["slot_detection"]["detector"] = "fused"
        lot = make_aisle_lot(VEH, params, np.random.default_rng(SEED))
        runs[name] = run_search(lot, localization_rng=np.random.default_rng(2), sensor_seed=3)
    return runs


@pytest.mark.parametrize("name", ["odometry", "gnss_ins", "truth"])
def test_fused_search_parks_with_each_estimator(fused_runs, name):
    res = fused_runs[name]
    assert res.estimator == name
    assert res.success, res.summary()
    assert not res.search.scenario.nearest_stall(*res.search.scenario.stall_center).occupied


def test_truth_with_a_sensor_detector_has_no_estimation_error(fused_runs):
    m = fused_runs["truth"].metrics
    assert m["est_err_max_m"] == 0.0 and m["est_yaw_err_max_deg"] == 0.0


def test_gnss_error_stays_small_while_odometry_drifts_on_the_way_down_the_aisle(fused_runs):
    assert fused_runs["gnss_ins"].metrics["est_err_max_m"] < 0.15
    assert fused_runs["odometry"].metrics["est_err_max_m"] > fused_runs["gnss_ins"].metrics["est_err_max_m"]


def test_gnss_plan_starts_on_the_estimated_pose_in_the_map_frame(fused_runs):
    res = fused_runs["gnss_ins"]
    t, path = res.search.plans[0]
    i = int(np.searchsorted(res.log["t"], t))
    assert path.x[0] == pytest.approx(res.log["est_x"][i], abs=1e-6)
    assert path.y[0] == pytest.approx(res.log["est_y"][i], abs=1e-6)


def test_ideal_detector_with_odometry_keeps_slots_in_the_map_frame():
    """The ideal detector reads the true lot; only the car's pose is in the drifting frame."""
    params = params_with("odometry")
    params["slot_detection"]["detector"] = "ideal"
    lot = make_aisle_lot(VEH, params, np.random.default_rng(SEED))
    res = run_search(lot, localization_rng=np.random.default_rng(2))
    assert res.estimator == "odometry"
    assert res.metrics["slot_corner_error_m"] == 0.0
    assert res.metrics["est_err_max_m"] > 0.0
    assert res.planned is not None


def test_single_stall_runs_with_each_estimator():
    for name in ("odometry", "gnss_ins"):
        sc = make_perpendicular(VEH, params_with(name), np.random.default_rng(1))
        res = run_closed_loop(sc, localization_rng=np.random.default_rng(1))
        assert res.estimator == name
        assert res.metrics["completed"] == 1.0, res.summary()
        assert res.metrics["est_err_max_m"] > 0.0


def test_ground_truth_keeps_feeding_the_sensors_whatever_the_estimator(fused_runs):
    """Scoring uses the true pose: the log's x, y differ from the estimate the controller followed."""
    log = fused_runs["odometry"].log
    assert np.max(np.hypot(log["est_x"] - log["x"], log["est_y"] - log["y"])) > 0.0
    assert log["x"][0] == log["est_x"][0]


# ---------- metrics ----------
def test_runs_report_estimation_error_final_and_max(fused_runs):
    res = fused_runs["odometry"]
    m = res.metrics
    for key in ESTIMATION_METRICS:
        assert key in m and not math.isnan(m[key])
    assert m["est_err_max_m"] >= m["est_err_final_m"] > 0.0
    assert m["est_yaw_err_max_deg"] >= m["est_yaw_err_final_deg"] > 0.0
    assert m["est_drift_final_m"] < m["est_err_final_m"]      # the drive down the aisle is most of the error


def test_baseline_run_reports_zero_estimation_error(baseline_runs):
    _, res = baseline_runs[0]
    assert res.estimator == "truth"
    assert all(res.metrics[key] == 0.0 for key in ESTIMATION_METRICS)


def test_monte_carlo_csv_has_estimation_columns_except_for_truth():
    mc = load_script("monte_carlo")
    rows = []
    mc.batch(1, VEH, PARAMS, 0.0, 0.0, rows=rows)
    assert not set(ESTIMATION_METRICS) & set(rows[0])
    rows = []
    mc.batch(1, VEH, params_with("odometry"), 0.0, 0.0, rows=rows)
    assert set(ESTIMATION_METRICS) <= set(rows[0])
    assert rows[0]["est_err_final_m"] > 0.0


def test_search_trial_seeds_the_estimator_so_runs_repeat():
    params = params_with("odometry")
    params["slot_detection"]["detector"] = "ideal"
    a = search_trial(VEH, params, 4)[1].metrics
    b = search_trial(VEH, params, 4)[1].metrics
    a.pop("planning_time_s"), b.pop("planning_time_s")
    assert a == b


# ---------- export ----------
def test_mcap_has_the_estimated_pose_on_tf_and_state(tmp_path):
    pytest.importorskip("mcap")
    from test_mcap_export import read_mcap
    params = params_with("odometry")
    sc = make_perpendicular(VEH, params, np.random.default_rng(1))
    res = run_closed_loop(sc, localization_rng=np.random.default_rng(1))
    path = export_run(sc, res, tmp_path / "run.mcap")
    messages = read_mcap(path)[3]
    last = messages["/tf"][-1][1].transforms
    assert [t.child_frame_id for t in last] == ["base_link", "base_link_est"]
    assert all(t.parent_frame_id == "map" for t in last)
    assert last[0].translation.x == pytest.approx(res.log["x"][-1])
    assert last[1].translation.x == pytest.approx(res.log["est_x"][-1])
    assert last[0].translation.x != last[1].translation.x
    state = messages["/state"][-1][1]
    assert state["est_x"] == pytest.approx(res.log["est_x"][-1])
    assert state["est_yaw"] == pytest.approx(res.log["est_yaw"][-1])
    assert state["est_error_m"] == pytest.approx(res.metrics["est_err_final_m"])
    assert state["est_sigma_xy"] > 0.0


# ---------- plant realism ----------
def drive(params, steer, speed, ticks=500):
    veh = Vehicle(params, VehicleState(0.0, 0.0, 0.0), dt=DT)
    for _ in range(ticks):
        veh.step(steer, speed)
    return veh.state


def test_realism_is_off_by_default():
    assert VEH.tire_scrub_gain == 0.0 and VEH.road_grade_percent == 0.0


def scrubbing(**changes):
    params = VehicleParams(**{**VEH.__dict__, "tire_scrub_gain": 0.05})
    return VehicleParams(**{**params.__dict__, **changes})


def test_scrub_slips_sideways_at_crawl_in_proportion_to_the_steering_angle():
    plain_y = drive(VEH, 0.0, 0.1).y
    a, b = drive(scrubbing(), 0.05, 0.1), drive(scrubbing(), 0.1, 0.1)
    # compare with the same steering without scrub
    a0, b0 = drive(VEH, 0.05, 0.1), drive(VEH, 0.1, 0.1)
    assert plain_y == 0.0
    slip_a, slip_b = math.hypot(a.x - a0.x, a.y - a0.y), math.hypot(b.x - b0.x, b.y - b0.y)
    assert slip_a > 0.001
    assert slip_b == pytest.approx(2.0 * slip_a, rel=0.1)


def test_scrub_fades_out_above_the_crawl_speed():
    fast = scrubbing()
    a, a0 = drive(fast, 0.1, 0.9, ticks=1000), drive(VEH, 0.1, 0.9, ticks=1000)
    b, b0 = drive(fast, 0.1, 0.1, ticks=1000), drive(VEH, 0.1, 0.1, ticks=1000)
    assert math.hypot(a.x - a0.x, a.y - a0.y) < math.hypot(b.x - b0.x, b.y - b0.y) / 2.0


def test_scrub_needs_turning_wheels_at_standstill():
    veh = Vehicle(scrubbing(), VehicleState(0.0, 0.0, 0.0), dt=DT)
    for _ in range(100):
        veh.step(0.3, 0.0)      # turning the wheels at rest moves the car a little
    moved = abs(veh.state.y)
    assert moved > 0.0
    for _ in range(1000):
        veh.step(0.3, 0.0)      # then the wheels settle and it stops
    settled = abs(veh.state.y)
    for _ in range(1000):
        veh.step(0.3, 0.0)
    assert abs(veh.state.y) == settled


def test_odometry_cannot_see_scrub():
    """Wheel speeds and yaw rate don't measure sideways slip, so it becomes estimation error."""
    errors = {}
    for gain in (0.0, 0.1):
        params = copy.deepcopy(VEH)
        params.tire_scrub_gain = gain
        cfg = params_with("odometry", latency=0.0, **ODOMETRY_OFF)
        sc = make_perpendicular(params, cfg, np.random.default_rng(1))
        errors[gain] = run_closed_loop(sc).metrics["est_err_final_m"]
    assert errors[0.1] > errors[0.0] + 0.005


def test_grade_shifts_the_speed_loop_by_its_slope():
    params = VehicleParams(**{**VEH.__dict__, "road_grade_percent": 2.0})
    uphill, flat = drive(params, 0.0, 0.5), drive(VEH, 0.0, 0.5)
    expected = 9.81 * 0.02 * VEH.speed_tau      # steady droop of a first-order loop
    assert flat.v == pytest.approx(0.5, abs=1e-3)
    assert flat.v - uphill.v == pytest.approx(expected, rel=0.1)


def test_grade_helps_driving_the_other_way_and_downhill_reverse():
    params = VehicleParams(**{**VEH.__dict__, "road_grade_percent": 2.0})
    assert abs(drive(params, 0.0, -0.5).v) > 0.5     # reversing down the same slope


def test_brakes_hold_the_car_on_a_grade():
    params = VehicleParams(**{**VEH.__dict__, "road_grade_percent": 5.0})
    state = drive(params, 0.0, 0.0)
    assert state.v == 0.0 and state.x == 0.0


def test_maneuver_still_completes_with_mild_scrub_and_grade():
    params = copy.deepcopy(VEH)
    params.tire_scrub_gain, params.road_grade_percent = 0.02, 1.0
    res = run_closed_loop(make_perpendicular(params, PARAMS, np.random.default_rng(1)))
    assert res.metrics["completed"] == 1.0, res.summary()


# ---------- sweep script ----------
def test_sweep_runs_end_to_end_on_a_tiny_grid(tmp_path, capsys):
    sweep = load_script("sweep_localization")
    out, summary = tmp_path / "sweep.csv", tmp_path / "sweep.txt"
    code = sweep.main(["-n", "1", "--bias", "0.1", "--scale", "0.005", "--gnss", "1.0", "--csv", str(out), "--summary", str(summary)])
    assert code == 0
    lines = out.read_text().strip().splitlines()
    assert lines[0].split(",") == list(sweep.COLUMNS)
    assert len(lines) == 3      # header, one odometry cell, one gnss cell
    assert "Cells below 95% pass rate" in summary.read_text()


def test_sweep_summary_names_cells_below_the_floor_and_warns_about_slow_planning():
    sweep = load_script("sweep_localization")
    row = dict(estimator="odometry", yaw_rate_bias_deg_s=0.5, wheel_scale_error=0.02, gnss_error_factor=None, runs=10, passed=5,
               pass_rate=0.5, failures="heading:5", planning_max_s=1.8)
    gnss = dict(estimator="gnss_ins", yaw_rate_bias_deg_s=None, wheel_scale_error=None, gnss_error_factor=2.0, runs=10, passed=10,
                pass_rate=1.0, failures="", planning_max_s=0.5)
    text = sweep.summary_text([row, gnss], 0.95, 2.0)
    assert "Cells below 95% pass rate: 1" in text
    assert "bias 0.5 deg/s, scale 2.0%: 5/10 (50%)  heading:5" in text
    assert "WARNING" in text
    assert "WARNING" not in sweep.summary_text([gnss], 0.95, 2.0)


def test_sweep_cells_change_only_what_they_name():
    sweep = load_script("sweep_localization")
    cell = sweep.cell_params(PARAMS, "gnss_ins", gnss=2.0)
    assert cell["slot_detection"]["detector"] == "fused"
    assert cell["localization"]["estimator"] == "gnss_ins"
    assert cell["localization"]["gnss_ins"]["position_sigma_m"] == 2.0 * PARAMS["localization"]["gnss_ins"]["position_sigma_m"]
    assert PARAMS["localization"]["estimator"] == "auto"
    cell = sweep.cell_params(PARAMS, "odometry", bias=0.5, scale=0.01)
    assert cell["localization"]["odometry"]["yaw_rate_bias_deg_s"] == 0.5
    assert cell["localization"]["odometry"]["wheel_scale_error"] == 0.01
