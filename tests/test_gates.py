"""The gate machinery on small fixed inputs. A gate failing today is not a test failure: nothing here reads the
current values of the real run sets. Only test_gates_yaml_* read the real config, to hold its thresholds."""
import copy
import csv
import importlib.util
import math
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from park_sim import gates as G  # noqa: E402
from park_sim import hybrid_astar  # noqa: E402
from park_sim.scenario import load_params  # noqa: E402
from park_sim.scenario_files import list_scenarios, load_scenario_file  # noqa: E402
from park_sim.sensors.base import load_sensor_params  # noqa: E402


def load_script():
    spec = importlib.util.spec_from_file_location("check_gates", ROOT / "scripts" / "check_gates.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CFG = {
    "run_sets": {"baseline": {"flow": "single_stall", "seeds": {"first": 0, "count": 4}},
                 "search_fused": {"flow": "search", "detector": "fused", "seeds": {"first": 0, "count": 4}}},
    "safety": {"seeds": {"first": 1, "count": 2}, "scenarios": [], "single_stall_detector": "ideal", "search_detectors": ["ideal"]},
    "thresholds": {"pass_rate_min": {"baseline": 0.75, "search_fused": 0.5}, "body_contacts_max": 0,
                   "lowest_clearance_min_m": 0.3, "heading_err_p95_max_deg": 1.5, "actor_collisions_max": 0,
                   "default_maneuver": "reverse_in", "forward_camera_within_deg": 90.0},
    "planning_budget": {"max_expansions": 10, "wall_clock_limit_s": 100.0},
    "workers": 1,
}
PARAMS = {"scenario": {"maneuver": "reverse_in"}}
SENSORS = {"line_camera": {"cameras": [{"name": "cam_rear", "yaw_deg": 180.0}, {"name": "cam_left", "yaw_deg": 90.0},
                                       {"name": "cam_right", "yaw_deg": -90.0}]}}


def good(run_set, seed, **changes):
    values = dict(success=True, completed=True, lowest_clearance_m=0.5, heading_err_deg=0.5, actor_collisions=0.0,
                  planning_wall_max_s=0.2, planning_wall_total_s=0.2, plans=1, expansions_max=7)
    values.update(changes)
    return G.RunRecord(run_set, seed, **values)


def passing_records(**changes_to_seed_2):
    records = []
    for name in ("baseline", "search_fused"):
        records += [good(name, seed) for seed in range(4)]
    records += [good(G.SAFETY_SET, seed, label="ped/search/ideal") for seed in (1, 2)]
    if changes_to_seed_2:
        records[2] = good("baseline", 2, **changes_to_seed_2)
    return records


def by_name(results):
    return {r.gate: r for r in results}


def judged(records, cfg=CFG, params=PARAMS, sensors=SENSORS):
    return by_name(G.evaluate(cfg, records, params, sensors))


# ---------- a passing case ----------
def test_all_gates_pass_on_good_runs():
    results = G.evaluate(CFG, passing_records(), PARAMS, SENSORS)
    assert not G.any_failed(results)
    assert {r.gate for r in results if r.result != G.INFO} == {
        "pass_rate_baseline", "body_contacts_baseline", "lowest_clearance_baseline", "heading_err_p95_baseline",
        "pass_rate_search_fused", "body_contacts_search_fused", "lowest_clearance_search_fused", "heading_err_p95_search_fused",
        "actor_collisions", "default_maneuver", "forward_cameras"}


# ---------- a failing case for each gate type ----------
def test_pass_rate_gate_fails_and_names_seeds():
    records = passing_records()
    records[1] = good("baseline", 1, success=False)
    records[2] = good("baseline", 2, success=False)
    gate = judged(records)["pass_rate_baseline"]
    assert gate.result == G.FAIL and gate.value == 0.5 and gate.seeds == "1 2" and gate.note == "2/4"
    assert judged(passing_records(success=False))["pass_rate_baseline"].result == G.PASS     # 3/4 meets 0.75


@pytest.mark.parametrize("clearance", [0.0, -0.1])
def test_contact_gate_counts_zero_clearance(clearance):
    gate = judged(passing_records(lowest_clearance_m=clearance))["body_contacts_baseline"]
    assert gate.result == G.FAIL and gate.value == 1 and gate.seeds == "2"


def test_contact_gate_passes_just_above_zero():
    assert judged(passing_records(lowest_clearance_m=0.01))["body_contacts_baseline"].result == G.PASS


def test_clearance_gate_uses_the_lowest_run():
    gate = judged(passing_records(lowest_clearance_m=0.29))["lowest_clearance_baseline"]
    assert gate.result == G.FAIL and gate.value == 0.29 and gate.seeds == "2"


def test_clearance_gate_passes_at_the_limit():
    assert judged(passing_records(lowest_clearance_m=0.3))["lowest_clearance_baseline"].result == G.PASS


def test_heading_gate_uses_the_95th_percentile_of_magnitudes():
    records = [good("baseline", s, heading_err_deg=-2.0 if s < 3 else 0.1) for s in range(4)] + passing_records()[4:]
    gate = judged(records)["heading_err_p95_baseline"]
    assert gate.result == G.FAIL and gate.value == pytest.approx(2.0) and gate.seeds == "0 1 2"


def test_heading_gate_tolerates_a_rare_outlier():
    records = [good("baseline", s, heading_err_deg=0.1) for s in range(99)] + [good("baseline", 99, heading_err_deg=9.0)]
    gate = judged(records + passing_records()[4:])["heading_err_p95_baseline"]
    assert gate.result == G.PASS and gate.seeds == "99"


def test_heading_gate_counts_completed_runs_only():
    gate = judged(passing_records(completed=False, heading_err_deg=30.0))["heading_err_p95_baseline"]
    assert gate.result == G.PASS and gate.note == "3/4 runs completed"


def test_actor_collision_gate_fails_and_names_the_case():
    records = passing_records()
    records[-1] = good(G.SAFETY_SET, 2, label="ped/search/ideal", actor_collisions=1.0)
    gate = judged(records)["actor_collisions"]
    assert gate.result == G.FAIL and gate.value == 1.0 and gate.seeds == "ped/search/ideal:2"


def test_default_maneuver_gate():
    gate = judged(passing_records(), params={"scenario": {"maneuver": "head_in"}})["default_maneuver"]
    assert gate.result == G.FAIL and gate.value == "head_in"


@pytest.mark.parametrize("camera", [{"name": "cam_nose", "yaw_deg": 0.0}, {"name": "cam_nose", "yaw_deg": 360.0},
                                    {"name": "cam_nose", "yaw_deg": -60.0}, {"name": "cam_nose", "yaw_deg": 89.0},
                                    {"name": "cam_front", "yaw_deg": 180.0}])
def test_forward_camera_gate(camera):
    sensors = copy.deepcopy(SENSORS)
    sensors["line_camera"]["cameras"].append(camera)
    gate = judged(passing_records(), sensors=sensors)["forward_cameras"]
    assert gate.result == G.FAIL and gate.value == 1 and gate.note == camera["name"]


# ---------- what can't be computed fails ----------
def test_a_set_with_no_runs_fails_every_gate():
    records = [r for r in passing_records() if r.run_set != "baseline"]
    results = judged(records)
    for gate in ("pass_rate_baseline", "body_contacts_baseline", "lowest_clearance_baseline", "heading_err_p95_baseline"):
        assert results[gate].result == G.FAIL, gate


def test_no_safety_runs_fails():
    assert judged([r for r in passing_records() if r.run_set != G.SAFETY_SET])["actor_collisions"].result == G.FAIL


def test_safety_run_without_the_metric_fails():
    records = passing_records()
    records[-1] = good(G.SAFETY_SET, 2, label="ped/search/ideal", actor_collisions=math.nan)
    assert judged(records)["actor_collisions"].result == G.FAIL


def test_unmeasured_clearance_fails():
    results = judged(passing_records(lowest_clearance_m=math.nan))
    assert results["body_contacts_baseline"].result == G.FAIL
    assert results["lowest_clearance_baseline"].result == G.FAIL


def test_no_completed_run_fails_the_heading_gate():
    records = [good("baseline", s, completed=False) for s in range(4)] + passing_records()[4:]
    assert judged(records)["heading_err_p95_baseline"].result == G.FAIL


# ---------- thresholds come from the config ----------
@pytest.mark.parametrize("key,value,gate", [
    ("lowest_clearance_min_m", 0.6, "lowest_clearance_baseline"),
    ("heading_err_p95_max_deg", 0.4, "heading_err_p95_baseline"),
    ("default_maneuver", "head_in", "default_maneuver"),
    ("pass_rate_min", {"baseline": 1.01, "search_fused": 0.5}, "pass_rate_baseline"),
])
def test_thresholds_are_read_from_config(key, value, gate):
    cfg = copy.deepcopy(CFG)
    cfg["thresholds"][key] = value
    assert judged(passing_records(), cfg=cfg)[gate].result == G.FAIL


def test_contact_and_collision_limits_are_read_from_config():
    cfg = copy.deepcopy(CFG)
    cfg["thresholds"]["body_contacts_max"] = 1
    assert judged(passing_records(lowest_clearance_m=0.0), cfg=cfg)["body_contacts_baseline"].result == G.PASS
    cfg["thresholds"]["actor_collisions_max"] = 1
    records = passing_records()
    records[-1] = good(G.SAFETY_SET, 2, label="p", actor_collisions=1.0)
    assert judged(records, cfg=cfg)["actor_collisions"].result == G.PASS


def test_forward_camera_limit_is_read_from_config():
    cfg = copy.deepcopy(CFG)
    cfg["thresholds"]["forward_camera_within_deg"] = 91.0
    assert judged(passing_records(), cfg=cfg)["forward_cameras"].result == G.FAIL


def test_missing_threshold_is_an_error_not_a_default():
    cfg = copy.deepcopy(CFG)
    del cfg["thresholds"]["lowest_clearance_min_m"]
    with pytest.raises(KeyError):
        G.evaluate(cfg, passing_records(), PARAMS, SENSORS)


def test_gates_yaml_holds_the_agreed_thresholds():
    """Loosening a gate takes a deliberate edit here too."""
    t = G.load_gates()["thresholds"]
    assert t["pass_rate_min"] == {"baseline": 0.99, "search_ideal": 0.99, "search_fused": 0.95, "search_empty": 0.95}
    assert t["body_contacts_max"] == 0
    assert t["lowest_clearance_min_m"] == 0.30
    assert t["actor_collisions_max"] == 0
    assert t["heading_err_p95_max_deg"] == 1.5
    assert t["default_maneuver"] == "reverse_in"
    assert t["forward_camera_within_deg"] == 90.0


def test_gates_yaml_holds_the_agreed_seed_sets():
    cfg = G.load_gates()
    assert {k: (v["seeds"]["first"], v["seeds"]["count"]) for k, v in cfg["run_sets"].items()} == {
        "baseline": (0, 200), "search_ideal": (0, 200), "search_fused": (0, 200), "search_empty": (0, 100)}
    assert cfg["run_sets"]["search_empty"]["occupancy"] == 0.0
    assert (cfg["safety"]["seeds"]["first"], cfg["safety"]["seeds"]["count"]) == (1, 20)
    assert cfg["run_sets"]["search_ideal"]["detector"] == "ideal" and cfg["run_sets"]["search_fused"]["detector"] == "fused"


def test_gates_yaml_clearance_limit_agrees_with_the_criteria():
    assert G.load_gates()["thresholds"]["lowest_clearance_min_m"] == load_params()["criteria"]["min_clearance"] == 0.30


def test_gates_yaml_covers_every_scenario_with_actors():
    with_actors = {name for name in list_scenarios() if load_scenario_file(name).actors}
    assert set(G.load_gates()["safety"]["scenarios"]) == with_actors


def test_shipped_defaults_are_read_by_the_config_gates():
    """The gates can read the real files. Whether they pass is the gate's business, not a test's."""
    limits = G.load_gates()["thresholds"]
    results = by_name(G.default_config_gates(limits, load_params(), load_sensor_params()))
    assert results["default_maneuver"].value == load_params()["scenario"]["maneuver"]
    assert isinstance(results["forward_cameras"].value, int)


# ---------- the planning budget ----------
def test_budget_replaces_the_wall_clock_limits_and_leaves_the_input_alone():
    params = load_params()
    out = G.with_planning_budget(params, {"max_expansions": 123, "wall_clock_limit_s": 99.0})
    assert out["planner"]["max_expansions"] == 123
    assert out["search"]["planning_time_limit_s"] == 99.0 and out["safety"]["replan_time_limit_s"] == 99.0
    assert params == load_params()


def test_slow_machine_does_not_change_a_gate_run(monkeypatch):
    """A plan taking longer than the plain 2 s limit, as on a loaded machine, still gives the same result."""
    script = load_script()
    budget = G.load_gates()["planning_budget"]
    job = ("search_ideal", "search", "ideal", None, 0, budget)
    quick = script.one_run(job)
    assert quick.plans_failed == 0

    plan = hybrid_astar.HybridAStar.plan

    def slow_plan(self, start=None, time_limit_s=None):
        time.sleep(2.2)
        return plan(self, start, time_limit_s)

    monkeypatch.setattr(hybrid_astar.HybridAStar, "plan", slow_plan)
    slow = script.one_run(job)
    assert slow.planning_wall_max_s > load_params()["search"]["planning_time_limit_s"]
    for field in ("success", "completed", "lowest_clearance_m", "heading_err_deg", "plans_failed", "expansions_max"):
        assert getattr(slow, field) == getattr(quick, field), field


def test_budget_ends_a_plan_on_the_expansion_count():
    script = load_script()
    record = script.one_run(("search_ideal", "search", "ideal", None, 0, {"max_expansions": 1, "wall_clock_limit_s": 100.0}))
    assert record.plans_failed >= 1 and not record.success


def test_gate_run_matches_the_plain_baseline_row():
    """Under the budget, a baseline run comes out the same as monte_carlo.py's."""
    script = load_script()
    with open(ROOT / "tests" / "data" / "baseline_monte_carlo_200.csv") as f:
        rows = {int(r["seed"]): r for r in csv.DictReader(f)}
    record = script.one_run(("baseline", "single_stall", None, None, 3, G.load_gates()["planning_budget"]))
    assert record.success == (rows[3]["success"] == "True")
    assert record.heading_err_deg == pytest.approx(float(rows[3]["heading_err_deg"]), abs=1e-12)


# ---------- the script ----------
def test_jobs_cover_the_seed_sets_and_every_flow():
    cfg = copy.deepcopy(CFG)
    cfg["safety"]["scenarios"] = ["vehicle_stops_in_aisle", "pedestrian_stays"]
    jobs = load_script().jobs_for(cfg)
    assert [j[4] for j in jobs if j[0] == "baseline"] == [0, 1, 2, 3]
    safety = [(j[1], j[2], j[3], j[4]) for j in jobs if j[0] == G.SAFETY_SET]
    assert ("single_stall", "ideal", "vehicle_stops_in_aisle", 1) in safety     # it has no search flow
    assert ("search", "ideal", "pedestrian_stays", 2) in safety
    assert ("single_stall", "ideal", "pedestrian_stays", 1) in safety
    assert len(safety) == 2 + 2 + 2


def test_exit_codes_and_report(tmp_path):
    script = load_script()
    path = tmp_path / "gates.yaml"
    path.write_text(yaml.safe_dump(CFG))
    out = tmp_path / "report.csv"
    assert script.main(["--gates", str(path), "--csv", str(out)], collect=lambda cfg, w: passing_records()) == 0
    with open(out) as f:
        rows = {r["gate"]: r for r in csv.DictReader(f)}
    assert rows["pass_rate_baseline"]["result"] == G.PASS and rows["pass_rate_baseline"]["threshold"] == ">= 0.75"
    assert rows["planning_wall_max_s_baseline"]["result"] == G.INFO
    assert (tmp_path / "report_runs.csv").exists()
    failing = lambda cfg, w: passing_records(lowest_clearance_m=0.0)     # noqa: E731
    assert script.main(["--gates", str(path), "--csv", str(out)], collect=failing) == 1
    with open(out) as f:
        rows = {r["gate"]: r for r in csv.DictReader(f)}
    assert rows["body_contacts_baseline"]["result"] == G.FAIL and rows["body_contacts_baseline"]["seeds"] == "2"


def test_report_prints_value_threshold_and_seeds():
    text = G.format_report(G.evaluate(CFG, passing_records(lowest_clearance_m=0.2), PARAMS, SENSORS))
    line = next(l for l in text.splitlines() if "lowest_clearance_baseline" in l)
    assert line.startswith("FAIL") and "0.2000" in line and ">= 0.3 m" in line and "seeds: 2" in line
    assert text.splitlines()[-1].endswith("FAILED")


def test_script_exits_non_zero_end_to_end(tmp_path):
    """One real run per set, from a gates file with a limit nothing can meet, as a real process."""
    cfg = G.load_gates()
    for run_set in cfg["run_sets"].values():
        run_set["seeds"] = {"first": 1, "count": 1}
    cfg["safety"]["seeds"] = {"first": 1, "count": 1}
    cfg["safety"]["scenarios"] = ["vehicle_stops_in_aisle"]
    cfg["thresholds"]["pass_rate_min"]["baseline"] = 1.01
    path = tmp_path / "gates.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "report.csv"
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_gates.py"), "--gates", str(path), "--csv", str(out),
                           "--workers", "1"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 1, done.stderr
    assert "FAIL  pass_rate_baseline" in done.stdout
    with open(out) as f:
        rows = {r["gate"]: r for r in csv.DictReader(f)}
    assert rows["pass_rate_baseline"]["result"] == G.FAIL
    assert float(rows["actor_collisions"]["value"]) == 0.0
    assert rows["actor_collisions"]["note"] == "1 runs"
