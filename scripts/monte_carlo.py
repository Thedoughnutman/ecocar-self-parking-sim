#!/usr/bin/env python3
"""Run many random start poses / lot layouts and report pass rate against the criteria.

    python3 scripts/monte_carlo.py -n 300
    python3 scripts/monte_carlo.py -n 200 --noise-xy 0.03 --noise-yaw 0.3
    python3 scripts/monte_carlo.py -n 100 --steer-sweep 20 25 30 36
    python3 scripts/monte_carlo.py -n 200 --noise-xy 0.03 --noise-yaw 0.3 --mcap-failures out/failures/
    python3 scripts/monte_carlo.py -n 50 --mode search     # whole aisle: occupancy, speed and slot random
    python3 scripts/monte_carlo.py -n 50 --mode search --detector ideal
    python3 scripts/monte_carlo.py -n 50 --mode search --estimator gnss_ins
    python3 scripts/monte_carlo.py -n 50 --scenario pedestrian_walks_into_slot     # actors and faults: config/scenarios/
    python3 scripts/monte_carlo.py -n 50 --mode search --safety                    # monitor on with nothing there
    python3 scripts/monte_carlo.py -n 50 --mode search --planner-map sensed        # plan from what the sensors saw

Runs with the safety monitor on add its metrics to the CSV (collisions, min distance to actors, stop
distance after detection, time paused, outcome and reason). Other runs keep their columns.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from park_sim.localization import ESTIMATION_METRICS, ESTIMATORS  # noqa: E402
from park_sim.mcap_export import export_run, failure_filename, require_backend  # noqa: E402
from park_sim.scenario import load_params, make_perpendicular  # noqa: E402
from park_sim.scenario_files import list_scenarios, load_scenario_file, run_scenario_file  # noqa: E402
from park_sim.simulator import run_closed_loop, search_trial  # noqa: E402
from park_sim.vehicle import VehicleParams  # noqa: E402


def batch(n, veh, params, noise_xy, noise_yaw, seed0=0, rows=None, tag="", mcap_dir=None, mode="maneuver", scenario=None):
    passed, fails = 0, {}
    for k in range(n):
        seed = seed0 + k
        drawn = {}
        if scenario is not None:
            flow = "search" if mode == "search" else "single_stall"
            sc, r, drawn = run_scenario_file(scenario, seed, flow, veh, params)
        elif mode == "search":
            sc, r, drawn = search_trial(veh, params, seed, noise_xy, noise_yaw)
        else:
            sc = make_perpendicular(veh, params, np.random.default_rng(seed))
            r = run_closed_loop(sc, np.random.default_rng(10_000 + seed), noise_xy, noise_yaw,
                                localization_rng=np.random.default_rng(20_000 + seed))
        if r.success:
            passed += 1
        elif r.planned is None:
            key = "no_path" if mode != "search" or r.reason.startswith("no path") else r.reason
            fails[key] = fails.get(key, 0) + 1
        else:
            for name, ok in r.checks.items():
                if not ok:
                    fails[name] = fails.get(name, 0) + 1
        if mcap_dir is not None and not r.success:
            print("  saved", export_run(sc, r, Path(mcap_dir) / failure_filename(seed, r)))
        if rows is not None:
            row = {"tag": tag, "seed": seed, "success": r.success, "reason": r.reason}
            row.update(drawn)
            row.update(r.metrics)
            if r.estimator == "truth":      # always zero, and the truth CSV stays byte-identical
                for key in ESTIMATION_METRICS:
                    row.pop(key, None)
            if r.safety is not None:
                row.update(outcome=r.safety.outcome, outcome_reason=r.safety.outcome_reason)
            rows.append(row)
    return passed, fails


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)


def report(rows, label):
    ok = [r for r in rows if "lateral_err" in r]
    if not ok:
        return
    def p95(key, f=abs):
        return float(np.percentile([f(r[key]) for r in ok], 95))
    print(f"  {label} 95th pct: |lat| {p95('lateral_err')*100:.1f} cm, |head| {p95('heading_err_deg'):.2f} deg, "
          f"|depth| {p95('depth_err')*100:.1f} cm, min clearance (5th pct) "
          f"{np.percentile([r['min_clearance'] for r in ok], 5):.2f} m, "
          f"gears max {max(r['gear_changes'] for r in ok):.0f}, time 95th {p95('time_s'):.1f} s")
    if "planning_time_s" in ok[0]:
        print(f"  {label} search: planning max {max(r['planning_time_s'] for r in ok):.2f} s, "
              f"selection to stop 95th {p95('selection_to_stop_s'):.1f} s, "
              f"confirm to parked 95th {p95('confirm_to_parked_s'):.1f} s, "
              f"total 95th {p95('total_time_s'):.1f} s")
    detected = [r for r in rows if "slots_offered" in r]
    if detected:
        def total(key):
            return sum(r[key] for r in detected)
        errors = [r["slot_corner_error_m"] for r in detected if not math.isnan(r["slot_corner_error_m"])]
        confirm = [r["time_to_confirm_s"] for r in detected if not math.isnan(r["time_to_confirm_s"])]
        error_text = f"{np.mean(errors)*100:.1f} cm mean, {np.percentile(errors, 95)*100:.1f} cm 95th" if errors else "n/a"
        print(f"  {label} detection: corner error {error_text}, "
              f"confirm {np.mean(confirm) if confirm else float('nan'):.2f} s, "
              f"false positives {total('slots_false_positive'):.0f}/{total('slots_offered'):.0f} offered, "
              f"missed {total('free_stalls_missed'):.0f}/{total('free_stalls_passed'):.0f} free stalls passed, "
              f"replans {total('replans')/len(detected):.2f}/run, "
              f"stop adjustments {total('stop_adjustments')/len(detected):.2f}/run")
    estimated = [r for r in ok if "est_err_final_m" in r]
    if estimated:
        print(f"  {label} estimation error: final {np.percentile([r['est_err_final_m'] for r in estimated], 95)*100:.1f} cm 95th, "
              f"max {max(r['est_err_max_m'] for r in estimated)*100:.1f} cm, "
              f"final heading {np.percentile([r['est_yaw_err_final_deg'] for r in estimated], 95):.2f} deg 95th, "
              f"added by the maneuver {np.percentile([r['est_drift_final_m'] for r in estimated], 95)*100:.1f} cm 95th")
    watched = [r for r in rows if "outcome" in r]
    if watched:
        outcomes = {}
        for r in watched:
            key = r["outcome_reason"] or r["outcome"]
            outcomes[key] = outcomes.get(key, 0) + 1
        distances = [r["min_actor_distance_m"] for r in watched if not math.isnan(r["min_actor_distance_m"])]
        stops = [r["stop_distance_after_detection_m"] for r in watched if not math.isnan(r["stop_distance_after_detection_m"])]
        print(f"  {label} safety: outcomes {outcomes}, collisions {sum(r['actor_collisions'] for r in watched):.0f}, "
              f"min distance to actors {min(distances) if distances else float('nan'):.2f} m, "
              f"longest stop after detection {max(stops) if stops else float('nan'):.2f} m, "
              f"paused {np.mean([r['time_paused_s'] for r in watched]):.1f} s mean, "
              f"runs paused {sum(r['time_paused_s'] > 0 for r in watched)}, "
              f"emergency brakes {sum(r['emergency_brakes'] for r in watched):.0f}, "
              f"blocked replans {sum(r['blocked_replans'] for r in watched):.0f}, "
              f"lowest clearance {min(r['min_clearance'] for r in watched if 'min_clearance' in r):.2f} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--mode", choices=["maneuver", "search"], default="maneuver",
                    help="maneuver: start next to the stall; search: drive the aisle first")
    ap.add_argument("--noise-xy", type=float, default=0.0, help="pose noise std [m]")
    ap.add_argument("--noise-yaw", type=float, default=0.0, help="heading noise std [deg]")
    ap.add_argument("--steer-sweep", type=float, nargs="*", help="max road-wheel angles [deg]")
    ap.add_argument("--detector", choices=["ideal", "lines", "space", "fused"], default=None,
                    help="search mode: slot detector (default: slot_detection.detector in the config)")
    ap.add_argument("--estimator", choices=ESTIMATORS, default=None,
                    help="localization estimator (default: localization.estimator in the config)")
    ap.add_argument("--scenario", choices=list_scenarios(), default=None,
                    help="a named scenario from config/scenarios/, in --mode's flow")
    ap.add_argument("--safety", action="store_true", help="safety monitor on, even with no actors or faults")
    ap.add_argument("--planner-map", choices=["true_lot", "sensed"], default=None,
                    help="search mode: the planner's map (default: planner.map in the config)")
    ap.add_argument("--csv", default=None, help="default out/monte_carlo.csv, or out/monte_carlo_search.csv")
    ap.add_argument("--mcap-failures", default=None, metavar="DIR",
                    help="write one MCAP per failed run, named by seed and failed checks")
    a = ap.parse_args()
    csv_path = a.csv or ("out/monte_carlo_search.csv" if a.mode == "search" else "out/monte_carlo.csv")

    if a.mcap_failures:
        require_backend()   # fail now rather than partway through the batch
    params = load_params()
    if a.detector:
        params["slot_detection"]["detector"] = a.detector
    if a.estimator:
        params["localization"]["estimator"] = a.estimator
    if a.safety:
        params["safety"]["enabled"] = True
    if a.planner_map:
        params["planner"]["map"] = a.planner_map
    scenario = load_scenario_file(a.scenario) if a.scenario else None
    base = VehicleParams.from_yaml()
    rows = []
    limits = a.steer_sweep or [math.degrees(base.max_steer)]
    for lim in limits:
        veh = VehicleParams.from_yaml()
        veh.max_steer = math.radians(lim)
        veh.planner_steer = veh.max_steer * (base.planner_steer / base.max_steer)
        tag = f"lock_{lim:.0f}deg"
        start = len(rows)
        mcap_dir = a.mcap_failures
        if mcap_dir and len(limits) > 1:   # the same seed can fail at several steering limits
            mcap_dir = Path(mcap_dir) / tag
        passed, fails = batch(a.n, veh, params, a.noise_xy, a.noise_yaw, rows=rows, tag=tag,
                              mcap_dir=mcap_dir, mode=a.mode, scenario=scenario)
        print(f"{tag}: {passed}/{a.n} pass ({100*passed/a.n:.1f}%)  "
              f"Rmin {veh.min_turn_radius:.2f} m  failures by check: {fails or 'none'}")
        report(rows[start:], tag)

    write_csv(csv_path, rows)
    print("saved", csv_path)


if __name__ == "__main__":
    main()
