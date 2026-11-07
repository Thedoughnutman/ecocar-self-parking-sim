#!/usr/bin/env python3
"""Monte Carlo of the harder straight-stall cases, per case and detector, search flow.

    python scripts/straight_stall_cases.py -n 50 --workers 4
    python scripts/straight_stall_cases.py -n 3 --cases cart_in_stall --detectors fused

Writes out/straight_stall_cases.csv (one row per case and detector) and, next to it, _runs.csv (one row per
run). pass_rate counts runs that parked and met every criterion, so a case where refusing is right can
correctly score 0; met_expect counts runs that did what the file's expect block says a safe car does.
--workers: as in sweep_localization.py, planning has a 2 s wall-clock limit, so don't overload the machine.
"""
import argparse
import csv
import multiprocessing
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from park_sim.scenario import load_params  # noqa: E402
from park_sim.scenario_files import (STRAIGHT_STALL_CASES, case_outcome, load_scenario_file, lowest_clearance,  # noqa: E402
                                     meets_expect, run_scenario_file, tried_target)
from park_sim.vehicle import VehicleParams  # noqa: E402

COLUMNS = ("case", "detector", "runs", "pass_rate", "outcomes", "lowest_clearance_m", "contacts", "met_expect",
           "tried_target", "planning_max_s", "contact_seeds", "unexpected_seeds")
RUN_COLUMNS = ("case", "detector", "seed", "success", "outcome", "reason", "lowest_clearance_m", "contact", "met_expect",
               "tried_target", "planning_time_s", "failed_checks", "obstacle_stops", "emergency_brakes", "time_paused_s")


def one_run(job):
    try:
        return _one_run(job)
    except StopIteration as e:      # pool.map would quietly take it for the end of its work and lose results
        raise RuntimeError(f"StopIteration in {job}") from e


def _one_run(job):
    name, detector, seed = job
    scenario = load_scenario_file(name)
    params = load_params()
    params["slot_detection"]["detector"] = detector
    sc, res, drawn = run_scenario_file(scenario, seed, "search", VehicleParams.from_yaml(), params)
    target = drawn["chosen_slot"]
    outcome = case_outcome(sc, res, target)
    clearance = lowest_clearance(sc, res)
    return {"case": name, "detector": detector, "seed": seed, "success": res.success, "outcome": outcome,
            "reason": res.reason, "lowest_clearance_m": clearance, "contact": clearance <= 0.0,
            "met_expect": meets_expect(scenario.expect, outcome, clearance, res),
            "tried_target": tried_target(sc, res, target), "planning_time_s": res.metrics.get("planning_time_s", float("nan")),
            "failed_checks": " ".join(k for k, ok in res.checks.items() if not ok),
            "obstacle_stops": res.metrics.get("obstacle_stops", 0.0), "emergency_brakes": res.metrics.get("emergency_brakes", 0.0),
            "time_paused_s": res.metrics.get("time_paused_s", 0.0)}


def summarize(runs):
    planning = [r["planning_time_s"] for r in runs if r["planning_time_s"] == r["planning_time_s"]]
    return {"case": runs[0]["case"], "detector": runs[0]["detector"], "runs": len(runs),
            "pass_rate": sum(r["success"] for r in runs) / len(runs),
            "outcomes": " ".join(f"{k}:{v}" for k, v in sorted(Counter(r["outcome"] for r in runs).items())),
            "lowest_clearance_m": round(min(r["lowest_clearance_m"] for r in runs), 3),
            "contacts": sum(r["contact"] for r in runs), "met_expect": sum(r["met_expect"] for r in runs),
            "tried_target": sum(r["tried_target"] for r in runs),
            "planning_max_s": round(max(planning, default=float("nan")), 2),
            "contact_seeds": " ".join(str(r["seed"]) for r in runs if r["contact"]),
            "unexpected_seeds": " ".join(str(r["seed"]) for r in runs if not r["met_expect"])}


def write(path, columns, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=50, help="runs per case per detector")
    ap.add_argument("--cases", nargs="*", default=list(STRAIGHT_STALL_CASES))
    ap.add_argument("--detectors", nargs="*", default=["ideal", "fused"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--csv", default="out/straight_stall_cases.csv")
    a = ap.parse_args(argv)

    summary, all_runs = [], []
    pool = multiprocessing.Pool(a.workers) if a.workers > 1 else None
    try:
        for name in a.cases:
            for detector in a.detectors:
                jobs = [(name, detector, seed) for seed in range(a.n)]
                runs = pool.map(one_run, jobs) if pool else [one_run(j) for j in jobs]
                all_runs.extend(runs)
                summary.append(summarize(runs))
                s = summary[-1]
                print(f"{name:20s} {detector:6s} pass {s['pass_rate']*100:5.1f}%  expected {s['met_expect']}/{s['runs']}  "
                      f"contacts {s['contacts']}  lowest {s['lowest_clearance_m']:.2f} m  tried target {s['tried_target']}  "
                      f"plan max {s['planning_max_s']} s  {s['outcomes']}", flush=True)
                write(a.csv, COLUMNS, summary)      # kept current, so a cut-off run still leaves its results
                write(Path(a.csv).with_name(Path(a.csv).stem + "_runs.csv"), RUN_COLUMNS, all_runs)
    finally:
        if pool:
            pool.close()
            pool.join()
    print("saved", a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
