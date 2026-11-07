#!/usr/bin/env python3
"""Run the safety gates in config/gates.yaml. Prints every gate's value, threshold and pass/fail, names the
failing seeds, writes out/gates_report.csv (and _runs.csv next to it) and exits 1 if any gate fails.

    python scripts/check_gates.py
    python scripts/check_gates.py --workers 2

Plans end on an expansion budget instead of the 2 s wall clock, so CPU load can't fail a gate. Wall-clock
planning time is reported, not judged. 800 runs: about 8 min on the dev machine at 10 workers, 25 min at 3.
"""
import argparse
import csv
import dataclasses
import math
import multiprocessing
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from park_sim import gates as G  # noqa: E402
from park_sim.scenario import load_params, make_perpendicular  # noqa: E402
from park_sim.scenario_files import load_scenario_file, lowest_clearance, run_scenario_file  # noqa: E402
from park_sim.sensors.base import load_sensor_params  # noqa: E402
from park_sim.simulator import run_closed_loop, search_trial  # noqa: E402
from park_sim.vehicle import VehicleParams  # noqa: E402


def plan_stats(res) -> dict:
    """Wall-clock time and expansions of the plans this run asked for and got back."""
    if res.search is not None:
        kept = [p for _, p in res.search.plans]
        walls = [a.time_s for a in res.search.plan_attempts] + [p.plan_time_s for p in kept[1:]]
        failed = sum(not a.found for a in res.search.plan_attempts)
    else:
        kept = [res.planned] if res.planned is not None else []
        walls = [p.plan_time_s for p in kept]
        failed = 0 if kept else 1
    return {"planning_wall_max_s": max(walls, default=math.nan), "planning_wall_total_s": sum(walls),
            "plans": len(walls), "plans_failed": failed, "expansions_max": max((p.expansions for p in kept), default=0)}


def record_of(run_set, seed, sc, res, label="") -> G.RunRecord:
    m = res.metrics
    return G.RunRecord(run_set=run_set, seed=seed, success=res.success, completed=m.get("completed", 0.0) > 0.5,
                       lowest_clearance_m=lowest_clearance(sc, res), heading_err_deg=m.get("heading_err_deg", math.nan),
                       actor_collisions=m.get("actor_collisions", math.nan), label=label, reason=res.reason,
                       **plan_stats(res))


def one_run(job) -> G.RunRecord:
    try:
        return _one_run(job)
    except StopIteration as e:      # pool.map would quietly take it for the end of its work and lose results
        raise RuntimeError(f"StopIteration in {job}") from e


def _one_run(job) -> G.RunRecord:
    run_set, flow, detector, scenario_name, seed, budget, *occupancy = job
    params = G.with_planning_budget(load_params(), budget)
    veh = VehicleParams.from_yaml()
    if detector:
        params["slot_detection"]["detector"] = detector
    if occupancy and occupancy[0] is not None:
        params["search"]["monte_carlo"]["occupancy_prob"] = [occupancy[0], occupancy[0]]
    if scenario_name:
        sc, res, _ = run_scenario_file(load_scenario_file(scenario_name), seed, flow, veh, params)
        return record_of(run_set, seed, sc, res, f"{scenario_name}/{flow}/{detector}")
    if flow == "search":
        sc, res, _ = search_trial(veh, params, seed)
    else:
        sc = make_perpendicular(veh, params, np.random.default_rng(seed))
        res = run_closed_loop(sc, np.random.default_rng(10_000 + seed), localization_rng=np.random.default_rng(20_000 + seed))
    return record_of(run_set, seed, sc, res)


def jobs_for(cfg: dict) -> list:
    budget = cfg["planning_budget"]
    jobs = []
    for name, run_set in cfg["run_sets"].items():
        jobs += [(name, run_set["flow"], run_set.get("detector"), None, seed, budget, run_set.get("occupancy")) for seed in G.seed_range(run_set["seeds"])]
    safety = cfg["safety"]
    for scenario_name in safety["scenarios"]:
        for flow in load_scenario_file(scenario_name).flows:
            detectors = safety["search_detectors"] if flow == "search" else [safety["single_stall_detector"]]
            jobs += [(G.SAFETY_SET, flow, detector, scenario_name, seed, budget)
                     for detector in detectors for seed in G.seed_range(safety["seeds"])]
    return jobs


def collect_runs(cfg: dict, workers: int) -> list:
    jobs = jobs_for(cfg)
    if workers <= 1:
        return [one_run(j) for j in jobs]
    with multiprocessing.Pool(workers) as pool:
        records = []
        for i, record in enumerate(pool.imap(one_run, jobs), 1):    # job order, so the report doesn't depend on workers
            records.append(record)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} runs", flush=True)
    return records


def write_runs(path, records) -> None:
    columns = [f.name for f in dataclasses.fields(G.RunRecord)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(dataclasses.asdict(r) for r in records)


def main(argv=None, collect=collect_runs) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gates", default=None, help="default config/gates.yaml")
    ap.add_argument("--workers", type=int, default=None, help="default: workers in the gates file")
    ap.add_argument("--csv", default="out/gates_report.csv")
    a = ap.parse_args(argv)

    cfg = G.load_gates(a.gates)
    records = collect(cfg, a.workers if a.workers is not None else cfg["workers"])
    results = G.evaluate(cfg, records, load_params(), load_sensor_params())
    print(G.format_report(results))
    G.write_report(a.csv, results)
    write_runs(Path(a.csv).with_name(Path(a.csv).stem + "_runs.csv"), records)
    print("saved", a.csv)
    return 1 if G.any_failed(results) else 0


if __name__ == "__main__":
    sys.exit(main())
