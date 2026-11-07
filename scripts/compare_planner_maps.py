#!/usr/bin/env python3
"""Search Monte Carlo with the planner on the true lot and on the sensed map: same seeds, same detector.

    python scripts/compare_planner_maps.py -n 200
    python scripts/compare_planner_maps.py -n 200 --workers 4     # only while plans stay well under the 2 s limit

Writes out/compare_planner_maps.csv (per run) and out/compare_planner_maps.txt (the table). Clearance and scoring use
the true lot for both maps. A contact is a true clearance of 0 anywhere in the run.
"""
import argparse
import csv
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from park_sim.scenario import load_params  # noqa: E402
from park_sim.simulator import search_trial  # noqa: E402
from park_sim.vehicle import VehicleParams  # noqa: E402

MAPS = ("true_lot", "sensed")


def one_run(job):
    planner_map, detector, seed = job
    params = load_params()
    params["planner"]["map"] = planner_map
    params["slot_detection"]["detector"] = detector
    _, r, _ = search_trial(VehicleParams.from_yaml(), params, seed)
    m, attempts = r.metrics, r.search.plan_attempts
    return {"map": planner_map, "seed": seed, "success": r.success, "parked": bool(m.get("completed", 0.0)), "reason": r.reason,
            "failed_checks": "-".join(k for k, ok in r.checks.items() if not ok),
            "no_path": bool(attempts) and not r.search.plans,
            "plan_attempts": len(attempts), "plans_failed": sum(not a.found for a in attempts),
            "plan_time_mean_s": float(np.mean([a.time_s for a in attempts])) if attempts else math.nan,
            "plan_time_max_s": max((a.time_s for a in attempts), default=math.nan),
            "min_clearance": m.get("min_clearance", math.nan), "lateral_err": m.get("lateral_err", math.nan),
            "heading_err_deg": m.get("heading_err_deg", math.nan), "depth_err": m.get("depth_err", math.nan),
            "gear_changes": m.get("gear_changes", math.nan), "time_s": m.get("time_s", math.nan),
            "replans": sum(x.kind == "replan" for x in r.search.refinements),
            "replans_failed": sum(x.kind == "replan_failed" for x in r.search.refinements)}


def table(rows, n):
    lines = [f"Fused detector, {n} runs per map, true-lot scoring",
             f"{'map':10} {'pass':>9} {'contacts':>8} {'lowest clear':>12} {'no path':>7} {'plans failed':>12} "
             f"{'plan mean':>9} {'plan 95th':>9} {'plan max':>8} {'replans':>7} {'replans failed':>14}"]
    for name in MAPS:
        part = [r for r in rows if r["map"] == name]
        if not part:
            continue
        clear = [r["min_clearance"] for r in part if not math.isnan(r["min_clearance"])]
        times = [r["plan_time_max_s"] for r in part if not math.isnan(r["plan_time_max_s"])]
        means = [r["plan_time_mean_s"] for r in part if not math.isnan(r["plan_time_mean_s"])]
        passed = sum(r["success"] for r in part)
        lines.append(f"{name:10} {passed:>4}/{len(part):<4} {sum(c <= 0.0 for c in clear):>8} {min(clear):>10.2f} m "
                     f"{sum(r['no_path'] for r in part):>7} {sum(r['plans_failed'] for r in part):>12} "
                     f"{np.mean(means):>7.2f} s {np.percentile(times, 95):>7.2f} s {max(times):>6.2f} s "
                     f"{sum(r['replans'] for r in part):>7} {sum(r['replans_failed'] for r in part):>14}")
        failed = {}
        for r in part:
            if not r["success"]:
                key = r["failed_checks"] or r["reason"]
                failed.setdefault(key, []).append(r["seed"])
        lines.append(f"  failing seeds: {failed or 'none'}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--detector", choices=["ideal", "lines", "space", "fused"], default="fused")
    ap.add_argument("--maps", nargs="+", choices=MAPS, default=list(MAPS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--csv", default="out/compare_planner_maps.csv")
    a = ap.parse_args()
    jobs = [(name, a.detector, seed) for name in a.maps for seed in range(a.n)]
    if a.workers > 1:
        with Pool(a.workers) as pool:
            rows = pool.map(one_run, jobs, chunksize=1)
    else:
        rows = [one_run(job) for job in jobs]
    path = Path(a.csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = table(rows, a.n)
    path.with_suffix(".txt").write_text(text + "\n")
    print(text)
    print("saved", path)


if __name__ == "__main__":
    main()
