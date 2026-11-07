#!/usr/bin/env python3
"""Search-mode Monte Carlo (fused detector) across localization error sizes.

Runs every yaw-rate bias with every wheel-speed scale error (odometry), then each GNSS error size
(gnss_ins). Writes one row per cell to a CSV and a text summary of where the pass rate falls below
localization.sweep.pass_rate_floor. The grid defaults come from the config.

    python scripts/sweep_localization.py -n 40 --workers 6
    python scripts/sweep_localization.py -n 2 --bias 0.1 --scale 0.005 --gnss 1

--workers: planning has a 2 s wall-clock limit, so an overloaded machine fails runs that would
otherwise pass. The CSV records the slowest plan of each cell and the summary warns if any plan
came within 75% of the limit; rerun with fewer workers if so.
"""
import argparse
import copy
import csv
import multiprocessing
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from park_sim.scenario import load_params  # noqa: E402
from park_sim.simulator import search_trial  # noqa: E402
from park_sim.vehicle import VehicleParams  # noqa: E402

COLUMNS = ("estimator", "yaw_rate_bias_deg_s", "wheel_scale_error", "gnss_error_factor", "runs", "passed", "pass_rate",
           "failures", "est_err_final_p95_m", "est_err_max_m", "lateral_p95_m", "heading_p95_deg", "planning_max_s")
PLANNING_WARNING_FRACTION = 0.75


def cell_params(base: dict, estimator: str, bias=None, scale=None, gnss=None) -> dict:
    params = copy.deepcopy(base)
    params["slot_detection"]["detector"] = "fused"
    loc = params["localization"]
    loc["estimator"] = estimator
    if bias is not None:
        loc["odometry"]["yaw_rate_bias_deg_s"] = bias
    if scale is not None:
        loc["odometry"]["wheel_scale_error"] = scale
    if gnss is not None:
        loc["gnss_ins"]["position_sigma_m"] *= gnss
        loc["gnss_ins"]["heading_sigma_deg"] *= gnss
    return params


def one_run(job):
    params, seed = job
    _, res, _ = search_trial(VehicleParams.from_yaml(), params, seed)
    failed = [name for name, ok in res.checks.items() if not ok] if res.planned is not None else ["no_path"]
    if not res.checks and res.planned is None and res.reason:
        failed = [res.reason]
    return res.success, failed, res.metrics


def summarize_cell(results, estimator, bias, scale, gnss) -> dict:
    failures = {}
    for _, failed, _ in results:
        for name in failed:
            failures[name] = failures.get(name, 0) + 1
    metrics = [m for _, _, m in results if "est_err_final_m" in m]
    planning = [m["planning_time_s"] for _, _, m in results if "planning_time_s" in m and m["planning_time_s"] == m["planning_time_s"]]

    def p95(key):
        return float(np.percentile([abs(m[key]) for m in metrics], 95)) if metrics else float("nan")

    passed = sum(1 for ok, _, _ in results if ok)
    return {"estimator": estimator, "yaw_rate_bias_deg_s": bias, "wheel_scale_error": scale, "gnss_error_factor": gnss,
            "runs": len(results), "passed": passed, "pass_rate": passed / len(results),
            "failures": " ".join(f"{k}:{v}" for k, v in sorted(failures.items())),
            "est_err_final_p95_m": p95("est_err_final_m"),
            "est_err_max_m": max((m["est_err_max_m"] for m in metrics), default=float("nan")),
            "lateral_p95_m": p95("lateral_err"), "heading_p95_deg": p95("heading_err_deg"),
            "planning_max_s": max(planning, default=float("nan"))}


def summary_text(rows, floor: float, planning_limit: float) -> str:
    def name(r):
        if r["estimator"] == "gnss_ins":
            return f"gnss_ins, error x{r['gnss_error_factor']}"
        return f"odometry, bias {r['yaw_rate_bias_deg_s']} deg/s, scale {r['wheel_scale_error'] * 100:.1f}%"

    below = [r for r in rows if r["pass_rate"] < floor]
    lines = [f"Localization sweep: {len(rows)} cells, {rows[0]['runs']} runs each, search mode, fused detector.",
             f"Cells below {floor * 100:.0f}% pass rate: {len(below)}"]
    lines += [f"  {name(r)}: {r['passed']}/{r['runs']} ({r['pass_rate'] * 100:.0f}%)  {r['failures']}" for r in below]

    odometry = [r for r in rows if r["estimator"] == "odometry"]
    if odometry:
        scales = sorted({r["wheel_scale_error"] for r in odometry})
        lines += ["", "Odometry pass rate (%), yaw-rate bias (deg/s) down, wheel scale error (%) across:",
                  "        " + "".join(f"{s * 100:7.1f}" for s in scales)]
        for bias in sorted({r["yaw_rate_bias_deg_s"] for r in odometry}):
            cells = {r["wheel_scale_error"]: r["pass_rate"] for r in odometry if r["yaw_rate_bias_deg_s"] == bias}
            lines.append(f"{bias:7.2f} " + "".join(f"{cells[s] * 100:7.0f}" for s in scales))
    gnss = [r for r in rows if r["estimator"] == "gnss_ins"]
    if gnss:
        lines += ["", "gnss_ins pass rate (%) by error factor:"]
        lines += [f"  x{r['gnss_error_factor']}: {r['pass_rate'] * 100:.0f}" for r in sorted(gnss, key=lambda r: r["gnss_error_factor"])]

    slowest = max((r["planning_max_s"] for r in rows if r["planning_max_s"] == r["planning_max_s"]), default=float("nan"))
    lines += ["", f"Slowest plan: {slowest:.2f} s (limit {planning_limit:.1f} s)."]
    if slowest >= PLANNING_WARNING_FRACTION * planning_limit:
        lines.append("WARNING: planning came close to its limit; rerun with fewer workers before trusting the failures.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=40, help="runs per cell")
    ap.add_argument("--bias", type=float, nargs="*", default=None, help="yaw-rate biases [deg/s]")
    ap.add_argument("--scale", type=float, nargs="*", default=None, help="wheel scale errors, fractions")
    ap.add_argument("--gnss", type=float, nargs="*", default=None, help="factors on the GNSS error sizes")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes; see the note above")
    ap.add_argument("--csv", default="out/sweep_localization.csv")
    ap.add_argument("--summary", default="out/sweep_localization.txt")
    a = ap.parse_args(argv)

    base = load_params()
    grid = base["localization"]["sweep"]
    biases = grid["yaw_rate_bias_deg_s"] if a.bias is None else a.bias
    scales = grid["wheel_scale_error"] if a.scale is None else a.scale
    factors = grid["gnss_error_factor"] if a.gnss is None else a.gnss
    cells = [("odometry", b, s, None) for b in biases for s in scales] + [("gnss_ins", None, None, f) for f in factors]

    rows = []
    pool = multiprocessing.Pool(a.workers) if a.workers > 1 else None
    try:
        for estimator, bias, scale, gnss in cells:
            params = cell_params(base, estimator, bias, scale, gnss)
            jobs = [(params, seed) for seed in range(a.n)]
            results = pool.map(one_run, jobs) if pool else [one_run(j) for j in jobs]
            rows.append(summarize_cell(results, estimator, bias, scale, gnss))
            r = rows[-1]
            print(f"{estimator:9s} bias {bias} scale {scale} gnss x{gnss}: {r['passed']}/{r['runs']}  "
                  f"plan max {r['planning_max_s']:.2f} s  {r['failures']}", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()

    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    text = summary_text(rows, grid["pass_rate_floor"], base["search"]["planning_time_limit_s"])
    Path(a.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(a.summary).write_text(text)
    print(text)
    print("saved", a.csv, "and", a.summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
