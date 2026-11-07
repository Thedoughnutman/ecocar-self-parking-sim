#!/usr/bin/env python3
"""Run the whole flow on a full aisle: drive, search, select, stop, park.

    python3 scripts/run_search_demo.py --seed 1                        # out/search.mcap and out/search.png
    python3 scripts/run_search_demo.py --seed 1 --speed 15 --policy select_nearest_after_passing
    python3 scripts/run_search_demo.py --seed 1 --policy select_first_after --after-x 30
    python3 scripts/run_search_demo.py --seed 1 --detector ideal --policy select_by_index --slot 23
    python3 scripts/run_search_demo.py --seed 1 --policy never_select
    python3 scripts/run_search_demo.py --seed 1 --sensors --lidar      # with sensors in the MCAP
    python3 scripts/run_search_demo.py --seed 1 --scenario pedestrian_walks_into_slot     # random lot and slot, as in Monte Carlo
    python3 scripts/run_search_demo.py --seed 1 --planner-map sensed   # plan from what the sensors saw; /sensed_map in the MCAP

Without the optional MCAP packages it saves only the PNG. Exits 0 if the car parked and met
every criterion (or, with never_select, reached the end of the aisle without maneuvering).
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from park_sim.driver import POLICIES, DriverConfig  # noqa: E402
from park_sim.localization import ESTIMATORS  # noqa: E402
from park_sim.sensors import load_sensor_params, make_suite  # noqa: E402
from park_sim.mcap_export import export_run, require_backend  # noqa: E402
from park_sim.scenario import load_params, make_aisle_lot  # noqa: E402
from park_sim.scenario_files import list_scenarios, load_scenario_file, run_scenario_file  # noqa: E402
from park_sim.simulator import run_search  # noqa: E402
from park_sim.vehicle import VehicleParams  # noqa: E402
from park_sim.viz import plot_search  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--policy", choices=POLICIES, default=None)
    ap.add_argument("--slot", type=int, default=None, help="slot id for select_by_index; track numbers unless --detector ideal")
    ap.add_argument("--after-x", type=float, default=None, help="m, for select_first_after")
    ap.add_argument("--detector", choices=["ideal", "lines", "space", "fused"], default=None)
    ap.add_argument("--estimator", choices=ESTIMATORS, default=None, help="localization estimator")
    ap.add_argument("--speed", type=float, default=None, help="search speed [km/h]")
    ap.add_argument("--occupancy", type=float, default=None, help="chance each stall holds a car, 0-1")
    ap.add_argument("--maneuver", choices=["reverse_in", "head_in"], default=None)
    ap.add_argument("--mcap", default="out/search.mcap")
    ap.add_argument("--png", default="out/search.png")
    ap.add_argument("--sensors", action="store_true", help="run the sensors in config/sensors.yaml and log them")
    ap.add_argument("--lidar", action="store_true", help="also turn on the lidar; implies --sensors")
    ap.add_argument("--scenario", choices=list_scenarios(), default=None, help="actors and faults from config/scenarios/")
    ap.add_argument("--planner-map", choices=["true_lot", "sensed"], default=None,
                    help="the planner's map (default: planner.map in the config)")
    a = ap.parse_args()

    params = load_params()
    if a.maneuver:
        params["scenario"]["maneuver"] = a.maneuver
    if a.detector:
        params["slot_detection"]["detector"] = a.detector
    if a.estimator:
        params["localization"]["estimator"] = a.estimator
    if a.planner_map:
        params["planner"]["map"] = a.planner_map
    cfg = DriverConfig.from_params(params["search"])
    changes = {k: v for k, v in (("policy", a.policy), ("slot_index", a.slot), ("speed_kph", a.speed),
                                                  ("after_x", a.after_x))
               if v is not None}
    cfg = replace(cfg, **changes)

    if a.scenario:
        sc, res, _ = run_scenario_file(load_scenario_file(a.scenario), a.seed, "search", params=params)
    else:
        sc = make_aisle_lot(VehicleParams.from_yaml(), params, np.random.default_rng(a.seed), a.occupancy)
        print("free stalls:", [s.slot_id for s in sc.stalls if not s.occupied])
        suite = None
        if a.sensors or a.lidar:
            sensor_cfg = load_sensor_params()
            sensor_cfg["lidar"]["enabled"] = sensor_cfg["lidar"]["enabled"] or a.lidar
            suite = make_suite(sensor_cfg, a.seed)
        res = run_search(sc, cfg, sensors=suite, sensor_seed=a.seed)

    print(res.summary())
    if res.safety is not None:
        print(f"outcome: {res.safety.outcome} {res.safety.outcome_reason}")
    for tr in res.transitions:
        print(f"  {tr.t:6.2f} s  {tr.from_state.value} -> {tr.to_state.value}  ({tr.reason})")
    for attempt in res.search.plan_attempts:
        print(f"  plan to slot {attempt.slot_id}: {attempt.time_s:.2f} s, "
              f"{'found' if attempt.found else 'not found'}")
    for refinement in res.search.refinements:
        print(f"  {refinement.t:6.2f} s  {refinement.kind}: goal moved {refinement.moved_m*100:.1f} cm, {refinement.moved_deg:.2f} deg")
    for name in ("search_distance_m", "selection_to_stop_s", "confirm_to_parked_s", "total_time_s", "slots_detected",
                 "slot_corner_error_m", "time_to_confirm_s", "replans"):
        print(f"  {name}: {res.metrics[name]:.2f}")

    Path(a.png).parent.mkdir(parents=True, exist_ok=True)
    plot_search(res, a.png)
    print("saved", a.png)
    try:
        require_backend()
    except ImportError as e:
        print("no MCAP:", e)
    else:
        print("saved", export_run(sc, res, a.mcap))

    if cfg.policy == "never_select":
        return 0 if res.reason == "reached the end of the aisle" else 2
    return 0 if res.success else 2


if __name__ == "__main__":
    sys.exit(main())
