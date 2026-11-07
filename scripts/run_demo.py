#!/usr/bin/env python3
"""Plan and drive one perpendicular reverse-in maneuver, then save the requested outputs.

    python3 scripts/run_demo.py --seed 1 --mcap out/run.mcap     # replay in Foxglove
    python3 scripts/run_demo.py --seed 1 --static                # PNG summary
    python3 scripts/run_demo.py --seed 1 --anim out/park.gif     # animation
    python3 scripts/run_demo.py --seed 1 --mcap out/run.mcap --sensors --lidar
    python3 scripts/run_demo.py --seed 1 --scenario pedestrian_walks_into_slot --mcap out/ped.mcap
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from park_sim.localization import ESTIMATORS  # noqa: E402
from park_sim.sensors import load_sensor_params, make_suite  # noqa: E402
from park_sim.mcap_export import export_run, require_backend  # noqa: E402
from park_sim.scenario import load_params, make_perpendicular  # noqa: E402
from park_sim.scenario_files import list_scenarios, load_scenario_file, run_scenario_file  # noqa: E402
from park_sim.simulator import run_closed_loop  # noqa: E402
from park_sim.vehicle import VehicleParams  # noqa: E402
from park_sim.viz import animate_run, plot_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mcap", default="", help="save for Foxglove replay, e.g. out/run.mcap")
    ap.add_argument("--static", action="store_true", help="save the PNG summary (to --plot)")
    ap.add_argument("--plot", default=None, help="PNG path; implies --static (default out/park_run.png)")
    ap.add_argument("--anim", default="", help="out/park.gif or out/park.mp4")
    ap.add_argument("--maneuver", choices=["reverse_in", "head_in"], default=None)
    ap.add_argument("--sensors", action="store_true", help="run the sensors in config/sensors.yaml and log them")
    ap.add_argument("--lidar", action="store_true", help="also turn on the lidar; implies --sensors")
    ap.add_argument("--estimator", choices=ESTIMATORS, default=None, help="localization estimator")
    ap.add_argument("--scenario", choices=list_scenarios(), default=None, help="actors and faults from config/scenarios/")
    a = ap.parse_args()

    if a.mcap:
        require_backend()   # fail now rather than after the run
    params = load_params()
    if a.estimator:
        params["localization"]["estimator"] = a.estimator
    if a.maneuver:
        params["scenario"]["maneuver"] = a.maneuver
    veh = VehicleParams.from_yaml()
    sc = make_perpendicular(veh, params, np.random.default_rng(a.seed))
    suite = None
    if a.sensors or a.lidar:
        sensor_cfg = load_sensor_params()
        sensor_cfg["lidar"]["enabled"] = sensor_cfg["lidar"]["enabled"] or a.lidar
        suite = make_suite(sensor_cfg, a.seed)
    if a.scenario:
        sc, res, _ = run_scenario_file(load_scenario_file(a.scenario), a.seed, "single_stall", veh, params)
    else:
        res = run_closed_loop(sc, sensors=suite)
    print(res.summary())
    if res.safety is not None:
        print(f"outcome: {res.safety.outcome} {res.safety.outcome_reason}")
    if a.mcap:
        export_run(sc, res, a.mcap)
        print("saved", a.mcap)
    if res.planned is None:
        return 1
    print(f"planned {res.planned.length:.1f} m, {res.planned.expansions} expansions, "
          f"{res.planned.plan_time_s:.2f} s planning")
    if a.static or a.plot:
        plot = a.plot or "out/park_run.png"
        Path(plot).parent.mkdir(parents=True, exist_ok=True)
        plot_run(sc, res, plot)
        print("saved", plot)
    if a.anim:
        animate_run(sc, res, a.anim)
        print("saved", a.anim)
    return 0 if res.success else 2


if __name__ == "__main__":
    sys.exit(main())
