#!/usr/bin/env python3
"""Drive the car yourself, press the button, and let it park. Docs: docs/interactive.md

    python scripts/interactive.py                               # seed 1: sensors only, monitor on
    python scripts/interactive.py --empty                       # no cars: painted lines only
    python scripts/interactive.py --seed 151
    python scripts/interactive.py --scenario cart_in_stall --monitor off
    python scripts/interactive.py --replay out/hmi_sessions/SESSION.json            # watch a session again
    python scripts/interactive.py --replay out/hmi_sessions/SESSION.json --headless # just the result

A seed gives the same lot as that seed in monte_carlo.py --mode search. Every run saves its inputs, and its MCAP, in
interactive.sessions_dir. Exit code: 0 if the last run parked and met every criterion, 2 if not.
"""
import argparse
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from park_sim.hmi import InputLog, KeyState, ManualConfig, ManualDriver, Replay  # noqa: E402
from park_sim.scenario import load_params  # noqa: E402
from park_sim.scenario_files import list_scenarios, load_scenario_file, merged, run_scenario_file  # noqa: E402
from park_sim.sensors import make_suite  # noqa: E402
from park_sim.simulator import plans_on_sensed_map, search_trial  # noqa: E402
from park_sim.vehicle import VehicleParams  # noqa: E402

MONITOR = {"auto": "auto", "on": True, "off": False}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--detector", choices=["ideal", "lines", "space", "fused"], default=None)
    ap.add_argument("--planner-map", choices=["true_lot", "sensed"], default="sensed",
                    help="default sensed: the planner knows only what the sensors saw. true_lot hands it the real lot")
    ap.add_argument("--empty", action="store_true", help="no parked cars: painted lines only")
    ap.add_argument("--scenario", choices=list_scenarios(), default=None, help="obstacles, actors and faults from config/scenarios/")
    ap.add_argument("--monitor", choices=sorted(MONITOR), default="on", help="the safety monitor; default on")
    ap.add_argument("--replay", default=None, help="a saved session's inputs, on the setup it was saved with")
    ap.add_argument("--headless", action="store_true", help="with --replay: no window, print the result")
    ap.add_argument("--no-mcap", action="store_true", help="skip saving an MCAP")
    ap.add_argument("--units", choices=["imperial", "metric"], default=None, help="what the window starts in; the U key switches. Metric matches Foxglove")
    a = ap.parse_args(argv)
    if a.headless and not a.replay:
        ap.error("--headless needs --replay")
    return a


def setup_of(a) -> dict:
    return {"seed": a.seed, "detector": a.detector, "planner_map": a.planner_map, "scenario": a.scenario, "monitor": a.monitor, "empty": a.empty, "units": a.units}


def build_params(setup: dict) -> dict:
    params = load_params()
    params["search"]["timeout_s"] = params["interactive"]["timeout_s"]
    if setup.get("detector"):
        params["slot_detection"]["detector"] = setup["detector"]
    if setup.get("planner_map"):
        params["planner"]["map"] = setup["planner_map"]
    if setup.get("empty"):
        params["search"]["monte_carlo"]["occupancy_prob"] = [0.0, 0.0]
    if setup.get("units"):
        params["interactive"]["units"] = setup["units"]
    if setup.get("scenario"):       # its own overrides go first, so --monitor can still change them
        params = merged(params, load_scenario_file(setup["scenario"]).overrides)
    if setup.get("monitor"):
        params["safety"]["enabled"] = MONITOR[setup["monitor"]]
    return params


def subtitle_of(setup: dict, params: dict) -> str:
    """Which run this is, for the top of the window."""
    monitor = params["safety"]["enabled"]
    monitor = "auto" if monitor == "auto" else ("on" if monitor else "off")
    parts = [f"seed {setup['seed']}", params["slot_detection"]["detector"], "sensed map" if plans_on_sensed_map(params) else "true lot map", f"monitor {monitor}"]
    if setup.get("scenario"):
        parts.append(setup["scenario"].replace("_", " "))
    if setup.get("empty"):
        parts.append("no cars")
    return "   ·   ".join(parts)


def run_once(setup: dict, source, observer=None):
    """One hand-driven run, with the same seeding as Monte Carlo. Returns (lot, result, driver)."""
    params, vehicle = build_params(setup), VehicleParams.from_yaml()
    driver = ManualDriver(ManualConfig.from_params(params["interactive"]), vehicle.max_steer, params["controller"]["dt"], source)
    if observer is not None:
        observer.rejected_of = lambda: driver.rejected
        observer.min_clearance = params["criteria"]["min_clearance"]
        observer.subtitle = subtitle_of(setup, params)
    seed = setup["seed"]
    logged = make_suite(seed=seed) if params["slot_detection"]["detector"] != "ideal" else None
    if setup.get("scenario"):
        changes = {"safety": {"enabled": MONITOR[setup["monitor"]]}} if setup.get("monitor") else None    # the file's overrides go on again in there
        sc, res, _ = run_scenario_file(load_scenario_file(setup["scenario"]), seed, "search", vehicle, params, changes=changes,
                                       observer=observer, sensors=logged, driver=driver)
    else:
        sc, res, _ = search_trial(vehicle, params, seed, observer=observer, sensors=logged, driver=driver)
    return sc, res, driver


def session_stem(params: dict, setup: dict) -> Path:
    return Path(params["interactive"]["sessions_dir"]) / f"{time.strftime('%Y%m%d_%H%M%S')}_seed{setup['seed']}"


def save_mcap(sc, res, stem: Path) -> None:
    from park_sim.mcap_export import export_run
    try:
        print("saved", export_run(sc, res, stem.with_suffix(".mcap")), flush=True)
    except ImportError as e:
        print(f"No MCAP: {e}")


def report(res) -> None:
    print(res.summary(), flush=True)
    for tr in res.transitions:
        print(f"  {tr.t:6.2f} s  {tr.from_state.value} -> {tr.to_state.value}  ({tr.reason})")


def replay(a) -> int:
    source = Replay(a.replay)
    setup = source.setup
    if a.headless:
        _, res, _ = run_once(setup, source)
        report(res)
        return 0 if res.success else 2
    window, keys = open_window(setup)
    sc, res, _ = run_once(setup, lambda: replace(source(), stop=source_stopped(keys, source)), window)
    report(res)
    if not window.closed and window.next_action != "quit":
        window.next_action = None
        window.show_result(res, build_params(setup)["criteria"])
        window.wait_for_next()
    return 0 if res.success else 2


def source_stopped(keys: KeyState, source: Replay) -> bool:
    return keys.stop_requested or source.finished


def open_window(setup: dict):
    try:
        from park_sim.hmi_view import HmiWindow
    except ImportError:
        sys.exit("The window needs pygame-ce: pip install pygame-ce")
    keys = KeyState()
    window = HmiWindow(build_params(setup)["interactive"], keys)
    window.pump()
    return window, keys


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.replay:
        return replay(a)
    setup = setup_of(a)
    window, keys = open_window(setup)
    code = 2
    while True:
        keys.__init__()     # nothing carries over from the last run
        log = InputLog(keys.read)
        stem = session_stem(build_params(setup), setup)
        try:
            sc, res, _ = run_once(setup, log, window)
        except Exception:   # a crash still leaves the inputs, and a replay of them can bring it back
            print("saved", log.save(stem.with_suffix(".json"), setup), flush=True)
            traceback.print_exc()
            return 1
        print("saved", log.save(stem.with_suffix(".json"), setup), flush=True)
        report(res)
        if not a.no_mcap:
            save_mcap(sc, res, stem)
        code = 0 if res.success else 2
        if window.closed:
            return code
        if window.next_action is None:      # it can be set already: R, N and Esc all work mid-run
            window.show_result(res, build_params(setup)["criteria"])
        next_action = window.wait_for_next()
        if next_action == "quit":
            return code
        if next_action == "next":
            setup["seed"] += 1


if __name__ == "__main__":
    sys.exit(main())
