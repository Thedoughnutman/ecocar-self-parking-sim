#!/usr/bin/env python3
"""Run the sim and watch it live in Foxglove, including the planner's search. Docs: docs/live.md

    python scripts/run_live.py --wait                       # full search flow; connect, then press Enter
    python scripts/run_live.py --seed 3 --detector ideal --speed 2
    python scripts/run_live.py --flow single --wait         # one maneuver next to the stall
    python scripts/run_live.py --scenario pedestrian_walks_into_slot --wait
    python scripts/run_live.py --wait --mcap out/live.mcap  # also save the run for replay

Binds to 127.0.0.1 only. After the run it keeps serving, so the result stays on screen, until Ctrl+C. Exits 0 if the
car parked and met every criterion, 2 if not, 130 if stopped during the run, 1 if it couldn't start.
"""
import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from park_sim.live import LiveServer, LiveStream, require_live_backend  # noqa: E402
from park_sim.mcap_export import _backend, export_run  # noqa: E402
from park_sim.scenario import load_params, make_aisle_lot, make_perpendicular  # noqa: E402
from park_sim.scenario_files import list_scenarios, load_scenario_file, merged, run_scenario_file  # noqa: E402
from park_sim.sensors import make_suite  # noqa: E402
from park_sim.simulator import plans_on_sensed_map, run_closed_loop, run_search  # noqa: E402
from park_sim.vehicle import VehicleParams  # noqa: E402

LAYOUT = "config/foxglove_live_layout.json"
FLOWS = {"search": "search", "single": "single_stall"}


def parse_args(argv=None, live=None):
    live = live if live is not None else load_params()["live"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--flow", choices=sorted(FLOWS), default="search", help="full search flow, or a single stall")
    ap.add_argument("--detector", choices=["ideal", "lines", "space", "fused"], default=None)
    ap.add_argument("--scenario", choices=list_scenarios(), default=None, help="actors, faults and obstacles from config/scenarios/")
    ap.add_argument("--speed", type=float, default=live["speed"], help="playback rate, times real time (not the car's speed)")
    ap.add_argument("--port", type=int, default=live["port"], help="0 picks a free one")
    ap.add_argument("--wait", action="store_true", help="wait for Enter before starting, so you can connect first")
    ap.add_argument("--mcap", default="", help="also save the finished run, e.g. out/live.mcap")
    ap.add_argument("--sensors", action="store_true", help="show the sensors when the run wouldn't otherwise run them; not with --scenario")
    ap.add_argument("--planner-map", choices=["true_lot", "sensed"], default=None, help="the planner's map (default: planner.map in the config)")
    ap.add_argument("--exit-when-done", action="store_true", help="stop the server when the run ends, instead of on Ctrl+C")
    a = ap.parse_args(argv)
    if not a.speed > 0.0:
        ap.error("--speed must be above 0")
    return a


def wants_logged_suite(a, params: dict) -> bool:
    """Whether to pass the run a suite, so its sensors can be shown. A search run with a sensor-based detector or the sensed
    map makes this same suite for itself, seeded the same; passing it only logs it. Elsewhere a suite could change what
    the safety monitor hears, so it takes --sensors, and a named scenario doesn't get one."""
    effective = merged(params, load_scenario_file(a.scenario).overrides) if a.scenario else params
    runs_own = effective["slot_detection"]["detector"] != "ideal" or plans_on_sensed_map(effective)
    if a.flow == "search" and runs_own:
        return True
    return a.sensors and not a.scenario


def run(a, params: dict, stream):
    """The same calls, with the same seeding, as run_search_demo.py and run_demo.py. Returns the lot and the result."""
    suite = make_suite(seed=a.seed) if wants_logged_suite(a, params) else None
    if a.scenario:
        sc, res, _ = run_scenario_file(load_scenario_file(a.scenario), a.seed, FLOWS[a.flow], params=params, observer=stream, sensors=suite)
        return sc, res
    veh = VehicleParams.from_yaml()
    if a.flow == "search":
        sc = make_aisle_lot(veh, params, np.random.default_rng(a.seed))
        return sc, run_search(sc, sensors=suite, sensor_seed=a.seed, observer=stream)
    sc = make_perpendicular(veh, params, np.random.default_rng(a.seed))
    return sc, run_closed_loop(sc, sensors=suite, observer=stream)


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def main(argv=None, sleep=time.sleep):
    params = load_params()
    a = parse_args(argv, params["live"])
    if a.detector:
        params["slot_detection"]["detector"] = a.detector
    if a.planner_map:
        params["planner"]["map"] = a.planner_map
    if hasattr(signal, "SIGBREAK"):     # Ctrl+Break, and how a parent process can stop this one on Windows
        signal.signal(signal.SIGBREAK, _raise_interrupt)
    try:
        require_live_backend()
    except ImportError as e:
        print(e)
        return 1
    live = params["live"]
    try:
        server = LiveServer(a.port, live["server_name"], live["message_backlog_size"])
    except Exception as e:      # the SDK's error type for a port in use isn't part of its documented API
        print(f"Could not start the server on port {a.port}: {e}")
        return 1

    stream = LiveStream(server, _backend(), a.speed, sleep=sleep)
    code = 130
    try:
        print(f"Streaming at {server.url}", flush=True)
        print("In Foxglove: Open connection, Foxglove WebSocket, enter that address, Open.")
        print(f"Layout, once: Layouts, Import from file, {LAYOUT}", flush=True)
        if a.wait:
            try:
                input("Press Enter to start once you are connected.\n")
            except EOFError:
                pass
        sc, res = run(a, params, stream)
        stream.finish(res)
        print(res.summary(), flush=True)
        for tr in res.transitions:
            print(f"  {tr.t:6.2f} s  {tr.from_state.value} -> {tr.to_state.value}  ({tr.reason})")
        if a.mcap:
            print("saved", export_run(sc, res, a.mcap), flush=True)
        code = 0 if res.success else 2
        if not a.exit_when_done:
            print("Run finished; still serving. Ctrl+C to stop.", flush=True)
            while True:
                server.service()
                sleep(live["idle_poll_s"])
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
    finally:
        server.stop()
    return code


if __name__ == "__main__":
    sys.exit(main())
