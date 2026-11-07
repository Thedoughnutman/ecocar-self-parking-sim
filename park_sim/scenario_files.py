"""Named safety scenarios from config/scenarios/*.yaml: actors, injected faults, and what should happen.

A file has: name, description, flows (single_stall and/or search), and optionally overrides (merged
into park_params.yaml), actors (see actors.py), events (see scripted_driver.py), obstacles placed around the
target stall (search flow only; see scenario.add_target_obstacles) and expect. `expect` is data for a future
catalog runner; only meets_expect() reads it, and only for the fields it lists there.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from .actors import make_actors
from .scenario import Scenario, load_params, make_perpendicular
from .scripted_driver import events_from_config
from .sensors.base import SensorSuite
from .simulator import RunObserver, RunResult, maneuver_slice, run_closed_loop, search_trial
from .vehicle import VehicleParams

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "config" / "scenarios"
FLOWS = ("single_stall", "search")
STRAIGHT_STALL_CASES = ("neighbor_over_line", "pillar_at_corner", "narrow_aisle_6m", "narrow_stall_2p4m", "cart_in_stall")


@dataclass(frozen=True)
class ScenarioFile:
    name: str
    description: str
    flows: Tuple[str, ...]
    overrides: dict = field(default_factory=dict)
    actors: Tuple[dict, ...] = ()
    events: Tuple[dict, ...] = ()
    expect: dict = field(default_factory=dict)
    obstacles: Tuple[dict, ...] = ()


def list_scenarios() -> List[str]:
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))


def load_scenario_file(name: str) -> ScenarioFile:
    path = SCENARIO_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"no scenario {name!r}; expected one of {list_scenarios()}")
    with open(path) as f:
        doc = yaml.safe_load(f)
    flows = tuple(doc["flows"])
    unknown = [flow for flow in flows if flow not in FLOWS]
    if unknown:
        raise ValueError(f"{name}: unknown flows {unknown}; expected some of {FLOWS}")
    obstacles = tuple(doc.get("obstacles") or ())
    if obstacles and flows != ("search",):
        raise ValueError(f"{name}: obstacles only work in the search flow")
    return ScenarioFile(doc["name"], doc["description"], flows, doc.get("overrides") or {},
                        tuple(doc.get("actors") or ()), tuple(doc.get("events") or ()), doc.get("expect") or {}, obstacles)


def merged(base: dict, changes: dict) -> dict:
    """A deep copy of base with changes merged in, section by section."""
    out = copy.deepcopy(base)
    for key, value in changes.items():
        out[key] = merged(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def run_scenario_file(scenario: ScenarioFile, seed: int, flow: Optional[str] = None,
                      vehicle: Optional[VehicleParams] = None, params: Optional[dict] = None,
                      changes: Optional[dict] = None, observer: Optional[RunObserver] = None,
                      sensors: Optional[SensorSuite] = None, driver=None) -> Tuple[Scenario, RunResult, Dict[str, float]]:
    """One run, with the same seeding as monte_carlo.py. Returns the lot, the result and any values drawn.
    `changes` merge over the file's own overrides. `observer` as in run_search. `sensors`: a suite to log,
    in place of the one the run would make for itself; the search flow only. `driver`: as in run_search, the search
    flow only."""
    flow = flow or scenario.flows[0]
    if flow not in scenario.flows:
        raise ValueError(f"{scenario.name} has no {flow} flow; it has {scenario.flows}")
    vehicle = vehicle if vehicle is not None else VehicleParams.from_yaml()
    cfg = merged(merged(params if params is not None else load_params(), scenario.overrides), changes or {})
    actors = make_actors(scenario.actors, cfg["actors"]) if scenario.actors else None
    events: Sequence = events_from_config(scenario.events)
    if flow == "search":
        return search_trial(vehicle, cfg, seed, events=events, actors=actors, target_obstacles=scenario.obstacles,
                            sensors=sensors, observer=observer, driver=driver)
    sc = make_perpendicular(vehicle, cfg, np.random.default_rng(seed))
    res = run_closed_loop(sc, np.random.default_rng(10_000 + seed), events=events,
                          localization_rng=np.random.default_rng(20_000 + seed), actors=actors, sensor_seed=seed,
                          observer=observer)
    return sc, res, {}


# ---------- how a search run with target obstacles ended ----------
def lowest_clearance(sc: Scenario, res: RunResult) -> float:
    """Lowest true body clearance of the whole run, 0 on contact: every tick of the maneuver, every 5th
    before it. Unlike min_clearance in the metrics, this exists for runs that never got a plan."""
    log = res.log
    start, _ = maneuver_slice(log)
    ticks = list(range(0, start, 5)) + list(range(start, len(log["x"])))
    return min(sc.clearance_exact(log["x"][i], log["y"][i], log["yaw"][i]) for i in ticks)


def tried_target(sc: Scenario, res: RunResult, target_slot: int) -> bool:
    """Whether the car asked for a plan to the target stall, whatever came of it."""
    return any(sc.nearest_stall(*attempt.map_center).slot_id == target_slot for attempt in res.search.plan_attempts)


def final_stall_id(sc: Scenario, res: RunResult) -> Optional[int]:
    """The stall a completed run ended in; None for any other run."""
    if not res.metrics.get("completed"):
        return None
    off = sc.vehicle.center_offset()
    x, y, yaw = res.log["x"][-1], res.log["y"][-1], res.log["yaw"][-1]
    return sc.nearest_stall(x + off * np.cos(yaw), y + off * np.sin(yaw)).slot_id


def case_outcome(sc: Scenario, res: RunResult, target_slot: int) -> str:
    """parked_in_target | parked_elsewhere | no_path (a plan was tried; none found) | no_slot (nothing was
    selected before the end of the aisle) | aborted | fault | incomplete."""
    stall = final_stall_id(sc, res)
    if stall is not None:
        return "parked_in_target" if stall == target_slot else "parked_elsewhere"
    if res.search is not None and res.search.plan_attempts and not res.search.plans:
        return "no_path"
    if res.planned is None:
        return "no_slot"
    reason = res.reason
    if reason.startswith("fault"):
        return "fault"
    return "aborted" if reason.startswith("aborted") else "incomplete"


def meets_expect(expect: dict, outcome: str, clearance: float, res: RunResult) -> bool:
    """A run against its file's expect block: outcome_any_of, contacts: 0, and completed_meets_criteria."""
    if outcome not in expect["outcome_any_of"]:
        return False
    if expect.get("contacts") == 0 and clearance <= 0.0:
        return False
    return res.success or not (expect.get("completed_meets_criteria") and outcome.startswith("parked"))
