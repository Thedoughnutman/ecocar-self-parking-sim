"""Safety gates: judges a batch of finished runs and the default config against config/gates.yaml.

Works on plain records, not on the simulator, so it can be tested on small fixed inputs. scripts/check_gates.py
does the running. Every threshold comes from the config; a value that can't be computed (no runs, nan) fails.
"""
from __future__ import annotations

import copy
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml

GATES_PATH = Path(__file__).resolve().parents[1] / "config" / "gates.yaml"
SAFETY_SET = "safety"
REPORT_COLUMNS = ("gate", "value", "threshold", "result", "seeds", "note")
PASS, FAIL, INFO = "PASS", "FAIL", "info"


@dataclass(frozen=True)
class RunRecord:
    """What the gates need from one run. `label` says which case a safety run belongs to."""
    run_set: str
    seed: int
    success: bool = False
    completed: bool = False
    lowest_clearance_m: float = math.nan
    heading_err_deg: float = math.nan
    actor_collisions: float = math.nan
    planning_wall_max_s: float = math.nan
    planning_wall_total_s: float = 0.0
    plans: int = 0
    plans_failed: int = 0
    expansions_max: int = 0
    label: str = ""
    reason: str = ""


@dataclass(frozen=True)
class GateResult:
    gate: str
    value: object
    threshold: str
    result: str         # PASS | FAIL | info
    seeds: str = ""     # the seeds behind a failure, or over the limit
    note: str = ""


def load_gates(path: Optional[str] = None) -> dict:
    with open(path if path is not None else GATES_PATH) as f:
        return yaml.safe_load(f)


def seed_range(seeds: dict) -> range:
    return range(seeds["first"], seeds["first"] + seeds["count"])


def with_planning_budget(params: dict, budget: dict) -> dict:
    """A copy of params where a plan ends on an expansion count, not on the clock. Results then don't
    depend on how busy the machine is. Plans under the budget come out exactly as they would anyway."""
    out = copy.deepcopy(params)
    out["planner"]["max_expansions"] = budget["max_expansions"]
    out["search"]["planning_time_limit_s"] = budget["wall_clock_limit_s"]
    out["safety"]["replan_time_limit_s"] = budget["wall_clock_limit_s"]
    return out


def _seeds(records: Sequence[RunRecord]) -> str:
    return " ".join(f"{r.label}:{r.seed}" if r.label else str(r.seed) for r in records)


def _gate(name: str, value, threshold: str, ok: bool, seeds: Sequence[RunRecord] = (), note: str = "") -> GateResult:
    return GateResult(name, value, threshold, PASS if ok else FAIL, _seeds(seeds), note)


def _pass_rate(name: str, runs: Sequence[RunRecord], minimum: float) -> GateResult:
    rate = sum(r.success for r in runs) / len(runs) if runs else math.nan
    return _gate(f"pass_rate_{name}", rate, f">= {minimum:g}", rate >= minimum, [r for r in runs if not r.success],
                 f"{sum(r.success for r in runs)}/{len(runs)}")


def _unmeasured(runs: Sequence[RunRecord], attribute: str) -> List[RunRecord]:
    return [r for r in runs if math.isnan(getattr(r, attribute))]


def _contacts(name: str, runs: Sequence[RunRecord], maximum: float) -> GateResult:
    touching = [r for r in runs if r.lowest_clearance_m <= 0.0]
    unmeasured = _unmeasured(runs, "lowest_clearance_m")
    ok = bool(runs) and not unmeasured and len(touching) <= maximum
    return _gate(f"body_contacts_{name}", len(touching) if runs else math.nan, f"<= {maximum:g}", ok, touching + unmeasured)


def _clearance(name: str, runs: Sequence[RunRecord], minimum: float) -> GateResult:
    values = [r.lowest_clearance_m for r in runs]
    lowest = min(values) if values and not any(math.isnan(v) for v in values) else math.nan
    bad = [r for r in runs if not r.lowest_clearance_m >= minimum]
    return _gate(f"lowest_clearance_{name}", lowest, f">= {minimum:g} m", lowest >= minimum, bad)


def _heading(name: str, runs: Sequence[RunRecord], maximum: float) -> GateResult:
    done = [r for r in runs if r.completed and not math.isnan(r.heading_err_deg)]
    p95 = float(np.percentile([abs(r.heading_err_deg) for r in done], 95)) if done else math.nan
    return _gate(f"heading_err_p95_{name}", p95, f"<= {maximum:g} deg", p95 <= maximum,
                 [r for r in done if abs(r.heading_err_deg) > maximum], f"{len(done)}/{len(runs)} runs completed")


def _actor_collisions(runs: Sequence[RunRecord], maximum: float) -> GateResult:
    unmeasured = _unmeasured(runs, "actor_collisions")
    hit = [r for r in runs if r.actor_collisions > 0]
    total = sum(r.actor_collisions for r in runs) if runs and not unmeasured else math.nan
    return _gate("actor_collisions", total, f"<= {maximum:g}", total <= maximum, hit + unmeasured, f"{len(runs)} runs")


def forward_cameras(sensor_params: dict, within_deg: float) -> List[str]:
    """Cameras whose axis is nearer the car's +x than within_deg, or that are named as front cameras."""
    found = []
    for cam in sensor_params["line_camera"]["cameras"]:
        yaw = abs((cam["yaw_deg"] + 180.0) % 360.0 - 180.0)
        if yaw < within_deg or "front" in cam["name"].lower():
            found.append(cam["name"])
    return found


def default_config_gates(thresholds: dict, params: dict, sensor_params: dict) -> List[GateResult]:
    maneuver = params["scenario"]["maneuver"]
    wanted = thresholds["default_maneuver"]
    forward = forward_cameras(sensor_params, thresholds["forward_camera_within_deg"])
    return [_gate("default_maneuver", maneuver, wanted, maneuver == wanted),
            _gate("forward_cameras", len(forward), "= 0", not forward, note=" ".join(forward))]


def planning_rows(cfg: dict, records: Sequence[RunRecord]) -> List[GateResult]:
    """Wall-clock planning time, reported and never judged: it depends on what else the machine is doing."""
    rows = []
    for name in [*cfg["run_sets"], SAFETY_SET]:
        runs = [r for r in records if r.run_set == name]
        plans = sum(r.plans for r in runs)
        walls = [r.planning_wall_max_s for r in runs if not math.isnan(r.planning_wall_max_s)]
        mean = sum(r.planning_wall_total_s for r in runs) / plans if plans else math.nan
        rows.append(GateResult(f"planning_wall_max_s_{name}", max(walls, default=math.nan), "", INFO, "",
                               f"mean {mean:.3f} s, {plans} plans, {sum(r.plans_failed for r in runs)} failed, "
                               f"most expansions {max((r.expansions_max for r in runs), default=0)}"))
    return rows


def evaluate(cfg: dict, records: Sequence[RunRecord], params: dict, sensor_params: dict) -> List[GateResult]:
    """Every gate, then the information rows."""
    limits = cfg["thresholds"]
    results: List[GateResult] = []
    for name in cfg["run_sets"]:
        runs = [r for r in records if r.run_set == name]
        results += [_pass_rate(name, runs, limits["pass_rate_min"][name]),
                    _contacts(name, runs, limits["body_contacts_max"]),
                    _clearance(name, runs, limits["lowest_clearance_min_m"]),
                    _heading(name, runs, limits["heading_err_p95_max_deg"])]
    results.append(_actor_collisions([r for r in records if r.run_set == SAFETY_SET], limits["actor_collisions_max"]))
    results += default_config_gates(limits, params, sensor_params)
    return results + planning_rows(cfg, records)


def any_failed(results: Sequence[GateResult]) -> bool:
    return any(r.result == FAIL for r in results)


def format_value(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def format_report(results: Sequence[GateResult]) -> str:
    width = max(len(r.gate) for r in results)
    lines = []
    for r in results:
        line = f"{r.result:4s}  {r.gate:{width}s}  {format_value(r.value):>10s}  {r.threshold:14s}"
        if r.note:
            line += f"  {r.note}"
        if r.seeds:
            line += f"  seeds: {r.seeds}"
        lines.append(line)
    failed = sum(r.result == FAIL for r in results)
    judged = sum(r.result != INFO for r in results)
    lines.append(f"{judged - failed}/{judged} gates pass" + (f", {failed} FAILED" if failed else ""))
    return "\n".join(lines)


def write_report(path, results: Sequence[GateResult]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        w.writeheader()
        for r in results:
            w.writerow({"gate": r.gate, "value": r.value, "threshold": r.threshold, "result": r.result,
                        "seeds": r.seeds, "note": r.note})
