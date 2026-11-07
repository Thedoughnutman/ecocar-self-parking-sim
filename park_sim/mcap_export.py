"""Save a finished run as an MCAP file for scrub-able replay in Foxglove.

Channels, all timestamped from the sim clock (t = 0 at run start):
  /tf       foxglove.FrameTransforms  map -> base_link (rear axle, true) and map -> base_link_est
                                      (estimated pose), every tick
  /lot      foxglove.SceneUpdate      stall lines, parked cars, curbs, target stall, goal; once
  /plan     foxglove.SceneUpdate      planned path by direction, cusps as spheres; once
  /ego      foxglove.SceneUpdate      body box (base_link) and lookahead marker (map), every tick
  /trail    foxglove.SceneUpdate      driven rear-axle path so far, foxglove.trail_publish_every
  /state    JSON                      per-tick signals, see _state_row
  /supervisor JSON                    supervisor state, action, reason and HMI message, every tick
  /metrics  JSON                      RunResult metrics, checks and transitions; once, at the end
Search runs also have:
  /slots    foxglove.SceneUpdate      each detected slot when first seen, the chosen one when selected
and /plan appears when the plan is made, instead of at the start.
Runs with sensors also have, for the sensor kinds present (all in base_link):
  /sensors/camera_footprints  foxglove.SceneUpdate   each camera's ground footprint; once
  /sensors/lines              foxglove.SceneUpdate   detected segments, at each camera's rate
  /sensors/ultrasonic         foxglove.SceneUpdate   cones, with an arc at the current range
  /sensors/ultrasonic_ranges  JSON                   distance per sensor, null if none
  /sensors/lidar              foxglove.PointCloud    returns in the lidar's frame
Runs with the safety monitor on also have:
  /safety        JSON                  sweep, time-to-collision and requests, each tick the monitor ran
  /safety_sweep  foxglove.SceneUpdate  swept footprints (map frame, like /plan), red when blocked
and, with actors:
  /actors        foxglove.SceneUpdate  each actor (map frame, true pose), every tick; gone when it leaves
Runs that plan on the sensed map (planner.map: sensed) also have:
  /sensed_map    foxglove.SceneUpdate  free, occupied and assumed cells and paint seen, in the estimate's frame
                                       (like /plan and /slots, and shown in map), every sensed_map.snapshot_period_s
                                       and at each plan. Cells with nothing drawn are unknown

The MCAP packages are optional: this module imports without them, and they are only
loaded when something is exported. Call require_backend() to fail before a long batch.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np

from .hybrid_astar import PlannedPath
from .scenario import Scenario
from .sensed_map import ASSUMED, FREE, OCCUPIED, SensedMapFrame
from .sensors.lidar2d import LidarMeasurement
from .sensors.line_camera import FALSE_POSITIVE, LineCameraMeasurement
from .sensors.ultrasonic import UltrasonicMeasurement
from .simulator import RunResult, clearance_samples, maneuver_slice

TOPICS = ("/tf", "/lot", "/plan", "/ego", "/trail", "/state", "/supervisor", "/metrics")
SEARCH_TOPICS = TOPICS + ("/slots",)
SAFETY_TOPICS = ("/safety", "/safety_sweep")
ACTOR_TOPIC = "/actors"
SENSED_MAP_TOPIC = "/sensed_map"
MAP_FRAME, BASE_FRAME, EST_FRAME = "map", "base_link", "base_link_est"
STATIC_CHANNEL = {"static": "true"}   # Foxglove has no static flag; this records the intent


@lru_cache(maxsize=1)
def _backend() -> SimpleNamespace:
    try:
        from foxglove_schemas_protobuf.Color_pb2 import Color
        from foxglove_schemas_protobuf.CubePrimitive_pb2 import CubePrimitive
        from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
        from foxglove_schemas_protobuf.FrameTransforms_pb2 import FrameTransforms
        from foxglove_schemas_protobuf.KeyValuePair_pb2 import KeyValuePair
        from foxglove_schemas_protobuf.LinePrimitive_pb2 import LinePrimitive
        from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField
        from foxglove_schemas_protobuf.Point3_pb2 import Point3
        from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
        from foxglove_schemas_protobuf.Pose_pb2 import Pose
        from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
        from foxglove_schemas_protobuf.SceneEntity_pb2 import SceneEntity
        from foxglove_schemas_protobuf.SceneEntityDeletion_pb2 import SceneEntityDeletion
        from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
        from foxglove_schemas_protobuf.SpherePrimitive_pb2 import SpherePrimitive
        from foxglove_schemas_protobuf.TextPrimitive_pb2 import TextPrimitive
        from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
        from google.protobuf.descriptor_pb2 import FileDescriptorSet
        from google.protobuf.timestamp_pb2 import Timestamp
        from mcap.writer import Writer
    except ImportError as e:
        raise ImportError("MCAP export needs the optional packages: "
                          "pip install mcap foxglove-schemas-protobuf") from e
    return SimpleNamespace(**{k: v for k, v in locals().items() if k != "e"})


def require_backend() -> None:
    """Raise ImportError with install instructions if the MCAP packages are missing."""
    _backend()


def failed_checks(res: RunResult) -> List[str]:
    if res.planned is None:
        return ["no_path"]
    return [name for name, ok in res.checks.items() if not ok]


def failure_filename(seed: int, res: RunResult) -> str:
    """e.g. seed0042_heading-gear_changes.mcap"""
    return f"seed{seed:04d}_{'-'.join(failed_checks(res))}.mcap"


def export_run(sc: Scenario, res: RunResult, path) -> Path:
    """Write the run to `path`, creating parent folders. A run with no plan still gets a
    file, showing the lot and the start pose, so planning failures can be inspected too."""
    pb = _backend()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log = res.log if res.log else _start_only_log(sc, res)
    clearance = _running_min_clearance(sc, log, whole_run=res.search is not None)
    times = [int(round(t * 1e9)) for t in log["t"]]
    metrics = _metrics_doc(res)

    with open(path, "wb") as f:
        out = _Channels(pb, f)
        out.add_proto("/tf", pb.FrameTransforms)
        out.add_proto("/lot", pb.SceneUpdate, STATIC_CHANNEL)
        out.add_proto("/plan", pb.SceneUpdate, STATIC_CHANNEL)
        out.add_proto("/ego", pb.SceneUpdate)
        out.add_proto("/trail", pb.SceneUpdate)
        out.add_json("/state", _state_row(log, clearance, 0))
        out.add_json("/supervisor", _supervisor_row(log, res, 0))
        out.add_json("/metrics", metrics)
        if res.search is not None:
            out.add_proto("/slots", pb.SceneUpdate)
        topics = sensor_topics(res)
        if "/sensors/lines" in topics:
            out.add_proto("/sensors/camera_footprints", pb.SceneUpdate, STATIC_CHANNEL)
            out.add_proto("/sensors/lines", pb.SceneUpdate)
            cameras = {m.sensor for m in res.measurements if isinstance(m, LineCameraMeasurement)}
            out.write_proto("/sensors/camera_footprints", times[0], _footprint_update(pb, sc, res.sensor_suite, cameras, times[0]))
        if "/sensors/ultrasonic" in topics:
            out.add_proto("/sensors/ultrasonic", pb.SceneUpdate)
            out.add_json("/sensors/ultrasonic_ranges", _ultrasonic_ranges(
                [m for m in res.measurements if isinstance(m, UltrasonicMeasurement)][:1]))
        if "/sensors/lidar" in topics:
            out.add_proto("/sensors/lidar", pb.PointCloud)
        safety = res.safety
        samples = list(safety.samples) if safety is not None else []
        frames = list(safety.actor_frames) if safety is not None else []
        if safety is not None:
            out.add_json("/safety", _safety_row(samples[0]) if samples else _safety_sample_doc())
            out.add_proto("/safety_sweep", pb.SceneUpdate)
            if frames:
                out.add_proto(ACTOR_TOPIC, pb.SceneUpdate)
        map_frames = list(res.search.sensed_frames) if res.search is not None else []
        if map_frames:
            out.add_proto(SENSED_MAP_TOPIC, pb.SceneUpdate)

        # a search run has no plan until the car has stopped
        writer = RunWriter(pb, sc, res, out, _plan_messages(res, times[0]), _slot_events(res), res.measurements,
                           frames, samples, map_frames)
        writer.start(times[0])
        last = len(times) - 1
        for i, ns in enumerate(times):
            writer.tick(ns, log, i, clearance, last)

        out.write_json("/metrics", times[-1], metrics)
        out.finish()
    return path


class _Feed:
    """A list in time order, read up to a given time. A live run's list may grow between reads."""

    def __init__(self, items: Sequence, time_ns):
        self.items, self.time_ns, self.next = items, time_ns, 0

    def due(self, ns: int) -> list:
        first = self.next
        while self.next < len(self.items) and self.time_ns(self.items[self.next]) <= ns:
            self.next += 1
        return self.items[first:self.next]


def _ns(t: float) -> int:
    return int(round(t * 1e9))


class RunWriter:
    """Writes a run to `out` one tick at a time: everything export_run and the live stream have in common.

    `out` has write_proto(topic, ns, msg) and write_json(topic, ns, doc). `res` needs transitions, search (for
    the slots), sensor_suite and measurements. Plans and slot events are (ns, ...) lists; the rest are the
    run's own lists, read in time order, each item when the tick reaches its time."""

    def __init__(self, pb: SimpleNamespace, sc: Scenario, res, out, plans: Sequence, slot_events: Sequence,
                 measurements: Sequence, actor_frames: Sequence, safety_samples: Sequence, map_frames: Sequence):
        self.pb, self.sc, self.res, self.out = pb, sc, res, out
        self.plans = _Feed(plans, lambda p: p[0])
        self.slot_events = _Feed(slot_events, lambda e: e[0])
        self.measurements = _Feed(measurements, lambda m: _ns(m.t))
        self.actor_frames = _Feed(actor_frames, lambda f: _ns(f[0]))
        self.safety_samples = _Feed(safety_samples, lambda s: _ns(s.t))
        self.map_frames = _Feed(map_frames, lambda f: _ns(f.t))
        self.trail: List[tuple] = []
        self.drawn_actors: set = set()

    def start(self, ns: int) -> None:
        self.out.write_proto("/lot", ns, _lot_update(self.pb, self.sc, ns))

    def tick(self, ns: int, log: Dict[str, Sequence], i: int, clearance: Sequence, last: int, count: int = None) -> None:
        """One tick: sample i of `log` and of `clearance`. count: which tick of the run it is, if not i; the
        trail goes out every trail_publish_every ticks, and on the one numbered `last`."""
        pb, sc, res, out = self.pb, self.sc, self.res, self.out
        fg = sc.cfg["foxglove"]
        count = i if count is None else count
        for _, path in self.plans.due(ns):
            out.write_proto("/plan", ns, _plan_update(pb, sc, path, ns))
        for _, slot, selected in self.slot_events.due(ns):
            out.write_proto("/slots", ns, _slot_update(pb, res, slot, selected, ns))
        _write_measurements(out, pb, sc, res, self.measurements.due(ns), ns)
        xy = (float(log["x"][i]), float(log["y"][i]))
        est = (float(log["est_x"][i]), float(log["est_y"][i]))
        out.write_proto("/tf", ns, _tf(pb, xy, float(log["yaw"][i]), est, float(log["est_yaw"][i]), ns))
        out.write_proto("/ego", ns, _ego_update(pb, sc, log, i, ns))
        trail = self.trail
        if not trail or math.dist(trail[-1], xy) >= fg["trail_min_spacing"]:
            trail.append(xy)
        if count % fg["trail_publish_every"] == 0 or count == last:
            shown = trail if trail[-1] == xy else trail + [xy]
            out.write_proto("/trail", ns, _trail_update(pb, sc, shown, ns))
        out.write_json("/state", ns, _state_row(log, clearance, i))
        out.write_json("/supervisor", ns, _supervisor_row(log, res, i))
        for _, shown in self.actor_frames.due(ns):
            out.write_proto(ACTOR_TOPIC, ns, _actors_update(pb, sc, shown, self.drawn_actors, ns))
        for sample in self.safety_samples.due(ns):
            out.write_json("/safety", ns, _safety_row(sample))
            out.write_proto("/safety_sweep", ns, _sweep_update(pb, sc, sample, ns))
        for frame in self.map_frames.due(ns):
            out.write_proto(SENSED_MAP_TOPIC, ns, _sensed_map_update(pb, sc, frame, ns))


# ---------- MCAP plumbing ----------
class _Channels:
    def __init__(self, pb: SimpleNamespace, stream):
        self.pb = pb
        self.writer = pb.Writer(stream)
        self.writer.start(profile="", library="ecocar_park_sim")
        self.ids: Dict[str, int] = {}
        self.seq: Dict[str, int] = {}
        self.schemas: Dict[str, int] = {}

    def add_proto(self, topic: str, cls, metadata=None) -> None:
        name = cls.DESCRIPTOR.full_name
        if name not in self.schemas:
            self.schemas[name] = self.writer.register_schema(
                name=name, encoding="protobuf", data=_descriptor_set(self.pb, cls.DESCRIPTOR.file))
        self._add(topic, "protobuf", self.schemas[name], metadata)

    def add_json(self, topic: str, sample: dict) -> None:
        schema = self.writer.register_schema(
            name=topic.strip("/"), encoding="jsonschema",
            data=json.dumps(_json_schema(sample)).encode())
        self._add(topic, "json", schema, None)

    def _add(self, topic: str, encoding: str, schema_id: int, metadata) -> None:
        self.ids[topic] = self.writer.register_channel(
            topic=topic, message_encoding=encoding, schema_id=schema_id, metadata=metadata or {})
        self.seq[topic] = 0

    def write_proto(self, topic: str, ns: int, msg) -> None:
        self._write(topic, ns, msg.SerializeToString())

    def write_json(self, topic: str, ns: int, doc: dict) -> None:
        # allow_nan=False: a bare NaN is not valid JSON and Foxglove would drop the message
        self._write(topic, ns, json.dumps(doc, allow_nan=False).encode())

    def _write(self, topic: str, ns: int, data: bytes) -> None:
        self.writer.add_message(channel_id=self.ids[topic], log_time=ns, publish_time=ns,
                                data=data, sequence=self.seq[topic])
        self.seq[topic] += 1

    def finish(self) -> None:
        self.writer.finish()


def _descriptor_set(pb: SimpleNamespace, file_descriptor) -> bytes:
    """Self-contained FileDescriptorSet, dependencies first, as MCAP's protobuf profile expects."""
    fds, seen = pb.FileDescriptorSet(), set()

    def add(fd):
        if fd.name not in seen:
            seen.add(fd.name)
            for dep in fd.dependencies:
                add(dep)
            fd.CopyToProto(fds.file.add())

    add(file_descriptor)
    return fds.SerializeToString()


def _json_schema(sample: dict) -> dict:
    """Schema from an example document, so Foxglove can autocomplete message paths.

    Foxglove's schema parser takes one type per field, so a None sample maps to "number"
    (the only nullable fields are numeric).
    """
    props = {}
    for key, value in sample.items():
        if isinstance(value, dict):
            props[key] = _json_schema(value)
        elif isinstance(value, list):
            props[key] = {"type": "array", "items": _item_schema(value[0] if value else {})}
        elif isinstance(value, bool):
            props[key] = {"type": "boolean"}
        elif isinstance(value, str):
            props[key] = {"type": "string"}
        else:
            props[key] = {"type": "number"}
    return {"type": "object", "properties": props}


def _item_schema(item) -> dict:
    if isinstance(item, list):
        return {"type": "array", "items": _item_schema(item[0] if item else 0.0)}
    return _json_schema(item) if isinstance(item, dict) else {"type": "number"}


# ---------- JSON documents ----------
def _finite(value: float):
    """Float for JSON, with NaN and infinity as null."""
    value = float(value)
    return value if math.isfinite(value) else None


def _state_row(log: Dict[str, np.ndarray], clearance: np.ndarray, i: int) -> dict:
    return {
        "t": float(log["t"][i]),
        "x": float(log["x"][i]),
        "y": float(log["y"][i]),
        "yaw": float(log["yaw"][i]),
        "v": float(log["v"][i]),
        "delta": float(log["delta"][i]),
        "delta_deg": math.degrees(log["delta"][i]),
        "steer_cmd": float(log["steer_cmd"][i]),
        "steer_cmd_deg": math.degrees(log["steer_cmd"][i]),
        "speed_cmd": float(log["speed_cmd"][i]),
        "gear": int(log["gear"][i]),
        "controller_state": str(log["ctrl_state"][i]),
        "segment_index": int(log["seg_i"][i]),
        "distance_to_go": _finite(log["dist_to_go"][i]),            # whole remaining path, m
        "segment_distance_to_go": _finite(log["seg_dist_to_go"][i]),  # what the speed profile uses, m
        "min_clearance": _finite(clearance[i]),                     # lowest so far, m
        "est_x": float(log["est_x"][i]),                            # estimated pose
        "est_y": float(log["est_y"][i]),
        "est_yaw": float(log["est_yaw"][i]),
        "est_error_m": math.hypot(log["est_x"][i] - log["x"][i], log["est_y"][i] - log["y"][i]),
        "est_sigma_xy": float(log["est_sigma_xy"][i]),              # the estimator's own 1-sigma, m
    }


def _supervisor_row(log: Dict[str, np.ndarray], res: RunResult, i: int) -> dict:
    t = float(log["t"][i])
    latest = [tr for tr in res.transitions if tr.t <= t]
    return {
        "t": t,
        "state": str(log["sup_state"][i]),
        "action": str(log["sup_action"][i]),
        "abort_reason": str(log["sup_abort_reason"][i]),
        "transition_reason": latest[-1].reason if latest else "",
        "hmi_message": str(log["sup_hmi"][i]),
        "driver_must_shift": bool(log["sup_must_shift"][i]),
        "request_park": bool(log["sup_request_park"][i]),
    }


def _safety_sample_doc() -> dict:
    """The /safety fields with nothing in them, for the channel's schema when the monitor never ran."""
    return {"t": 0.0, "obstacle_in_path": False, "emergency_brake": False, "ttc_s": None, "gap_m": None,
            "stop_distance_m": 0.0, "horizon_m": 0.0, "brake_request_mps2": 0.0, "sweep": [[0.0]], "hazards": [[0.0]]}


def _safety_row(sample) -> dict:
    return {
        "t": sample.t,
        "obstacle_in_path": bool(sample.obstacle_in_path),
        "emergency_brake": bool(sample.emergency_brake),
        "ttc_s": sample.ttc_s,                      # null: nothing in the sweep, or stopped
        "gap_m": sample.gap_m,                      # along the path to it
        "stop_distance_m": sample.stop_distance_m,
        "horizon_m": sample.horizon_m,
        "brake_request_mps2": sample.brake_request,
        "sweep": sample.sweep.tolist(),             # [x, y, yaw] of each swept footprint
        "hazards": sample.hazards.tolist(),         # [x, y] of the returns in the sweep
    }


def _metrics_doc(res: RunResult) -> dict:
    doc = {
        "success": res.success,
        "reason": res.reason,
        "summary": res.summary(),
        "final_state": res.final_state.value,
        "metrics": {k: _finite(v) for k, v in res.metrics.items()},
        "checks": {k: bool(v) for k, v in res.checks.items()},
        "transitions": [{"t": tr.t, "from": tr.from_state.value, "to": tr.to_state.value,
                         "reason": tr.reason} for tr in res.transitions],
    }
    if res.search is not None:
        doc["slot_id"] = res.search.slot_id
        doc["slots"] = [{"slot_id": c.slot_id, "first_seen_t": c.first_seen_t, "entry_side": c.entry_side,
                         "slot_type": c.slot_type, "confidence": c.confidence} for c in res.search.slots]
        doc["plan_attempts"] = [{"t": a.t, "slot_id": a.slot_id, "time_s": a.time_s, "found": a.found}
                                for a in res.search.plan_attempts]
        doc["refinements"] = [{"t": r.t, "kind": r.kind, "moved_m": r.moved_m, "moved_deg": r.moved_deg}
                              for r in res.search.refinements]
    if res.safety is not None:
        doc["outcome"] = res.safety.outcome
        doc["outcome_reason"] = res.safety.outcome_reason
        doc["blocked_replans"] = [{"t": r.t, "kind": r.kind} for r in res.safety.replans]
    return doc


def _running_min_clearance(sc: Scenario, log: Dict[str, np.ndarray], whole_run: bool = False) -> np.ndarray:
    """Lowest clearance so far, from the samples score() uses, so that the last value
    equals the min_clearance metric exactly. Held between samples. Unless whole_run, the car
    has not moved before the maneuver, so those ticks show the clearance at its start."""
    start, maneuver = (0, log) if whole_run else maneuver_slice(log)
    idx, values = clearance_samples(sc, maneuver)
    clearance = np.full(len(log["x"]), math.inf)
    clearance[np.array(idx) + start] = values
    clearance[:start] = values[0]
    return np.minimum.accumulate(clearance)


def sensor_topics(res: RunResult) -> tuple:
    """The sensor channels this run's measurements fill."""
    kinds = {type(m) for m in res.measurements}
    topics = ()
    if LineCameraMeasurement in kinds:
        topics += ("/sensors/camera_footprints", "/sensors/lines")
    if UltrasonicMeasurement in kinds:
        topics += ("/sensors/ultrasonic", "/sensors/ultrasonic_ranges")
    if LidarMeasurement in kinds:
        topics += ("/sensors/lidar",)
    return topics


def _write_measurements(out: "_Channels", pb: SimpleNamespace, sc: Scenario, res: RunResult,
                        batch: Sequence, ns: int) -> None:
    """One tick's measurements, with each kind's sensors together in one message."""
    lines = [m for m in batch if isinstance(m, LineCameraMeasurement)]
    if lines:
        out.write_proto("/sensors/lines", ns, _lines_update(pb, sc, lines, ns))
    sonar = [m for m in batch if isinstance(m, UltrasonicMeasurement)]
    if sonar:
        out.write_proto("/sensors/ultrasonic", ns, _ultrasonic_update(pb, sc, res, sonar, ns))
        out.write_json("/sensors/ultrasonic_ranges", ns, _ultrasonic_ranges(sonar))
    for m in batch:
        if isinstance(m, LidarMeasurement):
            out.write_proto("/sensors/lidar", ns, _lidar_cloud(pb, sc, res, m, ns))


def _ultrasonic_ranges(batch: Sequence) -> dict:
    return {"t": batch[0].t, **{m.sensor: m.distance for m in batch}} if batch else {}


def _plan_messages(res: RunResult, first_ns: int) -> list:
    """(ns, path) for /plan: once at the start, or whenever the car got a new plan: in a search run,
    or after a replan for an obstacle."""
    if res.search is None or res.planned is None:
        replans = res.safety.plans if res.safety is not None else []
        if not replans:
            return [(first_ns, res.planned)]
        return [(int(round(t * 1e9)), path) for t, path in replans]
    return [(int(round(t * 1e9)), path) for t, path in res.search.plans]


def _slot_events(res: RunResult) -> list:
    """(ns, slot, selected) in time order: each slot when first seen, the chosen one again when selected."""
    if res.search is None:
        return []
    events = [(int(round(c.first_seen_t * 1e9)), c, False) for c in res.search.slots]
    selected = [tr.t for tr in res.transitions if tr.to_state.value == "SLOT_SELECTED"]
    if res.search.slot_id is not None and selected:
        chosen = next(c for c in res.search.slots if c.slot_id == res.search.slot_id)
        events.append((int(round(selected[-1] * 1e9)), chosen, True))
    return sorted(events, key=lambda e: e[0])


def _start_only_log(sc: Scenario, res: RunResult) -> Dict[str, np.ndarray]:
    """One-sample log at the start pose, for a run that never got a plan."""
    log = {k: np.zeros(1) for k in ("t", "v", "delta", "steer_cmd", "speed_cmd", "seg_i",
                                    "sup_must_shift", "sup_request_park", "est_sigma_xy", "est_sigma_yaw")}
    log.update(x=np.array([sc.start.x]), y=np.array([sc.start.y]), yaw=np.array([sc.start.yaw]),
               est_x=np.array([sc.start.x]), est_y=np.array([sc.start.y]), est_yaw=np.array([sc.start.yaw]),
               gear=np.ones(1), ctrl_state=np.array(["no_path"]),
               sup_state=np.array([res.final_state.value]), sup_action=np.array(["none"]),
               sup_abort_reason=np.array(["none"]), sup_hmi=np.array([res.reason]))
    if res.transitions:
        log["t"] = np.array([res.transitions[-1].t])
    for k in ("lookahead_x", "lookahead_y", "dist_to_go", "seg_dist_to_go", "t_maneuver"):
        log[k] = np.full(1, math.nan)
    return log


# ---------- protobuf builders ----------
def _stamp(pb: SimpleNamespace, ns: int):
    ts = pb.Timestamp()
    ts.FromNanoseconds(ns)
    return ts


def _pose(pb: SimpleNamespace, x=0.0, y=0.0, z=0.0, yaw=0.0):
    return pb.Pose(position=pb.Vector3(x=x, y=y, z=z), orientation=_yaw_quat(pb, yaw))


def _yaw_quat(pb: SimpleNamespace, yaw: float):
    return pb.Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def _color(pb: SimpleNamespace, rgba: Sequence[float]):
    return pb.Color(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])


def _line(pb: SimpleNamespace, line_type, points, z: float, thickness: float, rgba):
    return pb.LinePrimitive(
        type=line_type, pose=_pose(pb), thickness=thickness, scale_invariant=False,
        points=[pb.Point3(x=float(px), y=float(py), z=z) for px, py in points],
        color=_color(pb, rgba))


def _rect_cube(pb: SimpleNamespace, poly: np.ndarray, z_bottom: float, height: float, rgba):
    """Box over a geometry.rect() polygon (first edge = length, second edge = width)."""
    e0, e1 = poly[1] - poly[0], poly[2] - poly[1]
    cx, cy = poly.mean(axis=0)
    return pb.CubePrimitive(
        pose=_pose(pb, float(cx), float(cy), z_bottom + height / 2.0, math.atan2(e0[1], e0[0])),
        size=pb.Vector3(x=float(np.hypot(*e0)), y=float(np.hypot(*e1)), z=height),
        color=_color(pb, rgba))


def _sphere(pb: SimpleNamespace, x: float, y: float, z: float, diameter: float, rgba):
    return pb.SpherePrimitive(pose=_pose(pb, x, y, z),
                              size=pb.Vector3(x=diameter, y=diameter, z=diameter),
                              color=_color(pb, rgba))


def _entity(pb: SimpleNamespace, ns: int, frame: str, entity_id: str, static=False, **primitives):
    # lifetime left unset = never expires; frame_locked = follows its frame during playback
    metadata = [pb.KeyValuePair(key="static", value="true")] if static else []
    return pb.SceneEntity(timestamp=_stamp(pb, ns), frame_id=frame, id=entity_id,
                          frame_locked=True, metadata=metadata, **primitives)


def _tf(pb: SimpleNamespace, xy: tuple, yaw: float, est_xy: tuple, est_yaw: float, ns: int):
    return pb.FrameTransforms(transforms=[pb.FrameTransform(
        timestamp=_stamp(pb, ns), parent_frame_id=MAP_FRAME, child_frame_id=child,
        translation=pb.Vector3(x=at[0], y=at[1], z=0.0), rotation=_yaw_quat(pb, heading))
        for child, at, heading in ((BASE_FRAME, xy, yaw), (EST_FRAME, est_xy, est_yaw))])


def _lot_update(pb: SimpleNamespace, sc: Scenario, ns: int):
    fg, col, z = sc.cfg["foxglove"], sc.cfg["foxglove"]["colors"], sc.cfg["foxglove"]["z"]
    line_points = [p for segment in sc.lines for p in segment]
    update = pb.SceneUpdate(entities=[
        _entity(pb, ns, MAP_FRAME, "stall_lines", static=True, lines=[
            _line(pb, pb.LinePrimitive.LINE_LIST, line_points, z["stall_lines"],
                  fg["line_thickness"], col["stall_lines"])]),
        _entity(pb, ns, MAP_FRAME, "parked_cars", static=True, cubes=[
            _rect_cube(pb, car, 0.0, fg["parked_car_height"], col["parked_car"])
            for car in sc.parked_cars]),
        _entity(pb, ns, MAP_FRAME, "curbs", static=True, cubes=[
            _rect_cube(pb, curb, 0.0, fg["curb_height"], col["curb"]) for curb in sc.curbs]),
    ])
    if sc.lot_objects:
        update.entities.append(_entity(pb, ns, MAP_FRAME, "lot_objects", static=True, cubes=[
            _rect_cube(pb, o.polygon, 0.0, fg[f"{o.kind}_height"], col[o.kind]) for o in sc.lot_objects]))
    if sc.goal is not None:     # an aisle lot has no target until the driver chooses one
        goal_box = sc.vehicle.footprint(sc.goal.x, sc.goal.y, sc.goal.yaw)
        update.entities.extend([
            _entity(pb, ns, MAP_FRAME, "target_stall", static=True, cubes=[
                _rect_cube(pb, sc.target_stall, 0.0, z["target_stall"], col["target_stall"])]),
            _entity(pb, ns, MAP_FRAME, "goal_footprint", static=True, lines=[
                _line(pb, pb.LinePrimitive.LINE_LOOP, goal_box, z["goal"],
                      fg["line_thickness"], col["goal"])]),
        ])
    return update


def _slot_update(pb: SimpleNamespace, res: RunResult, slot, selected: bool, ns: int):
    """One slot's box and id label, in the detected or the selected color. Reuses the entity id,
    so selecting a slot recolors it."""
    sc = res.search.scenario
    fg, col, z = sc.cfg["foxglove"], sc.cfg["foxglove"]["colors"], sc.cfg["foxglove"]["z"]
    corners = np.array(slot.corners)
    cx, cy = corners.mean(axis=0)
    color = col["slot_selected"] if selected else col["slot_detected"]
    entity = _entity(pb, ns, MAP_FRAME, f"slot_{slot.slot_id}",
                     cubes=[_rect_cube(pb, corners, z["slots"], fg["slot_height"], color)],
                     texts=[pb.TextPrimitive(
                         pose=_pose(pb, float(cx), float(cy), z["slot_label"]), billboard=True,
                         font_size=fg["slot_label_size"], scale_invariant=False,
                         color=_color(pb, col["slot_label"]), text=str(slot.slot_id))])
    update = pb.SceneUpdate(entities=[entity])
    if selected:
        goal_box = sc.vehicle.footprint(sc.goal.x, sc.goal.y, sc.goal.yaw)
        update.entities.append(_entity(pb, ns, MAP_FRAME, "goal_footprint", lines=[
            _line(pb, pb.LinePrimitive.LINE_LOOP, goal_box, z["goal"],
                  fg["line_thickness"], col["goal"])]))
    return update


def _plan_update(pb: SimpleNamespace, sc: Scenario, path, ns: int):
    if path is None:
        return pb.SceneUpdate()
    fg, col = sc.cfg["foxglove"], sc.cfg["foxglove"]["colors"]
    z = fg["z"]["plan"]
    segments = path.segments()
    lines = [_line(pb, pb.LinePrimitive.LINE_STRIP, zip(path.x[a:b + 1], path.y[a:b + 1]), z,
                   fg["path_thickness"], col["plan_forward"] if d > 0 else col["plan_reverse"])
             for a, b, d in segments]
    cusps = [_sphere(pb, float(path.x[b]), float(path.y[b]), z, fg["cusp_diameter"], col["cusp"])
             for _, b, _ in segments[:-1]]
    return pb.SceneUpdate(entities=[
        _entity(pb, ns, MAP_FRAME, "plan", static=True, lines=lines, spheres=cusps)])


def _ego_update(pb: SimpleNamespace, sc: Scenario, log: Dict[str, np.ndarray], i: int, ns: int):
    fg, col, v = sc.cfg["foxglove"], sc.cfg["foxglove"]["colors"], sc.vehicle
    body = pb.CubePrimitive(
        pose=_pose(pb, v.center_offset(), 0.0, v.height / 2.0),
        size=pb.Vector3(x=v.wheelbase + v.front_overhang + v.rear_overhang, y=v.width, z=v.height),
        color=_color(pb, col["ego"]))
    update = pb.SceneUpdate(entities=[_entity(pb, ns, BASE_FRAME, "body", cubes=[body])])
    lx, ly = float(log["lookahead_x"][i]), float(log["lookahead_y"][i])
    if math.isnan(lx):   # no target while shifting or stopped: remove the previous marker
        update.deletions.add(timestamp=_stamp(pb, ns), id="lookahead",
                             type=pb.SceneEntityDeletion.MATCHING_ID)
    else:
        update.entities.append(_entity(pb, ns, MAP_FRAME, "lookahead", spheres=[
            _sphere(pb, lx, ly, v.height + fg["lookahead_roof_gap"],
                    fg["lookahead_diameter"], col["lookahead"])]))
    return update


def _trail_update(pb: SimpleNamespace, sc: Scenario, points: List[tuple], ns: int):
    fg = sc.cfg["foxglove"]
    return pb.SceneUpdate(entities=[_entity(pb, ns, MAP_FRAME, "trail", lines=[
        _line(pb, pb.LinePrimitive.LINE_STRIP, points, fg["z"]["trail"],
              fg["path_thickness"], fg["colors"]["trail"])])])


# ---------- safety ----------
def _actors_update(pb: SimpleNamespace, sc: Scenario, shown: Sequence[tuple], drawn: set, ns: int):
    """This tick's actors. One that is gone since the last tick gets a deletion. `drawn` tracks them."""
    fg, col = sc.cfg["foxglove"], sc.cfg["foxglove"]["colors"]
    entities = []
    for name, kind, polygon in shown:
        if kind == "vehicle":
            shape = {"cubes": [_rect_cube(pb, polygon, 0.0, fg["actor_vehicle_height"], col["actor_vehicle"])]}
        else:
            cx, cy = polygon.mean(axis=0)
            d = 2.0 * float(np.hypot(*(polygon[0] - polygon.mean(axis=0))))
            height = fg["pedestrian_height"]
            shape = {"spheres": [pb.SpherePrimitive(pose=_pose(pb, float(cx), float(cy), height / 2.0),
                                                    size=pb.Vector3(x=d, y=d, z=height),
                                                    color=_color(pb, col["pedestrian"]))]}
        entities.append(_entity(pb, ns, MAP_FRAME, f"actor_{name}", **shape))
    update = pb.SceneUpdate(entities=entities)
    now = {name for name, _, _ in shown}
    for name in sorted(drawn - now):
        update.deletions.add(timestamp=_stamp(pb, ns), id=f"actor_{name}", type=pb.SceneEntityDeletion.MATCHING_ID)
    drawn.clear()
    drawn.update(now)
    return update


def _sweep_update(pb: SimpleNamespace, sc: Scenario, sample, ns: int):
    fg, col = sc.cfg["foxglove"], sc.cfg["foxglove"]["colors"]
    color = col["sweep_blocked"] if sample.obstacle_in_path else col["sweep_clear"]
    lines = [_line(pb, pb.LinePrimitive.LINE_LOOP, sc.vehicle.footprint(float(x), float(y), float(yaw)),
                   fg["z"]["sweep"], fg["sweep_line_thickness"], color) for x, y, yaw in sample.sweep]
    return pb.SceneUpdate(entities=[_entity(pb, ns, MAP_FRAME, "sweep", lines=lines)])


# ---------- sensed map ----------
def _sensed_map_update(pb: SimpleNamespace, sc: Scenario, frame: SensedMapFrame, ns: int):
    """Each kind of cell as lines one cell wide, joined along x. Reuses the entity id, so each frame replaces the last."""
    fg, col = sc.cfg["foxglove"], sc.cfg["foxglove"]["colors"]
    z = fg["z"]["sensed_map"]
    lines = []
    for kind, color in ((FREE, col["map_free"]), (OCCUPIED, col["map_occupied"]), (ASSUMED, col["map_assumed"])):
        runs = frame.runs(kind)
        if runs:
            points = [p for x0, x1, y in runs for p in ((x0, y), (x1, y))]
            lines.append(_line(pb, pb.LinePrimitive.LINE_LIST, points, z, frame.resolution, color))
    if len(frame.paint):
        lines.append(_line(pb, pb.LinePrimitive.LINE_LIST, frame.paint.reshape(-1, 2), z, fg["paint_line_thickness"], col["map_paint"]))
    return pb.SceneUpdate(entities=[_entity(pb, ns, MAP_FRAME, "sensed_map", lines=lines)])


# ---------- planner search (live only) ----------
def _planner_search_update(pb: SimpleNamespace, sc: Scenario, tree: Sequence[tuple], best: Sequence[tuple],
                           shots: Sequence[tuple], path, ns: int):
    """One displayed frame of a search. tree: points in pairs, one line per pair. best: the branch closest to the
    goal so far, as points. shots: (x, y, collision free) of the analytic shots just tried. path: the result, if any.
    Each entity keeps its id, so each frame replaces the last, and an empty one clears it."""
    fg, col = sc.cfg["foxglove"], sc.cfg["foxglove"]["colors"]
    z, width = fg["z"], fg["search_line_thickness"]
    tree_lines = [_line(pb, pb.LinePrimitive.LINE_LIST, tree, z["search_tree"], width, col["search_tree"])] if tree else []
    best_lines = [_line(pb, pb.LinePrimitive.LINE_STRIP, best, z["search_best"], fg["search_best_thickness"], col["search_best"])] if len(best) > 1 else []
    shot_lines = [_line(pb, pb.LinePrimitive.LINE_STRIP, zip(x, y), z["search_shots"], width,
                        col["search_shot_free"] if free else col["search_shot_blocked"]) for x, y, free in shots]
    path_lines = [] if path is None else [_line(pb, pb.LinePrimitive.LINE_STRIP, zip(path.x, path.y), z["search_path"],
                                                fg["path_thickness"], col["search_path"])]
    return pb.SceneUpdate(entities=[
        _entity(pb, ns, MAP_FRAME, "search_tree", lines=tree_lines),
        _entity(pb, ns, MAP_FRAME, "search_best", lines=best_lines),
        _entity(pb, ns, MAP_FRAME, "search_shots", lines=shot_lines),
        _entity(pb, ns, MAP_FRAME, "search_path", lines=path_lines)])


# ---------- sensors ----------
def _footprint_update(pb: SimpleNamespace, sc: Scenario, suite, cameras: Sequence[str], ns: int):
    fg = sc.cfg["foxglove"]
    return pb.SceneUpdate(entities=[
        _entity(pb, ns, BASE_FRAME, f"footprint_{name}", static=True, lines=[
            _line(pb, pb.LinePrimitive.LINE_LOOP, suite.by_name[name].footprint, fg["z"]["footprint"],
                  fg["footprint_line_thickness"], fg["colors"]["camera_footprint"])])
        for name in sorted(cameras)])


def _lines_update(pb: SimpleNamespace, sc: Scenario, batch: Sequence, ns: int):
    """Each camera's segments replace its previous ones. False positives get their own color for
    review; the perception side couldn't tell them apart."""
    fg, col = sc.cfg["foxglove"], sc.cfg["foxglove"]["colors"]
    entities = []
    for m in batch:
        false = m.truth_line == FALSE_POSITIVE
        groups = [(m.segments[~false], col["camera_segment"]), (m.segments[false], col["false_segment"])]
        entities.append(_entity(pb, ns, BASE_FRAME, f"lines_{m.sensor}", lines=[
            _line(pb, pb.LinePrimitive.LINE_LIST, segs.reshape(-1, 2), fg["z"]["sensor_lines"],
                  fg["sensor_line_thickness"], color)
            for segs, color in groups if len(segs)]))
    return pb.SceneUpdate(entities=entities)


def _ultrasonic_update(pb: SimpleNamespace, sc: Scenario, res: RunResult, batch: Sequence, ns: int):
    fg, col = sc.cfg["foxglove"], sc.cfg["foxglove"]["colors"]
    z, n = fg["z"]["cones"], fg["cone_arc_points"]
    entities = []
    for m in batch:
        sensor = res.sensor_suite.by_name[m.sensor]
        half = math.radians(sensor.cone_deg) / 2.0
        angles = np.linspace(-half, half, n)
        arc = lambda r: sensor.mount.to_vehicle(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))  # noqa: E731
        origin = np.array([[sensor.mount.x, sensor.mount.y]])
        outline = np.vstack([origin, arc(sensor.range_max)])
        lines = [_line(pb, pb.LinePrimitive.LINE_LOOP, outline, z, fg["cone_line_thickness"], col["cone_clear"])]
        if m.distance is not None:
            lines.append(_line(pb, pb.LinePrimitive.LINE_STRIP, arc(m.distance), z, fg["cone_arc_thickness"], col["cone_hit"]))
        entities.append(_entity(pb, ns, BASE_FRAME, f"cone_{m.sensor}", lines=lines))
    return pb.SceneUpdate(entities=entities)


def _lidar_cloud(pb: SimpleNamespace, sc: Scenario, res: RunResult, m: LidarMeasurement, ns: int):
    mount = res.sensor_suite.by_name[m.sensor].mount
    points = m.points()
    return pb.PointCloud(
        timestamp=_stamp(pb, ns), frame_id=BASE_FRAME,
        pose=_pose(pb, mount.x, mount.y, sc.cfg["foxglove"]["z"]["lidar"], mount.yaw), point_stride=8,
        fields=[pb.PackedElementField(name="x", offset=0, type=pb.PackedElementField.FLOAT32),
                pb.PackedElementField(name="y", offset=4, type=pb.PackedElementField.FLOAT32)],
        data=points.astype("<f4").tobytes())
