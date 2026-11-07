"""Stream a run to Foxglove over a local WebSocket, as it runs, with the planner's search shown as it went.

Every topic export_run writes goes out through the same RunWriter, so the existing layouts work. On top of those:
  /planner_search  foxglove.SceneUpdate  explored tree, current best branch, shots tried, final path (planner's frame)
  /overview        JSON                  one place for the state, HMI message, gear, speed, steering, slots and result

The search is never slowed. The planner runs at full speed with a SearchRecorder attached, and when it has returned, the
run waits here while the search is replayed to the display. So the pacing sits outside the planner's time limit.

Pacing happens between ticks, and nothing here feeds back into the run. Times on the messages are the run's clock, which
stands still during a replay. The foxglove SDK is optional: this module imports without it, and only LiveServer needs it.
"""
from __future__ import annotations

import json
import math
import threading
import time
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence

from .hybrid_astar import SearchRecord, SearchRecorder
from .mcap_export import (ACTOR_TOPIC, SAFETY_TOPICS, SEARCH_TOPICS, SENSED_MAP_TOPIC, RunWriter, _backend, _descriptor_set,
                          _footprint_update, _json_schema, _metrics_doc, _ns, _planner_search_update)
from .scenario import Scenario
from .sensors.line_camera import LineCamera
from .simulator import RunObserver, RunResult, RunTick, _corners_in_map

LIVE_HOST = "127.0.0.1"     # never configurable: this has no authentication and shouldn't be reachable from a network
PLANNER_SEARCH_TOPIC, OVERVIEW_TOPIC = "/planner_search", "/overview"
SENSOR_TOPICS = ("/sensors/camera_footprints", "/sensors/lines", "/sensors/ultrasonic", "/sensors/ultrasonic_ranges", "/sensors/lidar")
LIVE_TOPICS = SEARCH_TOPICS + SAFETY_TOPICS + SENSOR_TOPICS + (ACTOR_TOPIC, SENSED_MAP_TOPIC, PLANNER_SEARCH_TOPIC, OVERVIEW_TOPIC)
# A client can connect at any time. These change rarely, so someone subscribing late gets the latest again
LATCHED_TOPICS = ("/lot", "/plan", "/sensors/camera_footprints", SENSED_MAP_TOPIC, PLANNER_SEARCH_TOPIC, OVERVIEW_TOPIC, "/metrics")
CUMULATIVE_TOPICS = ("/slots",)     # each message is about one slot, so late subscribers need them all
_GEARS = {1: "D", -1: "R"}


def require_live_backend() -> None:
    """Raise ImportError with install instructions if the live packages are missing."""
    _backend()
    try:
        import foxglove  # noqa: F401
    except ImportError as e:
        raise ImportError("Live streaming needs the optional package: pip install foxglove-sdk") from e


class LiveServer:
    """A Foxglove WebSocket server on 127.0.0.1, written to the way _Channels writes an MCAP. Channels are created when
    first written to, with the same schemas as the MCAP's."""

    def __init__(self, port: int, name: str, backlog: int):
        require_live_backend()
        import foxglove
        from foxglove.websocket import Capability, ServerListener

        self.pb = _backend()
        self._foxglove = foxglove
        self._context = foxglove.Context()
        self._channels: Dict[str, object] = {}
        self._latched: Dict[str, List[bytes]] = {}
        self._schemas: Dict[str, object] = {}
        self._last_ns = 0
        self._new_subscriptions: set = set()
        self._lock = threading.Lock()
        outer = self

        class Listener(ServerListener):
            def on_subscribe(self, client, channel):
                with outer._lock:
                    outer._new_subscriptions.add(channel.topic)

        self._server = foxglove.start_server(name=name, host=LIVE_HOST, port=port, context=self._context,
                                             capabilities=[Capability.Time], server_listener=Listener(),
                                             message_backlog_size=backlog)
        self.port = self._server.port

    @property
    def url(self) -> str:
        return f"ws://{LIVE_HOST}:{self.port}"

    def add_json(self, topic: str, sample: dict) -> None:
        schema = self._foxglove.Schema(name=topic.strip("/"), encoding="jsonschema", data=json.dumps(_json_schema(sample)).encode())
        self._channels[topic] = self._foxglove.Channel(topic, schema=schema, message_encoding="json", context=self._context)

    def write_proto(self, topic: str, ns: int, msg) -> None:
        if topic not in self._channels:
            descriptor = msg.DESCRIPTOR
            schema = self._foxglove.Schema(name=descriptor.full_name, encoding="protobuf",
                                           data=_descriptor_set(self.pb, descriptor.file))
            self._channels[topic] = self._foxglove.Channel(topic, schema=schema, message_encoding="protobuf", context=self._context)
        self._send(topic, ns, msg.SerializeToString())

    def write_json(self, topic: str, ns: int, doc: dict) -> None:
        if topic not in self._channels:
            self.add_json(topic, doc)
        self._send(topic, ns, json.dumps(doc, allow_nan=False).encode())

    def _send(self, topic: str, ns: int, data: bytes) -> None:
        self._channels[topic].log(data, log_time=ns)
        self._last_ns = ns
        if topic in CUMULATIVE_TOPICS:
            self._latched.setdefault(topic, []).append(data)
        elif topic in LATCHED_TOPICS:
            self._latched[topic] = [data]

    def time(self, ns: int) -> None:
        self._last_ns = ns
        self._server.broadcast_time(ns)

    def service(self) -> None:
        """Sends what newly subscribed topics have missed. Everyone subscribed gets it again, which is harmless."""
        with self._lock:
            topics, self._new_subscriptions = self._new_subscriptions, set()
        for topic in topics:
            for data in self._latched.get(topic, ()):
                self._channels[topic].log(data, log_time=self._last_ns)

    def stop(self) -> None:
        self._server.stop()
        for channel in self._channels.values():
            channel.close()
        self._channels.clear()


class LiveStream(RunObserver):
    """Watches a run and writes it to `sink` (write_proto, write_json, time, service), at `speed` times real time."""

    def __init__(self, sink, pb: SimpleNamespace, speed: float, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.perf_counter):
        if not speed > 0.0:
            raise ValueError(f"speed must be above 0, got {speed}")
        self.sink, self.pb, self.speed, self._sleep, self._clock = sink, pb, speed, sleep, clock
        self.planner = SearchRecorder()
        self.sc: Optional[Scenario] = None
        self.writer: Optional[RunWriter] = None
        self._suite = None
        self._search_flow = False
        self._facade = SimpleNamespace(transitions=(), search=None, sensor_suite=None, measurements=())
        self._plans: List[tuple] = []
        self._slot_events: List[tuple] = []
        self._seen_slots: set = set()
        self._selected: Optional[int] = None
        self._seen_transitions = 0
        self._last_path = None
        self._count = 0
        self._origin: Optional[float] = None
        self._clearance = math.inf
        self._before_maneuver: Optional[float] = None
        self._samples = 0
        self._last_doc: Optional[dict] = None
        self._ns = 0
        self._search_status = {"status": "idle", "expansions": 0, "shots": 0, "planning_time_s": None}
        self.searches_shown = 0

    # ---------- the run ----------
    def on_run_start(self, sc, suite, measurements, safety, map_frames, search_flow) -> None:
        self.sc, self._suite, self._search_flow = sc, suite, search_flow
        self._facade.sensor_suite, self._facade.measurements = suite, measurements
        self._facade.search = SimpleNamespace(scenario=sc)
        self.sink.time(0)
        self.writer = RunWriter(self.pb, sc, self._facade, self.sink, self._plans, self._slot_events, measurements,
                                safety.actor_frames if safety is not None else (),
                                safety.samples if safety is not None else (), map_frames)
        self.writer.start(0)
        cameras = {s.name for s in suite.sensors if isinstance(s, LineCamera)} if suite is not None else set()
        if cameras:
            self.sink.write_proto("/sensors/camera_footprints", 0, _footprint_update(self.pb, sc, suite, cameras, 0))

    def on_tick(self, tick: RunTick) -> None:
        ns = _ns(tick.t)
        self._ns = ns
        self.sink.service()
        self.sink.time(ns)
        self._replay_searches(ns)
        self._facade.transitions = tick.transitions
        if tick.path is not None and tick.path is not self._last_path:
            self._last_path = tick.path
            self._plans.append((ns, tick.path))
        self._note_slots(tick, ns)
        clearance = self._clearance_now(tick)
        # the trail goes out on `last`: a live run doesn't know which that is until it gets there
        self.writer.tick(ns, tick.row, 0, (clearance,), self._count if tick.final else -1, self._count)
        self._count += 1
        self._last_doc = self._overview(tick)
        if tick.final or self._count % self.sc.cfg["live"]["overview_every_ticks"] == 1:
            self.sink.write_json(OVERVIEW_TOPIC, ns, self._last_doc)
        self._pace(tick.t)

    def finish(self, res: RunResult) -> None:
        """After the run: any search not yet shown (a run can end on a failed plan), then the result."""
        self._replay_searches(self._ns)
        self.sink.write_json("/metrics", self._ns, _metrics_doc(res))
        doc = dict(self._last_doc) if self._last_doc is not None else self._empty_overview()
        doc.update(result="PASS" if res.success else "FAIL", summary=res.summary(), final_state=res.final_state.value)
        doc["planner"] = dict(self._search_status)
        self.sink.write_json(OVERVIEW_TOPIC, self._ns, doc)
        self.sink.service()

    # ---------- what the run has done ----------
    def _note_slots(self, tick: RunTick, ns: int) -> None:
        for slot in tick.slots:
            if slot.slot_id not in self._seen_slots:
                self._seen_slots.add(slot.slot_id)
                self._slot_events.append((ns, slot, False))
        selections = sum(1 for tr in tick.transitions[self._seen_transitions:] if tr.to_state.value == "SLOT_SELECTED")
        self._seen_transitions = len(tick.transitions)
        if selections and tick.selected_slot is not None:
            chosen = next((c for c in tick.slots if c.slot_id == tick.selected_slot), None)
            if chosen is not None:
                center = _corners_in_map(chosen, tick.true_pose, tick.slot_frame).mean(axis=0)
                self._facade.search = SimpleNamespace(scenario=self.sc.with_target(self.sc.nearest_stall(*center)))
                self._slot_events.append((ns, chosen, True))
        self._selected = tick.selected_slot

    def _clearance_now(self, tick: RunTick) -> float:
        """The same samples as mcap_export._running_min_clearance: every 5th from where its slice starts, and the last.
        Before a single-stall maneuver the car hasn't moved, so that value is the clearance at its start."""
        row = tick.row
        x, y, yaw = row["x"][0], row["y"][0], row["yaw"][0]
        if not self._search_flow and math.isnan(row["t_maneuver"][0]) and not tick.final:
            if self._before_maneuver is None:
                self._before_maneuver = self.sc.clearance_exact(x, y, yaw)
            return self._before_maneuver
        if self._samples % 5 == 0 or tick.final:
            self._clearance = min(self._clearance, self.sc.clearance_exact(x, y, yaw))
        self._samples += 1
        return self._clearance

    # ---------- /overview ----------
    def _empty_overview(self) -> dict:
        return {"t": 0.0, "state": "", "hmi_message": "", "gear": "", "speed_kph": 0.0, "steering_deg": 0.0, "distance_to_go_m": None,
                "min_clearance_m": None, "slots_detected": 0, "slot_ids": [0], "selected_slot": None, "last_transition": "",
                "transitions": [{"t": 0.0, "from": "", "to": "", "reason": ""}], "planner": dict(self._search_status),
                "result": "running", "summary": "", "final_state": ""}

    def _overview(self, tick: RunTick) -> dict:
        row, keep = tick.row, self.sc.cfg["live"]["overview_transitions"]
        latest = [{"t": tr.t, "from": tr.from_state.value, "to": tr.to_state.value, "reason": tr.reason} for tr in tick.transitions[-keep:]]
        last = latest[-1] if latest else None
        dist = float(row["dist_to_go"][0])
        return {"t": tick.t, "state": str(row["sup_state"][0]), "hmi_message": str(row["sup_hmi"][0]),
                "gear": _GEARS.get(int(row["gear"][0]), "?"), "speed_kph": float(row["v"][0]) * 3.6,
                "steering_deg": math.degrees(row["delta"][0]),
                "distance_to_go_m": dist if math.isfinite(dist) else None,
                "min_clearance_m": self._clearance if math.isfinite(self._clearance) else None,
                "slots_detected": len(tick.slots), "slot_ids": [c.slot_id for c in tick.slots], "selected_slot": tick.selected_slot,
                "last_transition": f"{last['t']:.2f} s {last['from']} -> {last['to']} ({last['reason']})" if last else "",
                "transitions": latest, "planner": dict(self._search_status), "result": "running", "summary": "", "final_state": ""}

    # ---------- pacing ----------
    def _pace(self, t: float) -> None:
        now = self._clock()
        if self._origin is None:
            self._origin = now - t / self.speed
        due = self._origin + t / self.speed
        if now < due:
            self._sleep(due - now)
        elif now - due > self.sc.cfg["live"]["max_lag_s"]:
            self._origin = now - t / self.speed

    # ---------- the search ----------
    def _replay_searches(self, ns: int) -> None:
        records = self.planner.take()
        for record in records:
            self._replay(record, ns)
        if records:
            self._origin = None     # the run stood still for all that: carry on from here

    def _replay(self, record: SearchRecord, ns: int) -> None:
        live = self.sc.cfg["live"]
        per_frame = live["search_expansions_per_frame"]
        tree, nodes = [], {}
        best, best_h = None, math.inf
        n_frames = math.ceil(len(record.expanded) / per_frame)
        shots_done, n_shots = 0, 0
        self.searches_shown += 1
        for frame in range(n_frames):
            chunk = record.expanded[frame * per_frame:(frame + 1) * per_frame]
            for node in chunk:
                nodes[node.index] = node
                if node.parent is not None:
                    points = [(float(x), float(y)) for x, y in zip(node.xs, node.ys)]
                    previous = nodes[node.parent]
                    points.insert(0, (previous.x, previous.y))
                    tree.extend(p for a, b in zip(points, points[1:]) for p in (a, b))
                if node.f - node.g < best_h:
                    best, best_h = node, node.f - node.g
            last_number = chunk[-1].number
            first_shot = shots_done
            while shots_done < len(record.shots) and record.shots[shots_done].number <= last_number:
                shots_done += 1
            n_shots = shots_done
            shown = record.shots[first_shot:shots_done][-live["search_shots_shown"]:]
            self._search_status = {"status": "searching", "expansions": last_number, "shots": n_shots, "planning_time_s": None}
            self.sink.write_proto(PLANNER_SEARCH_TOPIC, ns, _planner_search_update(
                self.pb, self.sc, tree, _branch(nodes, best), [(s.x, s.y, s.collision_free) for s in shown], None, ns))
            self._write_search_status(ns)
            self._sleep(live["search_frame_s"] / self.speed)
        found = record.path is not None
        self._search_status = {"status": "found" if found else f"no path ({record.why})", "expansions": record.expansions,
                               "shots": len(record.shots), "planning_time_s": record.elapsed_s}
        self.sink.write_proto(PLANNER_SEARCH_TOPIC, ns, _planner_search_update(
            self.pb, self.sc, tree, _branch(nodes, best) if not found else [], [], record.path, ns))
        self._write_search_status(ns)
        self._sleep(live["search_hold_s"] / self.speed)

    def _write_search_status(self, ns: int) -> None:
        doc = dict(self._last_doc) if self._last_doc is not None else self._empty_overview()
        doc["planner"] = dict(self._search_status)
        self.sink.write_json(OVERVIEW_TOPIC, ns, doc)


def _branch(nodes: Dict[int, object], node) -> List[tuple]:
    """Start to `node`, along the arcs that reached it."""
    if node is None:
        return []
    chain = []
    while node is not None:
        chain.append(node)
        node = nodes.get(node.parent) if node.parent is not None else None
    chain.reverse()
    points = [(chain[0].x, chain[0].y)]
    for n in chain[1:]:
        points.extend((float(x), float(y)) for x, y in zip(n.xs, n.ys))
    return points


def layout_topics(layout: dict) -> Sequence[str]:
    """Every topic a Foxglove layout file reads."""
    topics = set()
    for panel in layout["configById"].values():
        topics.update(panel.get("topics", {}))
        for path in panel.get("paths", ()):
            topics.add(path["value"].split(".", 1)[0])
        for key in ("topicPath", "path"):
            if panel.get(key):
                topics.add(panel[key].split(".", 1)[0])
    return sorted(topics)
