"""Live Foxglove streaming and the planner's search view. No Foxglove app here: a small WebSocket client stands in."""
import base64
import csv
import json
import math
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from park_sim.hybrid_astar import HybridAStar, PlannerObserver, SearchRecorder, observing  # noqa: E402
from park_sim.live import LIVE_HOST, LIVE_TOPICS, LiveStream, layout_topics  # noqa: E402
from park_sim.mcap_export import SEARCH_TOPICS, RunWriter, export_run  # noqa: E402
from park_sim.scenario import load_params, make_aisle_lot, make_perpendicular  # noqa: E402
from park_sim.scenario_files import load_scenario_file, run_scenario_file  # noqa: E402
from park_sim.sensors import make_suite  # noqa: E402
from park_sim.simulator import RunObserver, run_closed_loop, run_search  # noqa: E402
from park_sim.vehicle import VehicleParams  # noqa: E402

VEHICLE = VehicleParams.from_yaml()
PLANNING_METRICS = {"planning_time_s", "planning_time_max_s"}


def _params(detector="ideal"):
    params = load_params()
    params["slot_detection"]["detector"] = detector
    return params


def _pb():
    pytest.importorskip("foxglove_schemas_protobuf")
    pytest.importorskip("mcap")
    from park_sim.mcap_export import _backend
    return _backend()


def _same_metrics(a, b):
    assert set(a) == set(b)
    for key in set(a) - PLANNING_METRICS:
        assert a[key] == b[key] or (math.isnan(a[key]) and math.isnan(b[key])), key


def _same_result(a, b):
    _same_metrics(a.metrics, b.metrics)
    assert a.checks == b.checks and a.reason == b.reason and a.final_state == b.final_state
    assert [(t.t, t.from_state, t.to_state, t.reason) for t in a.transitions] == [(t.t, t.from_state, t.to_state, t.reason) for t in b.transitions]
    assert (a.planned is None) == (b.planned is None)
    if a.planned is not None:
        assert np.array_equal(a.planned.x, b.planned.x) and np.array_equal(a.planned.y, b.planned.y)
    for key in a.log:
        if a.log[key].dtype.kind == "f":
            assert np.array_equal(a.log[key], b.log[key], equal_nan=True), key
        else:
            assert np.array_equal(a.log[key], b.log[key]), key


class RecordingSink:
    """Stands in for LiveServer, and for an MCAP channel set."""

    def __init__(self):
        self.messages = []
        self.times = []

    def write_proto(self, topic, ns, msg):
        self.messages.append((topic, ns, msg.SerializeToString()))

    def write_json(self, topic, ns, doc):
        self.messages.append((topic, ns, json.dumps(doc, allow_nan=False).encode()))

    def add_json(self, topic, sample):
        pass

    def time(self, ns):
        self.times.append(ns)

    def service(self):
        pass

    def by_topic(self):
        out = defaultdict(list)
        for topic, ns, data in self.messages:
            out[topic].append((ns, data))
        return out


def _live(sink=None, speed=1000.0, sleep=lambda s: None):
    return LiveStream(sink if sink is not None else RecordingSink(), _pb(), speed, sleep=sleep)


# ---------- the planner's observer ----------
class Counting(PlannerObserver):
    def __init__(self):
        self.calls = defaultdict(int)
        self.end = None

    def on_search_start(self, start, target):
        self.calls["start"] += 1

    def on_expand(self, number, index, parent, x, y, g, f, xs, ys):
        self.calls["expand"] += 1
        assert number == self.calls["expand"]

    def on_shot(self, number, x, y, collision_free):
        self.calls["shot"] += 1

    def on_search_end(self, path, expansions, elapsed_s, why):
        self.calls["end"] += 1
        self.end = (path, expansions, why)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_planned_path_is_identical_with_an_observer(seed):
    sc = make_perpendicular(VEHICLE, load_params(), np.random.default_rng(seed))
    plain = HybridAStar(sc).plan()
    seen = Counting()
    watched = HybridAStar(sc, observer=seen).plan()
    assert plain is not None and watched is not None
    for name in ("x", "y", "yaw", "direction"):
        assert np.array_equal(getattr(plain, name), getattr(watched, name)), name
    assert plain.expansions == watched.expansions
    assert seen.calls["expand"] == plain.expansions and seen.calls["start"] == seen.calls["end"] == 1
    assert seen.end[0] is watched and seen.end[1:] == (plain.expansions, "found")
    assert seen.calls["shot"] >= 1


def test_recorder_is_what_a_replay_needs():
    sc = make_perpendicular(VEHICLE, load_params(), np.random.default_rng(1))
    recorder = SearchRecorder()
    path = HybridAStar(sc, observer=recorder).plan()
    record, = recorder.take()
    assert recorder.take() == []
    assert record.path is path and record.why == "found" and len(record.expanded) == path.expansions
    assert record.expanded[0].parent is None
    indices = {n.index for n in record.expanded}
    assert all(n.parent in indices for n in record.expanded[1:])
    assert record.shots[-1].collision_free and not any(s.collision_free for s in record.shots[:-1])


def test_observer_hears_each_way_a_search_can_fail():
    sc = make_perpendicular(VEHICLE, load_params(), np.random.default_rng(1))
    seen = Counting()
    assert HybridAStar(sc, observer=seen).plan(time_limit_s=0.0) is None
    assert seen.end == (None, 0, "time_limit")
    cfg = dict(sc.cfg["planner"], max_expansions=0)
    seen = Counting()
    assert HybridAStar(sc, cfg, observer=seen).plan() is None
    assert seen.end == (None, 0, "exhausted")
    blocked = sc.parked_cars[0].mean(axis=0)
    from park_sim.scenario import Pose
    seen = Counting()
    assert HybridAStar(sc, observer=seen).plan(Pose(float(blocked[0]), float(blocked[1]), 0.0)) is None
    assert seen.end == (None, 0, "start_blocked")


def test_observing_scope_ends_with_the_block():
    sc = make_perpendicular(VEHICLE, load_params(), np.random.default_rng(1))
    seen = Counting()
    with observing(seen):
        assert HybridAStar(sc).observer is seen
    assert HybridAStar(sc).observer is None


def test_no_observer_is_the_default():
    sc = make_perpendicular(VEHICLE, load_params(), np.random.default_rng(1))
    assert HybridAStar(sc).observer is None


# ---------- live mode gives the same run ----------
def _search_pair(seed, detector):
    params = _params(detector)
    suite = (lambda: make_suite(seed=seed)) if detector != "ideal" else (lambda: None)
    plain = run_search(make_aisle_lot(VEHICLE, params, np.random.default_rng(seed)), sensors=suite(), sensor_seed=seed)
    stream = _live()
    sc = make_aisle_lot(VEHICLE, params, np.random.default_rng(seed))
    watched = run_search(sc, sensors=suite(), sensor_seed=seed, observer=stream)
    stream.finish(watched)
    return sc, plain, watched, stream


@pytest.mark.parametrize("seed,detector", [(1, "ideal"), (2, "ideal"), (1, "fused"), (3, "fused")])
def test_live_search_matches_a_normal_run(seed, detector):
    _, plain, watched, stream = _search_pair(seed, detector)
    _same_result(plain, watched)
    assert stream.searches_shown >= 1


def test_live_search_matches_the_way_a_demo_runs_without_a_logged_suite():
    """run_search_demo passes no suite; the live script logs the one the run would make for itself."""
    params = _params("fused")
    plain = run_search(make_aisle_lot(VEHICLE, params, np.random.default_rng(4)), sensor_seed=4)
    stream = _live()
    watched = run_search(make_aisle_lot(VEHICLE, params, np.random.default_rng(4)), sensors=make_suite(seed=4), sensor_seed=4, observer=stream)
    _same_result(plain, watched)


@pytest.mark.parametrize("seed", [1, 2])
def test_live_single_stall_matches_a_normal_run(seed):
    plain = run_closed_loop(make_perpendicular(VEHICLE, load_params(), np.random.default_rng(seed)))
    stream = _live()
    watched = run_closed_loop(make_perpendicular(VEHICLE, load_params(), np.random.default_rng(seed)), observer=stream)
    stream.finish(watched)
    _same_result(plain, watched)


@pytest.mark.parametrize("name,flow", [("pedestrian_walks_into_slot", "search"), ("pedestrian_walks_into_slot", "single_stall")])
def test_live_scenario_matches_a_normal_run(name, flow):
    plain = run_scenario_file(load_scenario_file(name), 2, flow)[1]
    stream = _live()
    sc, watched, _ = run_scenario_file(load_scenario_file(name), 2, flow, observer=stream)
    stream.finish(watched)
    _same_result(plain, watched)
    topics = {t for t, _, _ in stream.sink.messages}
    assert {"/actors", "/safety", "/safety_sweep", "/planner_search"} <= topics


def test_sensed_map_run_matches_and_shows_the_map():
    params = load_params()
    params["planner"]["map"] = "sensed"
    plain = run_search(make_aisle_lot(VEHICLE, params, np.random.default_rng(1)), sensors=make_suite(seed=1), sensor_seed=1)
    stream = _live()
    watched = run_search(make_aisle_lot(VEHICLE, params, np.random.default_rng(1)), sensors=make_suite(seed=1), sensor_seed=1, observer=stream)
    stream.finish(watched)
    _same_result(plain, watched)
    assert "/sensed_map" in {t for t, _, _ in stream.sink.messages}


def test_slow_display_does_not_change_the_run():
    """Time spent displaying, including a sleep in every tick and frame, is outside the planner's limit."""
    params = _params("ideal")
    plain = run_search(make_aisle_lot(VEHICLE, params, np.random.default_rng(1)), sensor_seed=1)
    slept = []

    def slow(seconds):
        slept.append(seconds)
        time.sleep(0.0002)

    stream = _live(speed=1.0, sleep=slow)
    watched = run_search(make_aisle_lot(VEHICLE, params, np.random.default_rng(1)), sensor_seed=1, observer=stream)
    _same_result(plain, watched)
    assert plain.search.plan_attempts[-1].found and watched.search.plan_attempts[-1].time_s < params["search"]["planning_time_limit_s"]
    assert len(slept) > 10


# ---------- the same messages as an MCAP ----------
def _mcap_bytes(path):
    pytest.importorskip("mcap")
    from mcap.reader import make_reader

    out = defaultdict(list)
    with open(path, "rb") as f:
        for _, channel, message in make_reader(f).iter_messages(log_time_order=False):
            out[channel.topic].append((message.log_time, message.data))
    return out


# in a live run the plan appears when it's made and slots are as first seen, not as they ended
EXACT_TOPICS = ("/tf", "/ego", "/trail", "/state", "/supervisor", "/lot", "/sensors/lines", "/sensors/ultrasonic",
                "/sensors/ultrasonic_ranges", "/sensors/camera_footprints", "/actors", "/safety", "/safety_sweep", "/sensed_map")


def _assert_same_as_mcap(sc, res, sink, tmp_path, name):
    path = tmp_path / f"{name}.mcap"
    export_run(sc, res, path)
    mcap, live = _mcap_bytes(path), sink.by_topic()
    assert mcap
    for topic in EXACT_TOPICS:
        assert len(live.get(topic, [])) == len(mcap.get(topic, [])), topic
        assert live.get(topic, []) == mcap.get(topic, []), topic
    return mcap, live


def test_live_messages_are_the_mcap_messages_search(tmp_path):
    sc, _, watched, stream = _search_pair(1, "fused")
    mcap, live = _assert_same_as_mcap(sc, watched, stream.sink, tmp_path, "search")
    assert live["/metrics"][-1] == mcap["/metrics"][-1] or json.loads(live["/metrics"][-1][1])["summary"] == json.loads(mcap["/metrics"][-1][1])["summary"]
    # the same paths in the same order, each at the tick it was first followed
    assert [d for _, d in live["/plan"]] == [d for _, d in mcap["/plan"]]


def test_live_messages_are_the_mcap_messages_single_stall(tmp_path):
    stream = _live()
    sc = make_perpendicular(VEHICLE, load_params(), np.random.default_rng(1))
    res = run_closed_loop(sc, sensors=make_suite(seed=1), observer=stream)
    stream.finish(res)
    _assert_same_as_mcap(sc, res, stream.sink, tmp_path, "single")


def test_live_messages_are_the_mcap_messages_with_actors(tmp_path):
    stream = _live()
    sc, res, _ = run_scenario_file(load_scenario_file("pedestrian_walks_into_slot"), 3, "search", observer=stream, sensors=make_suite(seed=3))
    stream.finish(res)
    _assert_same_as_mcap(sc, res, stream.sink, tmp_path, "actors")


def test_export_and_live_share_one_writer():
    source = (ROOT / "park_sim" / "live.py").read_text()
    assert "RunWriter(" in source and "_tf(" not in source and "_ego_update" not in source and "_state_row" not in source
    assert "RunWriter(" in (ROOT / "park_sim" / "mcap_export.py").read_text()
    assert RunWriter is not None


def test_final_state_clearance_equals_the_metric():
    sc, _, watched, stream = _search_pair(1, "ideal")
    state = stream.sink.by_topic()["/state"]
    assert json.loads(state[-1][1])["min_clearance"] == pytest.approx(watched.metrics["min_clearance"])


def test_every_search_topic_of_the_mcap_is_a_live_topic():
    assert set(SEARCH_TOPICS) <= set(LIVE_TOPICS)


# ---------- planner search display ----------
def _decode_scene(data):
    from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
    return SceneUpdate.FromString(data)


@pytest.fixture(scope="module")
def replay():
    params = _params("ideal")
    stream = _live()
    sc = make_aisle_lot(VEHICLE, params, np.random.default_rng(1))
    res = run_search(sc, sensor_seed=1, observer=stream)
    stream.finish(res)
    return sc, res, stream


def test_search_frames_follow_the_configured_pacing(replay):
    sc, res, stream = replay
    live = sc.cfg["live"]
    frames = stream.sink.by_topic()["/planner_search"]
    expected = math.ceil(res.planned.expansions / live["search_expansions_per_frame"]) + 1
    assert len(frames) == expected


def test_replay_timing_comes_from_config():
    params = _params("ideal")
    slept = []
    stream = _live(speed=2.0, sleep=slept.append)
    sc = make_aisle_lot(VEHICLE, params, np.random.default_rng(1))
    res = run_search(sc, sensor_seed=1, observer=stream)
    live = params["live"]
    frames = math.ceil(res.planned.expansions / live["search_expansions_per_frame"])
    assert slept.count(live["search_frame_s"] / 2.0) == frames
    assert slept.count(live["search_hold_s"] / 2.0) == 1


def test_search_tree_grows_and_ends_on_the_final_path(replay):
    sc, res, stream = replay
    frames = [_decode_scene(d) for _, d in stream.sink.by_topic()["/planner_search"]]
    sizes = [len(next(e for e in f.entities if e.id == "search_tree").lines[0].points) if next(e for e in f.entities if e.id == "search_tree").lines else 0 for f in frames]
    assert sizes == sorted(sizes) and sizes[-1] > 0
    assert {e.id for e in frames[0].entities} == {"search_tree", "search_best", "search_shots", "search_path"}
    final = next(e for e in frames[-1].entities if e.id == "search_path")
    assert final.lines and len(final.lines[0].points) == len(res.planned.x)
    assert not next(e for e in frames[0].entities if e.id == "search_path").lines
    assert any(next(e for e in f.entities if e.id == "search_shots").lines for f in frames)
    assert any(next(e for e in f.entities if e.id == "search_best").lines for f in frames[:-1])


def test_search_is_shown_before_the_plan_it_made(replay):
    _, _, stream = replay
    order = [t for t, _, _ in stream.sink.messages if t in ("/planner_search", "/plan")]
    assert order[0] == "/planner_search" and order.index("/plan") > 0


def test_a_failed_plan_is_still_shown():
    params = _params("ideal")
    params["planner"]["max_expansions"] = 0
    stream = _live()
    sc = make_perpendicular(VEHICLE, params, np.random.default_rng(1))
    res = run_closed_loop(sc, observer=stream)
    stream.finish(res)
    assert res.planned is None
    by = stream.sink.by_topic()
    assert by["/planner_search"]
    last = json.loads(by["/overview"][-1][1])
    assert last["result"] == "FAIL" and last["planner"]["status"] == "no path (exhausted)"
    assert by["/metrics"]


def test_overview_has_what_the_layout_promises(replay):
    sc, res, stream = replay
    docs = [json.loads(d) for _, d in stream.sink.by_topic()["/overview"]]
    for key in ("state", "hmi_message", "gear", "speed_kph", "steering_deg", "distance_to_go_m", "slots_detected", "slot_ids",
                "selected_slot", "transitions", "last_transition", "planner", "result"):
        assert key in docs[0]
    assert docs[0]["result"] == "running"
    assert docs[-1]["result"] == ("PASS" if res.success else "FAIL") and docs[-1]["summary"] == res.summary()
    assert {"D", "R"} <= {d["gear"] for d in docs}
    assert any(d["planner"]["status"] == "searching" for d in docs)
    assert docs[-1]["selected_slot"] == res.search.slot_id
    assert docs[-1]["transitions"][-1]["to"] == res.transitions[-1].to_state.value


def test_pacing_keeps_to_the_requested_rate():
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    stream = LiveStream(RecordingSink(), _pb(), 4.0, sleep=sleep, clock=lambda: now[0])
    sc = make_perpendicular(VEHICLE, load_params(), np.random.default_rng(1))
    res = run_closed_loop(sc, observer=stream)
    live = sc.cfg["live"]
    shown = stream.searches_shown
    replay_s = (math.ceil(res.planned.expansions / live["search_expansions_per_frame"]) * live["search_frame_s"] + live["search_hold_s"]) / 4.0
    assert shown == 1
    assert now[0] == pytest.approx(res.log["t"][-1] / 4.0 + replay_s, abs=0.05)


@pytest.mark.parametrize("speed", [0.0, -1.0])
def test_speed_must_be_positive(speed):
    with pytest.raises(ValueError):
        LiveStream(RecordingSink(), None, speed)


# ---------- a small Foxglove WebSocket client ----------
class WsClient:
    """Just enough of RFC 6455 and the Foxglove WebSocket protocol to subscribe and read."""

    def __init__(self, port):
        self.sock = socket.create_connection((LIVE_HOST, port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET / HTTP/1.1\r\nHost: {LIVE_HOST}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
                           f"Sec-WebSocket-Protocol: foxglove.sdk.v1, foxglove.websocket.v1\r\n\r\n").encode())
        self.buffer = b""
        while b"\r\n\r\n" not in self.buffer:
            data = self.sock.recv(4096)
            if not data:
                raise ConnectionError("closed during the handshake")
            self.buffer += data
        head, self.buffer = self.buffer.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise ConnectionError(head.decode(errors="replace"))
        self.channels, self.subscriptions, self.received, self.times = {}, {}, defaultdict(list), []
        self.server_info = None
        self.next_sub = 1
        self.subscribe_all = True
        self.stop = threading.Event()

    def _read(self, n):
        while len(self.buffer) < n:
            data = self.sock.recv(1 << 16)
            if not data:
                raise ConnectionError("closed")
            self.buffer += data
        out, self.buffer = self.buffer[:n], self.buffer[n:]
        return out

    def _frame(self):
        first, second = self._read(2)
        opcode, length = first & 0x0F, second & 0x7F
        if length == 126:
            length = struct.unpack(">H", self._read(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", self._read(8))[0]
        return opcode, self._read(length)

    def _send(self, opcode, data):
        mask = os.urandom(4)
        head = bytes([0x80 | opcode])
        n = len(data)
        head += bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack(">H", n)
        self.sock.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def subscribe_new(self, channels):
        subs = []
        for channel in channels:
            self.subscriptions[self.next_sub] = channel["topic"]
            subs.append({"id": self.next_sub, "channelId": channel["id"]})
            self.next_sub += 1
        if subs:
            self._send(1, json.dumps({"op": "subscribe", "subscriptions": subs}).encode())

    def pump(self, until=None, timeout=60.0):
        deadline = time.time() + timeout
        while time.time() < deadline and not self.stop.is_set():
            try:
                opcode, data = self._frame()
            except (ConnectionError, OSError):
                return False
            if opcode == 1:
                doc = json.loads(data)
                if doc.get("op") == "serverInfo":
                    self.server_info = doc
                elif doc.get("op") == "advertise":
                    for channel in doc["channels"]:
                        self.channels[channel["topic"]] = channel
                    if self.subscribe_all:
                        self.subscribe_new(doc["channels"])
            elif opcode == 2:
                if data[0] == 1:
                    sub, ns = struct.unpack("<IQ", data[1:13])
                    self.received[self.subscriptions[sub]].append((ns, data[13:]))
                elif data[0] == 2:
                    self.times.append(struct.unpack("<Q", data[1:9])[0])
            elif opcode == 8:
                return False
            elif opcode == 9:
                self._send(10, data)
            if until is not None and until(self):
                return True
        return False

    def close(self):
        self.sock.close()


@pytest.fixture
def server():
    pytest.importorskip("foxglove")
    _pb()
    from park_sim.live import LiveServer
    live = load_params()["live"]
    try:
        s = LiveServer(0, live["server_name"], live["message_backlog_size"])
    except Exception as e:
        pytest.skip(f"couldn't start a server here: {e}")
    yield s
    s.stop()


def _connect(port):
    try:
        return WsClient(port)
    except (ConnectionError, OSError) as e:
        pytest.skip(f"couldn't connect to the local server here: {e}")


def test_client_receives_the_run_live(server):
    from foxglove_schemas_protobuf.FrameTransforms_pb2 import FrameTransforms

    client = _connect(server.port)
    client.pump(lambda c: c.server_info is not None, timeout=10)
    assert "time" in client.server_info["capabilities"]

    params = _params("fused")
    stream = LiveStream(server, _pb(), 50.0)
    sc = make_aisle_lot(VEHICLE, params, np.random.default_rng(1))
    box = {}
    done = threading.Event()

    def run():
        box["res"] = run_search(sc, sensors=make_suite(seed=1), sensor_seed=1, observer=stream)
        stream.finish(box["res"])
        while not done.is_set():    # as the script does once the run is over
            server.service()
            time.sleep(0.01)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        client.pump(lambda c: "/metrics" in c.received and any(json.loads(d)["result"] != "running" for _, d in c.received.get("/overview", [])), timeout=240)
    finally:
        done.set()
        worker.join()
        client.close()

    expected = {"/tf", "/lot", "/plan", "/ego", "/trail", "/state", "/supervisor", "/metrics", "/slots", "/planner_search", "/overview",
                "/sensors/lines", "/sensors/ultrasonic", "/sensors/camera_footprints", "/sensors/ultrasonic_ranges"}
    missing = expected - set(client.received)
    assert not missing, f"nothing on {missing}; got {sorted(client.received)}"
    assert set(client.received) <= set(LIVE_TOPICS)
    assert client.channels["/planner_search"]["schemaName"] == "foxglove.SceneUpdate"
    assert client.channels["/planner_search"]["encoding"] == "protobuf"
    assert client.channels["/state"]["encoding"] == "json"
    assert client.times and client.times == sorted(client.times)
    assert FrameTransforms.FromString(client.received["/tf"][-1][1]).transforms[0].child_frame_id == "base_link"
    assert client.received["/tf"][0][0] < client.received["/tf"][-1][0]
    assert json.loads(client.received["/metrics"][-1][1])["summary"] == box["res"].summary()


def test_late_client_gets_the_latched_topics(server):
    """Someone connecting after the run still sees the lot, plan, search, slots and result."""
    params = _params("ideal")
    stream = LiveStream(server, _pb(), 1000.0, sleep=lambda s: None)
    sc = make_aisle_lot(VEHICLE, params, np.random.default_rng(1))
    res = run_search(sc, sensor_seed=1, observer=stream)
    stream.finish(res)

    client = _connect(server.port)
    stop = threading.Event()

    def keep_serving():
        while not stop.is_set():
            server.service()
            time.sleep(0.01)

    thread = threading.Thread(target=keep_serving)
    thread.start()
    try:
        wanted = {"/lot", "/plan", "/planner_search", "/slots", "/metrics", "/overview"}
        client.pump(lambda c: wanted <= set(c.received), timeout=20)
    finally:
        stop.set()
        thread.join()
        client.close()
    assert wanted <= set(client.received), sorted(client.received)
    assert json.loads(client.received["/overview"][-1][1])["result"] == ("PASS" if res.success else "FAIL")


def _listening_addresses(port):
    tool = shutil.which("netstat")
    if tool is None:
        pytest.skip("no netstat here")
    text = subprocess.run([tool, "-an"], capture_output=True, text=True).stdout
    found = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].upper().startswith("TCP") and "LISTEN" in line.upper():
            local = next((p for p in parts[1:] if p.endswith(f":{port}") or p.endswith(f".{port}")), None)
            if local is not None:
                found.append(local.rsplit(":" if ":" in local else ".", 1)[0])
    return found


def test_server_binds_to_loopback_only(server):
    assert LIVE_HOST == "127.0.0.1"
    assert server.url == f"ws://127.0.0.1:{server.port}"
    addresses = _listening_addresses(server.port)
    assert addresses, "netstat didn't list the server"
    assert set(addresses) == {"127.0.0.1"}


def test_server_is_not_reachable_on_other_interfaces(server):
    try:
        names = {info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    except OSError:
        names = set()
    others = sorted(n for n in names if not n.startswith("127."))
    if not others:
        pytest.skip("this machine has no other interface")
    for address in others:
        with socket.socket() as s:
            s.settimeout(1.0)
            assert s.connect_ex((address, server.port)) != 0, address


def test_host_is_not_a_setting():
    live = load_params()["live"]
    assert "host" not in live and "address" not in live
    source = (ROOT / "scripts" / "run_live.py").read_text()
    assert "--host" not in source


# ---------- the script ----------
def test_script_defaults_come_from_config_and_match_the_prompt():
    import run_live
    a = run_live.parse_args([])
    assert a.port == 8765 and a.speed == 1.0 and a.seed == 1 and a.flow == "search" and not a.wait and a.mcap == ""
    a = run_live.parse_args(["--seed", "3", "--detector", "ideal", "--scenario", "cart_in_stall", "--speed", "2", "--port", "9000", "--wait"])
    assert (a.seed, a.detector, a.scenario, a.speed, a.port, a.wait) == (3, "ideal", "cart_in_stall", 2.0, 9000, True)
    with pytest.raises(SystemExit):
        run_live.parse_args(["--speed", "0"])


@pytest.mark.parametrize("args,detector,expected", [
    ([], "fused", True), ([], "ideal", False), (["--sensors"], "ideal", True), (["--flow", "single"], "fused", False),
    (["--scenario", "pedestrian_walks_into_slot"], "fused", True), (["--scenario", "pedestrian_walks_into_slot", "--sensors"], "ideal", False),
])
def test_only_passes_a_suite_when_it_cannot_change_the_run(args, detector, expected):
    import run_live
    assert run_live.wants_logged_suite(run_live.parse_args(args), _params(detector)) is expected


def _script(*args, **kw):
    return subprocess.Popen([sys.executable, str(ROOT / "scripts" / "run_live.py"), *args], cwd=ROOT, text=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)


def test_script_runs_and_saves_an_mcap(tmp_path):
    pytest.importorskip("foxglove")
    _pb()
    mcap = tmp_path / "live.mcap"
    p = _script("--detector", "ideal", "--port", "0", "--speed", "500", "--exit-when-done", "--mcap", str(mcap))
    out, err = p.communicate(timeout=180)
    assert p.returncode == 0, err
    assert "ws://127.0.0.1:" in out and "Foxglove WebSocket" in out and "foxglove_live_layout.json" in out and "saved" in out
    assert mcap.exists()
    plain = run_search(make_aisle_lot(VEHICLE, _params("ideal"), np.random.default_rng(1)), sensor_seed=1)
    assert plain.summary().split("  t ")[0] in out


def test_wait_holds_the_run_until_enter(tmp_path):
    pytest.importorskip("foxglove")
    _pb()
    p = _script("--detector", "ideal", "--port", "0", "--speed", "500", "--wait", "--exit-when-done")
    try:
        lines = []
        while "Press Enter" not in "".join(lines):
            line = p.stdout.readline()
            assert line, p.stderr.read()
            lines.append(line)
        time.sleep(1.0)
        assert p.poll() is None
        out, err = p.communicate("\n", timeout=180)
    finally:
        p.kill()
    assert p.returncode == 0, err
    assert "PASS" in out


def test_ctrl_c_during_a_run_stops_cleanly(monkeypatch, capsys):
    pytest.importorskip("foxglove")
    _pb()
    import run_live
    calls = []

    def interrupt(seconds):
        calls.append(seconds)
        if len(calls) > 3:
            raise KeyboardInterrupt

    assert run_live.main(["--detector", "ideal", "--port", "0", "--speed", "1"], sleep=interrupt) == 130
    assert "Stopped." in capsys.readouterr().out


def test_ctrl_c_after_a_run_stops_cleanly(capsys):
    pytest.importorskip("foxglove")
    _pb()
    import run_live

    state = {"idle": 0}

    def sleep(seconds):
        if seconds == load_params()["live"]["idle_poll_s"]:
            state["idle"] += 1
            if state["idle"] > 2:
                raise KeyboardInterrupt

    assert run_live.main(["--detector", "ideal", "--port", "0", "--speed", "1000"], sleep=sleep) == 0
    out = capsys.readouterr().out
    assert "still serving" in out and "Stopped." in out


@pytest.mark.skipif(not hasattr(signal, "CTRL_BREAK_EVENT"), reason="Windows only")
def test_break_signal_stops_the_process_and_frees_the_port():
    pytest.importorskip("foxglove")
    _pb()
    with socket.socket() as s:
        s.bind((LIVE_HOST, 0))
        port = s.getsockname()[1]
    p = _script("--detector", "ideal", "--port", str(port), "--speed", "1", creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        assert "Streaming" in p.stdout.readline()
        time.sleep(2.0)
        p.send_signal(signal.CTRL_BREAK_EVENT)
        out, err = p.communicate(timeout=30)
    finally:
        p.kill()
    assert p.returncode == 130, err
    assert "Stopped." in out
    with socket.socket() as s:
        s.bind((LIVE_HOST, port))


def test_port_in_use_is_a_clear_error():
    pytest.importorskip("foxglove")
    _pb()
    import run_live
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE if hasattr(socket, "SO_EXCLUSIVEADDRUSE") else socket.SO_REUSEADDR, 1)
        s.bind((LIVE_HOST, 0))
        s.listen(1)
        assert run_live.main(["--port", str(s.getsockname()[1]), "--detector", "ideal"]) == 1


# ---------- the layout ----------
LAYOUT = json.loads((ROOT / "config" / "foxglove_live_layout.json").read_text())


def _panels(node):
    if isinstance(node, str):
        return [node]
    return _panels(node["first"]) + _panels(node["second"])


def test_layout_is_well_formed():
    assert sorted(_panels(LAYOUT["layout"])) == sorted(LAYOUT["configById"])
    kinds = {p.split("!")[0] for p in LAYOUT["configById"]}
    assert {"3D", "Plot", "StateTransitions", "RawMessages", "Indicator"} == kinds


def test_layout_topics_all_exist():
    topics = layout_topics(LAYOUT)
    assert "/planner_search" in topics and "/overview" in topics
    assert set(topics) <= set(LIVE_TOPICS), set(topics) - set(LIVE_TOPICS)


def test_older_layouts_only_read_live_topics():
    for name in ("foxglove_layout.json", "foxglove_search_layout.json"):
        topics = layout_topics(json.loads((ROOT / "config" / name).read_text()))
        assert set(topics) <= set(LIVE_TOPICS), name


def test_layout_shows_the_search_top_down():
    view = LAYOUT["configById"]["3D!parking"]
    assert view["topics"]["/planner_search"]["visible"] and view["topics"]["/plan"]["visible"]
    assert view["followTf"] == "map" and view["cameraState"]["phi"] == 0
    search = json.loads((ROOT / "config" / "foxglove_search_layout.json").read_text())["configById"]
    assert view["cameraState"] == search["3D!parking"]["cameraState"]


def test_layout_keeps_the_search_layouts_plots():
    search = json.loads((ROOT / "config" / "foxglove_search_layout.json").read_text())["configById"]
    for panel, cfg in search.items():
        if panel.startswith("Plot"):
            assert LAYOUT["configById"][panel]["paths"] == cfg["paths"]


def test_layout_fields_exist_in_the_messages(replay):
    _, _, stream = replay
    by = stream.sink.by_topic()
    docs = {t: json.loads(by[t][0][1]) for t in ("/state", "/supervisor", "/overview")}
    for panel in LAYOUT["configById"].values():
        paths = [p["value"] for p in panel.get("paths", ())] + [panel[k] for k in ("topicPath", "path") if k in panel]
        for path in paths:
            topic, _, field = path.partition(".")
            if field and topic in docs:
                assert field in docs[topic], path
    assert {r["rawValue"] for r in LAYOUT["configById"]["Indicator!result"]["rules"]} == {"PASS", "FAIL", "running"}


# ---------- plain runs are unchanged ----------
def test_plain_baseline_csv_is_byte_identical_to_the_reference(tmp_path):
    out = tmp_path / "base.csv"
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "monte_carlo.py"), "-n", "4", "--csv", str(out)],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    fresh = out.read_text().splitlines()
    assert fresh == (ROOT / "tests" / "data" / "baseline_monte_carlo_200.csv").read_text().splitlines()[:len(fresh)]


@pytest.mark.parametrize("detector", ["ideal", "fused"])
def test_plain_search_csv_matches_the_reference_apart_from_planning_time(detector, tmp_path):
    """The first rows of the search CSVs from before this feature. The full 200 are compared by hand: see PROGRESS.md.
    A column can be empty in these rows and only exist because of a later one, so a missing column reads as empty."""
    out = tmp_path / "search.csv"
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "monte_carlo.py"), "-n", "4", "--mode", "search",
                           "--detector", detector, "--csv", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    with open(out, newline="") as f:
        fresh = list(csv.DictReader(f))
    with open(ROOT / "tests" / "data" / f"search_{detector}_first_4.csv", newline="") as f:
        reference = list(csv.DictReader(f))
    assert len(fresh) == len(reference) == 4
    for new, old in zip(fresh, reference):
        for key in (set(new) | set(old)) - {"planning_time_s", "planning_time_max_s"}:     # both wall clock; the second is on sensed-map runs
            assert new.get(key, "") == old.get(key, ""), (new.get("seed"), key)


# ---------- optional dependency ----------
def test_park_sim_imports_without_foxglove_sdk():
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'foxglove':\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import park_sim, park_sim.simulator, park_sim.hybrid_astar, park_sim.live as live\n"
        "assert 'foxglove' not in sys.modules\n"
        "try:\n"
        "    live.LiveServer(0, 'x', 1)\n"
        "except ImportError as e:\n"
        "    assert 'pip install foxglove-sdk' in str(e)\n"
        "    print('clear-error')\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "clear-error" in done.stdout


def test_observer_defaults_change_nothing():
    """RunObserver's own methods do nothing, and a run with one is the run without it."""
    params = _params("ideal")
    plain = run_search(make_aisle_lot(VEHICLE, params, np.random.default_rng(2)), sensor_seed=2)
    watched = run_search(make_aisle_lot(VEHICLE, params, np.random.default_rng(2)), sensor_seed=2, observer=RunObserver())
    _same_result(plain, watched)
