"""MCAP export: reopen files with the mcap reader and check what Foxglove would receive."""
import copy
import importlib.util
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from park_sim.hybrid_astar import PlannedPath  # noqa: E402
from park_sim.mcap_export import TOPICS, export_run, failure_filename  # noqa: E402
from park_sim.scenario import load_params, make_perpendicular  # noqa: E402
from park_sim.simulator import RunResult, run_closed_loop  # noqa: E402
from park_sim.vehicle import VehicleParams  # noqa: E402

SEED = 1   # reverse-in with an initial D -> R shift, so every controller state is visited


def read_mcap(path):
    """Messages per topic in file order, decoded using only what the file itself embeds."""
    pytest.importorskip("mcap")
    from google.protobuf.descriptor_pb2 import FileDescriptorSet
    from google.protobuf.descriptor_pool import DescriptorPool
    from google.protobuf.message_factory import GetMessageClass
    from mcap.reader import make_reader

    decoders, out = {}, defaultdict(list)
    with open(path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        channels = {c.topic: c for c in summary.channels.values()}
        schemas = {channels[t].topic: summary.schemas[channels[t].schema_id] for t in channels}
        for schema, channel, message in reader.iter_messages(log_time_order=False):
            if channel.message_encoding == "json":
                doc = json.loads(message.data, parse_constant=_reject_constant)
            else:
                if schema.id not in decoders:
                    pool = DescriptorPool()
                    for fd in FileDescriptorSet.FromString(schema.data).file:
                        pool.Add(fd)
                    decoders[schema.id] = GetMessageClass(pool.FindMessageTypeByName(schema.name))
                doc = decoders[schema.id].FromString(message.data)
            out[channel.topic].append((message.log_time, doc))
    return channels, schemas, summary, out


def _reject_constant(name):
    raise AssertionError(f"{name} is not valid JSON")


@pytest.fixture(scope="module")
def scenario():
    return make_perpendicular(VehicleParams.from_yaml(), load_params(), np.random.default_rng(SEED))


@pytest.fixture(scope="module")
def result(scenario):
    return run_closed_loop(scenario)


@pytest.fixture(scope="module")
def exported(scenario, result, tmp_path_factory):
    path = tmp_path_factory.mktemp("mcap") / "nested" / "run.mcap"
    export_run(scenario, result, path)
    return read_mcap(path)


# ---------- optional dependency ----------
def test_package_imports_without_optional_deps():
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('mcap', 'foxglove_schemas_protobuf', 'foxglove'):\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import park_sim, park_sim.simulator, park_sim.viz, park_sim.mcap_export as m\n"
        "try:\n"
        "    m.require_backend()\n"
        "except ImportError as e:\n"
        "    assert 'pip install mcap' in str(e)\n"
        "    print('clear-error')\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "clear-error" in done.stdout


# ---------- every channel ----------
def test_every_channel_exists_with_expected_schema(exported):
    channels, schemas, _, messages = exported
    assert set(channels) == set(TOPICS)
    assert all(messages[t] for t in TOPICS)
    assert schemas["/tf"].name == "foxglove.FrameTransforms"
    for topic in ("/lot", "/plan", "/ego", "/trail"):
        assert schemas[topic].name == "foxglove.SceneUpdate"
        assert channels[topic].message_encoding == "protobuf"
    for topic in ("/state", "/supervisor", "/metrics"):
        assert channels[topic].message_encoding == "json"
        assert schemas[topic].encoding == "jsonschema"


def test_timestamps_monotonic_from_zero(exported, result):
    _, _, summary, messages = exported
    for topic in TOPICS:
        times = [t for t, _ in messages[topic]]
        assert all(b >= a for a, b in zip(times, times[1:])), topic
    tf_times = np.array([t for t, _ in messages["/tf"]])
    assert tf_times[0] == 0
    assert np.all(np.diff(tf_times) > 0)
    assert np.allclose(tf_times / 1e9, result.log["t"], atol=1e-9)
    assert summary.statistics.message_start_time == 0
    assert summary.statistics.message_end_time == tf_times[-1]


def test_tf_count_matches_sim_ticks(exported, result):
    """One /tf per logged tick. The log holds the control ticks plus the terminal state."""
    tf = exported[3]["/tf"]
    dt = load_params()["controller"]["dt"]
    assert len(tf) == len(result.log["t"])
    assert len(tf) == round(result.log["t"][-1] / dt) + 1


def test_tf_is_map_to_rear_axle(exported, result):
    log = result.log
    for i, (ns, msg) in [(0, exported[3]["/tf"][0]), (len(log["t"]) - 1, exported[3]["/tf"][-1])]:
        tr = msg.transforms[0]
        assert (tr.parent_frame_id, tr.child_frame_id) == ("map", "base_link")
        assert tr.timestamp.ToNanoseconds() == ns
        assert tr.translation.x == pytest.approx(log["x"][i])
        assert tr.translation.y == pytest.approx(log["y"][i])
        assert 2 * math.atan2(tr.rotation.z, tr.rotation.w) == pytest.approx(log["yaw"][i])
        assert tr.rotation.z ** 2 + tr.rotation.w ** 2 == pytest.approx(1.0)


# ---------- /lot and /plan ----------
def test_lot_written_once_and_static(exported, scenario):
    channels, _, _, messages = exported
    assert len(messages["/lot"]) == 1
    ns, lot = messages["/lot"][0]
    assert ns == 0
    assert channels["/lot"].metadata["static"] == "true"
    by_id = {e.id: e for e in lot.entities}
    assert set(by_id) == {"stall_lines", "parked_cars", "curbs", "target_stall", "goal_footprint"}
    for e in lot.entities:
        assert e.frame_id == "map"
        assert e.lifetime.ToNanoseconds() == 0      # never expires
        assert {(kv.key, kv.value) for kv in e.metadata} == {("static", "true")}
    assert len(by_id["stall_lines"].lines[0].points) == 2 * len(scenario.lines)
    assert len(by_id["parked_cars"].cubes) == len(scenario.parked_cars)
    assert len(by_id["curbs"].cubes) == len(scenario.curbs)
    assert len(by_id["goal_footprint"].lines[0].points) == 4


def test_parked_car_boxes_match_collision_polygons(exported, scenario):
    cars = {e.id: e for e in exported[3]["/lot"][0][1].entities}["parked_cars"].cubes
    for cube, poly in zip(cars, scenario.parked_cars):
        q = cube.pose.orientation
        yaw = 2 * math.atan2(q.z, q.w)
        hl, hw = cube.size.x / 2, cube.size.y / 2
        c, s = math.cos(yaw), math.sin(yaw)
        corners = [(cube.pose.position.x + c * a - s * b, cube.pose.position.y + s * a + c * b)
                   for a, b in [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]]
        assert np.allclose(corners, poly, atol=1e-5)


def test_scenario_splits_obstacles_into_cars_and_curbs(scenario):
    assert len(scenario.parked_cars) + len(scenario.curbs) == len(scenario.obstacles)
    assert all(a is b for a, b in zip(scenario.parked_cars + scenario.curbs, scenario.obstacles))


def test_plan_colors_directions_and_marks_cusps(scenario, result, tmp_path):
    """Forward, then reverse, then forward: two cusps, on a hand-made path."""
    x = np.arange(0.0, 3.0, 0.5)
    direction = np.array([1, 1, 1, -1, -1, 1])
    path = PlannedPath(x, np.zeros(6), np.zeros(6), direction)
    res = RunResult(path, result.log, result.metrics, result.checks)
    export_run(scenario, res, tmp_path / "cusps.mcap")
    plan = read_mcap(tmp_path / "cusps.mcap")[3]["/plan"]
    assert len(plan) == 1
    entity = plan[0][1].entities[0]
    assert len(entity.lines) == len(path.segments()) == 3
    colors = [(l.color.r, l.color.g, l.color.b) for l in entity.lines]
    assert colors[0] == colors[2] != colors[1]
    assert [s.pose.position.x for s in entity.spheres] == [x[b] for _, b, _ in path.segments()[:-1]]


def test_plan_follows_planned_path(exported, result):
    entity = exported[3]["/plan"][0][1].entities[0]
    points = [p for line in entity.lines for p in line.points]
    assert points[-1].x == pytest.approx(result.planned.x[-1])
    assert points[-1].y == pytest.approx(result.planned.y[-1])


# ---------- /ego and /trail ----------
def test_ego_body_box_uses_real_height(exported, scenario):
    ego = exported[3]["/ego"]
    assert len(ego) == len(exported[3]["/tf"])
    v = scenario.vehicle
    assert v.height == pytest.approx(1.623)
    body = {e.id: e for e in ego[0][1].entities}["body"]
    assert body.frame_id == "base_link"
    cube = body.cubes[0]
    assert (cube.size.x, cube.size.y, cube.size.z) == pytest.approx((4.996, v.width, 1.623))
    assert cube.pose.position.x == pytest.approx(v.center_offset())
    assert cube.pose.position.z == pytest.approx(1.623 / 2)   # sits on the ground


def test_lookahead_marker_only_while_driving(exported, result):
    states = result.log["ctrl_state"]
    assert {"shifting", "driving", "done"} <= set(states.tolist())
    for i, (_, update) in enumerate(exported[3]["/ego"]):
        ids = {e.id for e in update.entities}
        if np.isnan(result.log["lookahead_x"][i]):
            assert "lookahead" not in ids and [d.id for d in update.deletions] == ["lookahead"]
        else:
            assert "lookahead" in ids and not update.deletions
    assert np.isnan(result.log["lookahead_x"][states == "shifting"]).all()


def test_trail_grows_and_ends_at_final_pose(exported, result):
    trail = exported[3]["/trail"]
    fg = load_params()["foxglove"]
    assert len(trail) < len(exported[3]["/tf"]) / (fg["trail_publish_every"] - 1)
    counts = [len(msg.entities[0].lines[0].points) for _, msg in trail]
    assert counts[0] == 1
    assert all(b >= a for a, b in zip(counts, counts[1:]))
    last = trail[-1][1].entities[0].lines[0].points
    assert trail[-1][0] == exported[3]["/tf"][-1][0]
    assert (last[-1].x, last[-1].y) == pytest.approx((result.log["x"][-1], result.log["y"][-1]))
    # stored points are thinned by spacing, with the live pose appended to each message
    gaps = np.hypot(np.diff([p.x for p in last[:-1]]), np.diff([p.y for p in last[:-1]]))
    assert gaps.min() >= fg["trail_min_spacing"]


# ---------- /state and /metrics ----------
def test_state_has_all_requested_fields_every_tick(exported, result):
    state = exported[3]["/state"]
    assert len(state) == len(result.log["t"])
    required = {"t", "x", "y", "yaw", "v", "delta", "steer_cmd", "speed_cmd", "gear",
                "controller_state", "segment_index", "distance_to_go", "min_clearance"}
    assert all(required <= set(doc) for _, doc in state)
    for ns, doc in state:
        assert doc["t"] * 1e9 == pytest.approx(ns, abs=1)
    assert np.array_equal([d["controller_state"] for _, d in state], result.log["ctrl_state"])
    assert np.allclose([d["speed_cmd"] for _, d in state], result.log["speed_cmd"])
    assert np.allclose(np.radians([d["delta_deg"] for _, d in state]), result.log["delta"])


def test_state_schema_covers_every_field(exported):
    _, schemas, _, messages = exported
    props = json.loads(schemas["/state"].data)["properties"]
    assert set(props) == set(messages["/state"][0][1])
    assert props["controller_state"]["type"] == "string"
    assert all(isinstance(p["type"], str) for p in props.values())


def test_min_clearance_is_running_min_ending_at_metric(exported, result):
    values = [d["min_clearance"] for _, d in exported[3]["/state"]]
    assert all(b <= a for a, b in zip(values, values[1:]))
    assert values[-1] == result.metrics["min_clearance"]


def test_distance_to_go_counts_down_to_zero(exported, result):
    state = [d for _, d in exported[3]["/state"]]
    assert state[0]["distance_to_go"] is None      # no plan yet
    state = [s for s in state if s["distance_to_go"] is not None]
    assert state[0]["distance_to_go"] == pytest.approx(result.planned.length, abs=0.01)
    assert state[-1]["distance_to_go"] == 0.0
    assert state[-1]["controller_state"] == "done"
    d = np.array([s["distance_to_go"] for s in state])
    assert d.max() <= result.planned.length + 0.01
    assert np.diff(d).max() < 0.05    # counts down (small allowance for the projection refinement)


def test_distance_to_go_spans_all_segments():
    """Multi-segment path: total = current segment + later ones, and advances across a cusp."""
    from park_sim.controller import ParkingController
    veh, cfg = VehicleParams.from_yaml(), load_params()["controller"]
    x = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    path = PlannedPath(x, np.zeros(5), np.zeros(5), np.array([1, 1, 1, -1, -1]))
    ctrl = ParkingController(path, veh, cfg)
    assert [s.length for s in ctrl.segments] == [2.0, 2.0]
    ctrl.seg_dist_to_go = 1.5
    assert ctrl.dist_to_go == pytest.approx(3.5)
    ctrl.seg_i = 1
    assert ctrl.dist_to_go == pytest.approx(1.5)
    ctrl.seg_i = 2
    assert ctrl.dist_to_go == 0.0


def test_metrics_written_once_at_end(exported, result):
    metrics = exported[3]["/metrics"]
    assert len(metrics) == 1
    ns, doc = metrics[0]
    assert ns == exported[3]["/tf"][-1][0]
    assert doc["metrics"] == pytest.approx(result.metrics)
    assert doc["checks"] == {k: bool(v) for k, v in result.checks.items()}
    assert doc["success"] is True and doc["summary"] == result.summary()
    assert doc["final_state"] == "COMPLETED"
    assert [(tr["from"], tr["to"]) for tr in doc["transitions"]] == [
        (tr.from_state.value, tr.to_state.value) for tr in result.transitions]


def test_supervisor_channel_every_tick(exported, result):
    channels, schemas, _, messages = exported
    rows = messages["/supervisor"]
    assert len(rows) == len(messages["/tf"])
    assert [ns for ns, _ in rows] == [ns for ns, _ in messages["/tf"]]
    states = [d["state"] for _, d in rows]
    assert states[0] == "SEARCHING" and states[-1] == "COMPLETED"
    assert "MANEUVERING" in states
    assert rows[-1][1]["request_park"] is True
    assert rows[-1][1]["transition_reason"] == "controller finished"
    assert rows[0][1]["transition_reason"] == "feature button"
    assert all(isinstance(p["type"], str)
               for p in json.loads(schemas["/supervisor"].data)["properties"].values())
    assert json.loads(schemas["/metrics"].data)["properties"]["transitions"]["type"] == "array"


def test_aborted_run_shows_reason_on_supervisor_channel(scenario, tmp_path):
    res = run_closed_loop(scenario, events=[(3.0, {"steering_torque_nm": 5.0})])
    rows = read_mcap(export_run(scenario, res, tmp_path / "abort.mcap"))[3]["/supervisor"]
    last = rows[-1][1]
    assert last["state"] == "ABORTED" and last["abort_reason"] == "steering_override"
    assert last["driver_must_shift"] is True
    assert "steering torque" in last["transition_reason"]


# ---------- runs that went wrong ----------
def test_run_without_plan_still_exports(scenario, tmp_path):
    res = RunResult(None, reason="no path found")
    channels, _, _, messages = read_mcap(export_run(scenario, res, tmp_path / "np.mcap"))
    assert set(channels) == set(TOPICS)
    assert len(messages["/tf"]) == 1
    assert messages["/tf"][0][1].transforms[0].translation.x == pytest.approx(scenario.start.x)
    assert not messages["/plan"][0][1].entities
    assert messages["/state"][0][1]["controller_state"] == "no_path"
    assert messages["/state"][0][1]["distance_to_go"] is None
    assert messages["/metrics"][0][1]["success"] is False
    assert messages["/metrics"][0][1]["reason"] == "no path found"


def test_failure_filename_names_seed_and_failed_checks():
    bad = RunResult(object(), checks={"lateral": True, "heading": False, "gear_changes": False})
    assert failure_filename(42, bad) == "seed0042_heading-gear_changes.mcap"
    assert failure_filename(7, RunResult(None, reason="no path found")) == "seed0007_no_path.mcap"


# ---------- scripts ----------
def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_monte_carlo_writes_only_failures(tmp_path):
    pytest.importorskip("mcap")
    mc = _load_script("monte_carlo")
    veh, params = VehicleParams.from_yaml(), load_params()

    passed, _ = mc.batch(1, veh, params, 0.0, 0.0, seed0=SEED, mcap_dir=tmp_path / "ok")
    assert passed == 1 and not (tmp_path / "ok").exists()

    strict = copy.deepcopy(params)
    strict["criteria"]["lateral_tol"] = -1.0   # impossible, so the run fails on this check only
    passed, _ = mc.batch(1, veh, strict, 0.0, 0.0, seed0=SEED, mcap_dir=tmp_path / "bad")
    assert passed == 0
    path = tmp_path / "bad" / f"seed{SEED:04d}_lateral.mcap"
    assert path.is_file()
    assert read_mcap(path)[3]["/metrics"][0][1]["checks"]["lateral"] is False


def _run_demo(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "run_demo.py"), "--seed", str(SEED), *args],
                          capture_output=True, text=True)


def test_run_demo_mcap_flag_and_static_is_opt_in(tmp_path):
    pytest.importorskip("mcap")
    mcap, default_png = tmp_path / "run.mcap", ROOT / "out" / "park_run.png"
    before = default_png.stat().st_mtime_ns if default_png.exists() else None
    done = _run_demo("--mcap", str(mcap))
    assert done.returncode == 0, done.stderr
    assert mcap.is_file()
    assert set(read_mcap(mcap)[0]) == set(TOPICS)
    after = default_png.stat().st_mtime_ns if default_png.exists() else None
    assert before == after   # no PNG unless asked for


def test_run_demo_static_saves_png(tmp_path):
    png = tmp_path / "summary.png"
    done = _run_demo("--static", "--plot", str(png))
    assert done.returncode == 0, done.stderr
    assert png.read_bytes().startswith(b"\x89PNG")


# ---------- layout ----------
def _layout_panels(node):
    if isinstance(node, str):
        return [node]
    return _layout_panels(node["first"]) + _layout_panels(node["second"])


def test_layout_matches_what_the_export_writes(exported):
    layout = json.loads((ROOT / "config" / "foxglove_layout.json").read_text())
    panels = layout["configById"]
    assert sorted(_layout_panels(layout["layout"])) == sorted(panels)
    kinds = [p.split("!")[0] for p in panels]
    assert {"3D", "Plot", "RawMessages", "StateTransitions"} == set(kinds)

    fields = set(exported[3]["/state"][0][1])
    plotted = {path["value"] for pid, cfg in panels.items()
               if pid.startswith(("Plot", "StateTransitions")) for path in cfg["paths"]}
    assert {"/state.controller_state", "/supervisor.state", "/state.distance_to_go", "/state.v",
            "/state.speed_cmd", "/state.delta_deg", "/state.steer_cmd_deg"} == plotted
    assert {p.split(".", 1)[1] for p in plotted if p.startswith("/state")} <= fields
    assert "state" in exported[3]["/supervisor"][0][1]
    assert panels["RawMessages!state"]["topicPath"] == "/state"

    view = panels["3D!parking"]
    assert view["followTf"] == "map" and view["cameraState"]["phi"] == 0     # top-down on map
    assert set(view["topics"]) == {"/lot", "/plan", "/ego", "/trail", "/actors", "/safety_sweep", "/sensed_map"}
    assert all(t["visible"] for t in view["topics"].values())


def test_layout_camera_centered_on_configured_lot():
    panels = json.loads((ROOT / "config" / "foxglove_layout.json").read_text())["configById"]
    sc = load_params()["scenario"]
    target = panels["3D!parking"]["cameraState"]["targetOffset"]
    assert target[0] == pytest.approx(sc["n_stalls"] * sc["stall_width"] / 2)
    assert target[1] == pytest.approx((2 * sc["stall_depth"] + sc["aisle_width"]) / 2)
