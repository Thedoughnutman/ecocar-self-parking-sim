"""Parking supervisor: unit tests per transition and abort reason, then scripted closed-loop runs."""
import ast
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from park_sim.scenario import load_params, make_perpendicular  # noqa: E402
from park_sim.simulator import run_closed_loop  # noqa: E402
from park_sim.supervisor import (AbortReason, ControllerAction, ControllerStatus,  # noqa: E402
                                 ParkingSupervisor, PlannerResult, PlannerStatus, SensorHealth,
                                 SlotCandidate, State, SupervisorConfig, SupervisorInput)
from park_sim.vehicle import VehicleParams  # noqa: E402

SEED = 1      # reverse-in, 13 s maneuver, D -> R shift at the start
SLOT = 3
PARAMS = load_params()
CFG = PARAMS["supervisor"]
KPH = 1 / 3.6


class Bench:
    """Feeds a supervisor one tick at a time with a healthy, fresh, slot-present baseline."""

    def __init__(self, **cfg_changes):
        self.sup = ParkingSupervisor(SupervisorConfig(**{**CFG, **cfg_changes}))
        self.t = 0.0
        self.out = None

    def tick(self, dt=0.02, **kw):
        kw.setdefault("slots", (SlotCandidate(SLOT),))
        kw.setdefault("sensors", self.fresh())
        self.out = self.sup.step(SupervisorInput(t=self.t, **kw))
        self.t += dt
        return self.out

    def fresh(self):
        return tuple(SensorHealth(name, True, self.t) for name in CFG["sensor_timeouts_s"])

    def hold(self, seconds, **kw):
        end = self.t + seconds
        while self.t < end:
            self.tick(**kw)
        return self.out

    def press_feature(self):
        self.tick(feature_button=True)
        return self.tick()

    @property
    def state(self):
        return self.sup.state

    def reach(self, target):
        """Walk the normal path to `target` (one of the states on the way to parking)."""
        order = [State.OFF, State.SEARCHING, State.SLOTS_AVAILABLE, State.SLOT_SELECTED,
                 State.WAIT_CONFIRM, State.PLANNING, State.MANEUVERING, State.PAUSED]
        steps = {
            State.SEARCHING: dict(feature_button=True, slots=()),
            State.SLOTS_AVAILABLE: dict(),
            State.SLOT_SELECTED: dict(selected_slot_id=SLOT),
            State.WAIT_CONFIRM: dict(),
            State.PLANNING: dict(confirm_button=True),
            State.MANEUVERING: dict(planner=PlannerResult(PlannerStatus.FOUND, SLOT)),
            State.PAUSED: dict(brake=0.5),
        }
        for state in order[1:order.index(target) + 1]:
            self.tick(**steps[state])
            assert self.state is state
        return self


def last_transition(bench):
    return bench.sup.transitions[-1]


# ---------- the normal path ----------
def test_starts_off_with_no_action():
    bench = Bench()
    assert bench.tick().state is State.OFF
    assert bench.out.action is ControllerAction.NONE
    assert bench.out.abort_reason is AbortReason.NONE


def test_off_to_searching_on_button_below_speed_limit():
    bench = Bench()
    bench.tick(feature_button=True, speed=19 * KPH, slots=())
    assert bench.state is State.SEARCHING


@pytest.mark.parametrize("kph", [20.0, 40.0])
def test_off_stays_off_at_or_above_search_speed(kph):
    bench = Bench()
    out = bench.tick(feature_button=True, speed=kph * KPH)
    assert out.state is State.OFF and "20 km/h" in out.hmi_message


def test_held_feature_button_only_acts_once():
    bench = Bench()
    bench.tick(feature_button=True, slots=())
    bench.tick(feature_button=True, slots=())
    assert bench.state is State.SEARCHING


def test_searching_to_slots_available_needs_confirmed_slot():
    bench = Bench().reach(State.SEARCHING)
    bench.tick(slots=(SlotCandidate(SLOT, confirmed=False),))
    assert bench.state is State.SEARCHING
    bench.tick()
    assert bench.state is State.SLOTS_AVAILABLE


def test_slots_available_back_to_searching_when_slots_vanish():
    bench = Bench().reach(State.SLOTS_AVAILABLE)
    bench.tick(slots=())
    assert bench.state is State.SEARCHING


def test_selection_goes_to_slot_selected():
    bench = Bench().reach(State.SLOT_SELECTED)
    assert bench.sup.selected_slot == SLOT


def test_selecting_unknown_or_unconfirmed_slot_is_ignored():
    bench = Bench().reach(State.SLOTS_AVAILABLE)
    bench.tick(selected_slot_id=99)
    bench.tick(selected_slot_id=7, slots=(SlotCandidate(SLOT), SlotCandidate(7, confirmed=False)))
    assert bench.state is State.SLOTS_AVAILABLE


def test_slot_selected_waits_for_stop_then_wait_confirm():
    bench = Bench().reach(State.SLOT_SELECTED)
    bench.tick(speed=1.0)
    assert bench.state is State.SLOT_SELECTED
    bench.tick(speed=0.0)
    assert bench.state is State.WAIT_CONFIRM


def test_wait_confirm_back_to_slot_selected_if_car_moves():
    bench = Bench().reach(State.WAIT_CONFIRM)
    bench.tick(speed=0.5)
    assert bench.state is State.SLOT_SELECTED


@pytest.mark.parametrize("from_state", [State.SLOT_SELECTED, State.WAIT_CONFIRM])
def test_selected_slot_lost_returns_to_slots_available(from_state):
    bench = Bench().reach(from_state)
    bench.tick(slots=(SlotCandidate(SLOT + 1),))
    assert bench.state is State.SLOTS_AVAILABLE


def test_confirm_goes_to_planning_and_requests_plan():
    bench = Bench().reach(State.PLANNING)
    assert bench.out.request_plan and bench.out.plan_slot_id == SLOT
    assert bench.out.action is ControllerAction.HOLD


def test_confirm_already_held_on_arrival_does_not_start():
    bench = Bench().reach(State.SLOT_SELECTED)
    bench.tick(confirm_button=True)
    assert bench.state is State.WAIT_CONFIRM
    bench.tick(confirm_button=True)
    assert bench.state is State.WAIT_CONFIRM
    bench.tick()
    bench.tick(confirm_button=True)
    assert bench.state is State.PLANNING


def test_planning_to_maneuvering_when_path_found():
    bench = Bench().reach(State.MANEUVERING)
    assert bench.out.action is ControllerAction.DRIVE
    assert not bench.out.request_plan


def test_planning_back_to_slots_available_on_no_path():
    bench = Bench().reach(State.PLANNING)
    out = bench.tick(planner=PlannerResult(PlannerStatus.NOT_FOUND, SLOT))
    assert out.state is State.SLOTS_AVAILABLE
    assert out.hmi_message == "no path to this slot"
    assert out.action is ControllerAction.NONE
    assert last_transition(bench).reason == "no path to this slot"
    # A new selection works and clears the message
    assert bench.tick(selected_slot_id=SLOT).hmi_message != "no path to this slot"
    assert bench.state is State.SLOT_SELECTED


def test_planning_times_out_as_no_path():
    bench = Bench()
    bench.reach(State.PLANNING)
    bench.hold(CFG["planning_timeout_s"] + 0.1)
    assert bench.state is State.SLOTS_AVAILABLE
    assert bench.out.hmi_message == "no path to this slot"
    assert last_transition(bench).reason == "planning timed out"


def test_planning_ignores_result_for_another_slot():
    bench = Bench().reach(State.PLANNING)
    bench.tick(planner=PlannerResult(PlannerStatus.FOUND, SLOT + 1))
    assert bench.state is State.PLANNING


def test_maneuvering_to_completed_requests_park():
    bench = Bench().reach(State.MANEUVERING)
    bench.tick(controller=ControllerStatus.SEGMENT_DONE)
    assert bench.state is State.MANEUVERING
    out = bench.tick(controller=ControllerStatus.FINISHED)
    assert out.state is State.COMPLETED and out.request_park
    assert out.action is ControllerAction.HOLD and not out.driver_must_shift


def test_completed_to_off_on_button():
    bench = Bench().reach(State.MANEUVERING)
    bench.tick(controller=ControllerStatus.FINISHED)
    bench.press_feature()
    assert bench.state is State.OFF


@pytest.mark.parametrize("from_state", [State.SEARCHING, State.SLOTS_AVAILABLE,
                                        State.SLOT_SELECTED, State.WAIT_CONFIRM])
def test_feature_button_before_planning_returns_to_off(from_state):
    bench = Bench().reach(from_state)
    if from_state is State.SEARCHING:
        bench.tick(slots=())            # reach() left the button down; release it
    assert bench.state is from_state
    bench.tick(feature_button=True)
    assert bench.state is State.OFF


# ---------- pause and resume ----------
@pytest.mark.parametrize("kw,reason", [(dict(brake=0.5), "brake pressed"),
                                       (dict(obstacle_in_path=True), "obstacle in path")])
def test_maneuvering_to_paused(kw, reason):
    bench = Bench().reach(State.MANEUVERING)
    out = bench.tick(speed=0.5, **kw)
    assert out.state is State.PAUSED and last_transition(bench).reason == reason
    assert out.action is ControllerAction.STOP
    assert bench.tick(speed=0.0, **kw).action is ControllerAction.HOLD


def test_light_brake_contact_does_not_pause():
    bench = Bench().reach(State.MANEUVERING)
    bench.tick(brake=CFG["brake_pressed"])
    assert bench.state is State.MANEUVERING


def test_resumes_after_delay_when_confirm_held():
    bench = Bench().reach(State.PAUSED)
    bench.hold(CFG["resume_delay_s"] - 0.1, confirm_button=True)
    assert bench.state is State.PAUSED
    bench.hold(0.2, confirm_button=True)
    assert bench.state is State.MANEUVERING
    assert last_transition(bench).reason == "path clear and confirm held"


def test_without_confirm_waits_then_resumes_on_confirm():
    bench = Bench().reach(State.PAUSED)
    bench.hold(CFG["resume_delay_s"] * 5)
    assert bench.state is State.PAUSED
    bench.tick(confirm_button=True)
    assert bench.state is State.MANEUVERING


def test_new_obstacle_restarts_the_resume_delay():
    bench = Bench().reach(State.PAUSED)
    bench.hold(CFG["resume_delay_s"] - 0.1, confirm_button=True)
    bench.tick(confirm_button=True, obstacle_in_path=True)
    bench.hold(CFG["resume_delay_s"] - 0.1, confirm_button=True)
    assert bench.state is State.PAUSED


def test_still_blocked_never_resumes():
    bench = Bench().reach(State.PAUSED)
    bench.hold(CFG["resume_delay_s"] * 3, confirm_button=True, brake=0.5)
    bench.hold(CFG["resume_delay_s"] * 0.5, confirm_button=True)
    assert bench.state is State.PAUSED


# ---------- every abort reason, from every state where the system has the car ----------
def stale_sensors(bench):
    return tuple(SensorHealth(name, True, bench.t - CFG["sensor_timeouts_s"][name] - 0.01)
                 for name in CFG["sensor_timeouts_s"])


ABORTS = [
    (AbortReason.STEERING_OVERRIDE, lambda b: dict(steering_torque_nm=2.1)),
    (AbortReason.STEERING_OVERRIDE, lambda b: dict(steering_torque_nm=-2.1)),
    (AbortReason.ACCELERATOR_OVERRIDE, lambda b: dict(accelerator=0.06)),
    (AbortReason.GEAR_REQUEST, lambda b: dict(gear_request=1)),
    (AbortReason.DOOR_OPEN, lambda b: dict(door_open=True)),
    (AbortReason.SEATBELT_OFF, lambda b: dict(seatbelt_on=False)),
    (AbortReason.SENSOR_STALE, lambda b: dict(sensors=stale_sensors(b))),
    (AbortReason.SENSOR_STALE, lambda b: dict(sensors=())),
    (AbortReason.OVERSPEED, lambda b: dict(speed=-5.1 * KPH)),
    (AbortReason.TOO_MANY_GEAR_CHANGES, lambda b: dict(gear_changes=CFG["max_gear_changes"] + 1)),
    (AbortReason.DRIVER_CANCEL, lambda b: dict(feature_button=True)),
]


@pytest.mark.parametrize("from_state", [State.PLANNING, State.MANEUVERING, State.PAUSED])
@pytest.mark.parametrize("reason,inputs", ABORTS)
def test_abort_is_immediate(from_state, reason, inputs):
    bench = Bench().reach(from_state)
    out = bench.tick(**inputs(bench))
    assert out.state is State.ABORTED and out.abort_reason is reason


@pytest.mark.parametrize("inputs", [dict(steering_torque_nm=2.0), dict(accelerator=0.05),
                                    dict(gear_changes=CFG["max_gear_changes"]),
                                    dict(speed=5.0 * KPH)])
def test_values_exactly_at_the_limit_do_not_abort(inputs):
    bench = Bench().reach(State.MANEUVERING)
    assert bench.tick(**inputs).state is State.MANEUVERING


def test_abort_on_maneuver_time():
    bench = Bench(max_time_s=1.0).reach(State.MANEUVERING)
    bench.hold(0.9)
    assert bench.state is State.MANEUVERING
    bench.hold(0.2)
    assert bench.state is State.ABORTED
    assert bench.out.abort_reason is AbortReason.MANEUVER_TIMEOUT


def test_maneuver_time_keeps_counting_while_paused():
    bench = Bench(max_time_s=1.0).reach(State.PAUSED)
    bench.hold(1.1, brake=0.5)
    assert bench.out.abort_reason is AbortReason.MANEUVER_TIMEOUT


def test_one_stale_sensor_is_enough_and_fresh_ones_pass():
    bench = Bench().reach(State.MANEUVERING)
    sensors = list(bench.fresh())
    assert bench.tick(sensors=tuple(sensors)).state is State.MANEUVERING
    sensors[0] = SensorHealth(sensors[0].name, True, bench.t - 10.0)
    out = bench.tick(sensors=tuple(sensors))
    assert out.abort_reason is AbortReason.SENSOR_STALE
    assert sensors[0].name in last_transition(bench).reason


def test_abort_stops_then_holds_then_hands_back():
    bench = Bench().reach(State.MANEUVERING)
    out = bench.tick(speed=0.8, steering_torque_nm=5.0)
    assert out.action is ControllerAction.STOP and not out.driver_must_shift
    out = bench.tick(speed=0.3)
    assert out.state is State.ABORTED and out.action is ControllerAction.STOP
    out = bench.tick(speed=0.0)
    assert out.action is ControllerAction.HOLD and out.driver_must_shift
    assert "steering_override" in out.hmi_message and not out.request_park


def test_aborted_does_not_clear_while_moving_and_then_returns_to_off():
    bench = Bench().reach(State.MANEUVERING)
    bench.tick(speed=0.8, door_open=True)
    bench.tick(speed=0.8, feature_button=True)
    assert bench.state is State.ABORTED
    bench.tick(speed=0.0)
    out = bench.tick(speed=0.0, feature_button=True)
    assert out.state is State.OFF and out.abort_reason is AbortReason.NONE
    assert out.action is ControllerAction.NONE and not out.driver_must_shift


@pytest.mark.parametrize("from_state", [State.SEARCHING, State.SLOTS_AVAILABLE,
                                        State.SLOT_SELECTED, State.WAIT_CONFIRM])
@pytest.mark.parametrize("inputs", [dict(steering_torque_nm=5.0), dict(accelerator=0.5),
                                    dict(gear_request=-1), dict(door_open=True),
                                    dict(sensors=())])
def test_driver_keeps_the_car_before_confirm(from_state, inputs):
    """Steering, throttle and shifting are normal while the driver is still driving."""
    bench = Bench().reach(from_state)
    assert bench.tick(**{"slots": (), **inputs}).state is not State.ABORTED


# ---------- fault ----------
@pytest.mark.parametrize("from_state", [State.SEARCHING, State.WAIT_CONFIRM])
def test_unhealthy_sensor_before_control_faults_but_leaves_driver_the_car(from_state):
    bench = Bench().reach(from_state)
    out = bench.tick(sensors=tuple(SensorHealth(s.name, s.name != "localization", bench.t)
                                   for s in bench.fresh()), speed=2.0)
    assert out.state is State.FAULT and out.abort_reason is AbortReason.SENSOR_UNHEALTHY
    assert out.action is ControllerAction.NONE and not out.driver_must_shift


def test_unhealthy_sensor_while_maneuvering_faults_and_stops():
    bench = Bench().reach(State.MANEUVERING)
    sensors = tuple(SensorHealth(s.name, s.name != "localization", bench.t) for s in bench.fresh())
    out = bench.tick(sensors=sensors, speed=0.8)
    assert out.state is State.FAULT and out.action is ControllerAction.STOP
    out = bench.tick(sensors=sensors, speed=0.0)
    assert out.action is ControllerAction.HOLD and out.driver_must_shift
    assert "localization" in last_transition(bench).reason


def test_stale_takes_priority_over_unhealthy_when_both():
    bench = Bench().reach(State.MANEUVERING)
    sensors = tuple(SensorHealth(name, False, bench.t - 10.0) for name in CFG["sensor_timeouts_s"])
    out = bench.tick(sensors=sensors)
    assert out.state is State.ABORTED and out.abort_reason is AbortReason.SENSOR_STALE


def test_fault_to_off_on_button_once_stopped():
    bench = Bench().reach(State.MANEUVERING)
    sensors = tuple(SensorHealth(s.name, s.name != "localization", bench.t) for s in bench.fresh())
    bench.tick(sensors=sensors, speed=0.0)
    bench.tick(feature_button=True, speed=0.0)
    assert bench.state is State.OFF


# ---------- transition log, structure ----------
def test_every_transition_logged_with_time_from_to_reason():
    bench = Bench().reach(State.PAUSED)
    log = bench.sup.transitions
    assert [(tr.from_state, tr.to_state) for tr in log] == [
        (State.OFF, State.SEARCHING), (State.SEARCHING, State.SLOTS_AVAILABLE),
        (State.SLOTS_AVAILABLE, State.SLOT_SELECTED), (State.SLOT_SELECTED, State.WAIT_CONFIRM),
        (State.WAIT_CONFIRM, State.PLANNING), (State.PLANNING, State.MANEUVERING),
        (State.MANEUVERING, State.PAUSED)]
    assert all(tr.reason for tr in log)
    assert [tr.t for tr in log] == pytest.approx([0.02 * i for i in range(7)])


def test_supervisor_only_depends_on_the_standard_library():
    tree = ast.parse((ROOT / "park_sim" / "supervisor.py").read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative imports: keep the supervisor free of sim internals"
            modules.add(node.module.split(".")[0])
    assert modules <= {"__future__", "dataclasses", "enum", "typing"}


def test_doc_diagram_names_every_state_and_abort_reason():
    doc = (ROOT / "docs" / "supervisor.md").read_text(encoding="utf-8")
    assert "stateDiagram-v2" in doc
    for state in State:
        assert state.value in doc
    for reason in AbortReason:
        assert reason is AbortReason.NONE or reason.value in doc


# ---------- closed loop with scripted events ----------
def run(events=(), params=None, seed=SEED, **kw):
    params = params or PARAMS
    sc = make_perpendicular(VehicleParams.from_yaml(), params, np.random.default_rng(seed))
    return sc, run_closed_loop(sc, events=events, **kw)


def path_of(res):
    return [tr.to_state for tr in res.transitions]


def test_scripted_driver_walks_the_normal_path_and_parks():
    _, res = run()
    assert res.success, res.summary()
    assert path_of(res) == [State.SEARCHING, State.SLOTS_AVAILABLE, State.SLOT_SELECTED,
                            State.WAIT_CONFIRM, State.PLANNING, State.MANEUVERING, State.COMPLETED]
    assert res.final_state is State.COMPLETED
    assert res.log["sup_request_park"][-1] == 1.0
    assert res.log["sup_state"][0] == "SEARCHING" and res.log["sup_state"][-1] == "COMPLETED"


def test_car_waits_at_start_until_the_maneuver_begins():
    sc, res = run()
    waiting = np.isnan(res.log["t_maneuver"])
    assert 0 < waiting.sum() < 10
    assert np.all(res.log["x"][waiting] == sc.start.x) and np.all(res.log["v"][waiting] == 0.0)


def test_time_metric_is_maneuver_time_not_run_time():
    _, res = run()
    lead_up = res.log["t"][np.isnan(res.log["t_maneuver"])].size * PARAMS["controller"]["dt"]
    assert res.metrics["time_s"] == pytest.approx(res.log["t"][-1] - lead_up)


def test_override_stays_until_replaced():
    _, res = run([(1.0, {"brake": 0.5}), (3.0, {"brake": 0.0})])
    pausing = res.log["sup_state"] == "PAUSED"
    assert pausing.any()
    assert res.log["t"][pausing][0] == pytest.approx(1.0, abs=0.03)
    assert res.log["t"][pausing][-1] >= 3.0


@pytest.mark.parametrize("t_tap", [0.5, 2.0, 5.0, 8.0, 11.0])
def test_brake_tap_pauses_resumes_and_still_meets_criteria(t_tap):
    _, res = run([(t_tap, {"brake": 0.5}), (t_tap + 0.5, {"brake": 0.0})])
    assert State.PAUSED in path_of(res)
    assert res.success, res.summary()
    assert res.final_state is State.COMPLETED
    paused = res.log["t"][res.log["sup_state"] == "PAUSED"]
    assert paused[0] == pytest.approx(t_tap, abs=0.03)
    assert paused[-1] - paused[0] == pytest.approx(0.5 + CFG["resume_delay_s"], abs=0.1)


def test_brake_tap_brings_the_car_to_a_stop_before_resuming():
    _, res = run([(5.0, {"brake": 0.5}), (7.0, {"brake": 0.0})])
    log = res.log
    paused = log["sup_state"] == "PAUSED"
    assert np.all(log["speed_cmd"][paused] == 0.0)
    assert np.abs(log["v"][paused]).min() <= CFG["standstill_speed"]
    assert np.abs(log["v"][log["t"] > 8.5]).max() > 0.2       # moving again


def test_brake_tap_without_confirm_waits_for_confirm():
    _, res = run([(1.0, {"confirm_button": False}), (5.0, {"brake": 0.5}), (5.5, {"brake": 0.0}),
                  (12.0, {"confirm_button": True})])
    paused = res.log["t"][res.log["sup_state"] == "PAUSED"]
    assert paused[-1] == pytest.approx(12.0, abs=0.05)
    assert res.success, res.summary()


def test_obstacle_pauses_and_clears():
    _, res = run([(5.0, {"obstacle_in_path": True}), (8.0, {"obstacle_in_path": False})])
    assert [tr.reason for tr in res.transitions if tr.to_state is State.PAUSED] == ["obstacle in path"]
    assert res.success, res.summary()


def test_steering_spike_aborts_within_one_tick_and_stops():
    t_spike = 5.0
    _, res = run([(t_spike, {"steering_torque_nm": 5.0}), (t_spike + 0.02, {"steering_torque_nm": 0.0})])
    log = res.log
    first_after = np.flatnonzero(log["t"] >= t_spike)[0]
    assert log["sup_state"][first_after - 1] == "MANEUVERING"
    assert log["sup_state"][first_after] == "ABORTED"
    assert log["sup_abort_reason"][first_after] == "steering_override"
    assert np.all(log["speed_cmd"][first_after:] == 0.0)
    assert abs(log["v"][first_after]) > 0.2                   # it was really moving
    assert abs(log["v"][-1]) <= CFG["standstill_speed"]
    assert log["sup_action"][-1] == "hold" and log["sup_must_shift"][-1] == 1.0
    assert res.final_state is State.ABORTED
    assert not res.success and res.reason == "aborted: steering_override"
    assert not res.checks["completed"]


def frozen_sensors(t):
    return tuple(SensorHealth(name, True, t) for name in CFG["sensor_timeouts_s"])


@pytest.mark.parametrize("override,reason", [
    ({"steering_torque_nm": 5.0}, "steering_override"),
    ({"accelerator": 0.3}, "accelerator_override"),
    ({"gear_request": 1}, "gear_request"),
    ({"door_open": True}, "door_open"),
    ({"seatbelt_on": False}, "seatbelt_off"),
    ({"sensors": frozen_sensors(5.0)}, "sensor_stale"),
    ({"speed": 2.0}, "overspeed"),
    ({"feature_button": True}, "driver_cancel"),
])
def test_scripted_abort_stops_the_car(override, reason):
    _, res = run([(5.0, override)])
    assert res.final_state is State.ABORTED
    assert res.log["sup_abort_reason"][-1] == reason
    assert abs(res.log["v"][-1]) <= CFG["standstill_speed"]
    assert not res.success and reason in res.reason


def test_sensor_dropout_aborts_only_after_its_timeout():
    _, res = run([(5.0, {"sensors": frozen_sensors(5.0)})])
    first = min(CFG["sensor_timeouts_s"].values())
    aborted = res.log["t"][res.log["sup_state"] == "ABORTED"][0]
    assert 5.0 + first < aborted <= 5.0 + first + 0.05


def with_supervisor(**changes):
    params = copy.deepcopy(PARAMS)
    params["supervisor"].update(changes)
    return params


def test_scripted_abort_on_maneuver_time():
    _, res = run(params=with_supervisor(max_time_s=3.0))
    assert res.final_state is State.ABORTED and "maneuver_timeout" in res.reason
    assert res.metrics["time_s"] < 5.0


def test_scripted_abort_on_gear_changes():
    _, res = run(params=with_supervisor(max_gear_changes=0))
    assert res.final_state is State.ABORTED and "too_many_gear_changes" in res.reason


def test_scripted_fault_stops_the_car():
    sensors = tuple(SensorHealth(name, name != "localization", 1e9) for name in CFG["sensor_timeouts_s"])
    _, res = run([(5.0, {"sensors": sensors})])
    assert res.final_state is State.FAULT and res.reason == "fault: sensor_unhealthy"
    assert abs(res.log["v"][-1]) <= CFG["standstill_speed"]


def test_no_path_goes_back_to_slots_available():
    params = copy.deepcopy(PARAMS)
    params["planner"]["max_expansions"] = 0
    sc = make_perpendicular(VehicleParams.from_yaml(), params, np.random.default_rng(SEED))
    sc.start.x, sc.start.y = sc.goal.x + 8.0, sc.goal.y + 8.0   # not reachable by the first shot
    res = run_closed_loop(sc, planner_cfg=params["planner"])
    assert res.planned is None and res.reason == "no path found"
    assert res.final_state is State.SLOTS_AVAILABLE
    assert res.transitions[-1].reason == "no path to this slot"


def test_noise_results_are_repeatable_through_the_supervisor():
    _, a = run(noise_rng=np.random.default_rng(5), pose_noise_xy=0.01, pose_noise_yaw_deg=0.1)
    _, b = run(noise_rng=np.random.default_rng(5), pose_noise_xy=0.01, pose_noise_yaw_deg=0.1)
    assert a.metrics == b.metrics
