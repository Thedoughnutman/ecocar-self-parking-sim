# Safety monitor, actors and fault injection

`park_sim/safety.py` holds the parts that could run on the car: the monitor, the stream watchdog and
the steering check, using only the standard library and numpy. `park_sim/actors.py` (moving obstacles)
and `park_sim/safety_layer.py` (wiring into a run, metrics) are sim-only. All numbers are in
`config/park_params.yaml` under `safety:` and `actors:`, and in `config/lyriq_vehicle.yaml` under
`brake:`. Every one is [est].

## When it is on

`safety.enabled: auto` turns the monitor on for runs with actors or injected faults. Plain runs don't
get it, so a sensor return can't change the baseline and search results. `--safety` on
`monte_carlo.py` forces it on. The watchdog always runs: it only reports stream ages, which stay inside
their limits in a healthy run.

## Monitor

Each tick the system has the car, the monitor sweeps the body along the remaining planned path:

- **Length:** stop distance plus `standoff_m`. Stop distance is `v * (sensor_latency_s + brake dead time +
  decel / 2 jerk) + v^2 / 2 decel` at the service deceleration. A stopped car uses the speed it would
  pull away at, so an obstacle that only just let it stop doesn't count as clear.
- **Width:** the body plus `lateral_margin_m` sideways and `end_margin_m` front and rear.
- **Frame:** the estimated pose, like the controller. No ground truth.
- **Start:** the path point nearest the car now. The controller isn't stepped while the car is stopped, so
  its own nearest point is where the brakes came on. Starting there would spend the horizon on ground
  already driven, and an obstacle the car had just stopped for would look clear.

A return in the sweep sets `obstacle_in_path`. If time-to-collision (distance along the path divided by
the current speed) is under `ttc_emergency_s`, it also asks for the emergency deceleration directly,
without going through the supervisor.

### Where an ultrasonic return is

A reading is a distance with no bearing, so the object is somewhere on an arc across the cone. A reading
counts only if the **whole arc** is in the sweep. A parked car at the edge of a cone is on the arc but
outside the sweep, so it doesn't read as something in the path. Placing returns on the sensor's axis, and
matching readings between sensors, both stopped the car at parked neighbors in most runs.

The cost is range. With a 60 deg cone, the rear inner sensors only get the whole arc inside at about
1.3 m, and side and corner sensors rarely do. So within `close_range_m` any part of the arc counts.

`confirm_hits` consecutive readings are needed before a return counts. Lidar returns are real positions,
so they count directly, one scan at a time.

### None means "nothing seen or in the blind zone"

A return is forgotten after `clear_readings` readings of None, but only while its point is outside that
sensor's blind zone. Inside it, None says nothing and the return stays. It goes when the sensor reports
something further off, when the car has moved so that the point is back in view, or when another sensor
or the lidar has a clear view past it.

## Watchdog

`StreamWatchdog` produces the supervisor's `SensorHealth` inputs from what actually arrived, in place of
the scripted drivers' always-healthy values. There is still one stale check, in the supervisor.

| Stream | Last update | Limit |
|---|---|---|
| `localization` | when the current estimate was measured, so latency counts | `sensor_timeouts_s` |
| `wheel_speed` | every tick, unless injected away | `sensor_timeouts_s` |
| `slot_perception` | newest new reading from the run's sensors; with no sensors, every tick | `sensor_timeouts_s` |
| each ultrasonic, lidar | its last new reading | `watchdog.*_max_age_s`, added to the timeouts when the monitor runs |

A reading identical to the last one while the car moves isn't new. Real readings carry noise, so that
is a stuck sensor with fresh timestamps. **Not caught:** a sensor stuck on "nothing", because repeated
None is normal, and repeats at a stop, because those are normal too.

`SteeringMonitor` reports `steering` as unhealthy, which sends the supervisor to FAULT, once the road-wheel
angle has been more than `steer_error_deg` off its command and moving slower than
`steer_stuck_rate_deg_s` for `steer_fault_time_s`. Slewing at the rate limit doesn't trigger it.

## Supervisor

- Paused for an obstacle for `replan_after_s`: `request_replan`, repeated every `replan_after_s`.
- Blocked for `abort_after_s`: ABORTED with `obstacle_timeout`.
- The blocked clock keeps running through a flicker. It restarts only after the path has been clear for
  `resume_delay_s`. A pressed brake never starts it.

### Blocked replans, the 2-replan cap and the gear limit

They reuse the simulator's replan path and the same `gear_changes_with` / `continued_on` helpers.

- **Cap:** they have their own, `max_blocked_replans`, counting accepted replans. They don't use up
  `slot_detection.refinement.max_replans`, and refinement doesn't use up theirs. The two solve different
  problems, and a car that had already refined its goal twice should still be able to go around someone.
- **Gear limit:** the same rule as refinement. A plan that would take the maneuver over
  `max_gear_changes` is refused, because the supervisor would abort as soon as the count passed the limit.
  The car stays paused and the abort timer keeps running.
- **Failed plans** don't count against the cap, and the car keeps its old path.
- **Planner's view:** the true lot plus the ground-truth footprints of actors standing still. A moving
  actor is invisible to it. This is a limitation: the car has no such knowledge. With `planner.map: sensed`
  ([sensed_map.md](sensed_map.md)) it gets no footprints, only what the sensors have seen.

**The real planner rarely finds a way around.** With a pedestrian on the direct route, plans ran out the
2 s limit in all 27 probed stops, and in the 4 retried, the 20 s limit too. It only found a path when
the pedestrian was away from the direct route. So in practice a blocked car waits, then aborts. Only
the mechanism after a plan is found has been tested end to end, using a stub planner.

## Brakes

`Vehicle.step(steer, speed, brake_decel)`: the request acts after `dead_time_s` and builds at `jerk_mps3`,
on top of the speed loop. It can only slow the car, and it does nothing unless asked. In a run with the
monitor on, the supervisor's `stop` action asks for `service_decel_mps2` and an emergency for
`emergency_decel_mps2`. In a plain run, stops stay with the speed loop as before. From 1 m/s the plant
stops in 0.36 m at the service deceleration.

## Actors

Pedestrians are a 0.5 m diameter circle (the prompt's "0.5 m circle" could also mean radius) at
0.5-1.5 m/s. Vehicles are a rectangle at 5-15 km/h. They follow a polyline, in the map frame or relative
to the target stall, start on a time or a state plus a delay, and can stop for a while or for good. They
appear in the ultrasonics, the lidar, the camera's occlusion and the ground-truth checks.

**Actors yield by default.** Each holds while the space `yield_gap_m` ahead comes within `yield_side_m` of
the car, because a script that walks into a stopped car would count as a collision the feature can't
avoid. Set `yield_gap_m: 0` for one that doesn't.

## Faults

Scripted events (`scripted_driver.py`) inject them, timed on the run clock or from a state:

```python
run_closed_loop(sc, events=[(("MANEUVERING", 3.0), {"sensor_frozen": ["ultrasonic"]})])
```

| Key | Effect |
|---|---|
| `sensor_dropout: names` | the streams stop arriving |
| `sensor_frozen: names` | the streams keep arriving, with new timestamps and the last value |
| `pose_delay_s: s` | localization falls this far behind (prompt 6's latency; odometry and gnss_ins only) |
| `steer_not_followed: true` | the road wheels stop answering |
| `eps_saturation_deg: deg` | the road wheels go no further than this |

## Scenarios

`config/scenarios/*.yaml` holds each named case: actors, events, any config overrides, and an `expect` block
for prompt 9's catalog. `scenario_files.run_scenario_file` runs one, and nothing checks `expect` yet.

```
python scripts/run_demo.py --seed 1 --scenario pedestrian_walks_into_slot --mcap out/ped.mcap
python scripts/monte_carlo.py -n 50 --scenario pedestrian_stays --mode search
```

## Metrics

Runs with the monitor on add these to `RunResult.metrics` and the CSV, and only those runs, so the
baseline CSV stays byte-identical: `actor_collisions`, `min_actor_distance_m`,
`stop_distance_after_detection_m` (the longest, measured on the true path), `time_paused_s`,
`emergency_brakes`, `obstacle_stops`, `blocked_replans`, plus `outcome` and `outcome_reason` (in
`RunResult.safety` and the CSV). `actor_collision` joins the pass criteria.

## MCAP

`/actors` and `/safety_sweep` are SceneUpdates, and `/safety` is JSON with the sweep, time-to-collision and
requests. Both layouts show them, but they haven't been checked in the Foxglove app.

## Known limits

- **Seed 151 (fused):** the monitor fires 0.16 s before the contact, too late to avoid it, and then
  stops the car and aborts instead of finishing. The object is off the planned sweep until the body
  is almost on it, because the car is 0.45 m from where it thinks it is.
- **Sideways approach:** side and corner sensors only count inside `close_range_m`. The nose swinging
  toward something that wasn't there at planning time can get close: 0.17 m was the lowest in
  `vehicle_passes_in_aisle` over 40 runs (20 per flow), with no contact.
- **Mutual yield:** in `vehicle_passes_in_aisle`, 9 of those 40 runs abort with `obstacle_timeout`. The
  car stops for the vehicle while the vehicle's yield rule stops it for the car, and neither moves again
  (0.00 m in the last 15 s of the four aborted runs checked). Replans fail or are refused, so the car
  aborts at `abort_after_s`. It is clean and there is no contact, but a real driver would go around. It
  comes from the scripted actor's rule as much as from the monitor.
- **Results with replans aren't fully repeatable:** a blocked replan has a wall-clock limit
  (`replan_time_limit_s`), so whether it finds a path in time depends on machine load. Seed 9 of
  `vehicle_passes_in_aisle` aborted in one run and completed in the next, with no behavior change between.
  Gates on these scenarios should allow for that. Plain runs don't blocked-replan and stay byte-identical.
- **Search phase:** the monitor only acts while the system has the car. The scripted driver doesn't brake
  for anything in the aisle.
- **Time-to-collision** uses the car's speed along the path, not the obstacle's own motion.
