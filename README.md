# ecocar_park_sim — 2D parking sandbox

Pure-Python planning and control sandbox for a perpendicular self-parking feature.
No ROS, no GPU. Same toolchain as the LCC/ACC nodes (Python 3.10, NumPy, SciPy, YAML, pytest),
so the modules can drop into `ament_python` nodes in Phase 3.

## Run it

```bash
pip install -r requirements.txt                              # ffmpeg optional for .mp4

python3 scripts/run_demo.py --seed 1 --mcap out/run.mcap     # one maneuver, replay in Foxglove
python3 scripts/run_demo.py --seed 1 --static                # PNG summary, out/park_run.png
python3 scripts/run_demo.py --seed 1 --anim out/park.gif     # GIF animation
python3 scripts/run_demo.py --seed 4 --maneuver head_in --mcap out/head_in.mcap
python3 scripts/monte_carlo.py -n 300                        # pass rate vs. criteria, CSV in out/
python3 scripts/monte_carlo.py -n 200 --noise-xy 0.03 --noise-yaw 0.3 --mcap-failures out/failures/
python3 scripts/monte_carlo.py -n 100 --steer-sweep 20 25 30 36
python3 scripts/run_search_demo.py --seed 1                  # whole flow on a full aisle, PNG + MCAP
python3 scripts/monte_carlo.py -n 50 --mode search           # same, random occupancy, speed and slot
python3 scripts/monte_carlo.py -n 50 --mode search --detector ideal   # with the true slots
python3 scripts/run_search_demo.py --seed 1 --sensors --lidar # with synthetic sensors in the MCAP
python3 scripts/monte_carlo.py -n 50 --mode search --estimator gnss_ins   # OxTS-like pose instead of odometry
python3 scripts/sweep_localization.py -n 40 --workers 6      # pass rate across localization error sizes
python3 scripts/run_demo.py --seed 1 --scenario pedestrian_walks_into_slot --mcap out/ped.mcap   # named case from config/scenarios/
python3 scripts/monte_carlo.py -n 50 --scenario pedestrian_stays --mode search   # the same, many seeds
python3 scripts/straight_stall_cases.py -n 50 --workers 4     # harder stalls: tight neighbor, pillar, narrow aisle or stall, cart
python3 scripts/check_gates.py                                # the safety gates in config/gates.yaml; exits 1 if any gate fails
python3 -m pytest -q
```

`run_demo.py` saves only the outputs you ask for: `--mcap`, `--static` (or `--plot PATH`), `--anim`.

## Viewing runs in Foxglove

1. Save a run: `python3 scripts/run_demo.py --seed 1 --mcap out/run.mcap`
2. In Foxglove, open the `.mcap` file (**Open local file**, or drag it into the window).
3. Import the layout once: **Layouts → Import from file →** `config/foxglove_layout.json`.
4. Drag the timeline to scrub. Time starts at 0 at the beginning of the run, so Foxglove may
   show the clock as a date in 1970.

`--mcap-failures out/failures/` saves one file per failed run, named by seed and failed checks,
such as `seed0042_heading-gear_changes.mcap`. A run with no plan is saved as `seed0007_no_path.mcap`
and shows the lot and start pose. With `--steer-sweep`, each steering limit gets its own subfolder.

For search runs (`scripts/run_search_demo.py`), import `config/foxglove_search_layout.json` instead:
it frames the whole aisle and shows `/slots`, the detected slots appearing as the car passes them.

Add `--sensors` (or `--lidar`) to either demo to see camera footprints, detected lines, ultrasonic
cones and lidar points. The search layout shows them already; with the other layout, turn on the
`/sensors/*` topics in the 3D panel.

`/state` holds the per-tick signals (SI units, plus `delta_deg` and `steer_cmd_deg`). `distance_to_go`
covers the whole remaining path; `segment_distance_to_go` is the part the speed profile brakes on.
`/supervisor` holds the supervisor state, action, reasons and HMI message. `/metrics` holds the final
scores, checks and state transitions. The car box, parked cars and colors are set in
`config/lyriq_vehicle.yaml` and the `foxglove:` section of `config/park_params.yaml`.

`mcap` and `foxglove-schemas-protobuf` are optional: `park_sim/` imports without them and their
tests are skipped.

### Watching a run live

```
python scripts/run_live.py --wait
```

It prints `ws://127.0.0.1:8765` and waits for Enter. In Foxglove, choose **Open connection**, **Foxglove WebSocket**, enter that
address, and import `config/foxglove_live_layout.json` once. Press Enter to start, and Ctrl+C to stop. You see the same topics as
an MCAP, while the run happens, plus `/planner_search`, the planner's search replayed slowly enough to watch, and `/overview`.
It gives the same result as a normal run for the same seed. Options and details are in [docs/live.md](docs/live.md). It needs
`foxglove-sdk` as well, and hasn't yet been seen in the Foxglove app.

## Layout

| File | What it does |
|---|---|
| `config/lyriq_vehicle.yaml` | Published LYRIQ geometry `[pub]` + actuator placeholders `[est]` |
| `config/park_params.yaml` | Lot geometry, start region, planner, controller, **pass criteria** |
| `park_sim/vehicle.py` | Kinematic bicycle (rear-axle ref) + steering dead time / rate limit / lag, speed lag |
| `park_sim/scenario.py` | Perpendicular lot (2.7 × 5.5 m stalls, 7.3 m aisle), parked cars, curbs, distance field |
| `park_sim/reeds_shepp.py` | Reeds–Shepp curves; every candidate is integrated and checked against the goal |
| `park_sim/hybrid_astar.py` | Hybrid A* with RS analytic expansion and a straight final approach into the stall |
| `park_sim/controller.py` | Segment state machine (drive → stop → shift → pre-steer), pure pursuit fwd/rev, delay compensation |
| `park_sim/supervisor.py` | Feature state machine: driver inputs and system events in, actions out ([docs](docs/supervisor.md)) |
| `park_sim/scripted_driver.py` | Scripted driver and timed input overrides for tests |
| `park_sim/driver.py` | Search-run driver: lane keeping, slot selection policies, brakes and confirms ([docs](docs/search.md)) |
| `park_sim/slot_detection.py` | `SlotDetector` interface; ideal, line-based, space-based and fused detectors ([docs](docs/slot_detection.md)) |
| `park_sim/slot_tracker.py` | Slot tracks: Kalman filter on corners, confirmation, stable slot numbers |
| `park_sim/localization.py` | Estimated pose: truth, white noise, odometry or GNSS/INS, chosen in the config ([docs](docs/localization.md)) |
| `park_sim/safety.py` | Safety monitor (swept path against ultrasonic and lidar returns, emergency brake), stream watchdog, steering check ([docs](docs/safety.md)) |
| `park_sim/actors.py` | Scripted pedestrians and vehicles that sensors and collision checks see |
| `park_sim/safety_layer.py` | Wires actors, fault injection, the watchdog and the monitor into a run; safety metrics |
| `park_sim/scenario_files.py`, `config/scenarios/` | Named cases: actors, faults, config overrides and expected outcome |
| `park_sim/gates.py`, `config/gates.yaml`, `scripts/check_gates.py` | Safety gates: fixed seed sets and thresholds every change has to clear; report in `out/gates_report.csv` |
| `config/sensors.yaml` | Sensor mount poses, rates, noise and effects, all `[est]` |
| `park_sim/sensed_map.py` | Obstacle map from ultrasonics, line cameras and the selected slot, for `planner.map: sensed` ([docs](docs/sensed_map.md)) |
| `park_sim/sensors/` | Line cameras, ultrasonics and optional lidar with a common interface ([docs](docs/sensors.md)) |
| `park_sim/simulator.py` | Closed loops at 50 Hz through the supervisor (`run_closed_loop`, `run_search`), scoring, pass/fail per criterion |
| `park_sim/viz.py` | Plots (path, steering, speed; search with slots over time) and GIF/MP4 animation |
| `park_sim/mcap_export.py` | Saves a run as MCAP for replay in Foxglove (optional dependencies) |
| `config/foxglove_layout.json` | Foxglove layout: top-down 3D view, plots, raw state, state transitions |
| `config/foxglove_search_layout.json` | The same layout framed on the full aisle, with `/slots` |
| `park_sim/live.py`, `scripts/run_live.py` | Streams a run to Foxglove as it happens, with the planner's search (optional dependency) |
| `config/foxglove_live_layout.json` | Layout for live runs: top-down view with the search, run overview, plots |

## Vehicle numbers

| Parameter | Value | Source |
|---|---|---|
| Wheelbase | 3.094 m | published |
| Length / width (no mirrors) | 4.996 / 1.977 m | published |
| Height | 1.623 m | published (visualization only) |
| Track | 1.674 m | published |
| Curb-to-curb turning circle | 12.1 m | published |
| Max road-wheel angle | 36.4° | derived from turning circle |
| Planner steering limit | 92% of lock (33.5°), rear-axle R_min 4.68 m | design choice |
| Overhangs | 0.951 m front / rear | **estimate** (total split evenly) — measure |
| Steering rate / lag / dead time | 35°/s, 0.10 s, 0.10 s | **placeholder** — measure with torque requests |
| Speed lag, shift time | 0.30 s, 1.0 s | **placeholder** |

## Pass criteria (config/park_params.yaml)

Lateral ±0.15 m · heading ±2° · depth ±0.20 m · clearance ≥ 0.30 m · ≤ 3 gear changes · ≤ 5 km/h · ≤ 60 s

## Results so far (reverse-in, both neighbors occupied, random lot + start)

| Run | Pass rate | 95th pct errors |
|---|---|---|
| 200 runs, no noise | 100% | lat 2.3 cm, head 0.50°, depth 0.2 cm, clearance ≥ 0.36 m (5th pct) |
| 100 runs, pose noise 1 cm / 0.1° | 100% | lat 3.6 cm, head 0.61° |
| 200 runs, pose noise 3 cm / 0.3° (white, 50 Hz) | 85.5% | head 2.76° — fails on heading |
| Steering lock 30° / 25° / 20° | 99% / 98% / 99% | 20° needs up to 5 gear changes, ~34 s |
| Search, ideal detector, 200 runs: full aisle, occupancy 50–95%, 5–15 km/h, random free slot | 98.5% | lat 3.9 cm, head 0.72°, clearance ≥ 0.41 m (5th pct); planning at most 1.41 s |
| Search, fused detector, 200 runs, default sensor noise and odometry errors (0.5% scale, 0.1°/s bias) | 99% | lat 6.4 cm, head 1.13°, depth 4.4 cm, clearance ≥ 0.39 m (5th pct); 0/1154 false slots, 51/1129 free stalls missed. Seed 151 touches an obstacle (clearance 0.0 m) |
| Search, fused detector, 200 runs, `gnss_ins` estimator | 99% | lat 5.9 cm, head 1.05°, depth 3.3 cm, clearance ≥ 0.38 m (5th pct) |
| Search, lines detector, 50 runs | 98% | lat 5.3 cm, head 0.95°; 0/265 false slots, 36/287 free stalls missed |
| Search, space detector, 50 runs (ignores the paint) | 88% | lat 16.7 cm, depth 16.3 cm; 0/265 false slots, 33/285 free stalls missed |

The ideal and sensor-based runs use different driver policies (a set slot id, or the first slot past a
place along the aisle), so their pass rates aren't a direct comparison. All three ideal-detector failures
are on clearance (0.08–0.20 m), after searching at about 14 km/h. See [docs/search.md](docs/search.md) and
[docs/slot_detection.md](docs/slot_detection.md). Search lines and space rows predate the larger odometry
errors in [docs/localization.md](docs/localization.md).

Takeaways: the planner and controller meet the criteria cleanly; **localization jitter** is the
first thing that breaks it (filter the pose or use a smoothed OxTS output), and steering authority
is not the bottleneck for this start region.

## Known simplifications / next steps

- Footprint uses body width; mirrors (2.207 m) not in the collision outline yet.
- Steers at standstill before each segment (production-like) — confirm the EPS can do this with torque requests.
- Only perpendicular slots; parallel parking would add a scenario + criteria.
- `run_closed_loop` starts "just driven past the slot"; `run_search` covers stopping further past it.
- Phase 2: feed noisy synthetic slot detections instead of the true goal. Phase 3: wrap planner/controller in ROS 2 nodes.
