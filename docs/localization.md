# Localization

`park_sim/localization.py` is where the car's estimated pose comes from. Every estimator has one
interface, `update(measurements, dt) -> PoseEstimate` (x, y, yaw and a 3x3 covariance).

`localization.estimator` in `config/park_params.yaml` picks one:

| Estimator | Pose | Frame | Used by default |
|---|---|---|---|
| `truth` | Ground truth, no error | map | Single stall; search with the ideal detector |
| `white` | The default estimator, plus independent noise each time the controller reads it | as that estimator | Whenever `--noise-xy` / `--noise-yaw` are given |
| `odometry` | Dead reckoning from wheel speeds and yaw rate | odom | Search with a sensor-based detector |
| `gnss_ins` | Truth plus an OxTS-like error | map | Never; ask for it |

`auto` (the config default) picks `truth` or `odometry` as shown. Scripts take `--estimator`:

```bash
python scripts/monte_carlo.py -n 200 --mode search --detector fused --estimator gnss_ins
python scripts/run_search_demo.py --seed 1 --estimator gnss_ins
python scripts/run_demo.py --seed 1 --estimator odometry
python scripts/sweep_localization.py -n 40 --workers 6
```

## Odometry

Each error source has its own key under `localization.odometry` and is off at 0.

| Key | Default `[est]` | Effect |
|---|---|---|
| `rate_hz` | 50 | Sample rate; the estimate holds between samples |
| `wheel_source` | `rear_average` | Or `four_wheel`: all four wheels, the fronts reduced to their part along the heading |
| `wheel_scale_error` | 0.5% | Fixed size, sign drawn per run |
| `wheel_noise_std` | 0.01 m/s | Per wheel, per sample; a stopped wheel has none |
| `wheel_quantization` | 0.01 m/s | |
| `yaw_rate_bias_deg_s` | 0.1 deg/s | Fixed size, sign drawn per run |
| `yaw_rate_noise_std_deg_s` | 0.05 deg/s | |
| `yaw_rate_quantization_deg_s` | 0.01 deg/s | |
| `standstill_calibration` | on | See below |

`latency_s` (shared with `gnss_ins`) delays the output. It's off by default because the detectors don't
match pose to sensor time yet: they place this tick's camera segments using a pose from `latency_s` ago,
so at search speed a 20 ms delay moves a slot by several centimeters along the aisle. A real node would
use the sensor timestamps.

**Standstill calibration.** With all wheels stopped the car isn't turning, so the gyro's reading is its
bias. The estimator holds heading while stopped, averages the readings, and after
`standstill_min_time_s` removes that average from later readings. It uses only wheel speeds and the
gyro, never the true pose.

Without it, a 0.1 deg/s bias alone gives about 15 cm of error over a 17 s maneuver (1.6 deg of heading).
A car that stops first, as in a search run, starts its maneuver with the bias mostly removed. The
single-stall run only stands still for 0.1 s before moving, so there the bias is first removed at the
first gear change.

Scale error can't be calibrated this way. It moves the estimate by 0.5% of the net displacement, about
4 cm on a 9 m maneuver, and that is most of what is left.

## gnss_ins

Approximates the OxTS output, all values `[est]` until replaced with OxTS spec values. The pose is the
true pose plus a first-order Gauss-Markov error, 2 cm per axis and 0.05 deg of heading, with a 30 s time
constant. Normal heading error stays under 0.2 deg.

Degraded periods start at random (`degraded.rate_per_s`, one per 200 s) or at fixed times
(`degraded.windows`). During one, an extra error grows at `position_growth_m_s` and
`heading_growth_deg_s` in a direction drawn at its start, then decays with `recovery_tau_s`. The
covariance grows and recovers with it.

## Frames

- **Odometry** starts on the true start pose and drifts from it.
- **gnss_ins** is in the map frame, with its error.

How each part uses them:

| Part | Pose used |
|---|---|
| Sensor models, scoring | True pose |
| Sensor-based detector, tracker | Estimate; slots are in the estimator's frame |
| Ideal detector | True pose; slots are in the map frame |
| Controller | Estimate, always; `run_closed_loop` too |
| Planner, map frame estimate | Starts from the estimate, goal from the slot as detected, obstacles from the true lot |
| Planner, odometry | Car and slot go to the map frame using the current drift; the path comes back into the odometry frame |

The odometry plan-time conversion uses the true pose, because by default the planner's obstacle map is
the true lot. That is the open issue from `docs/slot_detection.md`, and it's optimistic. With
`planner.map: sensed` ([sensed_map.md](sensed_map.md)) there is no conversion: the map, the slot and the car
are all in the estimate's frame, and the true pose isn't read.

## Metrics

Added to `RunResult.metrics` and, except for `truth` runs, the Monte Carlo CSV. Leaving them out for
`truth` keeps the baseline CSV byte-identical.

| Metric | Meaning |
|---|---|
| `est_err_final_m`, `est_yaw_err_final_deg` | Estimate against truth at the end of the maneuver |
| `est_err_max_m`, `est_yaw_err_max_deg` | The largest during the maneuver |
| `est_drift_final_m` | The position error the maneuver itself added, with the error at its start removed |

For odometry in a search run, the first four include the drive down the aisle, before the first stop
has calibrated the gyro. They can reach a meter or more without hurting the park, because slots are
tracked in the same drifting frame. `est_drift_final_m` is the one to compare with the pass criteria.

The MCAP has `map -> base_link_est` on `/tf`, and `est_x`, `est_y`, `est_yaw`, `est_error_m` and
`est_sigma_xy` on `/state`. The log has `est_x`, `est_y`, `est_yaw`, `est_sigma_xy`, `est_sigma_yaw`.

## Plant realism

Both live under `realism:` in `config/lyriq_vehicle.yaml` and are off by default.

- **Tire scrub** (`tire_scrub_gain`): sideways slip proportional to the road-wheel angle, fading out at
  `tire_scrub_crawl_speed`. At standstill it only acts while the wheels are being turned. Wheel speeds and
  yaw rate can't see it, so with odometry it becomes estimation error.
- **Road grade** (`road_grade_percent`): a slope along map +x, as a disturbance in the speed loop. Brakes
  hold the car at a speed command of zero. The speed loop has no integrator, so a grade shifts the speed
  by `g * grade * tau`, about 0.03 m/s per percent. Creep speed is 0.15 m/s, so a steep uphill can stall a
  maneuver.

## Sweep

`scripts/sweep_localization.py` runs the search Monte Carlo with the fused detector for every yaw-rate
bias with every wheel scale error, then each GNSS error size. The grid is in `localization.sweep`. It
writes `out/sweep_localization.csv` and `out/sweep_localization.txt`, which lists the cells below 95%.

Planning has a 2 s wall-clock limit, so too many workers can fail runs that would otherwise pass. Each
cell records its slowest plan, and the summary warns if any plan came within 75% of the limit. Very large
errors can also make the planner hit the limit on its own, so if the warning appears, rerun the slow cells
with no workers to tell the two apart.

Results, 40 runs per cell. Below 95%: bias 0.2 deg/s with scale 1% or more, all of bias 0.5 deg/s, and GNSS
error x5 or more (10 cm per axis, 0.25 deg). The defaults (bias 0.1 deg/s, scale 0.5%; GNSS x1) pass 40/40.

| Odometry pass rate | Scale 0.2% | 0.5% | 1% | 2% |
|---|---|---|---|---|
| Bias 0.02 deg/s | 100 | 100 | 98 | 100 |
| 0.05 | 100 | 100 | 98 | 100 |
| 0.1 | 100 | 100 | 98 | 95 |
| 0.2 | 98 | 98 | 92 | 85 |
| 0.5 | 15 | 10 | 10 | 28 |

| gnss_ins error factor | x0.5 | x1 | x2 | x5 | x10 |
|---|---|---|---|---|---|
| Pass rate | 100 | 100 | 95 | 80 | 20 |
