# Slot detection and tracking

`slot_detection.detector` in `config/park_params.yaml` picks how a search run finds slots:

| Detector | Finds slots from | Confidence |
|---|---|---|
| `ideal` | The true lot, with no errors | 1.0 |
| `lines` | Painted stall dividers, with ultrasonics (and lidar) for occupancy | 0.9 |
| `space` | The space between parked cars, from the side ultrasonics | 0.5 |
| `fused` (default) | Lines first; space between cars where the lines don't give a slot | 0.9 or 0.5 |

```bash
python3 scripts/run_search_demo.py --seed 1 --detector fused
python3 scripts/monte_carlo.py -n 50 --mode search --detector fused
```

The sensor detectors produce track numbers as slot ids, not stall ids, so `select_by_index` isn't useful
with them. `select_first_after` aims for a place along the aisle instead.

## Frames

Sensor detectors work in the frame of the active estimator, by default odometry, which starts on the
true start pose and drifts. See [localization.md](localization.md) for the estimators and their errors.

- **Detector and controller** use the estimated pose, so slot and car errors show up in the final pose,
  which is scored against the true stall.
- **Planner:** by default its obstacle map is still the true lot. `planner.map: sensed` plans from what the
  sensors saw instead ([sensed_map.md](sensed_map.md)), and none of the rest of this item applies. At
  plan time the car and slot go into the map frame using the current odometry error, and the path comes
  back into the odometry frame. The planner therefore always sees the car correctly placed among the
  obstacles, which is optimistic compared with a real obstacle map. With `gnss_ins`, already in the map
  frame, the planner starts from the estimate instead.

## Lines

Every `fit_period_s`, using camera segments from the last `window_s`:

1. **Transform** each segment to the estimator's frame with the estimated pose.
2. **Merge** collinear segments into lines, with a total-least-squares fit. A line needs `min_segments`
   segments, which removes one-off false positives.
3. **Find dividers:** lines roughly square to the direction of travel with a real end on the car's side.
   Ends cut off by a camera footprint or the ego body don't count as real ends.
4. **Find corners:** if a tick or other crossing line meets the divider's end, the corner is where they cross.
5. **Fit a slot** between neighboring dividers `spacing` apart, parallel, with ends aligned. The opening comes
   from the corners. The depth comes from a rear line if one is seen, or else `standard_depth`.

A missing divider leaves its neighbors about 5.4 m apart, which is outside `spacing`, so the slot isn't
offered rather than being offered wrongly.

From the aisle centerline the rear line is out of camera reach, so depth starts as an assumption and is
corrected during the maneuver.

## Occupancy

A slot is offered only if it is confirmed, the car has passed it by `pass_margin`, and the sensors show it free:

- **Enough evidence:** at least `min_readings` side-ultrasonic readings across the slot. No coverage means not
  offered, rather than assumed free.
- **Nothing inside:** fewer than `min_hits` returns inside the slot, shrunk by `margin`.

An ultrasonic can't tell where in its cone a return came from. The return is placed on the sensor's axis, and
the part of the slot where cone spread could have picked up a car next door isn't used. Cones wider than
`max_cone_deg` can't place a return in a slot at all, so only the 30-degree side pair counts. Lidar returns are
used directly.

## Space

The side ultrasonics read the neighboring cars, then nothing, then the next car. The open stretch is corrected
for cone spread, then filled with as many stalls as fit. The opening is placed from the neighbors' bumper line.
The sensors can't reach the back of a stall, so `space.depth` is assumed if the gap reads clear as far as they
can see.

A gap needs a car at each end, so space at the very start or end of a row isn't found. Corners are about 10 cm
off on average, against 2-4 cm for lines, so these slots have lower confidence.

## Tracker

Each slot is a track with four corners and a 2 x 2 covariance per corner, updated by a simple Kalman filter.

- A detection matches a track within `gate_distance` and `gate_heading_deg`.
- A track is confirmed after `confirm_detections` matches and then gets a slot number it never loses.
- An unconfirmed track with no match for `unconfirmed_timeout_s` is dropped, and doesn't use up a number.
- In `fused`, a line detection takes over a space track. A space detection never moves a line track.

## Refinement during the maneuver

Once a slot is selected, its occupancy is frozen and lines in view keep correcting it, one slot axis at a time:

- **A divider** corrects the corners across it, but only within `line_reach` of the stretch in view. A short
  line's small tilt would become a large error several meters away.
- **A tick** corrects the entry corners along the slot, and **a rear line** corrects the rear ones.
- **With one divider in view,** the other side is placed one slot width away.

During the maneuver, if the goal moves more than `replan_threshold_m` or `replan_threshold_deg`:

| Where the car is | What happens | Logged as |
|---|---|---|
| Before the final straight | Replan from the current pose | `replan` |
| Same, but no path | Keep the current path | `replan_failed` |
| Same, but the new path would exceed `max_gear_changes` | Keep the current path | `replan_refused` |
| Inside the final straight | Only the stop point moves, along the straight | `stop_adjust` |

Replans are also limited by `max_replans` and `min_replan_interval_s`. Every event is in
`RunResult.search.refinements` and in the MCAP's `/metrics`, and each new plan is published on `/plan`.

## Metrics

These are added to `RunResult.metrics` and the Monte Carlo CSV, and the Monte Carlo prints a summary.

| Metric | Meaning |
|---|---|
| `slot_corner_error_m` | Mean corner error against the true stall, when each slot is first offered, with odometry drift removed |
| `selected_corner_error_m` | The same, for the selected slot at the end of the run |
| `time_to_confirm_s` | First detection to confirmation, mean over confirmed tracks |
| `slots_offered`, `slots_false_positive` | A false positive isn't within half a stall width of a free stall's center |
| `free_stalls_passed`, `free_stalls_missed` | A missed stall is one the car passed, free, and never offered |
| `replans`, `stop_adjustments` | Counts from the refinement log |

## Limits

- Only perpendicular slots, in a lot whose rows run roughly along the direction of travel.
- By default, the planner's obstacle map is the true lot, as described above.
- Slots are only found while the car passes them. A stall the car has already passed isn't found on the way back.
- No re-localization: odometry drift is only corrected relative to the selected slot, by refinement.
