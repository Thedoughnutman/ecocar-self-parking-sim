# Search phase

`simulator.run_search` runs the whole feature: drive, search, select, stop, park.

```bash
python3 scripts/run_search_demo.py --seed 1 --policy select_by_index --slot 35
python3 scripts/monte_carlo.py -n 50 --mode search
```

Settings are in `config/park_params.yaml` under `aisle_lot:`, `slot_detection:` and `search:`.

## What happens in a run

1. The car enters the aisle at `speed_kph`, already moving, and the driver presses the feature button.
2. The slot detector ([slot_detection.md](slot_detection.md)) reports free stalls as the car passes them, and
   the supervisor moves to SLOTS_AVAILABLE.
3. After `reaction_time_s`, the driver selects a slot according to its policy and brakes.
4. Once the car is stopped, the supervisor asks for confirmation, and the driver presses confirm
   after `confirm_delay_s`.
5. In PLANNING, the planner gets the car's actual pose and a goal worked out from the slot's corners
   (with `planner.map: sensed`, the estimated pose and the sensed map: [sensed_map.md](sensed_map.md)).
6. The maneuver runs as in `run_closed_loop`.

If planning fails or takes `planning_time_limit_s` or longer, the supervisor returns to
SLOTS_AVAILABLE. The driver won't pick that slot again: it selects another one its policy allows,
or else drives on. With nothing left to select, the run ends at the end of the aisle.

## Driver policies

| Policy | Selects |
|---|---|
| `select_first` | The slot seen earliest |
| `select_nearest_after_passing` | The slot whose opening is nearest the car when the driver decides |
| `select_by_index` | Only the slot with id `slot_index`; if it never appears, the car drives to the end of the aisle |
| `select_first_after` | The slot seen earliest with its center past `after_x`; at the end of the aisle, the nearest slot |
| `never_select` | Nothing; the car drives the aisle with the supervisor in SEARCHING or SLOTS_AVAILABLE |

With the ideal detector, slot ids are `index` in the first row (y = 0 side) and `stalls_per_row + index`
across the aisle, with index 0 at the entry end. Sensor-based detectors number slots as they confirm them.

## Slot detection

This section describes the ideal detector; the sensor-based ones are in
[slot_detection.md](slot_detection.md). A free stall becomes a candidate when both of these are true:

- **In range:** both ends of its opening have been within `detection_range` of the side of the car's
  body nearest them, at the same time.
- **Passed:** the rear axle is at least `pass_margin` past the far end of the opening.

The ideal detector has no occlusion or noise, reads occupancy from the lot, and never forgets a slot.

To use another detector, subclass `SlotDetector`, implement `update(t, pose)` to return the slots on
offer, and pass it as `run_search(sc, detector=...)`. It can also take sensor data through `observe()`
and ask for the estimated pose with `uses_estimated_pose`. Slot ids can be your own. The car aims for the slot as reported, and the final pose is scored
against the true stall nearest it, so detection error shows up in the lateral and depth results.

A `SlotCandidate` holds `slot_id`, `corners` (four map-frame points, opening first), `entry_side`
(`"+y"` or `"-y"`, the map direction out to the aisle), `slot_type`, `occupied`, `confidence` and
`first_seen_t`. The supervisor reads only `slot_id` and `confirmed`.

## Metrics

These are added to `RunResult.metrics`. Times are `nan` for phases a run didn't reach.

| Metric | Meaning |
|---|---|
| `search_distance_m` | Driven from the feature button to the selection |
| `selection_to_stop_s` | Slot selected until the car is stopped |
| `confirm_to_parked_s` | Confirm until the maneuver finishes, including planning |
| `total_time_s` | The whole run |
| `planning_time_s` | Wall-clock time of the last plan; every attempt is in `RunResult.search.plan_attempts` |
| `slots_detected` | Slots reported by the end of the run |

The final-pose checks and limits are the same as `run_closed_loop`, and speed and time cover the
maneuver only. `min_clearance` covers the whole run, including the drive down the aisle.

## Things to know

- **Plant limits:** `lyriq_vehicle.yaml`'s 1 m/s and 0.6 m/s² are the parking controller's limits, not
  the car's. While the driver has the car, the plant uses `speed_kph` and `brake_decel` instead. The
  parking limits return as soon as the system has the car.
- **Planning limit is wall clock:** the sim clock doesn't advance during planning, so the limit is
  measured in real time, and a very slow or busy machine can fail a plan that would otherwise pass.
  `planning_timeout_s` in the supervisor is set to the same 2 s.
- **Free stalls:** `guaranteed_free` is a minimum. With `occupancy_prob: 1.0` a lot has exactly 2 or 3
  free stalls, and at lower occupancy it usually has more.
- **Long reverses:** stopping well past a stall at high search speed means a long reverse at full
  controller speed before a tight turn. Steering then saturates and tracking error reaches about
  0.28 m, which is the main source of search Monte Carlo failures.

## Viewing a run

`run_search_demo.py` saves a PNG, with slots shaded by when they were first seen and a timeline of
the slot count. It also saves an MCAP if the optional packages are installed. In Foxglove, import
`config/foxglove_search_layout.json`. `/slots` gains each slot when first seen and recolors the chosen
one when selected; `/plan` appears when the plan is made. `/metrics` lists the slots and plan attempts.
