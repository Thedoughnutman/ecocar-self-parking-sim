# Synthetic sensors

`park_sim/sensors/` models what the car's sensors would report in the 2D lot. The sensor-based slot
detectors ([slot_detection.md](slot_detection.md)) use these measurements. The planner's obstacle map
is the true lot by default, or built from them with `planner.map: sensed` ([sensed_map.md](sensed_map.md)). The
driver still uses the true lot. Settings are in `config/sensors.yaml`, and every value there is
a placeholder `[est]`.

```python
from park_sim.sensors import make_suite
res = run_search(sc, sensors=make_suite(seed=1))     # or run_closed_loop; one suite per run
res.measurements                                     # in time order
```

```bash
python3 scripts/run_search_demo.py --seed 1 --sensors --lidar
```

## Interface

Every sensor has a `name`, a `mount` (pose in the vehicle frame: rear-axle origin, x forward, y left),
a `rate_hz`, and `measure(scenario, vehicle_state, t, rng)`, which returns a timestamped dataclass.

`SensorSuite` calls each sensor at its own rate. Each sensor has its own random stream, derived from
the seed and its name, so the same seed gives identical measurements and turning the lidar on doesn't
change the other sensors' noise.

## Line cameras

Three bird's-eye-view cameras (rear, left, right), each with a 5 m x 4 m ground footprint that
starts 0.5 m behind its mount. There is no front camera: it is permanently out on this LYRIQ, so
nothing should rely on one. All ultrasonics work. A `LineCameraMeasurement` has:

- `segments`: `(K, 2, 2)` start and end points, vehicle frame, clipped to the footprint.
- `truth_line`: index into `scenario.lines`, or `FALSE_POSITIVE`. This is sim-only, for tests and
  display; a real perception stack wouldn't have it.

| Effect | Setting | Behavior |
|---|---|---|
| Noise | `noise` | Gaussian on each endpoint coordinate, new every frame |
| Faded paint | `faded` | A whole line is missing, with probability `p_faded` |
| Gaps | `gaps` | Random stretches of a line are missing, at `rate_per_m` |
| Parked cars | `occlusion.parked_cars` | Line parts with a car on the straight line from the camera are hidden |
| Own body | `occlusion.ego_body` | Line parts under the ego car are hidden |
| False positives | `false_positives` | Short random segments, at `fp_rate` per second per camera, new every frame |

Each one has its own switch. Faded paint and gaps are wear on the lot, not camera noise, so they stay
put for the whole run and all the cameras see the same wear. The wear is drawn once per lot.

Visibility is tested at points every `sample_step` along a line. A fully visible line keeps its exact
ends, but an end created by occlusion or a gap lands on that grid.

From the aisle centerline, a 4 m footprint barely reaches past parked cars' bumpers, so occlusion
removes little paint there. It matters more once the car is closer to a row or maneuvering.

## Lot markings

`markings:` under `scenario:` and `aisle_lot:` sets each lot's paint, with no effect on planning:

- `rear_line`: the line across the back of each stall, now one line per stall so that wear applies per stall.
- `aisle_ticks`: short ticks of `tick_length` across the aisle end of each stall line.

## Ultrasonics

Twelve sensors (6 front, 6 rear, including 2 side-corner ones at each end) plus a side-looking pair for
finding the space between cars during search. Each `UltrasonicMeasurement` has `distance`, the meters
to the nearest obstacle in the cone, or `None`.

- **Cone:** 60 degrees, 0.2-4.5 m, computed exactly rather than with a fan of rays. The nearest point
  in a cone can't be hidden by anything, so no occlusion test is needed.
- **Blind zone:** an obstacle closer than 0.2 m reads as `None`, not as a close object, so a reader
  can't treat `None` as "clear".
- **Side pair:** 30 degrees rather than 60. From the aisle centerline, a 60-degree cone also catches
  the cars on each side of a free stall, so it would miss the gap.

## Lidar

Optional and off by default (`lidar.enabled`, or `--lidar` on the demos). Casts `beams` rays over 360
degrees against the obstacle polygons. A `LidarMeasurement` has `angles` and `ranges` (nan = no return)
in the lidar's frame, and `points()` gives the returns as x, y. The ego body doesn't block it.

## Speed

Ray casting and clipping are vectorized with NumPy. A test runs the plant, lane keeping and all 19
sensors down a full lot at 50 Hz and requires at least 5x real time. It measured 14-16x on the
development machine. That excludes planning and final scoring.

## Viewing in Foxglove

These channels appear only for the sensor types that ran, all in `base_link`:

| Topic | Content |
|---|---|
| `/sensors/camera_footprints` | Each camera's footprint, once |
| `/sensors/lines` | Detected segments, with false positives in their own color |
| `/sensors/ultrasonic` | Cones, with an arc at the current range |
| `/sensors/ultrasonic_ranges` | JSON, distance per sensor, null if none; can be plotted |
| `/sensors/lidar` | `foxglove.PointCloud` |

`config/foxglove_search_layout.json` shows the 3D ones.

## Limits

- 2D only: no heights, so a low curb blocks an ultrasonic the same as a car, and nothing can be seen over.
- Curbs and walls block ultrasonics and lidar, but only parked cars hide paint from cameras.
- No lens model: a camera sees its whole footprint evenly, with no blur or error growing with distance.
- Ultrasonics have no crosstalk or angle-dependent echo strength, and read at the same instant.
- No motion between the start and end of a lidar sweep.
