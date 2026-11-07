# What to measure on the LYRIQ

The sim's results rest on placeholder values, marked `[est]` in `config/`. This is what to collect from the car to replace them, most important first. The order
is a judgment from which recent fixes depend on what. The planned sensitivity sweep will test it, so it may change.

Collect these with the team's own logging tools, on a closed lot, at crawling speed, with someone ready to take over. Nothing in this repo talks to the car. For each test, log every signal with
its timestamp (a rosbag or timestamped CSV), and note the date, tire pressure and load.

## 1. Measure first

These decide whether the recent fixes hold on the real car.

| # | What | Now | Config | How | What depends on it |
|---|---|---|---|---|---|
| 1 | **Steering rate**, road-wheel deg/s | 35 | `lyriq_vehicle.yaml` `steering.rate_limit_deg_s` | Lock to lock under the same steering command path the feature will use, stopped and at crawl, both directions. Log commanded and actual angle. Take the steepest sustained slope. Convert from handwheel with the real ratio (the sim assumes about 16) | The speed cap at steering flips. If the car is slower than this, it runs wide again |
| 2 | **Full-lock road-wheel angle**, both ways | 36.4 deg, worked out from the turning circle | `steering.max_road_wheel_angle_deg` | Read it at full lock. Or measure the turning circle in reverse, left and right | Plans use 92% of this, leaving about 3 deg to correct with |
| 3 | **Ultrasonic cone and how returns behave** | 60 deg, sharp edge, always the nearest point | `sensors.yaml` `ultrasonic.cone_deg` | Car parked, a thin pole at a grid of ranges (0.3-4 m) and bearings (0-60 deg off axis). Log every sensor's raw distance. Note where each stops seeing it and whether the reading changes near the edge | The sensed map's ambiguous-return rule, the path check and the safety monitor. Soft cone edges could change all three |
| 4 | **Ultrasonics reversing into a real stall** | Simulated | Used to check #3 | Reverse slowly into an empty stall with cars both sides, several times. Log raw distances and pose. Repeat with a cart in the stall | Direct test of the phantom wall, and of when a real object gets marked. This log can be replayed against the map |
| 5 | **Which ultrasonics exist, and where** | 12 corner and bumper sensors plus 2 side sensors, all assumed | `ultrasonic.sensors`, `side_sensors` | Check the real list, then tape-measure x, y, and facing angle from the center of the rear axle. Confirm the 2 side sensors exist: the search relies on them | Everything built on ultrasonics |
| 6 | **Dropouts and rates** | 1% empty, 15 Hz, 0.02 m noise | `ultrasonic.dropout_prob`, `rate_hz`, `noise_std`, `range` | From #3 and #4 logs: gaps in the stream, spread at a fixed target, nearest and farthest distance seen | A dropped reading currently marks free space through parked cars. How often real ones happen sets how much that matters |
| 7 | **Body dimensions** | Overhangs split evenly at 0.951 m. Mirrors aren't in the footprint | `geometry.front_overhang`, `rear_overhang` | Tape-measure front and rear overhang from the axle centers. Measure mirrors out to the side, and their height compared with neighboring cars' bodies | Stop position and every clearance number. Mirrors stick out about 0.115 m each side |

## 2. Measure next

These set how well the car follows a plan and how fast it stops.

| # | What | Now | Config | How |
|---|---|---|---|---|
| 8 | **Steering lag** | 0.10 s delay, 0.10 s time constant | `steering.dead_time_s`, `time_constant_s` | Small step commands. Log commanded and actual angle |
| 9 | **Speed response** | 0.30 s lag, 0.5 m/s² up, 0.6 m/s² down | `longitudinal.*` | Speed steps between 0 and 1 m/s, forward and reverse. Check that it holds a steady 0.15 m/s creep |
| 10 | **Gear change time** | 1.0 s | `longitudinal.shift_time_s` | D to R and back from a stop, request to moving |
| 11 | **Braking from 1 m/s** | 0.10 s delay, 15 m/s³ jerk, 2.0 and 5.0 m/s² | `brake.*` | Service and emergency stops from 1 m/s. Log request, speed and distance. This sets how close an obstacle can get before the car can't stop in time |
| 12 | **Delays end to end** | Off in the sim | `park_params.yaml` `localization.latency_s` | Compare sensor timestamps with arrival times at the node. At 20 ms, the fused detector dropped from 198/200 to 194/200 before the steering fix |

## 3. Measure when you can

| # | What | Now | Config | How |
|---|---|---|---|---|
| 13 | **Wheel speed errors** | 0.5% scale, 0.01 m/s noise and steps | `localization.odometry.wheel_*` | Drive a taped distance in a straight line, several times, both directions |
| 14 | **Yaw rate errors** | 0.1 deg/s bias, 0.05 noise | `localization.odometry.yaw_rate_*` | Stopped for a few minutes for the bias and noise. Then a known turn |
| 15 | **Camera line detection** | 5 x 4 m footprint, 30 Hz, 0.03 m noise, 1 false line/s, 5% of lines faded | `sensors.yaml` `line_camera.*` | Log the rear, left and right camera detections driving past stalls with measured line positions. Mounts by tape. The front camera is out |
| 16 | **OxTS GNSS/INS** | Placeholder error model | `localization.gnss_ins.*` | Spec sheet values, then a parked and a slow-drive log in the test lot, including near buildings |
| 17 | **Your test lot** | Stall and aisle sizes from `scenario:` | `park_params.yaml`, or `config/scenarios/my_lot.yaml` | Tape-measure stall width and depth, aisle width, line width and how far parked cars actually sit from the lines |
| 18 | **Tire scrub, slope** | Off | `realism.*` | Only if tracking on the car turns out worse than the model. Compare a logged tight turn with the model |

## When data comes in

Send the logs or summary numbers. Each value then replaces its `[est]`, the baseline, search and gates get rerun, and any change gets explained. A value that differs a lot from the placeholder is itself worth knowing,
since it may move a fix that was tuned against the placeholder.
