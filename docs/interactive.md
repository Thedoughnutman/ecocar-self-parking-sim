# Testing by hand

`scripts/interactive.py` opens a window where you drive the car, press the button and let it park. It's a test tool for working on the
feature, not a mock-up of what a driver would see in the car.

```
python scripts/interactive.py                               # seed 1: sensors only, monitor on
python scripts/interactive.py --empty                       # no cars: painted lines only
python scripts/interactive.py --seed 151
python scripts/interactive.py --scenario cart_in_stall --monitor off
```

It needs the optional `pygame-ce` package. Nothing else in the project does.

| Option | Default | Meaning |
|---|---|---|
| `--seed` | 1 | The same lot as this seed in `monte_carlo.py --mode search` |
| `--detector` | config (`fused`) | `lines`, `space`, `fused`, or `ideal`, the ground-truth reference, which always plans on the true lot |
| `--planner-map` | `sensed` | `sensed`: only what the sensors saw. `true_lot` hands the planner ground truth |
| `--empty` | off | No parked cars |
| `--scenario` | none | Obstacles, actors and faults from `config/scenarios/` |
| `--monitor` | `on` | The safety monitor: `on`, `off`, `auto` |
| `--units` | config (`imperial`) | What the window starts in: `imperial` or `metric` |
| `--replay FILE` | none | Run a saved session again, on the setup it was saved with |
| `--headless` | off | With `--replay`: no window, print the result |
| `--no-mcap` | off | Don't save an MCAP |

Exit code: 0 if the last run parked and met every criterion, 2 if not, 1 on a crash.

## A run

1. **Drive.** W or up speeds up, S or down slows down and then reverses. The speed stays where you leave it. A or left and D or right steer,
   and the wheel returns to straight when you let go.
2. **P** turns the feature on. It needs to be under 12.4 mph, which the keys can't exceed.
3. **Stalls appear** once the car has passed them and read across them. A stall is available only while it's in range: within `slot_detection.offer_max_distance_m` (15 m), and not driven `offer_max_driven_m` (30 m) since it was last seen. Out of range it's gone, and it comes back when the car returns.
4. **Click a stall**, moving or stopped, and **B** to stop if you're moving. The next step waits for a standstill.
5. **Press Enter.** The car plans, then parks. Press again to pause, and again to carry on, after about 1 s. Enter is a toggle, so the car keeps going until you press it again or something stops it. After a no-path or a finished maneuver it resets, so the next stall needs its own press.
6. If it can't park, a red box says why and how far away the stall is, and the result banner explains a failed run. **R** runs the same seed again, **N** the next seed, **Esc** quits. All three work mid-run too.

Available stalls are numbered 1, 2, 3 in the order found. A stall keeps its number for the run and no number is reused, so "stall 4" means one stall. Unavailable stalls have no number. These aren't the slot ids in logs and MCAPs, which count every track.

| Key | Does |
|---|---|
| T | Hands on the wheel. During a maneuver this aborts it, as a real takeover would |
| Z | Zoom in on the car and back. The zoomed view follows the car |
| U | Switch units |

| Stall outline | Means |
|---|---|
| Gray | Seen, not confirmed yet |
| Blue | Offered: you can click it |
| Green | Selected |
| Red | No path from where the car was. Clicks on it are ignored until the car has moved `interactive.retry_after_driven_m` (1 m), when it goes back to blue |

Also on screen: the plan (purple), the path driven (teal), ultrasonic returns as amber arcs, paint the cameras see in cyan, carts and pillars in solid amber, and actors in pink. The bar across the car is its windshield.

**U** switches the window between imperial (mph, feet, inches) and metric (m/s, meters, cm), at any time. Metric matches Foxglove's plots. It starts in `interactive.units`, or `--units`. The sim, config, logs, MCAPs and console output are always metric.

The cards under the lot show the state and the feature's message, speed and gear, steering (amber near the limit), distance to go, clearance, and stalls. The clearance
reading is the true distance to the nearest thing, turning red under the pass limit, which is the tick on its bar. "Estimate off by" is how far the car's own estimate is from the truth.

**The picture is the truth, not what the car believes.** The lot and car are drawn where they really are. The plan and slots come from the car's estimate and are
moved onto the lot using the current error, so a stall outline sitting off its painted lines is a real detection error.

## What it's the same as, and what it isn't

`ManualDriver` (`park_sim/hmi.py`) plugs into `run_search` in place of the scripted driver, and the window is a read-only `RunObserver`. So the supervisor, detectors,
planner, controller and monitor are the ones Monte Carlo uses, and the same seed gives the same lot. The run itself won't match a Monte Carlo run of that seed, since you
drive differently.

- **Pausing is in `ManualDriver`, not the supervisor.** Enter going off presses the brake, and the brake is what pauses a maneuver. There's no dead-man: a real system would need one, in the feature itself.
- **If the window loses focus, held keys count as released**, since their release would never arrive. Enter stays as it is, so a maneuver carries on.
- **A no-path stall can be tried again** once the car has moved `retry_after_driven_m`. "No path" is only true of where the car was.
- **It runs on pygame-ce.** Each frame is presented through SDL's renderer, on the GPU where there is one. Shapes are drawn by the CPU, in SDL's C code. About 31 frames a second at 0.99x real time on the dev machine, zoomed or not (seed 1004, fused). The drawing can lag the run, but keys are read every tick.
- **The window freezes while the planner runs**, up to 2 s. The plan has a wall-clock limit, so plans near it can differ between runs, and other programs add load.
- Only generated key events and a scripted person have tried it, plus the owner's hand tests. The speeds and steering rates in `interactive:` in the config may need changing.

## Sessions and replay

Every run saves `out/hmi_sessions/DATE_TIME_seedN.json`, its setup and inputs by tick, and an MCAP of the same name for Foxglove. If the sim crashes the inputs are still saved, though not an MCAP.

```
python scripts/interactive.py --replay out/hmi_sessions/SESSION.json            # watch it again
python scripts/interactive.py --replay out/hmi_sessions/SESSION.json --headless # just the result
```

A replay feeds the same inputs on the same ticks, so a bug you hit by hand can be run again and looked into. The exception is anything that depends on the 2 s wall-clock planning limit. A replay of a run from before a
code change can go differently after it: the inputs are the same, but the car's response isn't.

## How it fits together

| File | What |
|---|---|
| `park_sim/hmi.py` | No window code. `HmiInput` and `HmiStatus`, the two messages between the window and a run; `KeyState`, `ManualDriver`, `StallNumbers`, `Units`, `explain_result`, `InputLog`, `Replay` |
| `park_sim/hmi_view.py` | `HmiWindow`, in pygame-ce: the `RunObserver` that draws, and the key and mouse handlers |
| `scripts/interactive.py` | Options, the loop over runs, saving |
| `config/park_params.yaml` | `interactive:`, every number `[est]` |
