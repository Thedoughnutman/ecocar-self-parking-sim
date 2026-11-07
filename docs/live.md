# Watching a run live in Foxglove

`scripts/run_live.py` streams a run to Foxglove over a local WebSocket while it runs, including the planner's search. It
needs the optional `foxglove-sdk`, `foxglove-schemas-protobuf` and `mcap` packages. `park_sim/` imports without them.

## Connect

```
python scripts/run_live.py --wait
```

1. Run the command. It prints the address, `ws://127.0.0.1:8765`, then waits for Enter.
2. In Foxglove: **Open connection**, **Foxglove WebSocket**, enter the address, **Open**.
3. Once: **Layouts**, **Import from file**, `config/foxglove_live_layout.json`.
4. Press Enter in the terminal. The run starts.
5. When the run is over the server keeps going, so the result stays on screen. Ctrl+C stops it.

Without `--wait` the run starts at once. Connecting late still works, but only for what's still current: you get the lot, the
latest plan and search, all slots and the overview. The rest of the past is gone, unlike with an MCAP.

## Commands

```
python scripts/run_live.py --wait                                     # full search flow, seed 1, fused detector
python scripts/run_live.py --seed 3 --detector ideal --speed 2 --wait # twice real time
python scripts/run_live.py --flow single --wait                       # one maneuver next to the stall
python scripts/run_live.py --scenario pedestrian_walks_into_slot --wait
python scripts/run_live.py --scenario cart_in_stall --seed 2 --wait
python scripts/run_live.py --planner-map sensed --wait                # plan from what the sensors saw; shows /sensed_map
python scripts/run_live.py --wait --mcap out/live.mcap                # also save the finished run
```

| Option | Default | Meaning |
|---|---|---|
| `--seed` | 1 | Same seed, same run as `run_search_demo.py` / `run_demo.py` |
| `--flow` | `search` | `search` or `single` |
| `--detector` | config (`fused`) | `ideal`, `lines`, `space`, `fused` |
| `--scenario` | none | A file in `config/scenarios/`, seeded as in Monte Carlo |
| `--speed` | 1.0 (`live.speed`) | Playback rate, times real time. Not the car's speed |
| `--port` | 8765 (`live.port`) | 0 picks a free one |
| `--wait` | off | Wait for Enter before starting |
| `--mcap PATH` | none | Save the finished run with `export_run`, as the other scripts do |
| `--sensors` | off | Show sensors on a run that doesn't otherwise need them. Not with `--scenario` |
| `--planner-map` | config | `true_lot` or `sensed` |
| `--exit-when-done` | off | Stop when the run ends, rather than on Ctrl+C |

Exit code: 0 parked and met every criterion, 2 didn't, 130 stopped during the run, 1 couldn't start.

`--speed` is a target. Measured on seed 1, fused, with no client connected (a 24 s run, 2 searches shown): `--speed 1` took 29 s, the extra
being the search replays; `--speed 5` took 7.6 s; `--speed 20` took 4.0 s, about 6x, because that is as fast as the sim itself runs here.

The address is always `127.0.0.1`, with no option to change it. The server has no authentication, so it stays off any network.

## What's on it

Every topic `export_run` writes, built by the same code (`RunWriter` in `park_sim/mcap_export.py`), so the older layouts work too.
Two more are live only, and aren't in the MCAP that `--mcap` saves, which stays exactly what the other scripts write:

| Topic | Schema | What |
|---|---|---|
| `/planner_search` | `foxglove.SceneUpdate` | Entities `search_tree` (explored arcs), `search_best` (the branch closest to the goal so far), `search_shots` (the latest analytic shots: green free, red blocked), `search_path` (the result) |
| `/overview` | JSON | `state`, `hmi_message`, `gear`, `speed_kph`, `steering_deg`, `distance_to_go_m`, `min_clearance_m`, `slots_detected`, `slot_ids`, `selected_slot`, `last_transition`, `transitions`, `planner` (status, expansions, shots, time), `result` (`running`, `PASS`, `FAIL`), `summary` |

Differences from the MCAP of the same run, on these topics only:

- `/plan`: a single-stall MCAP shows the plan from t = 0. Live, it appears when it's made.
- `/slots`: an MCAP draws each slot still on the list at the end, with its final corners, when it was first seen. Live can't know
  that, so it draws each slot as it was when first seen, including any that later drop off the list.
- `/metrics`: the same document, except for wall-clock planning times.

The other topics are identical message for message, with the same times (`tests/test_live.py`).

## The search view

Planning takes well under a second, so a plain search would be a flash. The planner runs at full speed with a `SearchRecorder`
attached, then when it has returned, the run waits while the search is replayed to the display. Pacing is in `live:` in
`config/park_params.yaml`:

| Setting | Default | Meaning |
|---|---|---|
| `search_expansions_per_frame` | 1 | Expansions in each displayed frame |
| `search_frame_s` | 0.05 | Seconds per frame at `--speed 1` |
| `search_hold_s` | 1.0 | How long the finished search stays before the run carries on |
| `search_shots_shown` | 6 | Latest shots drawn per frame |

Things to know when watching:

- **Many searches are tiny.** Often an early analytic shot works, so there are a few expansions and hardly any tree. Pick a seed
  with a real search from the table below.
- **The run's clock stands still during a replay.** The plots don't move and the tree grows. Every replay frame has the time of the tick it
  belongs to.
- **Replans show too**, and a run with the fused detector often has them. Failed plans show, with `planner.status` saying why.
- **The tree is in the planner's frame.** On the true lot with odometry, the planner works in the map frame while `/plan` is drawn
  in the estimate's frame, so `search_path` and `/plan` are apart by the drift so far. That difference is real. On the sensed map
  both are in the estimate's frame.
- A sensed-map plan that fails its check of the final straight never starts a search, so it has nothing to show.

Search sizes, seeds 1-20 at the defaults, expansions in the first plan (out/scratch_11/probe_seeds.py):

| Flow | Worth watching | The rest |
|---|---|---|
| Search, fused | Seed 7 (60), 19 (59), 5 (58), 3 (57), 15 (57), 11 (55), 14 (55), 1 (50), 6 (48), 20 (45), 2 (35) | 5 expansions |
| Single stall | Seed 15 (45), 10 (23), 3 (14), 12 (13), 17 (10) | 1 or 2 |

Replans in those runs took 1 expansion each. So the default, seed 1 with the fused detector, has a real search: 50 expansions, about
2.5 s at `--speed 1`. A plan that runs out its 2 s can reach about 400 expansions, about 20 s to watch: raise
`search_expansions_per_frame` or `--speed` if that's too long.

## Why it can't change a result

- The observer only records. With no observer, `plan()` takes the same path through the code, apart from `is None` checks.
- Recording is a list append per event, inside the planner's time limit: 6 microseconds for the 4 events of a 2 ms plan. Replay and
  pacing all happen after `plan()` has returned, outside the limit.
- `RunObserver.on_tick` runs between ticks, and reads only. The run has no clock of its own there.
- The tests compare live and plain runs: every log column, the metrics apart from planning time, checks, transitions and path.

A run with plans close to the 2 s limit can still differ between any two runs, live or not, because that limit is wall clock. A heavily loaded machine makes that more likely. Foxglove on the same machine adds load.

## How it fits together

- `park_sim/hybrid_astar.py`: `PlannerObserver`, `SearchRecorder`, and `observing()`, which scopes an observer over the layers between a run
  and its planner.
- `park_sim/simulator.py`: `RunObserver`, `RunTick`, and `observer=` on `run_closed_loop`, `run_search` and `search_trial`.
- `park_sim/mcap_export.py`: `RunWriter` writes a run tick by tick to whatever `out` it's given; `export_run` uses it over a finished run.
- `park_sim/live.py`: `LiveServer` (the SDK, the channels, resending to late subscribers) and `LiveStream` (the observer: pacing, replay,
  `/overview`).
- Channels are created at their first message, so a client can't have subscribed yet when that message goes out. The topics that
  change rarely are resent when someone subscribes (`LATCHED_TOPICS`, `CUMULATIVE_TOPICS`). A streaming topic just loses that one message.
- Messages use the Time capability: the app's clock is the run's clock, starting at 0, so Foxglove may show a date in 1970, as with an MCAP.

## Not checked

None of this has been seen in the Foxglove app. A small WebSocket client in the tests connects, subscribes and receives the topics, and
decodes some of them. Whether it all looks right is untested. When you first connect, please check:

1. The connection opens, and the lot appears top-down, framed on the aisle.
2. The search tree grows while the clock stands still. This is the part most likely to misbehave: if the 3D panel only redraws when
   the clock moves, the tree will jump rather than grow. Try `--seed` from the table above.
3. Search tree, best branch, shots and final path are told apart by their colors and not hidden under `/plan` or `/lot`.
4. The car, its trail and the plan move smoothly at `--speed 1`, and plots scroll with the run.
5. The Result panel goes Running, then PASS or FAIL. The Indicator panel's settings were written from memory of its format.
6. The HMI panel and Overview panel read well. The Overview's `transitions` list may need expanding by hand.
7. `/sensors/camera_footprints` is off in this layout's 3D panel to reduce clutter. Turn it on if you want it.
8. Ctrl+C in a real console. The tests cover an interrupt during sleep and Ctrl+Break to a child process, not a keypress.
