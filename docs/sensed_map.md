# Planning from a sensed map

With `planner.map: sensed`, the planner works from what the car's sensors have seen rather than from the true lot.
**This is the default, by the owner's decision.** The ideal detector, a labeled reference, always plans on the true lot, and so do single-stall runs.

```
python scripts/monte_carlo.py -n 200 --mode search --planner-map sensed
python scripts/compare_planner_maps.py -n 200 --workers 4     # both maps, same seeds: pass, contacts, no-path, planning time
python scripts/run_search_demo.py --seed 3 --planner-map sensed --mcap out/search_sensed.mcap
```

It works in the search flow only. A single-stall run starts next to the stall having seen nothing, so it plans on the lot it's given.

## What goes in

`park_sim/sensed_map.py` imports only the standard library, numpy and `scipy.ndimage` (test-enforced). It gets readings and a
pose, and everything goes in the frame of that pose: the estimate. `simulator._SensedMapFeed` passes it each tick's
measurements and `loc.estimate`, and never a camera measurement's `truth_line`.

| Source | What it adds |
|---|---|
| Ultrasonic return at range r | Cells in the cone short of r are free, and any return placed on them earlier is wiped. Arc cells at r not already known free get a hit |
| Ultrasonic with no return | Ignored until the sensor has read empty `empty_readings` times running, since one may be a dropout: it used to clear the whole cone, and about 30% of the inside of parked cars read as free. After that, cells in the cone out to its range lose some occupancy, and nothing gets wiped |
| Return with the whole arc known free | Something new is there. The arc loses its free evidence at once, so the planner stays out even if the car never moves |
| Return whose arc touches the selected slot and reaches past its lines | Ambiguous: see below. Adds nothing to the slot or its margin, only to what's past the margin and to cells already occupied |
| Line camera | Seeing paint means seeing the ground, so the triangle from camera to segment gets weak free evidence. A one-frame false segment stays unknown, and the camera alone can't clear a return or take a cell lower than a later return can undo. Segments are also kept for display |
| The car's own body | The ground under it is free |
| Selected slot from the tracker | The ASSUMED area, below |

Lidar isn't used, even if it's on. A sensor standing still adds a reading only if it changed by `reading_change_m`, since
repeating a reading would pile evidence on whatever arc it happens to draw.

## How unseen space is treated

**Unknown is blocked, with one stated exception.** The planner can't go anywhere the car has no evidence for, apart from
the slot it was told to park in.

The exception is needed because the sensors can't reach the back of a stall from the aisle: 4.5 m of range from a side sensor
reaches about 1.8 m in. So the selected slot's rectangle, plus `assumed_slot_margin_m` (0.3 m) all round, is ASSUMED. The
planner may enter ASSUMED cells, and they are drawn in their own color. The assumption rests on the detector's own check,
which read across the slot and found nothing, and it only ever covers unknown cells: anything seen in it blocks it. The
margin is there because the car at its goal is 0.36 m from each stall line and 0.25 m from the rear line, under the 0.35 m
planning margin at the rear and almost at it on the sides. Parked cars stand about 0.375 m off a line, so 0.3 m stops short
of them, and the unknown past it stands in for a neighbor's unseen body. It also assumes nothing stands over the rear line or
a side line by more than that, until the sensors see it.

The alternative, treating unknown as free, would look better on paper, but it's the same optimism the true lot gives, without
saying so.

The planner also gets only `planning_window_m` around the car and the goal, because the oldest data has drifted most, and
the edge of that window is a wall.

## What plans use it

The first plan, refinement replans and blocked replans all go through `plan_to` in `run_search`, so they read the live map and
anything newly seen counts. `_plan_on_sensed_map` takes the map, the slot, the estimated pose, the vehicle and the config: no lot
and no true pose. Because map, slot and car are in the estimate's frame, no drift conversion is needed.

Two differences from a plan on the true lot:

- It checks the goal and the final straight before searching. The search itself stops at the staging pose, so a true-lot plan
  never checks them. This means a stall with something seen in it fails in about 0.02 s.
- Actors' ground-truth footprints aren't added, and `known_to_planner` means nothing. The planner knows what was seen.

A replan happens when the slot moves or the path is reported blocked.

## The path check

A plan is only as good as the map it was made on, and the map keeps filling in. So while the car is maneuvering, `_SensedMapFeed.path_blocked` sweeps the body over the next `horizon_m` of the path, and any cells in the sweep with
`min_log_odds` or more count. With `min_cells` of them for `confirm_checks` checks running, it reports the path blocked. That goes to the supervisor as `obstacle_in_path`, the same input the monitor uses, so it pauses the car, asks for a replan after 5 s and aborts after 20 s.
Unknown cells don't count. It's only as far-seeing as the map, so it's a second layer behind the monitor, not a replacement.

| Setting tried | False stops, plain | Result |
|---|---|---|
| 1 cell, 2 returns, 1 check | 13/60 | Caught every cart. Far too many stops, 2 runs aborted |
| 8 cells, 4 returns, 3 checks | 0/60, then 2/140 unseen | Looked perfect on the seeds it was picked from. On unseen seeds it hit a cart 1/50, a 0.3 m post 6/50, a 0.1 m post 13/50. A thin object only builds strong evidence where its arcs cross |
| **2 cells, 4 returns, 3 checks** | 9/200 in a replay (before the dropout fix) | Triggered with 0.9 m to stop on all 151 runs that would have hit something |

Then, with the dropout fix, on seeds nothing was chosen from (1000 and up, 50 runs each, closest to the object):

| Object | Monitor off | Monitor on |
|---|---|---|
| Cart | 0 contacts, 0.18 m | 0 contacts, 0.79 m |
| 0.3 m post | 0 contacts, 0.17 m | 0 contacts, 0.86 m |
| 0.1 m post | 0 contacts, 0.20 m | 0 contacts, 0.93 m |

False stops were 2/200 on those seeds, both parked. They aren't always harmless: with the monitor off there's no replan, so seed 105 stopped for a stray mark, waited 20 s and aborted. All of this rests on the sim's sharp-edged cones. It doesn't exist on the true lot, where no map gets built.

The car stops for the object, but it can still finish badly elsewhere: on seed 1042 with the 0.3 m post, monitor off, the slot's depth came out short and a stop adjustment moved the goal 0.51 m deeper, so it finished 0.02 m from the curb. That's prompt 15, item 7.

## Weak marks in the slot

Inside the assumed slot, a mark needs `slot_mark_min_log_odds` (2.2, or 4 returns) before it blocks a plan. Weaker marks stay ASSUMED. All 37 marks that refused a first plan were phantom. With the rule, no object was hit on seeds 1000-1049.

Clearance and scoring keep using the true lot: that is the check on whether the map was good enough.

## Results

Current, fused with odometry, plain runs (monitor off): seeds 0-199 pass 199/200, lowest clearance 0.37 m, seed 105 aborted on a false stop. Seeds 1000-1199, which chose nothing, pass 194/200, lowest 0.40 m, with 5 no path (1065, 1085, 1118, 1131, 1192) and 1 heading (1143). No contacts. Those are after the rules above and the stall range limits.

Before the weak-mark rule, from `scripts/compare_planner_maps.py`, 2 workers:

| Map | Pass | Contacts | Lowest clearance | Runs with no path | Plans that failed | Plan time mean / 95th / max | Replans done / failed |
|---|---|---|---|---|---|---|---|
| true_lot | 200/200 (100%) | 0 | 0.34 m | 0 | 0 | 0.18 / 0.65 / 1.16 s | 186 / 0 |
| sensed | 197/200 (98.5%) | 0 | 0.36 m | 2 | 72 | 0.17 / 0.62 / 2.01 s | 162 / 2 |

The settings are `[est]`, and the sim's cones have a sharp edge, so the real numbers could differ.

## Ambiguous returns

An ultrasonic gives a distance, not a direction. When the nearest thing in a rear cone is the side of a neighbor at the cone's
edge, the arc at that range runs from the neighbor right across the empty stall. Marking it drew a wall across the slot about 1.3 m in
front of the curb, moving back with the car, and it refused nearly every replan: 145 of 177 over these seeds, 42 of 44 traced at the
final-straight check. The hits were mostly the two rear inner sensors, near their axis, so weighing arcs by how far off-axis a cell is
wouldn't have helped.

So a return whose arc touches the slot or its margin, and reaches past the slot's lines to somewhere an object could be, is explained
out there. It adds nothing to the slot or margin, only to what's past the margin and to cells already occupied.

- It tests against the lines, not the margin. Neighbors stand about 0.375 m off a line, just past the 0.3 m margin, so an arc's tip can
  stop short of them.
- Nothing goes in the margin because the goal sits 0.36 m from a line and plans need 0.35 m, so one stray cell there refuses the plan.
- Only an ultrasonic rules out an object past the lines: a camera seeing paint past a cell says little about a car standing on it. If
  ultrasonics have cleared everything past the lines, the return must be in the slot, and the usual rule applies.
- An arc that doesn't touch the slot isn't affected. A refused slot stays selected while the car drives on, so without this every return
  down the aisle took the rule (seed 153 stopped finding a path).

**The cost:** something real in the stall goes unmarked until the car is close enough for its arc to fit between the lines. In
cart_in_stall, 20 runs, the cart was first marked as close as 1.25 m, against 2.05 m without the rule (snapshots 1 s apart). No contacts
with the monitor on. This rests on the sim's sharp-edged cones, so it needs checking against real sensor logs.

2 of the 164 replans still fail, not traced.

## Parameter history

Settings came from reasoning about the sensors, and were tried on seeds 1000-1011 only, before the 200 runs. Two changes came
before those runs: a return further out now wipes earlier returns inside it, after a probe (seeds 1000-1005) showed plans failing
at once on leftover arcs in the mouths of free stalls, and a return on an all-free arc now clears its free evidence, after a unit
test showed a new object staying free for a stopped car. Nothing was changed after the 200-run result. The sensed-run tests
use seeds 1000 and up for the same reason.

## Harder cases

Fused, seeds 1000-1049, monitor on, 50 runs each, with every rule above. No contacts. The true-lot column is from seeds 0-49, so it's only a rough comparison.

| Case | Pass, sensed | Pass, true lot | Lowest clearance, sensed | Parked in target / elsewhere, sensed |
|---|---|---|---|---|
| neighbor_over_line | 58% | 60% | 0.43 m | 12 / 30 |
| pillar_at_corner | 80% | 80% | 0.43 m | 2 / 40 |
| narrow_aisle_6m | 94% | 98% | 0.40 m | 39 / 8 |
| narrow_stall_2p4m | 52% | 74% | 0.37 m | 16 / 23 |
| cart_in_stall | 28% | 16% | 0.39 m | 0 / 14 |

The pillar and neighbor aren't given to the sensed planner. Why the target was so often passed over wasn't traced, and neither was the
narrow stall's lower rate. Pillar, neighbor, narrow aisle and cart each lost 2-6 runs against the same seeds before the stall range limits and sensed default; that isn't traced either.

## Display

`/sensed_map` (`foxglove.SceneUpdate`) shows free, occupied and assumed cells and paint, every `snapshot_period_s` and at each
plan. It is in the estimate's frame, drawn in `map` like `/plan` and `/slots`, so it sits on `base_link_est`, and drifts against
`/lot` as the estimate does. Unknown cells aren't drawn. It's in both Foxglove layouts, and only exists for sensed runs.
