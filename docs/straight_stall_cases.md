# Harder straight-stall cases

Five named cases in `config/scenarios/`, search flow, reverse-in, at 90 deg to the aisle. Run them with
`python scripts/straight_stall_cases.py -n 50 --workers 4`, or one at a time through `--scenario` on
`monte_carlo.py --mode search` and `run_search_demo.py`.

| Case | What changes | Planner told? |
|---|---|---|
| `neighbor_over_line` | The car in the next stall is exactly 0.3 m over the line into the target stall | Yes, it is a parked car like any other |
| `pillar_at_corner` | 0.5 m square pillar at the target's aisle-side corner, +x side, its face on the stall line | Yes, `known_to_planner: true`, like a curb |
| `narrow_aisle_6m` | 6.0 m aisle instead of 7.3 m | Yes |
| `narrow_stall_2p4m` | 2.4 m stalls instead of 2.7 m | Yes |
| `cart_in_stall` | 0.9 x 0.6 m cart inside a stall still marked free, position and angle drawn per run | **No** |

## How obstacles are added

A file's `obstacles:` list goes to `scenario.add_target_obstacles`, around the stall `search_trial` drew. Nothing else
in the draw changes, and the cart's position comes from its own random stream. Sizes are in `lot_objects:` in
`config/park_params.yaml`, all `[est]`.

Pillars and carts are `LotObject`s. Each one is always in `Scenario.obstacles`, so it is in:

- the sensors' view: ultrasonic and lidar echoes, and it hides paint from the line cameras,
- the clearance checks and the score,
- `/lot` in the MCAP, as the entity `lot_objects`, which only appears when there are some.

The planner plans on `Scenario.planner_view()`, which leaves out anything not `known_to_planner`. That flag defaults
to false, so nothing gets on the planner's map without a file saying so. The first plan, refinement replans and blocked
replans all get their map from `SafetyLayer.planning_scene`, which starts from that view. With `planner.map: sensed`
([sensed_map.md](sensed_map.md)), `known_to_planner` has no effect: the planner knows only what the sensors saw.

## Run settings

- All five files set `safety.enabled: true`. `auto` would leave the monitor off in these runs, and in the cart case
  nothing else could stop the car.
- With target obstacles, a sensor-based run's driver stays in the target's row (`search.row_y`), so it reaches the
  stall under test rather than a free stall across the aisle. Plain runs don't set it. At the end of the aisle it
  takes any row, as before.
- The ideal detector still offers these stalls: `Stall.occupied` knows about parked cars in their own stall, not a
  cart or a neighbor's overhang. So the ideal runs test what happens after the offer.

## Reading the results

`out/straight_stall_cases.csv` has one row per case and detector; `_runs.csv` next to it has one row per run.

- `pass_rate`: parked and met every criterion, in any stall. In a case where refusing is right, 0 can be correct.
- `outcomes`: `parked_in_target`, `parked_elsewhere`, `no_path` (asked for a plan, none came, nothing else worked),
  `no_slot` (nothing selected by the end of the aisle), `aborted`, `fault`, `incomplete`.
- `lowest_clearance_m`: true body clearance over the whole run, every tick of the maneuver and every 5th before it.
- `contacts`: runs where that reached 0.
- `met_expect`: runs doing what the file's `expect` says. Those blocks were written before anything was measured and
  must not be changed to fit the results.
- `tried_target`: runs that asked for a plan to the target stall. If this is low, the case mostly tested the detector
  refusing the stall, or something else on offer first.

The current numbers and what they mean are in `PROGRESS.md`, entry 08.
