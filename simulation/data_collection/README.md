# data_collection

Scripted and random policies record the episodes the world model is trained on: one folder per episode, a
mix of policy kinds per dataset, both cameras and the height map at every step.

```
collect.py           collect a dataset: --config (the mix), --episodes, --out, --workers, --seed
scripted_policy.py   ScriptedPickPlace (a waypoint state machine on the exact state), RandomPolicy, make_policy
route.py             A* on a 2 cm grid around the obstacle footprints, line-of-sight smoothing (the scripted policy's route)
writer.py            EpisodeWriter: data.csv, meta.json, images/ per episode
```

```bash
python -m data_collection.collect --config configs/full_mix.yml --episodes 400 --out data/episodes/episodes_raw --workers 4 --seed 1000
python -m data_collection.collect --config configs/rules_mix.yml --episodes 150 --out data/episodes_rules/episodes_raw --workers 4 --seed 5000
python play.py success --seed 34          # watch one kind in the viewer with the live reward plot
python plot_rewards.py data/episodes/episodes_raw/episode_000003   # the rewards of a recorded episode
```

The collector appends: run it again with the number still missing and another `--seed`. With `--workers N`
each worker owns a simulator. Rendering is offscreen (EGL).

## Kinds

The scripted policy approaches around tall obstacles and over low ones, hovers, turns the fingers to the
cube, descends, closes, lifts, carries along the route, lowers, releases, retreats; it retries a missed grasp
twice. Each kind changes one thing so the data holds the situations the planners get into:

| kind | what is different | why |
|---|---|---|
| success | clean, no noise | the task |
| scripted | slowly drifting action noise | the task, noisy |
| miss | the first close 2.5-7 cm beside or above the cube, then a clean pick | D must know a close beside the cube is no hold |
| lateral | a low sideways approach, up to 2.5 cm off, fingers turned half the time | the way the planner grasps |
| wander | 2-4 random table points first, fingers shut half the time, a detour while carrying | the hand far from the cube |
| travel_close | the fingers shut for 8 steps 4-20 cm from the cube on the way in, then a clean pick | closes mid-travel (the gripper envelope stood in for this) |
| short_release | the cube let go 5-14 cm short of the goal, then picked up and placed | releases that are not placements (the release rule stood in for this) |
| drop, collide | let go mid carry; carry low through the obstacles | contacts and drops for R |
| random | random actions with momentum | coverage |

`configs/full_mix.yml` (success .2, scripted .1, miss .15, lateral .15, wander .25, drop .05, collide .05,
random .05; 0-3 obstacles) made the training data. `configs/rules_mix.yml` (travel_close .4, short_release .4,
miss .2; 0-2 obstacles) is the data meant to retire the two planner rules.

## A recorded episode

```
data/<run>/episodes_raw/
  index.csv                 one row per episode (kind, object, ends, steps, success, collisions, seed)
  episode_000000/
    data.csv                one row per control step: action, p (20), cube pose, place, every reward component, flags
    meta.json               the layout (replayable), kind and its parameters, seed, success, the config
    heightmaps.npy          the 8x8 height map per step
    images/static_1.jpg     one static and one wrist frame per step
```

`world_model/prepare_episodes.py` turns such a folder into the run folder the world model trains on.
