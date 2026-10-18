# configs

`default.yml` is the base configuration and lists every knob with a comment: simulator step, control rate
and IK gains, cameras, robot home pose, table, task (where the cube and the goal can be, the object pool,
the obstacles), episode limits, reward weights and the data collection mix. Every other file here overrides
only the keys it lists; `load_config(path)` merges it into the default.

```
default.yml           the base, every knob
full_mix.yml          the collection mix of the training data: cube and goal anywhere, 0-3 obstacles, every policy kind
hack_mix.yml          the extra data for the planner's rules (fingers shut mid-travel, releases short of the goal, misses)
grade_e.yml           the fixed scene the old SAC was trained on (no posture noise, a simpler mix)
grade_e_layout.json   that scene's layout (cube, goal, obstacles), the controllers' default --layout
```

A mix file has to list every default kind, zeroed if unwanted, because the merge keeps the default's kinds.
The controllers and `rl.rl_baseline --random-ends` do not read these files: they take the config stored in
the model package (`models/v3c/info.json`), so a scene is sampled the way the training data was.
