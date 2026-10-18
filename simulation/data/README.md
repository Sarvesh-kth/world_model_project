# data

Run folders, all local except `clutter_scenes/` (`collection.json` and `scene_settings.json`: the six
held-out obstacle layouts that `controller.control_clutter` replays). The models the controllers run with
are packaged in `../models/`.

Made by the commands in the READMEs:

```
episodes/             the training run: episodes_raw/ (the collected episodes), manifest.json, features/ (the
                      latents, bases and maps) and attempts/<tag>/ (the trained models); train.sh makes it
episodes_rules/       the extra episodes of configs/rules_mix.yml (closes mid-travel, releases short of the goal)
sac_*/                runs of rl.rl_baseline (the packaged one is ../models/sac)
test_<n>obstacles_*/  what ../../evaluate.sh writes, one folder per obstacle count and controller
<other runs>/         controller runs (frames, gifs, a per-step csv per episode); summaries go to ../../results/
```
