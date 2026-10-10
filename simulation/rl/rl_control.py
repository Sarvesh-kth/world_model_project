import json
import shutil
import time

import numpy as np

from data_collection.scripted_policy import ScriptedPickPlace
from world_model.common import write_json
from .task_control import GoalReward, TaskSession, make_rl_env

# Training the exact-state SAC policy: scripted demonstrations -> behaviour cloning warm start ->
# SAC with the demonstrations in the replay buffer and a BC term on the actor. Validation every
# eval_every steps picks the checkpoint that is saved as sac_best.zip.


# Run the policy on fresh seeds and count placements
def validate(model, cfg, layout, args):
  session = TaskSession(cfg, layout, args.position_jitter)
  results = []
  try:
    for i in range(args.validation_episodes):
      seed = args.seed + 10000 + i
      obs = session.reset(seed)
      for _ in range(cfg["episode"]["max_steps"]):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, timeout, _ = session.step(action)
        if done or timeout:
          break
      results.append({"seed": seed, **session.result()})
  finally:
    session.close()
  return {"episodes": results, "successes": sum(r["task_success"] for r in results),
          "success_rate": float(np.mean([r["task_success"] for r in results])),
          "mean_final_goal_distance_cm": float(np.mean([r["final_goal_distance_cm"] for r in results]))}


# Record transitions from the scripted policy until args.demos episodes succeeded, failed attempts are
# dropped. Saved to demos.npz so a rerun does not collect again.
def demonstrations(path, cfg, layout, args):
  if path.exists():
    with np.load(path) as data:
      return {k: data[k] for k in data.files}
  session = TaskSession(cfg, layout, args.position_jitter)
  transitions, reports = [], []
  try:
    for attempt in range(3 * args.demos):
      seed = args.seed + attempt
      obs = session.reset(seed)
      policy = ScriptedPickPlace(session.sim, np.random.default_rng(seed))
      rows = []
      for _ in range(cfg["episode"]["max_steps"]):
        action = np.asarray(policy.act(), np.float32)
        nxt, reward, done, timeout, _ = session.step(action)
        rows.append((obs.copy(), action.copy(), nxt.copy(), reward, done, timeout, attempt))
        obs = nxt
        if done or timeout or policy.done:
          break
      result = session.result()
      reports.append({"seed": seed, **result})
      print(f"demo {attempt + 1}: success={result['task_success']} steps={result['steps']}", flush=True)
      if result["task_success"]:
        transitions.extend(rows)
      if sum(r["task_success"] for r in reports) >= args.demos:
        break
  finally:
    session.close()

  write_json(path.with_suffix(".json"), {"attempts": reports, "requested_successes": args.demos})
  if sum(r["task_success"] for r in reports) < args.demos:
    raise RuntimeError("the scripted policy did not produce enough successful demos, see demos.json")
  names = ("obs", "action", "next_obs", "reward", "terminated", "truncated", "episode")
  data = {name: np.asarray([row[i] for row in transitions]) for i, name in enumerate(names)}
  np.savez_compressed(path, **data)
  return data


def train(root, cfg, layout, args):
  import torch
  import stable_baselines3
  from stable_baselines3 import SAC
  from stable_baselines3.common.callbacks import BaseCallback
  from stable_baselines3.common.logger import configure
  from stable_baselines3.common.monitor import Monitor

  torch.set_num_threads(args.threads)
  torch.manual_seed(args.seed)
  folder = root / "models"
  folder.mkdir(parents=True, exist_ok=True)
  if (folder / "rl_training.json").exists():
    print("SAC training already complete", flush=True)
    return

  data = demonstrations(root / "demos.npz", cfg, layout, args)
  demo_obs = torch.as_tensor(data["obs"], dtype=torch.float32)
  demo_actions = torch.as_tensor(data["action"], dtype=torch.float32)
  env = Monitor(make_rl_env(cfg, layout, args.position_jitter, args.seed + 1000),
                filename=str(root / "rl_monitor.csv"), info_keywords=("task_success",))
  start = time.monotonic()
  try:
    model = SAC("MlpPolicy", env, device="cpu", seed=args.seed, verbose=1,
                policy_kwargs={"net_arch": [128, 128]}, learning_rate=args.learning_rate,
                buffer_size=max(100000, len(data["obs"]) + args.rl_steps),
                learning_starts=1000, batch_size=256, ent_coef=args.ent_coef, gamma=GoalReward.gamma)

    # 1. behaviour cloning: fit the actor mean to the demonstrated actions, the gripper counts double
    rng = np.random.default_rng(args.seed)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
    for epoch in range(args.bc_epochs):
      loss_sum = 0.0
      for ids in np.array_split(rng.permutation(len(demo_obs)), max(1, (len(demo_obs) + 255) // 256)):
        predicted = model.actor(demo_obs[ids], deterministic=True)
        error = (predicted - demo_actions[ids]).square()
        loss = error[:, :4].mean() + 2 * error[:, 4].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * len(ids)
      if (epoch + 1) % 20 == 0 or epoch + 1 == args.bc_epochs:
        print(f"BC epoch {epoch + 1}: action MSE={loss_sum / len(demo_obs):.5f}", flush=True)

    # start exploring with a small spread around the cloned actions
    with torch.no_grad():
      model.actor.log_std.weight.zero_()
      model.actor.log_std.bias.fill_(-2)

    # 2. the demonstrations go into the replay buffer so the critic sees successful episodes from the start
    for i in range(len(data["obs"])):
      timeout = bool(data["truncated"][i])
      model.replay_buffer.add(data["obs"][i:i + 1], data["next_obs"][i:i + 1], data["action"][i:i + 1],
                              np.array([data["reward"][i]]), np.array([bool(data["terminated"][i]) or timeout]),
                              [{"TimeLimit.truncated": timeout}])
    initial = validate(model, cfg, layout, args)
    model.save(folder / "bc_initial.zip")
    history = {"BC_initial_validation": initial, "validation": [], "best": None}
    print(f"BC only: {initial['successes']}/{args.validation_episodes} placements", flush=True)
    model.set_logger(configure(str(root / "rl_logs"), ["stdout", "csv"]))

    # 3. a BC loss added to every SAC actor update, so RL does not wander away from the demonstrations
    def demonstration_gradient(optimizer, positional, keyword):
      if not model.actor.mu.weight.requires_grad or not args.bc_weight:
        return
      ids = torch.randint(len(demo_obs), (model.batch_size,))
      error = (model.actor(demo_obs[ids], deterministic=True) - demo_actions[ids]).square()
      loss = args.bc_weight * (error[:, :4].mean() + 2 * error[:, 4].mean())
      loss.backward()
      model.logger.record("train/demonstration_loss", float(loss.detach()))

    model.actor.optimizer.register_step_pre_hook(demonstration_gradient)

    # 4. fit the critic on the demonstrations first with the actor frozen, a random critic would pull
    # the cloned actor apart in the first updates
    if args.critic_warmup:
      for parameter in model.actor.parameters():
        parameter.requires_grad_(False)
      model.train(gradient_steps=args.critic_warmup, batch_size=256)
      for parameter in model.actor.parameters():
        parameter.requires_grad_(True)
      print(f"critic warmup: {args.critic_warmup} updates with the actor frozen", flush=True)

    # 5. SAC, validated every eval_every steps, the best checkpoint by placements then goal distance is kept
    def checkpoint():
      evaluation = {"training_steps": model.num_timesteps, **validate(model, cfg, layout, args)}
      history["validation"].append(evaluation)
      path = folder / f"sac_{model.num_timesteps:06d}.zip"
      model.save(path)
      best = history["best"]
      if best is None or ((evaluation["success_rate"], -evaluation["mean_final_goal_distance_cm"])
                          > (best["success_rate"], -best["mean_final_goal_distance_cm"])):
        history["best"] = evaluation
        history["best_checkpoint_file"] = path.name
      for old in folder.glob("sac_0*.zip"):
        if old.name != history["best_checkpoint_file"]:
          old.unlink()
      print(f"SAC validation at {model.num_timesteps}: {evaluation['successes']}/{args.validation_episodes} "
            f"placements; best={history['best']['success_rate']:.1%}", flush=True)

    class ValidationCallback(BaseCallback):
      def _on_step(self):
        if self.num_timesteps % args.eval_every == 0:
          checkpoint()
        return True

    model.learn(args.rl_steps, callback=ValidationCallback())
    if not history["validation"] or history["validation"][-1]["training_steps"] != model.num_timesteps:
      checkpoint()
    shutil.copyfile(folder / history["best_checkpoint_file"], folder / "sac_best.zip")
    write_json(folder / "rl_training.json", {
      **history, "SB3_version": stable_baselines3.__version__, "torch_version": torch.__version__,
      "seconds": time.monotonic() - start,
      "settings": {"rl_steps": args.rl_steps, "critic_warmup": args.critic_warmup, "bc_weight": args.bc_weight,
                   "learning_rate": args.learning_rate, "ent_coef": args.ent_coef, "bc_epochs": args.bc_epochs,
                   "demos": args.demos, "seed": args.seed, "network": [128, 128], "gamma": model.gamma,
                   "validation_seeds": [args.seed + 10000 + i for i in range(args.validation_episodes)]}})
  finally:
    env.close()
