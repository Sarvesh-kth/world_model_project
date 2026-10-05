"""Train optional SB3 SAC on real MuJoCo transitions, with a demonstration warm start."""
import json
import shutil
import time

import numpy as np

from data_collection.scripted_policy import ScriptedPickPlace
from .data import digest, write_json
from .task_control import GoalReward, TaskSession, make_rl_env


def validate(model, cfg, layout, args):
    session = TaskSession(cfg, layout, args.position_jitter)
    results = []
    try:
        for i in range(args.validation_episodes):
            obs = session.reset(args.seed+10000+i)
            for _ in range(cfg["episode"]["max_steps"]):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, timeout, _ = session.step(action)
                if done or timeout:
                    break
            results.append({"seed": args.seed+10000+i, **session.result()})
    finally:
        session.close()
    return {"episodes": results, "successes": sum(r["task_success"] for r in results),
            "success_rate": float(np.mean([r["task_success"] for r in results])),
            "mean_final_goal_distance_cm": float(np.mean([r["final_goal_distance_cm"] for r in results]))}


def demonstrations(path, cfg, layout, args):
    if path.exists():
        with np.load(path) as data:
            return {k: data[k] for k in data.files}
    session = TaskSession(cfg, layout, args.position_jitter)
    transitions, reports = [], []
    try:
        for attempt in range(3*args.demos):
            seed = args.seed+attempt
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
            print(f"demo {attempt+1}: success={result['task_success']} steps={result['steps']}", flush=True)
            if result["task_success"]:
                transitions.extend(rows)
            if sum(r["task_success"] for r in reports) >= args.demos:
                break
    finally:
        session.close()
    write_json(path.with_suffix(".json"), {"attempts": reports, "requested_successes": args.demos})
    if sum(r["task_success"] for r in reports) < args.demos:
        raise RuntimeError("scripted reference did not supply enough successful full-task demos; inspect demos.json")
    names = ("obs", "action", "next_obs", "reward", "terminated", "truncated", "episode")
    data = {name: np.asarray([row[i] for row in transitions]) for i, name in enumerate(names)}
    np.savez_compressed(path, **data)
    return data


def train(root, cfg, layout, args, *, demonstrations_fn=demonstrations,
          env_factory=make_rl_env, validate_fn=validate):
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
    report_path = folder / "rl_training.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if digest(folder / "sac_best.zip") != report["checkpoint_sha256"]:
            raise ValueError("completed SAC checkpoint changed")
        print("SAC training already complete", flush=True)
        return
    data = demonstrations_fn(root / "demos.npz", cfg, layout, args)
    demo_obs = torch.as_tensor(data["obs"], dtype=torch.float32)
    demo_actions = torch.as_tensor(data["action"], dtype=torch.float32)
    env = Monitor(env_factory(cfg, layout, args.position_jitter, args.seed+1000),
                  filename=str(root / "rl_monitor.csv"), info_keywords=("task_success",))
    progress = folder / "rl_progress.json"
    start = time.monotonic()
    try:
        if progress.exists():
            history = json.loads(progress.read_text())
            last = folder / history["last_checkpoint_file"]
            if digest(last) != history["last_checkpoint_sha256"]:
                raise ValueError("interrupted SAC checkpoint hash mismatch")
            if digest(root / "demos.npz") != history["demo_sha256"]:
                raise ValueError("interrupted demonstration data changed")
            model = SAC.load(last, env=env, device="cpu")
            replay = folder / history["replay_file"]
            if digest(replay) != history["replay_sha256"]:
                raise ValueError("interrupted replay buffer changed")
            model.load_replay_buffer(replay)
            print(f"Resuming SAC from {model.num_timesteps} steps; simulator resets on resume", flush=True)
        else:
            model = SAC("MlpPolicy", env, device="cpu", seed=args.seed, verbose=1,
                        policy_kwargs={"net_arch": [128, 128]},
                        learning_rate=getattr(args, "learning_rate", 3e-4),
                        buffer_size=max(100000, len(data["obs"])+args.rl_steps),
                        learning_starts=1000, batch_size=256,
                        ent_coef=getattr(args, "ent_coef", "auto_0.05"), gamma=GoalReward.gamma)
            if getattr(args, "start_checkpoint", None):
                source = SAC.load(args.start_checkpoint, device="cpu")
                model.policy.load_state_dict(source.policy.state_dict())
                del source
                # A new fit keeps the requested seed/optimizers/entropy settings;
                # interrupted fits instead restore the full SAC and replay above.
                print(f"New SAC actor/critic weights initialized from preserved checkpoint {args.start_checkpoint}", flush=True)
            rng = np.random.default_rng(args.seed)
            obs = torch.as_tensor(data["obs"], dtype=torch.float32)
            actions = torch.as_tensor(data["action"], dtype=torch.float32)
            optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
            for epoch in range(args.bc_epochs):
                loss_sum = 0.0
                for ids in np.array_split(rng.permutation(len(obs)), max(1, (len(obs)+255)//256)):
                    predicted = model.actor(obs[ids], deterministic=True)
                    error = (predicted-actions[ids]).square()
                    loss = error[:, :4].mean()+2*error[:, 4].mean()
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    loss_sum += float(loss.detach())*len(ids)
                if (epoch+1) % 20 == 0 or epoch+1 == args.bc_epochs:
                    print(f"BC warm start epoch {epoch+1}: action MSE={loss_sum/len(obs):.5f}", flush=True)
            # Small initial exploration around the demonstrator-initialized mean.
            with torch.no_grad():
                model.actor.log_std.weight.zero_(); model.actor.log_std.bias.fill_(-2)
            for i in range(len(data["obs"])):
                timeout = bool(data["truncated"][i])
                model.replay_buffer.add(data["obs"][i:i+1], data["next_obs"][i:i+1],
                    data["action"][i:i+1], np.array([data["reward"][i]]),
                    np.array([bool(data["terminated"][i]) or timeout]), [{"TimeLimit.truncated": timeout}])
            initial = validate_fn(model, cfg, layout, args)
            model.save(folder / "bc_initial.zip")
            history = {"BC_initial_validation": initial, "validation": [], "best": None,
                       "demo_sha256": digest(root / "demos.npz")}
            print(f"BC ONLY validation: {initial['successes']}/{args.validation_episodes}; not an RL result", flush=True)
        model.set_logger(configure(str(root / "rl_logs"), ["stdout", "csv"]))

        def demonstration_gradient(optimizer, positional, keyword):
            if not model.actor.mu.weight.requires_grad or not args.bc_weight:
                return
            ids = torch.randint(len(demo_obs), (model.batch_size,))
            error = (model.actor(demo_obs[ids], deterministic=True)-demo_actions[ids]).square()
            loss = args.bc_weight*(error[:, :4].mean()+2*error[:, 4].mean())
            # Add to the SAC actor gradient before its ONE optimizer step.
            # This hook is reattached after load; optimizer state remains SB3's.
            loss.backward()
            model.logger.record("train/demonstration_loss", float(loss.detach()))

        model.actor.optimizer.register_step_pre_hook(demonstration_gradient)
        if not progress.exists() and args.critic_warmup:
            # Fit Bellman targets on real demos before a random critic steers the
            # actor. These are critic updates; the initialized actor stays fixed.
            for parameter in model.actor.parameters():
                parameter.requires_grad_(False)
            model.train(gradient_steps=args.critic_warmup, batch_size=256)
            for parameter in model.actor.parameters():
                parameter.requires_grad_(True)
            print(f"Critic warmup: {args.critic_warmup} demonstration Bellman updates; actor frozen", flush=True)

        def checkpoint():
            evaluation = {"training_steps": model.num_timesteps, **validate_fn(model, cfg, layout, args)}
            history["validation"].append(evaluation)
            rank = (evaluation.get("selection_success_rate", evaluation["success_rate"]), -evaluation["mean_final_goal_distance_cm"])
            best = history["best"]
            last = folder / f"resume_{model.num_timesteps:09d}.zip"
            replay = last.with_suffix(".pkl")
            model.save(last)
            model.save_replay_buffer(replay)
            if best is None or rank > (best.get("selection_success_rate", best["success_rate"]), -best["mean_final_goal_distance_cm"]):
                history["best"] = evaluation
                history["best_checkpoint_file"] = last.name
            history["last_checkpoint_file"] = last.name
            history["replay_file"] = replay.name
            history["last_checkpoint_sha256"] = digest(last)
            history["replay_sha256"] = digest(replay)
            write_json(progress, history)
            # Commit the pointer only after both files are complete; keep the
            # selected best weights and latest replay, discard older resume slots.
            keep = {last.name, replay.name, history["best_checkpoint_file"]}
            for path in folder.glob("resume_*"):
                if path.name not in keep:
                    path.unlink()
            print(f"SAC validation at {model.num_timesteps}: {evaluation['successes']}/{args.validation_episodes} "
                  f"full placements; best={history['best']['success_rate']:.1%}", flush=True)

        class ValidationCallback(BaseCallback):
            def _on_step(self):
                if self.num_timesteps % args.eval_every == 0:
                    checkpoint()
                return True

        remaining = args.rl_steps-model.num_timesteps
        if remaining > 0:
            model.learn(remaining, callback=ValidationCallback(), reset_num_timesteps=False)
        if not history["validation"] or history["validation"][-1]["training_steps"] != model.num_timesteps:
            checkpoint()
        shutil.copyfile(folder / history["best_checkpoint_file"], folder / "sac_best.zip")
        write_json(report_path, {**history, "checkpoint_sha256": digest(folder / "sac_best.zip"),
            "SB3_version": stable_baselines3.__version__, "torch_version": torch.__version__,
            "seconds_this_session": time.monotonic()-start, "settings": {
                "algorithm": "SB3 SAC; scripted BC initialization, demonstration replay and declared actor demonstration regularization",
                "reward": "GoalReward: gamma*Phi(next)-Phi(current), strict placement bonus15, real penalties and time cost0.01",
                "rl_steps": args.rl_steps, "critic_warmup": args.critic_warmup, "bc_weight": args.bc_weight,
                "learning_rate": getattr(args, "learning_rate", 3e-4),
                "start_checkpoint_sha256": digest(args.start_checkpoint) if getattr(args, "start_checkpoint", None) else None,
                "ent_coef": getattr(args, "ent_coef", "auto_0.05"),
                "bc_epochs": args.bc_epochs, "demos": args.demos, "seed": args.seed,
                "actor_inputs": 41, "actions": 5, "network": [128, 128], "gamma": model.gamma,
                "validation_seeds": getattr(args, "validation_seeds", [args.seed+10000+i for i in range(args.validation_episodes)]),
                "demo_loss": "bc_weight*(movement/yaw MSE+2*gripper MSE), added to SAC actor gradient",
                "checkpoint_selection": "minimum per-condition validation success, then goal distance; RL checkpoints only"
                                        if "selection_success_rate" in history["best"] else
                                        "validation placement success, then final goal distance; RL checkpoints only"}})
    finally:
        env.close()
