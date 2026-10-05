"""Full-task paired collection, forced-release scenarios and audits; no GPU imports."""
import copy
import json
import pathlib
import time

import numpy as np
from PIL import Image

from environment.config import Config
from data_collection.scripted_policy import ScriptedPickPlace
from world_model.prepare import P_COLUMNS, A_COLUMNS
from .collect import record
from .data import write_json, provenance
from .task_control import TaskSession

CAMPAIGN = "full_task_clutter_v1"
CASES = ("normal", "drop_early", "drop_middle", "drop_late")
VIEWS = ("empty", "clutter")


def layouts(base, seed):
    """Pair A/B/object/yaw seeds; sample route-clear cuboids in the far table strip."""
    rng = np.random.default_rng(seed)
    empty = copy.deepcopy(base)
    empty["pick"] = (np.asarray(base["pick"])+rng.uniform(-.01, .01, 2)).tolist()
    empty["obstacles"] = []
    clutter = copy.deepcopy(empty)
    # ponytail: far edge only. Blocking obstacles need obstacle-aware RL and collision forecasts.
    ys = rng.choice(np.linspace(-.34, .34, 9), int(rng.integers(1, 4)), replace=False)
    for y in sorted(ys):
        half = rng.uniform([.018, .018, .025], [.03, .028, .1])
        clutter["obstacles"].append({"kind": "box", "pos": [float(rng.uniform(.40, .425)), float(y)],
            "yaw": float(rng.uniform(-.3, .3)), "size": half.tolist()})
    return {"empty": empty, "clutter": clutter}


class RecoverySession(TaskSession):
    """A physical opening intervention; measured object values only trigger/evaluate the test."""
    def __init__(self, cfg, layout, jitter=0, case="normal"):
        super().__init__(cfg, layout, jitter)
        if case not in CASES:
            raise ValueError(f"unknown case {case}")
        self.case = case

    def reset(self, seed):
        obs = super().reset(seed)
        self.initial_xyz = self.obs["state"][:3].copy()
        self.release_remaining = 0
        self.intervention_step = None
        self.release_end_step = None
        self.regrasp_step = None
        return obs

    def phase(self):
        xyz = self.obs["state"][:3]
        held = self.sim._check_contacts()[0]
        if self.release_remaining:
            return "forced_release"
        if self.intervention_step is not None and self.regrasp_step is None:
            return "recovery"
        if self.success or (self.lifted and not held and np.linalg.norm(xyz[:2]-self.goal[:2]) < .07):
            return "release_settle"
        if held:
            if np.linalg.norm(xyz[:2]-self.goal[:2]) < .07:
                return "lower"
            return "carry" if xyz[2]-self.rest_z >= .04 else "grasp_lift"
        return "approach"

    def step(self, action):
        action = np.asarray(action, np.float32).copy()
        xyz = self.obs["state"][:3]
        held = self.sim._check_contacts()[0]
        original_distance = np.linalg.norm(self.initial_xyz[:2]-self.goal[:2])
        progress = 1-np.linalg.norm(xyz[:2]-self.goal[:2])/max(original_distance, 1e-6)
        threshold = {"drop_early": .15, "drop_middle": .50, "drop_late": .75}.get(self.case)
        if (threshold is not None and self.intervention_step is None and held
                and xyz[2]-self.rest_z >= .04 and progress >= threshold):
            self.intervention_step = self.sim.step_count+1
            self.release_remaining = 7
        intervened = self.release_remaining > 0
        if intervened:
            # Hold the hand in place and open, letting gravity drop the object; no teleport/reset.
            action = np.array([0, 0, 0, 0, 1], np.float32)
            self.release_remaining -= 1
        obs, reward, done, timeout, info = super().step(action)
        restart = intervened and self.release_remaining == 0
        if restart:
            self.release_end_step = self.sim.step_count
        if (self.release_end_step is not None and self.sim.step_count > self.release_end_step
                and info["held_endpoint"] and self.regrasp_step is None):
            self.regrasp_step = self.sim.step_count
        info.update(executed_action=action.tolist(), forced_release=intervened,
                    restart_scripted=restart, phase=self.phase())
        return obs, reward, done, timeout, info

    def result(self):
        return {**super().result(), "case": self.case,
                "intervention_triggered": self.intervention_step is not None,
                "intervention_step": self.intervention_step, "release_end_step": self.release_end_step,
                "regrasp_step": self.regrasp_step,
                "regrasped_after_intervention": self.regrasp_step is not None,
                "regrasp_seconds": (self.regrasp_step-self.release_end_step)/self.cfg.control.hz
                                    if self.regrasp_step is not None else None,
                "recovery_placed": self.regrasp_step is not None and self.success}


def collect(root, signature):
    cfg = Config.nested(signature["config"])
    actor = None
    if signature.get("frozen_baseline"):
        from stable_baselines3 import SAC
        actor = SAC.load(pathlib.Path(signature["frozen_baseline"]["run"]) / "models/sac_best.zip", device="cpu")
    path = root / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
    else:
        manifest = {"schema": "vision_consequences_v1", "campaign": CAMPAIGN, "complete": False,
            "settings": {"pair_tolerance": 1e-5, "collection": signature["settings"]},
            "config": cfg, "camera": "static", "control_hz": cfg.control.hz,
            "p_columns": P_COLUMNS, "action_columns": A_COLUMNS,
            "states": [], "rollouts": [], "scene_settings": [], "provenance": provenance()}
    completed = {r["id"] for r in manifest["rollouts"]}
    for group in signature["groups"]:
        pair = layouts(signature["layout"], group["seed"])
        for case in CASES:
            empty_actions = None
            for view in VIEWS:
                name = f"{group['scene']}_{case}_{view}"
                if name in completed:
                    existing = next(r for r in manifest["rollouts"] if r["id"] == name)
                    if view == "empty":
                        empty_actions = existing["actions"]
                    continue
                folder = root / "observations" / name
                if folder.exists():
                    backup = root / "interrupted" / f"{name}_{time.time_ns()}"
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    folder.rename(backup)
                session = RecoverySession(cfg, pair[view], case=case)
                states, actions, history = [], [], []
                try:
                    session.reset(group["seed"])
                    scripted = ScriptedPickPlace(session.sim, np.random.default_rng(group["seed"]))

                    def capture(action=None, info=None, reward=0):
                        image = session.sim.render("static")
                        filename = folder / f"static_{len(states):04d}.jpg"
                        filename.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(image).save(filename, quality=cfg.data.jpeg_quality)
                        history.append(str(filename.relative_to(root)))
                        state = record(session.sim, f"{name}:{len(states):04d}", group["scene"],
                                       group["split"], history, action, info, reward)
                        state.update(view=view, case=case, phase=session.phase(),
                                     source_controller="frozen_SAC" if case == "normal" and actor is not None else "scripted_recovery",
                                     rl_observation=session.observation().tolist(),
                                     forced_release=bool((info or {}).get("forced_release")),
                                     task_success=session.success)
                        states.append(state)

                    capture()
                    # Empty reference supplies the SAME executed actions to its physical clutter pair.
                    for step in range(cfg.episode.max_steps):
                        if view == "clutter":
                            if step >= len(empty_actions):
                                break
                            action = empty_actions[step]
                        else:
                            action = actor.predict(session.observation(), deterministic=True)[0] if case == "normal" and actor is not None else scripted.act()
                        _, reward, done, timeout, info = session.step(action)
                        executed = info["executed_action"]
                        if view == "clutter" and not np.allclose(executed, action, atol=1e-7):
                            raise RuntimeError(f"intervention timing changed in pair {name}; inspect obstacle contacts")
                        actions.append(executed)
                        capture(executed, info, reward)
                        if info["restart_scripted"]:
                            scripted = ScriptedPickPlace(session.sim, np.random.default_rng(group["seed"]+step+1))
                        if view == "empty" and (done or timeout):
                            break
                    start = len(manifest["states"])
                    manifest["states"].extend(states)
                    result = session.result()
                    manifest["rollouts"].append({"id": name, "scene": group["scene"], "split": group["split"],
                        "placement": view, "branch": case, "view": view, "case": case,
                        "states": list(range(start, start+len(states))), "actions": actions,
                        "restore_p_error": 0.0, "restore_integration_error": 0.0,
                        "pair_contract": "same initial layout except rectangles, same reset seed and executed actions; independent resets, not restored branches",
                        "result": result})
                    manifest["scene_settings"].append({"rollout": name, "seed": group["seed"], "layout": pair[view]})
                    write_json(path, manifest)
                    completed.add(name)
                    if view == "empty":
                        empty_actions = actions
                    print(f"COLLECT {name} split={group['split']} steps={len(actions)} "
                          f"placed={result['task_success']} intervention={result['intervention_triggered']} "
                          f"regrasp={result['regrasped_after_intervention']} contacts={result['obstacle_contact_steps']}", flush=True)
                finally:
                    session.close()
    manifest["complete"] = True
    write_json(path, manifest)
    write_json(root / "scene_settings.json", manifest["scene_settings"])
    print(f"COLLECTION COMPLETE: {len(manifest['states'])} states, {len(manifest['rollouts'])} rollouts", flush=True)


def audit_pairs(manifest, problems):
    """Extra checks used by the existing encoder/trainer's common structural audit."""
    groups = {}
    for r in manifest["rollouts"]:
        groups.setdefault((r["scene"], r["case"]), {})[r["view"]] = r
    errors = []
    for (scene, case), pair in groups.items():
        if set(pair) != set(VIEWS):
            problems.append(f"missing empty/clutter pair {scene}/{case}")
            continue
        a, b = pair["empty"], pair["clutter"]
        if a["actions"] != b["actions"]:
            problems.append(f"paired actions differ {scene}/{case}")
        aa, bb = (manifest["states"][r["states"][0]] for r in (a, b))
        error = float(np.max(np.abs(np.asarray(aa["p"])-bb["p"])))
        errors.append(error)
        if error > 1e-5 or not np.allclose(aa["object_xyz"], bb["object_xyz"], atol=1e-5, rtol=0):
            problems.append(f"initial paired robot/object state differs {scene}/{case}")
        for r in (a, b):
            if r["result"]["obstacle_contact_steps"]:
                problems.append(f"route-clear collection touched a rectangle: {r['id']}; inspect images/layout")
            if case != "normal" and not r["result"]["intervention_triggered"]:
                problems.append(f"recovery example never triggered: {r['id']}")
    for split in ("train", "val", "test"):
        for view in VIEWS:
            subset = [s for s in manifest["states"] if s["split"] == split and s["view"] == view]
            if not subset or not any(s["held"] for s in subset) or all(s["held"] for s in subset):
                problems.append(f"{split}/{view} needs both held classes")
    return [], errors


def recovery_demonstrations(root):
    """Only successful TRAIN scenes supply SAC/BC transitions. Q/D retain all collection outcomes."""
    manifest = json.loads((root / "manifest.json").read_text())
    rows, reports = [], []
    for r in manifest["rollouts"]:
        if r["split"] != "train" or not r["result"]["task_success"]:
            continue
        reports.append(r["id"])
        for j, action in enumerate(r["actions"]):
            a, b = [manifest["states"][i] for i in r["states"][j:j+2]]
            if b["forced_release"]:
                # External perturbations are valid D/Q examples, not actions the actor should imitate.
                continue
            rows.append((a["rl_observation"], action, b["rl_observation"], b["reward"],
                         j == len(r["actions"])-1, False, len(reports)-1))
    if not rows:
        raise ValueError("no successful full-task training demonstrations for recovery RL")
    names = ("obs", "action", "next_obs", "reward", "terminated", "truncated", "episode")
    data = {name: np.asarray([r[i] for r in rows]) for i, name in enumerate(names)}
    if data["obs"].shape[1] != 41 or not np.isfinite(data["obs"]).all():
        raise ValueError("invalid full-task SAC observations")
    return data, reports


def recovery_env(cfg, layout, jitter, seed, groups):
    import gymnasium as gym

    class MixedRecoveryEnv(gym.Env):
        def __init__(self):
            self.rng = np.random.default_rng(seed)
            self.session = None
            self.action_space = gym.spaces.Box(-1, 1, (5,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (41,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            if self.session is not None:
                self.session.close()
            group = groups[int(self.rng.integers(len(groups)))]
            view, case = VIEWS[int(self.rng.integers(2))], CASES[int(self.rng.integers(4))]
            self.session = RecoverySession(cfg, layouts(layout, group["seed"])[view], case=case)
            obs = self.session.reset(group["seed"])
            if case != "normal":
                # Initialize at a real post-drop state. No action overrides occur while
                # SB3 is collecting transitions, so its stored action is the executed action.
                scripted = ScriptedPickPlace(self.session.sim, np.random.default_rng(group["seed"]))
                for _ in range(cfg["episode"]["max_steps"]):
                    obs, _, done, timeout, info = self.session.step(scripted.act())
                    if info["restart_scripted"]:
                        break
                    if done or timeout:
                        raise RuntimeError("training recovery prefix did not reach its release")
                else:
                    raise RuntimeError("training recovery prefix timed out")
            return obs, {}

        def step(self, action):
            obs, reward, done, timeout, info = self.session.step(action)
            assert not info["forced_release"], "online replay must use the executed actor action"
            info["is_success"] = info["task_success"]
            return obs, reward, done, timeout, info

        def close(self):
            if self.session is not None:
                self.session.close()

    return MixedRecoveryEnv()


def validate_recovery(model, cfg, layout, groups):
    rows = []
    for group in groups:
        for view, scene_layout in layouts(layout, group["seed"]).items():
            for case in CASES:
                session = RecoverySession(cfg, scene_layout, case=case)
                try:
                    obs = session.reset(group["seed"])
                    for _ in range(cfg["episode"]["max_steps"]):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, _, done, timeout, _ = session.step(action)
                        if done or timeout:
                            break
                    rows.append({"seed": group["seed"], "view": view, **session.result()})
                finally:
                    session.close()
    conditions = []
    for view in VIEWS:
        for case in CASES:
            subset = [r for r in rows if r["view"] == view and r["case"] == case]
            successes = sum(r["task_success"] and (case == "normal" or r["recovery_placed"])
                            and not r["obstacle_contact_steps"] for r in subset)
            conditions.append({"view": view, "case": case, "successes": successes, "episodes": len(subset),
                               "success_rate": successes/len(subset)})
    return {"episodes": rows, "conditions": conditions, "successes": sum(r["task_success"] for r in rows),
        "success_rate": float(np.mean([r["task_success"] for r in rows])),
        "selection_success_rate": min(c["success_rate"] for c in conditions),
        "mean_final_goal_distance_cm": float(np.mean([r["final_goal_distance_cm"] for r in rows]))}


def train_recovery(root, signature, args):
    from types import SimpleNamespace
    import shutil
    from stable_baselines3 import SAC
    import torch
    from .data import digest
    from .rl_control import train
    torch.set_num_threads(args.threads)
    folder = root / "recovery_rl"
    checkpoint = args.baseline_run / "models/sac_best.zip"
    validation = [g for g in signature["groups"] if g["split"] == "val"]
    training = [g for g in signature["groups"] if g["split"] == "train"]
    probe_file = folder / "source_validation.json"
    if not probe_file.exists():
        probe = validate_recovery(SAC.load(checkpoint, device="cpu"), signature["config"], signature["layout"], validation)
        write_json(probe_file, {"source_checkpoint_sha256": digest(checkpoint), **probe})
    probe = json.loads(probe_file.read_text())
    if probe["source_checkpoint_sha256"] != digest(checkpoint):
        raise ValueError("recovery probe/source actor mismatch")
    needs_fit = probe["selection_success_rate"] < .8
    print(f"SOURCE recovery validation minimum-condition success={probe['selection_success_rate']:.1%}; fit_new_SAC={needs_fit}", flush=True)
    if needs_fit:
        data, ids = recovery_demonstrations(root)
        opts = SimpleNamespace(seed=args.seed+100000, position_jitter=0, threads=args.threads,
            demos=len(ids), bc_epochs=args.bc_epochs, bc_weight=100, critic_warmup=args.critic_warmup,
            rl_steps=args.rl_steps, eval_every=min(2500, args.rl_steps), validation_episodes=8*len(validation),
            validation_seeds=[g["seed"] for g in validation],
            learning_rate=3e-5, ent_coef="auto_0.005", start_checkpoint=checkpoint)
        def demos(path, cfg, layout, options):
            if not path.exists():
                np.savez_compressed(path, **data)
                write_json(path.with_suffix(".json"), {"training_rollouts": ids, "validation_test_used": False,
                    "external_intervention_actions_imitated": False})
            with np.load(path) as saved:
                return {k: saved[k] for k in saved.files}
        train(folder, signature["config"], signature["layout"], opts, demonstrations_fn=demos,
            env_factory=lambda cfg, layout, jitter, seed: recovery_env(cfg, layout, jitter, seed, training),
            validate_fn=lambda model, cfg, layout, options: validate_recovery(model, cfg, layout, validation))
    else:
        for name in ("sac_best.zip", "rl_training.json"):
            target = folder / "models" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.baseline_run / "models" / name, target)
    write_json(folder / "ready.json", {"complete": True, "new_recovery_fit": needs_fit,
        "source_checkpoint_sha256": digest(checkpoint), "checkpoint_sha256": digest(folder / "models/sac_best.zip"),
        "selection": "source probe and new checkpoint selection on validation scenes only; final control uses held-out test groups"})
