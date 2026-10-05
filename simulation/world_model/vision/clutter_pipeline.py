"""One resumable command: paired full trajectories -> two Q fits -> D -> offline/live tests."""
import argparse
import copy
import fcntl
import json
import os
import pathlib
import shutil
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

from .data import digest, write_json, load_run, load_features, state_arrays, outcome_metrics, provenance
from .pipeline import SIMULATION, run_command, archive_training
from .full_task import CAMPAIGN, CASES, VIEWS, RecoverySession, layouts, collect, train_recovery


@contextmanager
def run_lock(root):
    """Linux/macOS advisory lock; the kernel releases it even after a killed process."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another campaign process owns {root}; do not run concurrently") from error
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid())+"\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def offline(root, args):
    """Untouched scene groups; Q on REAL z/p first, then recursive D on executed actions."""
    from .train import load_model, readout, imagine, baseline_predict, q_gate
    import torch
    torch.set_num_threads(args.threads)
    manifest = load_run(root)
    z, _ = load_features(root, manifest)
    p, xyz, held = state_arrays(manifest)
    models = {view: load_model(root, "readout", f"q_{view}") for view in ("empty", "mixed")}
    models.update({view+"_p": load_model(root, "readout_p", f"q_{view}") for view in ("empty", "mixed")})
    d, dc = load_model(root, "dynamics", "shared_width")
    blind, bc = load_model(root, "no_vision", "mixed_dynamics")
    report = {"manifest_sha256": digest(root / "manifest.json"), "models": {}, "Q": [], "D": [],
              "contract": "All losses use measured simulator labels only as answers. No future p is supplied to D or the blind baseline."}
    rows = []
    for label, (q, ck) in models.items():
        report["models"][label] = digest(root / "attempts" / ck["training_settings"]["tag"] / f"models/{ck['role']}.pt")
        predicted, prob = readout(q, ck, z, p)
        for split in ("val", "test"):
            for view in VIEWS:
                ids = [i for i, s in enumerate(manifest["states"]) if s["split"] == split and s["view"] == view]
                phases = sorted({manifest["states"][i]["phase"] for i in ids})
                for phase in ("all", *phases):
                    subset = ids if phase == "all" else [i for i in ids if manifest["states"][i]["phase"] == phase]
                    metrics = outcome_metrics(predicted[subset], prob[subset], xyz[subset], held[subset])
                    metrics["false_held_count"] = int(((prob[subset] >= .5) & (held[subset] < .5)).sum())
                    metrics["missed_held_count"] = int(((prob[subset] < .5) & (held[subset] >= .5)).sum())
                    report["Q"].append({"model": label, "split": split, "view": view, "phase": phase,
                        "gate_passed": q_gate(metrics, ck["q_limits"]) if phase == "all" else None, **metrics})
                print(f"Q-{label} {split}/{view}: {report['Q'][-len(phases)-1]}", flush=True)
        for r in manifest["rollouts"]:
            if r["split"] not in ("val", "test"):
                continue
            ids = r["states"]
            for serial, i in enumerate(ids):
                rows.append({"model": label, "rollout": r["id"], "split": r["split"], "view": r["view"],
                    "case": r["case"], "serial": serial, "phase": manifest["states"][i]["phase"],
                    "true_held": bool(held[i]), "Q_held_probability": float(prob[i]),
                    **{f"true_{k}": float(xyz[i, j]) for j, k in enumerate("xyz")},
                    **{f"Q_{k}": float(predicted[i, j]) for j, k in enumerate("xyz")}})
            # Release diagnosis starts when bilateral held contact is actually lost, not at the command.
            losses = [j for j in range(1, len(ids)) if held[ids[j-1]] and not held[ids[j]]]
            for serial in losses:
                stop = next((j for j in range(serial+1, len(ids)) if held[ids[j]]), len(ids))
                detected = next((j for j in range(serial, stop) if prob[ids[j]] < .5), None)
                report.setdefault("release_lag", []).append({"model": label, "rollout": r["id"],
                    "split": r["split"], "view": r["view"], "serial": serial,
                    "detection_seconds": None if detected is None else (detected-serial)/manifest["control_hz"],
                    "censored_at_regrasp_or_episode_end": detected is None,
                    "observation_seconds": (stop-serial)/manifest["control_hz"]})

    # Fixed, predeclared windows include source states throughout reach/carry/release/recovery.
    for split in ("val", "test"):
        for view in VIEWS:
            for horizon in (1, 4, 8, 16):
                sums, count, changed, attempted = {}, 0, 0, 0
                invalid = []
                outcomes = {label: [[], [], [], []] for label in models}
                blind_outcomes = [[], [], [], []]
                for r in manifest["rollouts"]:
                    if r["split"] != split or r["view"] != view or len(r["actions"]) < horizon:
                        continue
                    for start in np.unique(np.linspace(0, len(r["actions"])-horizon, 24, dtype=int)):
                        attempted += 1
                        source, target = r["states"][start], r["states"][start+horizon]
                        actions = np.asarray(r["actions"][start:start+horizon], np.float32)
                        zz, pp = imagine(d, dc, z[source], p[source], actions)
                        # Invert movement and gripper commands to test action dependence, without executing them.
                        wrong = -actions
                        wrong_z, _ = imagine(d, dc, z[source], p[source], wrong)
                        if not np.isfinite(zz).all() or not np.isfinite(pp).all() or not np.isfinite(wrong_z).all():
                            invalid.append({"rollout": r["id"], "serial": int(start), "reason": "nonfinite_D"})
                            continue
                        predictions = {label: readout(q, ck, zz[-1:], pp[-1:]) for label, (q, ck) in models.items()}
                        blind_prediction = baseline_predict(blind, bc, p[source], actions)
                        if any(not np.isfinite(value).all() for pair in (*predictions.values(), blind_prediction) for value in pair):
                            invalid.append({"rollout": r["id"], "serial": int(start), "reason": "nonfinite_outcome"})
                            continue
                        scales = dc["z_std"].numpy()
                        for name, value in (("D_z_mse", np.square((zz[-1]-z[target])/scales).mean()),
                                            ("unchanged_z_mse", np.square((z[source]-z[target])/scales).mean()),
                                            ("wrong_action_z_mse", np.square((wrong_z[-1]-z[target])/scales).mean()),
                                            ("D_ee_mae_cm", 100*np.abs(pp[-1, 14:17]-p[target, 14:17]).mean()),
                                            ("D_width_mae_cm", 100*abs(pp[-1, 18]-p[target, 18]))):
                            sums[name] = sums.get(name, 0.0)+float(value)
                        changed += int(not np.allclose(actions, wrong))
                        for label, (predicted, prob) in predictions.items():
                            for bucket, value in zip(outcomes[label], (predicted[0], prob[0], xyz[target], held[target])):
                                bucket.append(value)
                        predicted, prob = blind_prediction
                        for bucket, value in zip(blind_outcomes, (predicted[0], prob[0], xyz[target], held[target])):
                            bucket.append(value)
                        count += 1
                if count:
                    row = {"split": split, "view": view, "horizon": horizon, "n": count,
                        "attempted_windows": attempted, "invalid_windows": invalid,
                        "metric_scope": "common finite forecasts only; inspect invalid_windows before comparing",
                        "changed_action_fraction": changed/count, **{k: v/count for k, v in sums.items()},
                        "D_Q": {label: outcome_metrics(*values) for label, values in outcomes.items()},
                        "blind": outcome_metrics(*blind_outcomes)}
                    report["D"].append(row)
                    print(f"D {split}/{view} h={horizon}: latent={row['D_z_mse']:.4f} "
                          f"unchanged={row['unchanged_z_mse']:.4f} wrong={row['wrong_action_z_mse']:.4f} "
                          f"invalid={len(invalid)}/{attempted}", flush=True)
                else:
                    report["D"].append({"split": split, "view": view, "horizon": horizon, "n": 0,
                        "attempted_windows": attempted, "invalid_windows": invalid, "metrics_available": False})
                    print(f"D {split}/{view} h={horizon}: no finite forecast windows; invalid={len(invalid)}/{attempted}", flush=True)
    from .decision import save_csv
    save_csv(root / "reports/Q_real_states.csv", rows)
    write_json(root / "reports/offline.json", report)


def controls(root, signature, args, visual):
    """Same scene/seed/intervention rule and exact frozen actor for every method."""
    import torch
    from stable_baselines3 import SAC
    from . import control_pipeline as cp
    from .train import load_model
    torch.set_num_threads(args.threads)
    checkpoint = root / "recovery_rl/models/sac_best.zip"
    ready = json.loads((root / "recovery_rl/ready.json").read_text())
    if digest(checkpoint) != ready["checkpoint_sha256"]:
        raise ValueError("selected recovery actor changed")
    policy = SAC.load(checkpoint, device="cpu")
    models, encoder = {}, None
    if visual:
        encoder = json.loads((root / "features/meta.json").read_text())
        models["empty"] = cp.WorldModels(root, "q_empty", encoder, dynamics_tag="shared_width")
        models["mixed"] = copy.copy(models["empty"])
        models["mixed"].q, models["mixed"].qc = load_model(root, "readout", "q_mixed")
    methods = (("rl_q", "empty"), ("rl_q", "mixed"), ("jepa_mpc", "empty"), ("jepa_mpc", "mixed")) if visual else (("scripted", "exact"), ("rl_true", "exact"))
    selected_groups = [g for g in signature["groups"] if g["split"] == "test"][:args.control_scenes]
    for group in selected_groups:
        for view, layout in layouts(signature["layout"], group["seed"]).items():
            for case in CASES:
                for method, label in methods:
                    output = root / "control" / f"{group['scene']}_{view}_{case}" / label
                    result = output / "episodes" / method / f"seed_{group['seed']}" / "result.json"
                    marker = result.parent / "complete.json"
                    if marker.exists():
                        known = json.loads(marker.read_text())
                        if all((result.parent / name).is_file() and digest(result.parent / name) == sha
                               for name, sha in known.items()):
                            print(f"REUSE {group['scene']}/{view}/{case}/{method}/{label}", flush=True)
                            continue
                        raise ValueError(f"completed control evidence changed: {result.parent}")
                    opts = SimpleNamespace(**vars(args))
                    opts.method, opts.episode_seed = method, group["seed"]
                    opts.position_jitter, opts.policy_checkpoint = 0, checkpoint
                    opts.models_run, opts.tag = root, f"q_{label}"
                    episode_signature = {"config": signature["config"], "layout": layout,
                        "known_rest_z": signature["known_rest_z"], "encoder": encoder,
                        "frozen_baseline": {"checkpoint_sha256": ready["checkpoint_sha256"]}}
                    print(f"CONTROL {group['scene']}/{view}/{case}/{method}/Q-{label}", flush=True)
                    cp.episode(output, episode_signature, opts,
                        session_factory=lambda cfg, layout, jitter: RecoverySession(cfg, layout, jitter, case),
                        models=models.get(label), policy=policy)
                    # Commit control completion only after frames, metrics and replay arrays are durable.
                    artifacts = list(result.parent.glob("*.json"))+list(result.parent.glob("*.csv"))
                    artifacts += [result.parent / "trajectory.npz", result.parent / "actual.gif"]
                    write_json(marker, {str(p.relative_to(result.parent)): digest(p) for p in artifacts})
    name = "visual" if visual else "reference"
    write_json(root / f"reports/{name}_complete.json", {"complete": True})


def summarize(root, signature, final=False):
    from .decision import save_csv
    rows = []
    for file in sorted((root / "control").glob("*/*/episodes/*/seed_*/result.json")):
        if not (file.parent / "complete.json").exists():
            continue
        result = json.loads(file.read_text())
        # Scenario name contains scene_####; explicit fields are read from each trajectory result.
        scenario = file.parents[4].name
        view = "clutter" if "_clutter_" in scenario else "empty"
        rows.append({"scenario": scenario, "view": view, "Q": file.parents[3].name,
            "method": result["method"], "seed": result["seed"], "case": result["case"],
            **{k: result[k] for k in ("task_success", "final_goal_distance_cm", "obstacle_contact_steps",
                "intervention_triggered", "regrasped_after_intervention", "recovery_placed", "model_fallback_steps")}})
    aggregate = []
    for method, label, view, case in sorted({(r["method"], r["Q"], r["view"], r["case"]) for r in rows}):
        subset = [r for r in rows if (r["method"], r["Q"], r["view"], r["case"]) == (method, label, view, case)]
        triggered = [r for r in subset if r["intervention_triggered"]]
        successes = sum(r["task_success"] for r in subset)
        rate = successes/len(subset)
        aggregate.append({"method": method, "Q": label, "view": view, "case": case,
            "episodes": len(subset), "placements": successes, "placement_rate": rate,
            "mean_final_goal_distance_cm": float(np.mean([r["final_goal_distance_cm"] for r in subset])),
            "triggered": len(triggered), "regrasped": sum(r["regrasped_after_intervention"] for r in subset),
            "recovered_placements": sum(r["recovery_placed"] for r in subset),
            "obstacle_contact_steps": sum(r["obstacle_contact_steps"] for r in subset),
            "fallback_steps": sum(r["model_fallback_steps"] for r in subset),
            "reference_gate": bool(len(subset) >= 5 and rate >= .8
                and not any(r["obstacle_contact_steps"] for r in subset)
                and (case == "normal" or (len(triggered) >= 5 and len(triggered)/len(subset) >= .8)))
            if method == "rl_true" else None})
    reference = [r for r in aggregate if r["method"] == "rl_true"]
    gate = len(reference) == len(CASES)*len(VIEWS) and all(r["reference_gate"] for r in reference)
    ready_file = root / "recovery_rl/ready.json"
    ready = json.loads(ready_file.read_text()) if ready_file.exists() else {}
    summary = {"complete": final, "reference_all_conditions_passed": gate, "control": aggregate,
        "source_SAC_sha256": signature["frozen_baseline"]["checkpoint_sha256"],
        "frozen_comparison_SAC_sha256": ready.get("checkpoint_sha256"),
        "new_recovery_fit": ready.get("new_recovery_fit"),
        "interpretation": "Pipeline completion does not mean control success. Reference requires >=80% placement over >=5 episodes per view/case, zero rectangle contacts, and >=5 triggered recovery episodes. Visual collision scores are unsupported; rectangles are route-clear."}
    write_json(root / "reports/summary.json", summary)
    save_csv(root / "reports/control_episodes.csv", rows)
    save_csv(root / "reports/control_summary.csv", aggregate)
    lines = ["Two-Q full-task/clutter campaign", f"complete={final}; exact RL all-condition gate={gate}"]
    for r in aggregate:
        lines.append(f"{r['method']}/Q-{r['Q']} {r['view']}/{r['case']}: "
            f"place={r['placements']}/{r['episodes']}; regrasp={r['regrasped']}/{r['triggered']}; "
            f"B={r['mean_final_goal_distance_cm']:.2f}cm; contacts={r['obstacle_contact_steps']}; fallbacks={r['fallback_steps']}")
    (root / "reports/summary.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines), flush=True)
    return summary


def export(root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    previous = destination / "pipeline.json"
    if previous.exists() and json.loads(previous.read_text())["inputs"] != json.loads((root / "pipeline.json").read_text())["inputs"]:
        raise ValueError("export belongs to another campaign; choose a fresh --export")
    files = list((root / "reports").glob("*"))+list((root / "logs").glob("*.txt"))
    files += list((root / "attempts").glob("*/reports/*"))
    files += list((root / "control").glob("*/*/episodes/*/seed_*/*.json"))
    files += list((root / "control").glob("*/*/episodes/*/seed_*/*.csv"))
    files += [root / "pipeline.json", root / "scene_settings.json"]
    files += list((root / "recovery_rl").glob("*.json"))
    files += list((root / "recovery_rl/models").glob("*.json"))
    index = {}
    for source in files:
        if source.is_file() and source.suffix in (".json", ".csv", ".txt"):
            relative = source.relative_to(root)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            index[str(relative)] = {"sha256": digest(target), "bytes": target.stat().st_size}
    write_json(destination / "files.json", index)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=pathlib.Path, default=pathlib.Path("data/q_clutter_v1"))
    p.add_argument("--baseline-run", type=pathlib.Path, default=pathlib.Path("data/rl_baseline_v1"))
    p.add_argument("--encoder-run", type=pathlib.Path, default=pathlib.Path("data/vision_v2"))
    p.add_argument("--export", type=pathlib.Path, default=pathlib.Path("../results/q_clutter_v1"))
    p.add_argument("--train-scenes", type=int, default=12)
    p.add_argument("--val-scenes", type=int, default=4)
    p.add_argument("--test-scenes", type=int, default=6)
    p.add_argument("--control-scenes", type=int, default=5)
    p.add_argument("--seed", type=int, default=20464005)
    p.add_argument("--training-seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--rl-steps", type=int, default=20000, help="new recovery SAC steps, only when validation probe fails")
    p.add_argument("--bc-epochs", type=int, default=240)
    p.add_argument("--critic-warmup", type=int, default=2000)
    p.add_argument("--rollout-steps", type=int, default=8)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--elites", type=int, default=8)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--terminal-weight", type=float, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--pilot", action="store_true", help="3/1/1 scenes, 3 epochs, one control scene: software pilot, not a reference pass")
    p.add_argument("--stage", choices=("preflight", "recovery_rl", "reference", "collect", "bind", "offline", "visual"), help=argparse.SUPPRESS)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.pilot:
        args.train_scenes, args.val_scenes, args.test_scenes, args.control_scenes = 3, 1, 1, 1
        args.epochs, args.population, args.elites, args.iterations = 3, 8, 2, 1
        args.rl_steps, args.bc_epochs, args.critic_warmup = 1000, 2, 20
    if min(args.train_scenes, args.val_scenes, args.test_scenes, args.control_scenes, args.epochs,
           args.rollout_steps, args.horizon, args.population, args.elites, args.iterations, args.threads,
           args.rl_steps, args.bc_epochs) < 1 or args.critic_warmup < 0:
        p.error("counts must be positive")
    if args.elites > args.population or args.population < 2 or args.control_scenes > args.test_scenes:
        p.error("require 2 <= population, elites <= population, control-scenes <= test-scenes")
    if args.terminal_weight < 0:
        p.error("terminal-weight cannot be negative")
    for key in ("run", "baseline_run", "encoder_run", "export"):
        setattr(args, key, getattr(args, key).resolve())
    root = args.run
    if args.export == root or root in args.export.parents or args.export in root.parents:
        p.error("export must be separate from the raw run")
    state_path = root / "pipeline.json"
    if args.stage:
        state = json.loads(state_path.read_text())
        signature = state["inputs"]
        if args.stage == "preflight":
            import torch, stable_baselines3, transformers, mujoco
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; use the notebook CUDA virtualenv")
            print(f"GPU={torch.cuda.get_device_name()} SB3={stable_baselines3.__version__} MuJoCo={mujoco.__version__}", flush=True)
            # Exercise the renderer before a long collection. OSMesa may run on the CPU.
            session = RecoverySession(signature["config"], signature["layout"])
            try:
                session.reset(args.seed)
                assert session.sim.render("static").shape == (256, 256, 3)
            finally:
                session.close()
        elif args.stage == "collect":
            collect(root, signature)
        elif args.stage == "recovery_rl":
            train_recovery(root, signature, args)
        elif args.stage == "offline":
            offline(root, args)
        elif args.stage == "bind":
            # Existing width trainer keeps these companions frozen beside the parent D.
            for relative in ("models/readout.pt", "models/readout_p.pt", "reports/readout_validation.json"):
                source = root / "attempts/q_mixed" / relative
                target = root / "attempts/mixed_dynamics" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        else:
            controls(root, signature, args, visual=args.stage == "visual")
        return

    with run_lock(root):
        from .control_pipeline import frozen_baseline
        baseline = json.loads((args.baseline_run / "pipeline.json").read_text())
        cfg, layout = baseline["inputs"]["config"], baseline["inputs"]["layout"]
        evidence = frozen_baseline(args.baseline_run, cfg, layout, baseline["inputs"]["options"]["position_jitter"])
        meta = json.loads((args.encoder_run / "features/meta.json").read_text())
        if (meta["camera"] != "static" or meta["clip_frames"] != 64
                or meta["pooling"] != "mean_all_encoder_tokens" or not meta.get("model_revision")):
            raise ValueError("source encoder must be pinned static/64-frame/mean-pooled")
        if cfg["cameras"]["record_hz"] != cfg["control"]["hz"]:
            raise ValueError("collection assumes one frame per control step")
        if layout["obstacles"] or layout["object"] != "cube" or layout["scale"] != 1:
            raise ValueError("this campaign requires the tested empty-table unit-cube reference")
        settings = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items() if k != "stage"}
        groups = []
        reserved = set(evidence["test_seeds"]+evidence["validation_seeds"])
        for split, n in (("train", args.train_scenes), ("val", args.val_scenes), ("test", args.test_scenes)):
            for _ in range(n):
                i = len(groups)
                seed = args.seed+i
                if seed in reserved:
                    raise ValueError("campaign seed overlaps the original reference evaluation")
                groups.append({"scene": f"scene_{i:04d}", "seed": seed, "split": split})
        code = list((SIMULATION / "environment").glob("*.py"))+list((SIMULATION / "world_model/vision").glob("*.py"))
        code += [SIMULATION / "world_model/train_dynamics.py", SIMULATION / "data_collection/scripted_policy.py", SIMULATION / "data_collection/route.py"]
        signature = {"campaign": CAMPAIGN, "settings": settings, "config": cfg, "layout": layout,
            "groups": groups, "frozen_baseline": evidence,
            "encoder": {k: meta[k] for k in ("model", "model_revision", "camera", "clip_frames", "pooling")},
            "known_rest_z": cfg["table"]["height"]+.0225,
            "code_sha256": {str(f.relative_to(SIMULATION)): digest(f) for f in code},
            "score_contract": "GoalReward supported terms only; no predicted obstacle contacts. Horizon20 includes15 settling steps; shared D; terminal critic disabled by default."}
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["inputs"] != signature:
                raise ValueError("run settings/code/source changed; use a new --run and --export to preserve evidence")
        else:
            if root.exists() and any(p.name != "pipeline.lock" for p in root.iterdir()):
                raise ValueError(f"nonempty directory without campaign state: {root}")
            state = {"inputs": signature, "complete": False, "stages": {}, "provenance": provenance()}
            write_json(state_path, state)

        def command(module, *extra):
            return [sys.executable, "-u", "-m", "world_model.vision."+module, *map(str, extra)]

        def worker(stage):
            # Forward the EXACT public settings to child processes, including pilot-adjusted values.
            options = []
            for key, value in settings.items():
                if key == "pilot":
                    if value:
                        options.append("--pilot")
                else:
                    options.extend(("--"+key.replace("_", "-"), str(value)))
            return command("clutter_pipeline", *options, "--stage", stage)

        work = [("preflight", worker("preflight"), []),
                ("collect", worker("collect"), [root / "manifest.json", root / "scene_settings.json"]),
                ("audit", command("audit", "--run", root), [root / "reports/audit.json"]),
                ("recovery_rl", worker("recovery_rl"), [root / "recovery_rl/ready.json",
                    root / "recovery_rl/source_validation.json", root / "recovery_rl/models/sac_best.zip",
                    root / "recovery_rl/models/rl_training.json"]),
                ("reference", worker("reference"), [root / "reports/reference_complete.json"]),
                ("encode", command("encode", "--run", root, "--model", meta["model"], "--revision", meta["model_revision"]),
                    [root / "features/meta.json", root / "features/latents.npy"])]
        for label in ("empty", "mixed"):
            output = root / f"attempts/q_{label}"
            files = [output / "models/readout.pt", output / "models/readout_p.pt",
                     output / "reports/readout_training.json", output / "reports/readout_p_training.json",
                     output / "reports/readout_validation.json"]
            work.append((f"train_Q_{label}", command("train", "readout", "--run", root,
                "--tag", f"q_{label}", "--view", "empty" if label == "empty" else "all",
                "--seed", args.training_seed, "--epochs", args.epochs), files))
        for stage, role in (("dynamics", "dynamics"), ("baseline", "no_vision")):
            output = root / "attempts/mixed_dynamics"
            options = ("--baseline-horizon", 20) if stage == "baseline" else ()
            work.append((f"train_{role}", command("train", stage, "--run", root, "--tag", "mixed_dynamics",
                "--seed", args.training_seed, "--epochs", args.epochs, "--rollout-steps", args.rollout_steps, *options),
                [output / f"models/{role}.pt", output / f"reports/{role}_training.json"]))
        work.append(("bind", worker("bind"), [root / "attempts/mixed_dynamics" / relative for relative in
            ("models/readout.pt", "models/readout_p.pt", "reports/readout_validation.json")]))
        output = root / "attempts/shared_width"
        work.append(("train_width", command("train", "width", "--run", root, "--tag", "shared_width",
            "--from-tag", "mixed_dynamics", "--width-input", "visual", "--seed", args.training_seed,
            "--epochs", args.epochs, "--rollout-steps", args.rollout_steps),
            [output / "models/dynamics.pt", output / "reports/dynamics_training.json"]))
        work += [("offline", worker("offline"), [root / "reports/offline.json", root / "reports/Q_real_states.csv"]),
                 ("visual", worker("visual"), [root / "reports/visual_complete.json"])]
        try:
            for i, (name, cmd, files) in enumerate(work, 1):
                known = state["stages"].get(name, {})
                if known.get("complete"):
                    if any(not f.is_file() or digest(f) != known["files"][str(f.relative_to(root))] for f in files):
                        raise ValueError(f"completed {name} artifact changed; do not mix runs")
                    print(f"[{i}/{len(work)}] REUSE {name}", flush=True)
                    continue
                if name.startswith("train_"):
                    if name == "train_width" and (root / "attempts/shared_width").exists():
                        archive_training([root / "attempts/shared_width"], root)
                    else:
                        archive_training(files, root)
                if name == "encode" and (root / "features").exists():
                    # A finished encode can outlive a killed parent; authenticate it before adopting.
                    if (root / "features/meta.json").is_file():
                        load_features(root, load_run(root))
                        cached = json.loads((root / "features/meta.json").read_text())
                        if cached["model"] != meta["model"] or cached["model_revision"] != meta["model_revision"]:
                            raise ValueError("completed encoder cache has another model/revision")
                        cmd = None
                    else:
                        cmd.append("--resume")
                print(f"[{i}/{len(work)}] START {name}", flush=True)
                state["stages"][name] = {"complete": False, "command": cmd, "started": time.time()}
                write_json(state_path, state)
                if cmd is not None:
                    run_command(cmd, root / f"logs/{name}.txt")
                state["stages"][name].update(complete=True, finished=time.time(),
                    files={str(f.relative_to(root)): digest(f) for f in files})
                write_json(state_path, state)
                if name in ("reference", "visual"):
                    summarize(root, signature)
                export(root, args.export)
            state["complete"] = True
            state.pop("error", None)
            write_json(state_path, state)
            summarize(root, signature, final=True)
            export(root, args.export)
            print(f"COMPLETE: shareable reports {args.export}; actual GIFs/frames/forecasts {root}/control", flush=True)
        except BaseException as error:
            state["complete"] = False
            state["error"] = repr(error)
            write_json(state_path, state)
            export(root, args.export)
            raise


if __name__ == "__main__":
    main()
