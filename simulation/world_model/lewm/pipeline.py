"""Same clutter recordings -> native LeWM -> goal-image CEM -> matched-scene reports."""
import argparse
from collections import defaultdict
import hashlib
import json
import pathlib
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from .model import DIM, UPSTREAM, NativeModel, Windows, build, indices_and_actions, pixels, plan
from .upstream.module import SIGReg
from world_model.vision.audit import audit
from world_model.vision.data import digest, load_run, load_features, provenance, write_json
from world_model.vision.decision import save_csv
from world_model.vision.pipeline import SIMULATION, run_command
from world_model.vision.clutter_pipeline import run_lock
from world_model.vision.full_task import CASES, VIEWS, RecoverySession


def torch_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def source_info(source):
    state = json.loads((source / "pipeline.json").read_text())
    if not state.get("complete"):
        raise ValueError("Finish q_clutter_v1 first. LeWM only reads the completed source campaign.")
    manifest = load_run(source)
    if manifest.get("campaign") != "full_task_clutter_v1" or not manifest.get("complete"):
        raise ValueError("expected a completed full_task_clutter_v1 dataset")
    for name, stage in state["stages"].items():
        if not stage.get("complete"):
            raise ValueError(f"incomplete source stage: {name}")
        for relative, sha in stage.get("files", {}).items():
            if digest(source / relative) != sha:
                raise ValueError(f"source artifact changed: {relative}")
    # These legacy files must match the running campaign; our package is outside its vision/*.py glob.
    for relative, sha in state["inputs"]["code_sha256"].items():
        if digest(SIMULATION / relative) != sha:
            raise ValueError(f"source simulator/campaign code changed: {relative}; use its recorded revision")
    ready = json.loads((source / "recovery_rl/ready.json").read_text())
    actor = source / "recovery_rl/models/sac_best.zip"
    if digest(actor) != ready["checkpoint_sha256"]:
        raise ValueError("frozen source actor changed")
    return state, manifest


def prepare(args):
    _, manifest = source_info(args.source_run)
    report = audit(args.source_run, manifest, images=True)
    if report["problems"]:
        raise ValueError("source audit failed: " + "; ".join(report["problems"][:5]))
    root = args.out
    # Reuse actual JPEG bytes; no frame conversion/recollection or second dataset copy.
    link = root / "observations"
    target = args.source_run / "observations"
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise ValueError("observation link points at another dataset")
    elif link.exists():
        raise ValueError("observations exists and is not our source-data link")
    else:
        link.symlink_to(target, target_is_directory=True)
    for state in manifest["states"]:
        if any(pathlib.Path(p).is_absolute() or pathlib.Path(p).parts[0] != "observations"
               or ".." in pathlib.Path(p).parts for p in state["frames"]):
            raise ValueError("expected relative observations/ camera paths")
    for name in ("manifest.json", "scene_settings.json"):
        destination = root / name
        if destination.exists() and digest(destination) != digest(args.source_run/name):
            raise ValueError(f"different existing {name}")
        shutil.copyfile(args.source_run/name, destination)
    files = sorted({p for s in manifest["states"] for p in s["frames"]})
    hashes = {p: digest(args.source_run/p) for p in files}
    write_json(root / "frame_hashes.json", hashes)
    write_json(root / "reports/data.json", {**report,
        "source_run": str(args.source_run), "manifest_sha256": digest(root/"manifest.json"),
        "frame_inventory_sha256": digest(root/"frame_hashes.json"),
        "identical_manifest_and_images": True,
        "training_inputs": "RGB frames and executed 5-value action blocks only; no p, xyz, held, reward or goal labels",
        "split_contract": "Unchanged whole scene groups; goal images are created after training, not used in fitting.",
        "source_reference_gate": json.loads((args.source_run/"reports/summary.json").read_text())["reference_all_conditions_passed"]})
    print(f"REUSED {len(manifest['states'])} states, {len(manifest['rollouts'])} trajectories, "
          f"{len(files)} JPEGs; exact same scene IDs, splits, cameras and actions", flush=True)


def train(args):
    root, manifest = args.out, load_run(args.out)
    torch.manual_seed(args.training_seed)
    torch.cuda.manual_seed_all(args.training_seed)
    generator = torch.Generator().manual_seed(args.training_seed)
    sets = {split: Windows(root, manifest, split, args.history, args.frameskip) for split in ("train", "val")}
    loaders = {split: DataLoader(data, batch_size=args.batch_size, shuffle=split == "train",
        drop_last=split == "train", num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0, generator=generator if split == "train" else None)
        for split, data in sets.items()}
    if len(loaders["train"]) == 0:
        raise ValueError("batch-size exceeds training window count")
    core = build(args.history, args.frameskip).to("cuda")
    regularizer = SIGReg(num_proj=1024).to("cuda")
    optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=.001)
    # One normalizer per command coordinate, fitted ONLY to real training actions.
    actions = torch.tensor([a for r in manifest["rollouts"] if r["split"] == "train" for a in r["actions"]])
    mean = actions.mean(0).repeat(args.frameskip)
    std = actions.std(0, unbiased=False).clamp_min(.001).repeat(args.frameskip)
    metadata = {"upstream_commit": UPSTREAM, "history_frames": args.history, "frameskip": args.frameskip,
        "action_mean": mean, "action_std": std, "manifest_sha256": digest(root/"manifest.json"),
        "frame_inventory_sha256": digest(root/"frame_hashes.json"), "training_seed": args.training_seed,
        "inputs": "pixels and executed actions only", "settings": json.loads((root/"pipeline.json").read_text())["inputs"]["settings"]}
    first, best, rows = 1, float("inf"), []
    latest = root / "models/last.pt"
    if latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=True)
        if any(ck[k] != metadata[k] for k in ("upstream_commit", "history_frames", "frameskip", "manifest_sha256", "settings")):
            raise ValueError("interrupted training has different settings/data")
        core.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        first, best, rows = ck["epoch"]+1, ck["best"], ck["history"]
        torch.set_rng_state(ck["rng_cpu"])
        torch.cuda.set_rng_state_all(ck["rng_cuda"])
        generator.set_state(ck["loader_rng"])
        print(f"RESUME LeWM after epoch {ck['epoch']}; best validation={best:.6f}", flush=True)
    mean, std = mean.to("cuda"), std.to("cuda")
    print(f"LeWM: {sum(p.numel() for p in core.parameters()):,} parameters; "
          f"train/val windows={len(sets['train'])}/{len(sets['val'])}; "
          f"context={args.history} frames, one transition={args.frameskip/manifest['control_hz']:.2f}s", flush=True)
    started = time.monotonic()

    def losses(batch):
        images, action = [value.to("cuda", non_blocking=True) for value in batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = core.encode({"pixels": images, "action": (action-mean)/std})
            emb = output["emb"]
            predicted = core.predict(emb[:, :args.history], output["act_emb"])
        # Same joint predictive+SIGReg objective as upstream; neither target encoder nor predictor is frozen.
        prediction = (predicted.float()-emb[:, 1:].float()).square().mean()
        reg = regularizer(emb.float().transpose(0, 1))
        return prediction+args.sigreg_weight*reg, prediction, reg, emb[:, -1].float()

    for epoch in range(first, args.epochs+1):
        core.train()
        totals, count = np.zeros(3), 0
        for batch in loaders["train"]:
            loss, prediction, reg, _ = losses(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.)
            optimizer.step()
            n = len(batch[0]); count += n
            totals += n*np.asarray([loss.item(), prediction.item(), reg.item()])
        core.eval()
        val_totals, val_count, embeddings = np.zeros(3), 0, []
        # Fix regularizer projections during validation; selection never accesses test images.
        with torch.inference_mode(), torch.random.fork_rng(devices=[0]):
            torch.manual_seed(1729)
            for batch in loaders["val"]:
                loss, prediction, reg, emb = losses(batch)
                n = len(batch[0]); val_count += n
                val_totals += n*np.asarray([loss.item(), prediction.item(), reg.item()])
                embeddings.append(emb.cpu())
        value = val_totals[0]/val_count
        spread = float(torch.cat(embeddings).std(0, unbiased=False).mean())
        if not np.isfinite(value) or not np.isfinite(spread):
            raise RuntimeError("nonfinite validation values")
        row = {"epoch": epoch, "train_loss": float(totals[0]/count),
            "train_prediction_mse": float(totals[1]/count), "train_sigreg": float(totals[2]/count),
            "val_loss": float(value), "val_prediction_mse": float(val_totals[1]/val_count),
            "val_sigreg": float(val_totals[2]/val_count), "val_mean_embedding_std": spread}
        rows.append(row)
        weights = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
        if value < best:
            best = float(value)
            torch_save(root/"models/lewm.pt", {**metadata, "epoch": epoch, "val_loss": best, "model": weights})
        write_json(root/"reports/training.json", {"history": rows, "best_validation_loss": best,
            "checkpoint_sha256": digest(root/"models/lewm.pt"), "upstream_commit": UPSTREAM,
            "selection": "minimum validation predictive+SIGReg loss, never test/control success",
            "session_seconds": time.monotonic()-started, "parameters": sum(p.numel() for p in core.parameters()),
            "train_windows": len(sets["train"]), "val_windows": len(sets["val"]),
            "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
            "device_capacity_gib": torch.cuda.get_device_properties(0).total_memory/2**30})
        # Commit resume state after the epoch's report, so a killed parent can adopt a finished fit.
        torch_save(latest, {**metadata, "model": weights, "optimizer": optimizer.state_dict(),
            "epoch": epoch, "best": best, "history": rows, "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(), "loader_rng": generator.get_state()})
        print(f"LeWM epoch {epoch:03d} train={row['train_loss']:.5f} val={value:.5f} "
              f"prediction={row['val_prediction_mse']:.5f} SIGReg={row['val_sigreg']:.5f} "
              f"embedding_std={spread:.4f} allocated={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
        if spread < .01:
            print("DIAGNOSTIC: low embedding spread; a small prediction loss alone does not prove useful dynamics", flush=True)
    print(f"LeWM checkpoint selected by validation: {root/'models/lewm.pt'}", flush=True)


def encode(args):
    root, manifest = args.out, load_run(args.out)
    model = NativeModel(root)
    output = root / "features"
    output.mkdir(exist_ok=True)
    signature = {"manifest_sha256": digest(root/"manifest.json"), "checkpoint_sha256": digest(root/"models/lewm.pt")}
    partial = output/"latents.partial.npy"
    progress_path = output/"progress.json"
    count = 0
    if progress_path.exists():
        known = json.loads(progress_path.read_text())
        if any(known[k] != v for k, v in signature.items()):
            raise ValueError("interrupted feature cache belongs to another checkpoint")
        count = known["encoded"]
    elif (output/"meta.json").exists():
        _, meta = load_features(root, manifest)
        if meta["checkpoint_sha256"] != signature["checkpoint_sha256"]:
            raise ValueError("completed feature cache has another checkpoint")
        return
    if not partial.exists() and count:
        if count == len(manifest["states"]) and (output/"latents.npy").exists():
            partial = output/"latents.npy"
        else:
            raise ValueError("missing interrupted feature array")
    vectors = np.lib.format.open_memmap(partial, mode="r+" if partial.exists() else "w+",
        dtype=np.float32, shape=(len(manifest["states"]), DIM))
    if not 0 <= count <= len(vectors) or not np.isfinite(vectors[:count]).all():
        raise ValueError("invalid interrupted cache")
    started = time.monotonic()
    with torch.inference_mode():
        for offset in range(count, len(vectors), args.batch_size):
            batch = []
            for s in manifest["states"][offset:offset+args.batch_size]:
                with Image.open(root/s["frames"][-1]) as image:
                    batch.append(pixels(image))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.core.encode({"pixels": torch.stack(batch).to("cuda")[:, None]})["emb"][:, 0]
            value = result.float().cpu().numpy()
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite encoding at offset {offset}")
            vectors[offset:offset+len(value)] = value
            vectors.flush()
            write_json(progress_path, {**signature, "encoded": offset+len(value)})
            print(f"LeWM encoded {offset+len(value)}/{len(vectors)} frames; "
                  f"allocated={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
    del vectors
    if partial != output/"latents.npy":
        partial.replace(output/"latents.npy")
    write_json(output/"meta.json", {**signature, "keys": [s["key"] for s in manifest["states"]],
        "latents_sha256": digest(output/"latents.npy"), "model": "LeWM trained on these training scenes",
        "model_revision": UPSTREAM, "pooling": "projected_ViT_CLS", "camera": "static", "clip_frames": 1,
        "history_frames": args.history, "frameskip": args.frameskip,
        "session_seconds": time.monotonic()-started, "provenance": provenance()})
    progress_path.unlink(missing_ok=True)


def goals(args):
    """A supplied target image of the SAME scene at B; never an intermediate future input."""
    from data_collection.scripted_policy import ScriptedPickPlace
    root, manifest = args.out, load_run(args.out)
    signature = json.loads((args.source_run/"pipeline.json").read_text())["inputs"]
    layouts = {s["rollout"]: s["layout"] for s in manifest["scene_settings"]}
    groups = {g["scene"]: g for g in signature["groups"]}
    index_path = root / "goals/index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for r in manifest["rollouts"]:
        if r["case"] != "normal":
            continue
        key = f"{r['scene']}_{r['view']}"
        filename = root / "goals" / f"{key}.jpg"
        if key in index:
            if not filename.exists() or digest(filename) != index[key]["sha256"]:
                raise ValueError(f"saved goal image changed: {key}")
            continue
        filename.parent.mkdir(exist_ok=True)
        target = manifest["states"][r["states"][-1]]
        if r["result"]["task_success"] and not target["held"]:
            shutil.copyfile(root/target["frames"][-1], filename)
            origin = "recorded successful normal endpoint supplied as task goal, not training data"
        else:
            session = RecoverySession(signature["config"], layouts[r["id"]], case="normal")
            try:
                session.reset(groups[r["scene"]]["seed"])
                scripted = ScriptedPickPlace(session.sim, np.random.default_rng(groups[r["scene"]]["seed"]))
                for _ in range(session.cfg.episode.max_steps):
                    _, _, done, _, _ = session.step(scripted.act())
                    if done:
                        break
                if not session.success:
                    raise RuntimeError(f"scripted goal creation failed: {key}; no artificial success image is substituted")
                Image.fromarray(session.sim.render("static")).save(filename, quality=session.cfg.data.jpeg_quality)
                origin = "separate successful scripted task specification; never used in LeWM training"
            finally:
                session.close()
        index[key] = {"file": str(filename.relative_to(root)), "sha256": digest(filename),
            "scene": r["scene"], "view": r["view"], "split": r["split"], "origin": origin,
            "meaning": "cube released at B, same camera/clutter, reference final arm pose"}
        write_json(index_path, index)
        print(f"GOAL {key}: {origin}", flush=True)


def goal_index(root):
    index = json.loads((root/"goals/index.json").read_text())
    for key, item in index.items():
        if digest(root/item["file"]) != item["sha256"]:
            raise ValueError(f"goal image changed: {key}")
    return index


def offline(args):
    root, manifest = args.out, load_run(args.out)
    z, _ = load_features(root, manifest)
    model = NativeModel(root)
    goal_files = goal_index(root)
    layouts = {s["rollout"]: s["layout"] for s in manifest["scene_settings"]}
    goal_z = {}
    for key, item in goal_files.items():
        with Image.open(root/item["file"]) as image:
            goal_z[key] = model.encode([image])[0]
    train_ids = [i for i, s in enumerate(manifest["states"]) if s["split"] == "train"]
    scale = np.maximum(z[train_ids].std(0), 1e-3)
    frame_hashes = json.loads((root/"frame_hashes.json").read_text())
    rows, alignment = [], []
    for split in ("val", "test"):
        for view in VIEWS:
            ids = [i for i, s in enumerate(manifest["states"]) if s["split"] == split and s["view"] == view]
            costs, distances, aligned_rows = [], [], []
            for i in ids:
                state = manifest["states"][i]
                key = f"{state['scene']}_{view}"
                value = float(model.cost(torch.as_tensor(z[i:i+1, None], device="cuda"), goal_z[key])[0])
                b = np.asarray(layouts[f"{state['scene']}_{state['case']}_{view}"]["place"])
                distance = float(100*np.linalg.norm(np.asarray(state["object_xyz"][:2])-b))
                costs.append(value); distances.append(distance)
                aligned_rows.append({"key": state["key"], "scene": state["scene"], "split": split,
                    "view": view, "phase": state["phase"], "latent_goal_cost": value,
                    "actual_cube_goal_distance_cm": distance, "actual_held": state["held"],
                    "actual_task_success": state["task_success"]})
            correlation = float(np.corrcoef(costs, distances)[0, 1]) if np.std(costs) > 1e-8 and np.std(distances) > 1e-8 else None
            alignment.append({"split": split, "view": view, "n": len(ids),
                "goal_cost_vs_cube_distance_correlation": correlation,
                "mean_latent_std": float(z[ids].std(0).mean()),
                "interpretation": "positive correlation is diagnostic, not a calibrated distance or success guarantee"})
            save_csv(root/f"reports/goal_alignment_{split}_{view}.csv", aligned_rows)
    for r in manifest["rollouts"]:
        if r["split"] not in ("val", "test"):
            continue
        goal = goal_z[f"{r['scene']}_{r['view']}"]
        for blocks in (1, 2, 4, 8):
            h = blocks*args.frameskip
            if len(r["actions"]) < h:
                continue
            # Shared time grid preserves branch matches even when trajectories have different lengths.
            for start in range(0, len(r["actions"])-h+1, args.frameskip):
                ids, past = indices_and_actions(r, int(start), args.history, args.frameskip)
                commands = np.asarray(r["actions"][start:start+h], np.float32).reshape(1, blocks, -1)
                predicted = model.rollout(z[ids], past, commands)
                wrong = model.rollout(z[ids], past, -commands)
                target = r["states"][start+h]
                value, bad = predicted[0, -1].cpu().numpy(), wrong[0, -1].cpu().numpy()
                finite = np.isfinite(value).all() and np.isfinite(bad).all()
                row = {"rollout": r["id"], "scene": r["scene"], "split": r["split"], "view": r["view"],
                    "case": r["case"], "source_serial": int(start), "source_phase": manifest["states"][ids[-1]]["phase"],
                    "horizon_control_steps": h, "horizon_seconds": h/manifest["control_hz"], "finite": bool(finite)}
                # Identical reset/layout and complete executed prefix identify genuine recorded branches.
                prefix = np.asarray(r["actions"][:start], np.float32).tobytes()
                context = [r["scene"], r["view"], int(start), h,
                    hashlib.sha256(prefix).hexdigest(), layouts[r["id"]],
                    [frame_hashes[manifest["states"][i]["frames"][-1]] for i in ids]]
                row["context_sha256"] = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
                row["actions_sha256"] = hashlib.sha256(commands.tobytes()).hexdigest()
                endpoint = np.asarray(manifest["states"][target]["object_xyz"][:2])
                row["actual_future_goal_distance_cm"] = float(100*np.linalg.norm(endpoint-layouts[r["id"]]["place"]))
                if finite:
                    row.update({"predicted_z_normalized_mse": float(np.square((value-z[target])/scale).mean()),
                        "unchanged_z_normalized_mse": float(np.square((z[ids[-1]]-z[target])/scale).mean()),
                        "wrong_action_z_normalized_mse": float(np.square((bad-z[target])/scale).mean()),
                        "predicted_goal_cost": float(model.cost(predicted, goal)[0]),
                        "actual_future_goal_cost": float(model.cost(torch.as_tensor(z[target:target+1, None], device="cuda"), goal)[0])})
                rows.append(row)
    summaries = []
    for split in ("val", "test"):
        for view in VIEWS:
            for h in (args.frameskip, 2*args.frameskip, 4*args.frameskip, 8*args.frameskip):
                subset = [r for r in rows if (r["split"], r["view"], r["horizon_control_steps"]) == (split, view, h)]
                finite = [r for r in subset if r["finite"]]
                summary = {"split": split, "view": view, "horizon_control_steps": h,
                    "attempted": len(subset), "finite": len(finite), "invalid": len(subset)-len(finite)}
                if finite:
                    for name in ("predicted_z_normalized_mse", "unchanged_z_normalized_mse", "wrong_action_z_normalized_mse"):
                        summary[name] = float(np.mean([r[name] for r in finite]))
                    summary["goal_cost_mae"] = float(np.mean([abs(r["predicted_goal_cost"]-r["actual_future_goal_cost"]) for r in finite]))
                summaries.append(summary)
                print(f"OFFLINE {summary}", flush=True)
    save_csv(root/"reports/forecasts.csv", rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["context_sha256"]].append(row)
    choices = []
    for context, members in groups.items():
        distinct = list({r["actions_sha256"]: r for r in members}.values())
        if len(distinct) < 2:
            continue
        finite = all(r["finite"] for r in distinct)
        distances = [r["actual_future_goal_distance_cm"] for r in distinct]
        choice = {"context_sha256": context, "scene": distinct[0]["scene"], "split": distinct[0]["split"],
            "view": distinct[0]["view"], "source_serial": distinct[0]["source_serial"],
            "horizon_control_steps": distinct[0]["horizon_control_steps"], "candidates": len(distinct),
            "all_predictions_finite": finite, "eligible_2cm": max(distances)-min(distances) >= 2.}
        if finite:
            selected = min(distinct, key=lambda r: r["predicted_goal_cost"])
            real_latent = min(distinct, key=lambda r: r["actual_future_goal_cost"])
            choice.update(selected_rollout=selected["rollout"],
                selected_distance_cm=selected["actual_future_goal_distance_cm"],
                best_recorded_distance_cm=min(distances),
                distance_regret_cm=selected["actual_future_goal_distance_cm"]-min(distances),
                real_latent_distance_regret_cm=real_latent["actual_future_goal_distance_cm"]-min(distances))
        choices.append(choice)
    save_csv(root/"reports/recorded_choices.csv", choices)
    choice_summary = []
    for split in ("val", "test"):
        subset = [r for r in choices if r["split"] == split and r["eligible_2cm"]]
        valid = [r for r in subset if r["all_predictions_finite"]]
        choice_summary.append({"split": split, "eligible_contexts": len(subset), "valid": len(valid),
            "predicted_goal_distance_regret_cm": float(np.mean([r["distance_regret_cm"] for r in valid])) if valid else None,
            "real_latent_goal_distance_regret_cm": float(np.mean([r["real_latent_distance_regret_cm"] for r in valid])) if valid else None})
    write_json(root/"reports/offline.json", {"prediction": summaries, "goal_alignment": alignment,
        "recorded_candidate_choices": choice_summary,
        "choice_contract": "Only exact reset/layout/prefix/image-history matches; alternatives are already executed recorded commands. Regret compares endpoint cube distance, not whole-task placement. Real-latent choice uses observed endpoints for diagnosis only, never live planning. Empty eligible sets provide no ranking evidence.",
        "checkpoint_sha256": digest(root/"models/lewm.pt"),
        "contract": "Native LeWM on real starting history and recorded actions; no future images, p or object labels enter prediction. Wrong actions are a model diagnostic, not newly executed simulator branches. No Q/readout is fitted.",
        "comparison_limit": "Do not compare raw latent MSE across V-JEPA and LeWM; representations, temporal contexts and scales differ."})


def episode(args, group, view, case, layout, model, goal_item):
    root = args.out
    signature = json.loads((root/"pipeline.json").read_text())["inputs"]["source_inputs"]
    folder = root/"control"/f"{group['scene']}_{view}_{case}"/"episodes/lewm_native"/f"seed_{group['seed']}"
    marker = folder/"complete.json"
    if marker.exists():
        known = json.loads(marker.read_text())
        if any(not (folder/name).is_file() or digest(folder/name) != sha for name, sha in known.items()):
            raise ValueError(f"completed LeWM episode evidence changed: {folder}")
        print(f"REUSE LeWM {group['scene']}/{view}/{case}", flush=True)
        return
    if folder.exists():
        backup = root/"interrupted"/f"{group['scene']}_{view}_{case}_{time.time_ns()}"
        backup.parent.mkdir(exist_ok=True)
        folder.rename(backup)
    folder.mkdir(parents=True)
    session = RecoverySession(signature["config"], layout, case=case)
    rng = np.random.default_rng(group["seed"])
    rows, frames, actions, robot, xyz, held, zs = [], [], [], [], [], [], []
    previous = None
    started = time.monotonic()
    with Image.open(root/goal_item["file"]) as image:
        goal = model.encode([image])[0]
    shutil.copyfile(root/goal_item["file"], folder/"goal.jpg")

    def capture():
        filename = folder/"frames"/f"static_{len(frames):04d}.jpg"
        filename.parent.mkdir(exist_ok=True)
        Image.fromarray(session.sim.render("static")).save(filename, quality=session.cfg.data.jpeg_quality)
        frames.append(filename)
        robot.append(session.obs["proprio"].copy())
        xyz.append(session.obs["state"][:3].copy())
        held.append(bool(session.sim._check_contacts()[0]))
        with Image.open(filename) as image:
            zs.append(model.encode([image])[0].cpu().numpy())

    try:
        session.reset(group["seed"])
        capture()
        for step in range(session.cfg.episode.max_steps):
            context_ids = [max(0, i) for i in range(step-(args.history-1)*args.frameskip, step+1, args.frameskip)]
            past = [[actions[i] if i >= 0 else np.zeros(5, np.float32)
                     for i in range(t, t+args.frameskip)] for t in range(step-(args.history-1)*args.frameskip, step, args.frameskip)]
            past = np.asarray(past, np.float32).reshape(args.history-1, 5*args.frameskip)
            decision_start = time.monotonic()
            decision = plan(model, np.asarray(zs)[context_ids], past, goal, args, rng, previous)
            decision_seconds = time.monotonic()-decision_start
            proposed = decision["action"]
            _, reward, done, _, info = session.step(proposed)
            executed = np.asarray(info["executed_action"], np.float32)
            actions.append(executed.copy())
            previous = None if info["forced_release"] else decision["sequence"]
            capture()
            observed_cost = float(model.cost(torch.as_tensor(zs[-1][None, None], device="cuda"), goal)[0])
            row = {"step": step+1, "time_s": float(session.sim.data.time), "phase": info["phase"],
                "latent_goal_cost_observed": observed_cost,
                "predicted_terminal_goal_cost": float(decision["costs"][decision["selected"]]),
                "valid_candidates": int(decision["valid"].sum()), "decision_seconds": decision_seconds,
                "actual_goal_distance_cm": float(100*np.linalg.norm(session.obs["state"][:2]-session.goal[:2])),
                "actual_held": info["held_endpoint"], "actual_lift_cm": float(100*(session.obs["state"][2]-session.rest_z)),
                "task_success": info["task_success"], "forced_release": info["forced_release"],
                "table_contact": info["table_contact"], "obstacle_contact": info["obstacle_contact"],
                "true_original_reward": info["original_reward"], "control_reward_for_evaluation_only": reward,
                "proposed_gripper": float(proposed[-1])}
            row.update({f"action_{name}": float(v) for name, v in zip(("dx", "dy", "dz", "dyaw", "gripper"), executed)})
            row.update({f"actual_object_{name}": float(v) for name, v in zip("xyz", session.obs["state"][:3])})
            rows.append(row)
            forecast = folder/"forecasts"/f"step_{step+1:04d}.npz"
            forecast.parent.mkdir(exist_ok=True)
            np.savez_compressed(forecast, actions=decision["actions"].reshape(args.population, args.horizon, 5),
                costs=decision["costs"], valid=decision["valid"], selected=decision["selected"],
                selected_z=decision["selected_z"], goal_z=goal.cpu().numpy(),
                context_z=np.asarray(zs)[context_ids], intervention=info["forced_release"],
                frameskip=args.frameskip, horizon_control_steps=args.horizon)
            if (step+1) % 10 == 0 or done:
                save_csv(folder/"steps.csv", rows)
                print(f"lewm_native {group['scene']}/{view}/{case} step={step+1} "
                      f"goal={row['actual_goal_distance_cm']:.1f}cm held={row['actual_held']} "
                      f"cost={observed_cost:.3f} success={row['task_success']}", flush=True)
            if done:
                break
        save_csv(folder/"steps.csv", rows)
        np.savez_compressed(folder/"trajectory.npz", p=np.asarray(robot), object_xyz=np.asarray(xyz),
            held=np.asarray(held), actions=np.asarray(actions), z=np.asarray(zs))
        with Image.open(frames[0]) as first:
            images = []
            for path in frames[2::2]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
            first.save(folder/"actual.gif", save_all=True, append_images=images, duration=200, loop=0)
        result = {"method": "lewm_native", "scene": group["scene"], "view": view, "seed": group["seed"],
            **session.result(), "seconds": time.monotonic()-started,
            "LeWM_checkpoint_sha256": digest(root/"models/lewm.pt"), "goal_image_sha256": goal_item["sha256"],
            "model_fallback_steps": 0, "planner": {"horizon_control_steps": args.horizon, "frameskip": args.frameskip,
                "population": args.population, "elites": args.elites, "iterations": args.iterations},
            "input_contract": "RGB history, previous executed commands, supplied goal image. No Q, D_p, SAC, physical object coordinates, numeric B or reward enter action selection.",
            "scoring": "official LeWM terminal squared latent distance; costs are feature units, not metres"}
        write_json(folder/"result.json", result)
        write_json(folder/"trajectory_meta.json", {"alignment": "state[t] -> actions[t] -> state[t+1]",
            "context_frames": args.history, "context_stride_control_steps": args.frameskip,
            "native_prediction_stride_control_steps": args.frameskip,
            "goal_specification": goal_item, "physical_labels": "evaluation only",
            "forecast_frames": "selected_z contains latent vectors, not images"})
        names = ("result.json", "trajectory_meta.json", "trajectory.npz", "steps.csv", "actual.gif", "goal.jpg")
        write_json(marker, {name: digest(folder/name) for name in names})
        print(f"FINISHED LeWM: {result['task_success']} B={result['final_goal_distance_cm']:.2f}cm", flush=True)
    finally:
        session.close()


def control(args):
    root = args.out
    manifest = load_run(root)
    signature = json.loads((root/"pipeline.json").read_text())["inputs"]["source_inputs"]
    goals_index = goal_index(root)
    model = NativeModel(root)
    # Saved collection layouts avoid even resampling the obstacle geometry.
    lookup = {s["rollout"]: s["layout"] for s in manifest["scene_settings"]}
    groups = [g for g in signature["groups"] if g["split"] == "test"][:args.control_scenes]
    for group in groups:
        for view in VIEWS:
            for case in CASES:
                layout = lookup[f"{group['scene']}_{case}_{view}"]
                episode(args, group, view, case, layout, model, goals_index[f"{group['scene']}_{view}"])
    write_json(root/"reports/control_complete.json", {"complete": True, "episodes": len(groups)*len(VIEWS)*len(CASES)})


def summarize(args, complete=False):
    root = args.out
    signature = json.loads((root/"pipeline.json").read_text())["inputs"]["source_inputs"]
    scenes = {g["scene"] for g in signature["groups"] if g["split"] == "test"}
    selected = [g["scene"] for g in signature["groups"] if g["split"] == "test"][:args.control_scenes]
    rows = []
    source_summary = json.loads((args.source_run/"reports/summary.json").read_text())
    source_episodes = list((args.source_run/"control").glob("*/*/episodes/*/seed_*/result.json"))
    native_episodes = list((root/"control").glob("*/episodes/lewm_native/seed_*/result.json"))
    for file in source_episodes+native_episodes:
        marker = file.parent/"complete.json"
        if not marker.exists():
            continue
        for name, sha in json.loads(marker.read_text()).items():
            if not (file.parent/name).is_file() or digest(file.parent/name) != sha:
                raise ValueError(f"completed episode evidence changed: {file.parent/name}")
        result = json.loads(file.read_text())
        native = result["method"] == "lewm_native"
        scenario = file.parents[3].name if native else file.parents[4].name
        scene = next((s for s in scenes if scenario.startswith(s+"_")), None)
        if scene not in selected:
            continue
        row = {"scene": scene, "scenario": scenario,
            "view": "clutter" if "_clutter_" in scenario else "empty",
            "method": result["method"], "Q": "none" if native else file.parents[3].name,
            "family": "LeWM native" if native else "existing source controller",
            "seed": result["seed"], "case": result["case"],
            "LeWM_checkpoint_sha256": result.get("LeWM_checkpoint_sha256"),
            "SAC_checkpoint_sha256": result.get("SAC_checkpoint_sha256"),
            **{k: result[k] for k in ("task_success", "final_goal_distance_cm", "obstacle_contact_steps",
                "intervention_triggered", "regrasped_after_intervention", "recovery_placed", "model_fallback_steps",
                "ever_held", "ever_lifted_4cm", "maximum_lift_cm", "table_contact_steps", "steps")}}
        rows.append(row)
    if complete:
        methods = (("scripted", "exact"), ("rl_true", "exact"), ("rl_q", "empty"),
            ("rl_q", "mixed"), ("jepa_mpc", "empty"), ("jepa_mpc", "mixed"), ("lewm_native", "none"))
        actual = {(r["scene"], r["view"], r["case"], r["method"], r["Q"]) for r in rows}
        expected = {(scene, view, case, method, q) for scene in selected
                    for view in VIEWS for case in CASES for method, q in methods}
        if actual != expected or len(rows) != len(expected):
            raise ValueError(f"incomplete matched comparison: missing={sorted(expected-actual)[:5]}")
    aggregate = []
    keys = sorted({(r["method"], r["Q"], r["view"], r["case"]) for r in rows})
    for method, q, view, case in keys:
        values = [r for r in rows if (r["method"], r["Q"], r["view"], r["case"]) == (method, q, view, case)]
        aggregate.append({"method": method, "Q": q, "view": view, "case": case, "episodes": len(values),
            "placements": sum(r["task_success"] for r in values),
            "placement_rate": float(np.mean([r["task_success"] for r in values])),
            "episodes_grasped": sum(r["ever_held"] for r in values),
            "episodes_lifted_4cm": sum(r["ever_lifted_4cm"] for r in values),
            "mean_steps": float(np.mean([r["steps"] for r in values])),
            "mean_final_goal_distance_cm": float(np.mean([r["final_goal_distance_cm"] for r in values])),
            "forced_release_episodes": sum(r["intervention_triggered"] for r in values),
            "regrasps": sum(r["regrasped_after_intervention"] for r in values),
            "recovered_placements": sum(r["recovery_placed"] for r in values),
            "obstacle_contact_steps": sum(r["obstacle_contact_steps"] for r in values),
            "table_contact_steps": sum(r["table_contact_steps"] for r in values),
            "fallback_steps": sum(r["model_fallback_steps"] for r in values)})
    save_csv(root/"reports/control_episodes.csv", rows)
    save_csv(root/"reports/comparison.csv", aggregate)
    summary = {"complete": complete, "source_reference_gate": source_summary["reference_all_conditions_passed"],
        "control": aggregate, "source_run": str(args.source_run), "upstream_commit": UPSTREAM,
        "comparison": "Same scene IDs, exact saved layouts, camera, simulator, initial reset seeds and forced-release rules. Entire systems differ: representation, predictor, goal format, scoring and planning budget; this is not an isolated encoder ablation.",
        "interpretation": "Completion is not success. Goal-image distance is not metres. Measured placements, recovery and contacts establish task performance; no Q was trained for LeWM."}
    write_json(root/"reports/summary.json", summary)
    lines = ["LeWM native goal-image comparison", f"complete={complete}; source exact-RL gate={summary['source_reference_gate']}"]
    for row in aggregate:
        lines.append(f"{row['method']}/Q-{row['Q']} {row['view']}/{row['case']}: "
            f"place={row['placements']}/{row['episodes']}; B={row['mean_final_goal_distance_cm']:.2f}cm; "
            f"regrasp={row['regrasps']}/{row['forced_release_episodes']}; contacts={row['obstacle_contact_steps']}")
    (root/"reports/summary.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines), flush=True)


def export(args):
    destination = args.export
    destination.mkdir(parents=True, exist_ok=True)
    files = list((args.out/"reports").glob("*"))+list((args.out/"logs").glob("*.txt"))
    files += list((args.out/"control").glob("*/episodes/lewm_native/seed_*/*.json"))
    files += list((args.out/"control").glob("*/episodes/lewm_native/seed_*/*.csv"))
    files += [args.out/"pipeline.json", args.out/"goals/index.json"]
    index = {}
    for source in files:
        if source.is_file() and source.suffix in (".json", ".csv", ".txt"):
            relative = source.relative_to(args.out)
            target = destination/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            index[str(relative)] = {"sha256": digest(target), "bytes": target.stat().st_size}
    write_json(destination/"files.json", index)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=pathlib.Path, default=pathlib.Path("data/q_clutter_v1"))
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/lewm_native_v1"))
    p.add_argument("--export", type=pathlib.Path, default=pathlib.Path("../results/lewm_native_v1"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--sigreg-weight", type=float, default=.09)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--training-seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--horizon", type=int, default=40, help="physical control actions; must be a multiple of frameskip")
    p.add_argument("--population", type=int, default=128)
    p.add_argument("--elites", type=int, default=16)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--control-scenes", type=int, default=5)
    p.add_argument("--pilot", action="store_true", help="2 epochs, batch8, 1 test group, CEM8x1; software pilot, not scientific evidence")
    p.add_argument("--stage", choices=("prepare", "train", "encode", "goals", "offline", "control"), help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.pilot:
        args.epochs, args.batch_size, args.control_scenes = 2, 8, 1
        args.population, args.elites, args.iterations = 8, 2, 1
    if min(args.epochs, args.batch_size, args.history, args.frameskip, args.threads, args.horizon,
           args.population, args.elites, args.iterations, args.control_scenes) < 1:
        p.error("counts must be positive")
    if args.batch_size < 2 or args.workers < 0 or args.lr <= 0 or args.sigreg_weight <= 0:
        p.error("require batch-size >=2, workers >=0, positive lr and SIGReg weight")
    if args.population < 2 or args.elites > args.population or args.horizon % args.frameskip:
        p.error("require population >=2, elites <= population and horizon divisible by frameskip")
    for key in ("source_run", "out", "export"):
        setattr(args, key, getattr(args, key).resolve())
    paths = (args.source_run, args.out, args.export)
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(paths) for b in paths[i+1:]):
        p.error("source, new run and export must be separate directories, not nested")
    if not torch.cuda.is_available():
        p.error("CUDA unavailable: use the notebook CUDA virtualenv")
    torch.set_num_threads(args.threads)
    state_path = args.out/"pipeline.json"
    if args.stage:
        state = json.loads(state_path.read_text())
        settings = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items() if k != "stage"}
        if state["inputs"]["settings"] != settings:
            raise ValueError("worker settings differ from recorded pipeline")
        globals()[args.stage](args)
        return
    source_state, manifest = source_info(args.source_run)
    if args.control_scenes > source_state["inputs"]["settings"]["control_scenes"]:
        p.error("control-scenes exceeds the source campaign's completed comparison scene count")
    settings = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items() if k != "stage"}
    code = list(pathlib.Path(__file__).parent.rglob("*.py"))
    inputs = {"settings": settings, "source_inputs": source_state["inputs"],
        "source_manifest_sha256": digest(args.source_run/"manifest.json"),
        "source_summary_sha256": digest(args.source_run/"reports/summary.json"),
        "source_actor_sha256": digest(args.source_run/"recovery_rl/models/sac_best.zip"),
        "upstream_commit": UPSTREAM,
        "code_sha256": {str(f.relative_to(SIMULATION)): digest(f) for f in code},
        "native_contract": "joint encoder/predictor; official terminal goal-image cost; no Q, robot predictor or RL guide"}
    with run_lock(args.out):
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["inputs"] != inputs:
                raise ValueError("settings/code/source changed; preserve this run and use new --out/--export")
        else:
            if any(f.name != "pipeline.lock" for f in args.out.iterdir()):
                raise ValueError("nonempty output without pipeline metadata; choose a fresh --out")
            state = {"inputs": inputs, "complete": False, "stages": {}, "provenance": provenance()}
            write_json(state_path, state)
        export_state = args.export/"pipeline.json"
        if export_state.exists() and json.loads(export_state.read_text())["inputs"] != inputs:
            raise ValueError("export contains another experiment")
        options = []
        for key, value in settings.items():
            if key == "pilot":
                if value:
                    options.append("--pilot")
            else:
                options.extend(("--"+key.replace("_", "-"), str(value)))
        stages = [("prepare", ["manifest.json", "scene_settings.json", "frame_hashes.json", "reports/data.json"]),
            ("train", ["models/lewm.pt", "models/last.pt", "reports/training.json"]),
            ("encode", ["features/meta.json", "features/latents.npy"]),
            ("goals", ["goals/index.json"]),
            ("offline", ["reports/offline.json", "reports/forecasts.csv", "reports/recorded_choices.csv"]),
            ("control", ["reports/control_complete.json"])]
        print(f"GPU={torch.cuda.get_device_name()}; LeWM native; no Q/readout; "
              f"same source scenes={len({s['scene'] for s in manifest['states']})}", flush=True)
        try:
            for index, (name, files) in enumerate(stages, 1):
                known = state["stages"].get(name, {})
                if known.get("complete"):
                    if any(not (args.out/f).is_file() or digest(args.out/f) != known["files"][f] for f in files):
                        raise ValueError(f"completed {name} artifact changed")
                    if name == "prepare":
                        for frame, sha in json.loads((args.out/"frame_hashes.json").read_text()).items():
                            if digest(args.out/frame) != sha:
                                raise ValueError(f"linked source image changed: {frame}")
                    if name == "goals":
                        goal_index(args.out)
                    print(f"[{index}/7] REUSE {name}", flush=True)
                    continue
                cmd = [sys.executable, "-u", "-m", "world_model.lewm.pipeline", *options, "--stage", name]
                print(f"[{index}/7] START {name}", flush=True)
                state["stages"][name] = {"complete": False, "started": time.time(), "command": cmd}
                write_json(state_path, state)
                run_command(cmd, args.out/f"logs/{name}.txt")
                state["stages"][name].update(complete=True, finished=time.time(),
                    files={f: digest(args.out/f) for f in files})
                write_json(state_path, state)
                export(args)
            print("[7/7] SUMMARIZE matched source controllers and native LeWM", flush=True)
            summarize(args, complete=True)
            state["stages"]["summary"] = {"complete": True, "finished": time.time(),
                "files": {f"reports/{name}": digest(args.out/"reports"/name) for name in
                          ("summary.json", "summary.txt", "comparison.csv", "control_episodes.csv")}}
            state["complete"] = True
            state.pop("error", None)
            write_json(state_path, state)
            export(args)
            print(f"COMPLETE: shareable reports {args.export}; GIFs/goals/forecasts {args.out}/control", flush=True)
        except BaseException as error:
            state["complete"] = False
            state["error"] = repr(error)
            write_json(state_path, state)
            export(args)
            raise


if __name__ == "__main__":
    main()
