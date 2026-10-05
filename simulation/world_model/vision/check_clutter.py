"""CPU software check (real physics, synthetic visual vectors, NOT a JEPA accuracy test).

Run from simulation: python -m world_model.vision.check_clutter --run data/q_clutter_check
Requires the normal simulation dependencies plus Torch/SB3; no CUDA/HF downloads.
"""
import argparse
import copy
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from environment.config import load_config
from . import clutter_pipeline as campaign
from . import control_pipeline as cp
from .audit import audit, previews
from .data import digest, write_json, state_arrays, load_run
from .full_task import collect, RecoverySession, recovery_demonstrations, recovery_env, train_recovery
from .train import load_model


def check_orchestration(root, cfg, layout):
    """Stub only the expensive child jobs; exercise the real parent/resume/hash/lock logic."""
    import shutil
    from unittest.mock import patch
    base = root / "orchestration_source"
    write_json(base / "pipeline.json", {"inputs": {"config": cfg, "layout": layout,
        "options": {"position_jitter": .01}}})
    evidence = {"run": str(base), "test_seeds": [], "validation_seeds": [], "checkpoint_sha256": "SOFTWARE_ONLY"}
    target = root / "orchestration"
    argv = ["clutter_pipeline", "--pilot", "--run", str(target), "--baseline-run", str(base),
            "--encoder-run", str(root), "--export", str(root / "orchestration_export")]
    calls = []
    def job(command, log):
        calls.append(command)
        module = command[command.index("-m")+1].split(".")[-1]
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("SOFTWARE STUB; no GPU experiment was run\n")
        if module == "clutter_pipeline":
            stage = command[command.index("--stage")+1]
            if stage in ("reference", "visual"):
                write_json(target / f"reports/{stage}_complete.json", {"complete": True, "stub": True})
            elif stage == "collect":
                for name in ("manifest.json", "scene_settings.json"):
                    shutil.copyfile(root / name, target / name)
            elif stage == "recovery_rl":
                write_json(target / "recovery_rl/ready.json", {"checkpoint_sha256": "STUB", "new_recovery_fit": True})
                write_json(target / "recovery_rl/source_validation.json", {"stub": True})
                folder = target / "recovery_rl/models"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "sac_best.zip").write_text("STUB")
                write_json(folder / "rl_training.json", {"stub": True})
            elif stage == "bind":
                for name in ("models/readout.pt", "models/readout_p.pt", "reports/readout_validation.json"):
                    dst = target / "attempts/mixed_dynamics" / name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(target / "attempts/q_mixed" / name, dst)
            elif stage == "offline":
                write_json(target / "reports/offline.json", {"stub": True})
                (target / "reports/Q_real_states.csv").write_text("stub\n")
        elif module == "encode":
            toy_cache(target)
        elif module == "audit":
            write_json(target / "reports/audit.json", {"stub": True})
        elif module == "train":
            stage = command[command.index(module)+1] if module in command else command[command.index("-m")+2]
            tag = command[command.index("--tag")+1]
            roles = ("readout", "readout_p") if stage == "readout" else ("no_vision",) if stage == "baseline" else ("dynamics",)
            for role in roles:
                file = target / f"attempts/{tag}/models/{role}.pt"
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("STUB checkpoint, never loaded")
                write_json(target / f"attempts/{tag}/reports/{role}_training.json", {"stub": True})
            if stage == "readout":
                write_json(target / f"attempts/{tag}/reports/readout_validation.json", {"stub": True})
        else:
            raise AssertionError(command)
    with patch.object(sys, "argv", argv), patch.object(cp, "frozen_baseline", return_value=evidence), patch.object(campaign, "run_command", job):
        campaign.main()
        assert len(calls) == 14, len(calls)
        campaign.main()
        assert len(calls) == 14, "finished stages reran"
        state = json.loads((target / "pipeline.json").read_text())
        del state["stages"]["encode"]
        write_json(target / "pipeline.json", state)
        campaign.main()
        assert len(calls) == 14, "completed encoder cache was not adopted"
        artifact = target / "attempts/q_empty/models/readout.pt"
        artifact.write_text("corrupt")
        try:
            campaign.main()
        except ValueError as error:
            assert "artifact changed" in str(error)
        else:
            raise AssertionError("tampered completed checkpoint was reused")
    with campaign.run_lock(root / "lock_check"):
        try:
            with campaign.run_lock(root / "lock_check"):
                raise AssertionError("concurrent runner was accepted")
        except RuntimeError as error:
            assert "another campaign process" in str(error)


def toy_cache(root):
    manifest = load_run(root)
    p, xyz, held = state_arrays(manifest)
    # Intentionally privileged synthetic fixture. It checks software, never JEPA performance.
    z = np.concatenate((xyz, held[:, None], p[:, 14:17], p[:, 18:19]), 1).astype(np.float32)
    folder = root / "features"
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "latents.npy", z)
    write_json(folder / "meta.json", {"manifest_sha256": digest(root / "manifest.json"),
        "latents_sha256": digest(folder / "latents.npy"), "keys": [s["key"] for s in manifest["states"]],
        "model": "SOFTWARE_CHECK_NOT_JEPA", "model_revision": "toy", "camera": "static",
        "clip_frames": 64, "pooling": "mean_all_encoder_tokens"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=pathlib.Path, default=pathlib.Path("data/q_clutter_check"))
    parser.add_argument("--reuse-collection", type=pathlib.Path, help="reuse an existing software-check collection")
    args = parser.parse_args()
    root = args.run.resolve()
    if root.exists():
        parser.error("use a fresh check directory")
    cfg = load_config("configs/grade_e.yml", overrides={"episode": {"terminate_on_success": False}})
    layout = json.loads(pathlib.Path("configs/grade_e_layout.json").read_text())
    signature = {"config": cfg, "layout": layout, "settings": {"software_check": True},
        "groups": [{"scene": f"scene_{i:04d}", "seed": 4405+i, "split": split}
                   for i, split in enumerate(("train", "val", "test"))]}
    if args.reuse_collection:
        import shutil
        source = args.reuse_collection.resolve()
        for name in ("manifest.json", "scene_settings.json"):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, root / name)
        # Copy paths remain relative to the run; one symlink avoids duplicating thousands of JPEGs.
        (root / "observations").symlink_to(source / "observations", target_is_directory=True)
    else:
        collect(root, signature)
    manifest = load_run(root)
    report = audit(root, manifest)
    assert not report["problems"], report["problems"]
    before = digest(root / "manifest.json")
    collect(root, signature)
    assert digest(root / "manifest.json") == before, "completed collection changed on resume"
    damaged = copy.deepcopy(manifest)
    damaged["rollouts"][0]["actions"][0][0] = .1234
    assert audit(root, damaged, images=False)["problems"], "audit missed a paired/alignment error"
    damaged = copy.deepcopy(manifest)
    damaged["states"][0]["split"] = "test"
    assert audit(root, damaged, images=False)["problems"], "audit missed scene leakage"
    previews(root, manifest)
    toy_cache(root)
    if not args.reuse_collection:
        from stable_baselines3 import SAC
        from .task_control import make_rl_env
        data, demo_ids = recovery_demonstrations(root)
        expected = sum(not manifest["states"][i]["forced_release"] for r in manifest["rollouts"]
                       if r["split"] == "train" and r["result"]["task_success"] for i in r["states"][1:])
        assert len(data["obs"]) == expected
        env = recovery_env(cfg, layout, 0, 44, signature["groups"][:1])
        try:
            env.reset(seed=3)
            proposed = np.array([.1, 0, 0, 0, -1], np.float32)
            _, _, _, _, info = env.step(proposed)
            np.testing.assert_array_equal(info["executed_action"], proposed)
            assert not info["forced_release"]
        finally:
            env.close()
        source = root / "recovery_source"
        env = make_rl_env(cfg, layout, 0, 44)
        try:
            prototype = SAC("MlpPolicy", env, device="cpu", policy_kwargs={"net_arch": [128, 128]},
                            learning_rate=3e-5, ent_coef="auto_0.005", buffer_size=100000)
            (source / "models").mkdir(parents=True)
            prototype.save(source / "models/sac_best.zip")
        finally:
            env.close()
        opts = SimpleNamespace(baseline_run=source, threads=2, seed=44, bc_epochs=2,
                               critic_warmup=2, rl_steps=1050)
        train_recovery(root, signature, opts)
        ready = json.loads((root / "recovery_rl/ready.json").read_text())
        assert ready["new_recovery_fit"]
        assert ready["checkpoint_sha256"] == digest(root / "recovery_rl/models/sac_best.zip")
        train_recovery(root, signature, opts)
        fitted = root / "recovery_rl/models/rl_training.json"
        report = json.loads(fitted.read_text())
        assert report["settings"]["validation_seeds"] == [4406]
        assert report["settings"]["ent_coef"] == "auto_0.005"
        fitted.unlink()  # Emulate interruption after the final replay/checkpoint commit.
        train_recovery(root, signature, opts)
        assert digest(root / "recovery_rl/models/sac_best.zip") == ready["checkpoint_sha256"]
        # The no-fit branch copies weights unchanged; its passing probe is a stub.
        copy_root = root / "reuse_source_check"
        write_json(source / "models/rl_training.json", {"software_fixture": True})
        write_json(copy_root / "recovery_rl/source_validation.json", {
            "source_checkpoint_sha256": digest(source / "models/sac_best.zip"), "selection_success_rate": 1.0})
        train_recovery(copy_root, signature, opts)
        reused = json.loads((copy_root / "recovery_rl/ready.json").read_text())
        assert not reused["new_recovery_fit"]
        assert reused["checkpoint_sha256"] == reused["source_checkpoint_sha256"]
    def train(stage, tag, *extra):
        subprocess.run([sys.executable, "-m", "world_model.vision.train", stage, "--run", str(root),
                        "--tag", tag, "--epochs", "2", "--rollout-steps", "4", *extra], check=True)
    train("readout", "q_empty", "--view", "empty")
    train("readout", "q_mixed")
    train("dynamics", "mixed_dynamics")
    train("baseline", "mixed_dynamics", "--baseline-horizon", "20")
    import shutil
    for name in ("models/readout.pt", "models/readout_p.pt", "reports/readout_validation.json"):
        shutil.copyfile(root / "attempts/q_mixed" / name, root / "attempts/mixed_dynamics" / name)
    train("width", "shared_width", "--from-tag", "mixed_dynamics")
    _, empty = load_model(root, "readout", "q_empty")
    _, mixed = load_model(root, "readout", "q_mixed")
    assert empty["training_settings"]["view"] == "empty"
    assert mixed["training_settings"]["view"] == "all"
    ids = [i for i, s in enumerate(manifest["states"]) if s["split"] == "train" and s["view"] == "empty"]
    robot, _, _ = state_arrays(manifest)
    np.testing.assert_allclose(empty["p_mean"], robot[ids].mean(0), atol=1e-6, rtol=0)
    fitting = json.loads((root / "attempts/q_empty/reports/readout_training.json").read_text())
    assert fitting["training_examples"] == len(ids)
    campaign.offline(root, SimpleNamespace(threads=2))
    # Poor numeric forecasts are exported as model failures, without aborting the campaign.
    from unittest.mock import patch
    offline_file = root / "reports/offline.json"
    original_report = offline_file.read_text()
    with patch("world_model.vision.train.imagine", return_value=(np.full((16, 8), np.nan), np.full((16, 20), np.nan))):
        campaign.offline(root, SimpleNamespace(threads=2))
    failed = json.loads(offline_file.read_text())
    assert all(r["n"] == 0 and len(r["invalid_windows"]) == r["attempted_windows"] for r in failed["D"])
    offline_file.write_text(original_report)
    # Exercise physical intervention plus recording using the shared live runner.
    opts = SimpleNamespace(method="scripted", episode_seed=4405, threads=2, position_jitter=0)
    cp.episode(root / "control/scene_0000_clutter_drop_middle/exact",
        {"config": cfg, "layout": campaign.layouts(layout, 4405)["clutter"], "known_rest_z": .7725}, opts,
        session_factory=lambda cfg, layout, jitter: RecoverySession(cfg, layout, jitter, "drop_middle"))
    folder = root / "control/scene_0000_clutter_drop_middle/exact/episodes/scripted/seed_4405"
    write_json(folder / "complete.json", {"result.json": digest(folder / "result.json")})
    result = json.loads((folder / "result.json").read_text())
    assert result["recovery_placed"] and result["regrasped_after_intervention"]
    trajectory = np.load(folder / "trajectory.npz")
    assert len(trajectory["p"]) == len(trajectory["actions"])+1
    assert np.array_equal(trajectory["actions"][result["intervention_step"]-1:result["release_end_step"], -1], np.ones(7))
    signature["frozen_baseline"] = {"checkpoint_sha256": "software-only-no-SAC"}
    summary = campaign.summarize(root, signature, final=True)
    assert summary["control"][0]["view"] == "clutter" and summary["control"][0]["Q"] == "exact"
    assert not summary["reference_all_conditions_passed"], "one scripted trajectory cannot pass the RL gate"
    exported = root.parent / (root.name+"_export")
    campaign.export(root, exported)
    assert all(digest(exported / name) == info["sha256"] for name, info in json.loads((exported / "files.json").read_text()).items())
    check_orchestration(root, cfg, layout)
    write_json(root / "software_check.json", {"passed": True, "real_states": len(manifest["states"]),
        "real_rollouts": len(manifest["rollouts"]), "real_scripted_recovery_placed": result["task_success"],
        "visual_vectors": "synthetic privileged fixture, NOT JEPA", "GPU_training_tested": False,
        "tiny_recovery_SAC_fit_tested": not bool(args.reuse_collection),
        "checks": ["physics/drop/regrasp", "same-action clutter pairs", "causal frames/action alignment",
            "whole-scene splits", "collection resume", "Q subset normalizers", "Q/D/blind/width fitting",
            "offline reports", "real GIF/trajectory recording", "failed reference gate", "export hashes",
            "14-stage orchestration (stub child jobs)", "finished-stage resume", "completed-cache adoption",
            "checkpoint tamper rejection", "concurrent-run lock", "nonfinite forecasts recorded"]
            + (["external opening excluded from actor imitation", "online replay action matches execution",
                "source actor/critic initialization", "tiny SAC updates/save/load/replay resume", "source-copy no-fit branch (stub passing probe)"]
               if not args.reuse_collection else [])})
    print(f"SOFTWARE CHECK PASS: {root}/software_check.json; synthetic vectors, no JEPA accuracy claim", flush=True)


if __name__ == "__main__":
    main()
