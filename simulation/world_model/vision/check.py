"""Small CPU regression check using toy features, never a V-JEPA performance result."""
import copy
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

from .audit import audit
from .data import digest, load_features, write_json


def fixture(root, extra=False):
  manifest = {"schema": "vision_consequences_v1", "complete": True, "camera": "static",
              "config": {"toy_check": True},
              "clip_frames": 64, "control_hz": 10, "settings": {"pair_tolerance": 1e-3},
              "states": [], "rollouts": []}
  image = "frame.jpg"
  Image.new("RGB", (32, 32), "teal").save(root / image)
  for scene_number, split in enumerate(("train", "train", "train", "val", "test")):
    scene = f"scene_{scene_number:04d}"
    for placement in ("under", "offset"):
      p0 = np.zeros(20, np.float32)
      p0[14:17] = [0.15+scene_number*.01, -.23, .78]
      p0[18:] = [.08, 1]
      pos = [float(p0[14]) + (.1 if placement == "offset" else 0), -.23, .775]
      source = len(manifest["states"])
      manifest["states"].append({"key": f"{scene}/{placement}/source", "scene": scene, "split": split,
                   "frames": [image], "p": p0.tolist(), "object_xyz": pos, "held": False, "time": 0})
      for branch in (("close_lift", "open_lift", "close_side_lift", "close_lift_release") if extra else ("close_lift", "open_lift")):
        ids, actions = [source], []
        for step in range(1, 9):
          close = branch != "open_lift" and not (branch == "close_lift_release" and step >= 6)
          action = [int(branch == "close_side_lift")*.1, 0, int(step > 2), 0, -1 if close else 1]
          held = placement == "under" and close
          p = p0.copy(); p[16] += max(step-2, 0)*.01; p[19] = action[-1]
          p[18] = .04 if held else (.08 if action[-1] == 1 else 0)
          xyz = pos.copy(); xyz[2] += max(step-2, 0)*.01 if held else 0
          ids.append(len(manifest["states"])); actions.append(action)
          manifest["states"].append({"key": f"{scene}/{placement}/{branch}/{step}", "scene": scene,
            "split": split, "frames": [image]*(step+1), "p": p.tolist(), "object_xyz": xyz,
            "held": held, "time": step*.1, "action_from_previous": action})
        manifest["rollouts"].append({"id": f"{scene}/{placement}/{branch}", "scene": scene,
          "split": split, "placement": placement, "branch": branch, "states": ids, "actions": actions,
          "restore_p_error": 0, "restore_integration_error": 0})
  write_json(root / "manifest.json", manifest)
  # Deliberately simple synthetic vectors check plumbing, not the real encoder.
  z = np.asarray([s["object_xyz"] + [float(s["held"])] + s["p"][14:18]
                  for s in manifest["states"]], np.float32)
  (root / "features").mkdir()
  np.save(root / "features/latents.npy", z)
  write_json(root / "features/meta.json", {"manifest_sha256": digest(root / "manifest.json"),
             "latents_sha256": digest(root / "features/latents.npy"),
             "keys": [s["key"] for s in manifest["states"]], "model": "TOY-CHECK-NOT-VJEPA"})
  return manifest


def main():
  with tempfile.TemporaryDirectory(prefix="vision-check-") as folder:
    root = pathlib.Path(folder)
    manifest = fixture(root)
    assert not audit(root, manifest)["problems"]
    broken = copy.deepcopy(manifest)
    broken["states"][1]["action_from_previous"] = [.5, 0, 0, 0, -1]
    assert any("action not aligned" in p for p in audit(root, broken)["problems"])
    broken = copy.deepcopy(manifest)
    broken["states"][1]["split"] = "test"
    assert any("leakage" in p for p in audit(root, broken)["problems"])
    manifest_path = root / "manifest.json"
    original = manifest_path.read_text(); manifest_path.write_text(original+" ")
    try:
      load_features(root, manifest)
      raise AssertionError("stale features accepted")
    except ValueError:
      pass
    manifest_path.write_text(original)

    def run(module, *args):
      result = subprocess.run([sys.executable, "-m", f"world_model.vision.{module}",
                               *args, "--run", str(root)], capture_output=True, text=True)
      assert result.returncode == 0, result.stdout+result.stderr
      print(f"PASS {module} {' '.join(args[:1])}", flush=True)

    run("audit")
    for stage in ("readout", "dynamics", "baseline"):
      run("train", stage, "--epochs", "2", "--batch-size", "32", "--rollout-steps", "4")
    for test in ("persistence", "action", "no-vision", "positions"):
      run("test", test, "--horizon", "8")
      report = json.loads((root / "reports" / f"{test}_test_h8.json").read_text())
      assert report["scene_groups"] == 1
      assert isinstance(report["Q_object_interpretation_gate_passed"], bool)
      if test in ("action", "positions"):
        assert report["eligible_pairs"] >= 1
    # An attempt/tag must still load the original data/cache contract.
    run("train", "baseline", "--epochs", "1", "--tag", "retry")
    # Exercise the complete runner with toy collection/encoding and real CPU model/test CLIs.
    # No GPU, real V-JEPA, or physics-quality claim is made by this fixture.
    from unittest.mock import patch
    from . import pipeline
    raw, exported = root / "campaign", root / "shared"
    original_runner = pipeline.run_command
    def toy_runner(command, log):
      if "-c" in command:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("TOY CPU CHECK: GPU preflight replaced, not a real encoder run\n")
      elif "world_model.vision.collect" in command:
        fixture(raw, extra=True)
      else:
        original_runner(command, log)
    arguments = ["pipeline", "--run", str(raw), "--export-dir", str(exported), "--profile", "controlled",
                 "--train-scenes", "3", "--val-scenes", "1", "--test-scenes", "1", "--epochs", "2",
                 "--training-seeds", "0", "--rollout-steps", "4", "--horizon", "8"]
    with patch.object(sys, "argv", arguments), patch.object(pipeline, "run_command", toy_runner):
      pipeline.main()
    state = json.loads((raw / "pipeline.json").read_text())
    assert state["complete"] and len(state["stages"]) == 14
    action_report = json.loads((raw / "attempts/seed_0/reports/action_test_h8.json").read_text())
    assert action_report["additional_action_comparisons"]["eligible_pairs"] > 0
    assert (exported / "summary.csv").exists()
    assert all(p.suffix in (".txt", ".json", ".csv") for p in exported.rglob("*") if p.is_file())
    assert not list(exported.rglob("*.pt")) and not list(exported.rglob("*.npy"))
    interrupted = raw / "attempts/interrupted/models/dynamics.pt"
    interrupted.parent.mkdir(parents=True); interrupted.write_bytes(b"incomplete-checkpoint")
    pipeline.archive_training([interrupted], raw)
    preserved = list((raw / "logs/interrupted").rglob("dynamics.pt"))
    assert not interrupted.exists() and len(preserved) == 1 and preserved[0].read_bytes() == b"incomplete-checkpoint"
    try:
      pipeline.run_command([sys.executable, "-c", "import sys; print('intentional failure check'); sys.exit(3)"], raw / "logs/failure.txt")
      raise AssertionError("failed subprocess accepted")
    except RuntimeError as error:
      assert "exited 3" in str(error)
    with patch.object(sys, "argv", arguments+["--resume"]), patch.object(pipeline, "run_command",
          side_effect=AssertionError("completed stage rerun")):
      pipeline.main()
    report_path = raw / "reports/audit.json"
    report_path.write_text(report_path.read_text()+" ")
    with patch.object(sys, "argv", arguments+["--resume"]):
      try:
        pipeline.main()
        raise AssertionError("modified stage artifact accepted")
      except ValueError as error:
        assert "artifacts changed" in str(error)
    assert not audit(raw, json.loads((raw / "manifest.json").read_text()))["problems"]
    print("PASS pipeline sequencing, additional branches, report export, resume and changed-artifact guard", flush=True)
    # Perfect predictions must beat persistence/swapped pairing and the blind position baseline.
    from .data import state_arrays
    from .test import action_test, position_test, persistence
    z, _ = load_features(root, manifest)
    p, xyz, grasp = state_arrays(manifest)
    exact = {}
    for r in manifest["rollouts"]:
      if r["split"] == "test":
        i = r["states"]
        exact[r["id"]] = {"rollout": r, "z": z[i], "p": p[i], "xyz": xyz[i], "grasp": grasp[i],
          "real_q_xyz": xyz[i], "real_q_grasp": grasp[i],
          "persistent_q_xyz": np.repeat(xyz[i[:1]], len(i), axis=0), "persistent_q_grasp": np.zeros(len(i)),
          "no_vision_xyz": xyz[i[:1]], "no_vision_grasp": np.zeros(1)}
    import torch
    contrast = action_test(exact, z, xyz, {"z_std": torch.ones(z.shape[1])}, 8, 2)
    assert contrast["matched_beats_swapped_fraction"] == 1 and contrast["height_effect_mae_cm"] == 0
    positions = position_test(exact, p, xyz, 8, 2, 1e-3)
    assert positions["D_Q_effect_mae_cm"] == 0 and positions["no_vision_effect_mae_cm"] > 5
    # The real implementation expects Torch tensors for checkpoint statistics.
    persistence_report = persistence(exact, z, p, xyz, grasp,
                        {"z_std": torch.ones(z.shape[1]), "p_std": torch.ones(20)}, [8])
    assert persistence_report["by_horizon"][0]["D_z_mse"] == 0
    assert persistence_report["by_horizon"][0]["persistence_z_mse"] > 0

    from .train import load_model
    load_model(root, "no_vision", "retry")
    from world_model.train_dynamics import SplitDynamics
    model = SplitDynamics(8, 20, 5)
    zz, pp = torch.randn(2, 8), torch.randn(2, 20)
    for _ in range(3):
      zz, pp = model(zz, pp, torch.randn(2, 5))
    zz.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.robot.parameters())
    print("PASS perfect-prediction comparisons and multi-step gradient isolation")
  print("PASS alignment/leakage/cache guards, training, tagged loading and all four test commands. "
        "Toy CPU check only; no real encoder/model-quality claim.")


if __name__ == "__main__":
  main()
