"""Compare D with two real actions taken from the same saved validation state."""

import argparse
import hashlib
import json
import pathlib

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

from environment import load_config
from .replay import read_episode, replay_branch
from .train_dynamics import Dynamics


def _next_clip(folder, camera, source, clip_frames, branch_jpeg):
  frames = []
  for serial in range(source - clip_frames + 2, source + 1):
    path = folder / "images" / f"{camera}_{max(1, serial)}.jpg"
    frame = cv2.imread(str(path))
    if frame is None:
      raise FileNotFoundError(path)
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  branch = cv2.imdecode(np.frombuffer(branch_jpeg, np.uint8), cv2.IMREAD_COLOR)
  frames.append(cv2.cvtColor(branch, cv2.COLOR_BGR2RGB))
  return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)


def _encode(model, processor, clip):
  inputs = processor(clip, return_tensors="pt").to("cuda")
  with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    tokens = model(**inputs, skip_predictor=True).last_hidden_state
  return tokens.float().mean(dim=1).squeeze(0).cpu().numpy()


def _predict(model, checkpoint, z, p, action, device):
  z_mean = checkpoint["z_mean"].numpy()
  z_std = checkpoint["z_std"].numpy()
  p_mean = checkpoint["p_mean"].numpy()
  p_std = checkpoint["p_std"].numpy()
  z_in = torch.as_tensor((z - z_mean) / z_std, dtype=torch.float32, device=device)[None]
  p_in = torch.as_tensor((p - p_mean) / p_std, dtype=torch.float32, device=device)[None]
  a_in = torch.as_tensor(action, dtype=torch.float32, device=device)[None]
  with torch.inference_mode():
    pred_z, pred_p = model(z_in, p_in, a_in)
  return (pred_z[0].cpu().numpy() * z_std + z_mean,
          pred_p[0].cpu().numpy() * p_std + p_mean)


def _errors(pred_z, pred_p, true_z, true_p, checkpoint):
  z_error = np.square((pred_z - true_z) / checkpoint["z_std"].numpy()).mean()
  p_error = np.square((pred_p - true_p) / checkpoint["p_std"].numpy()).mean()
  return {"z_mse": float(z_error), "p_mse": float(p_error),
          "total": float(z_error + p_error)}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", required=True, type=pathlib.Path)
  parser.add_argument("--features", required=True, type=pathlib.Path)
  parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
  parser.add_argument("--out", type=pathlib.Path,
                      default=pathlib.Path("artifacts/grade_e/action_contrast"))
  parser.add_argument("--sample-index", type=int,
                      help="zero-based index among validation transitions; default: most x clearance")
  args = parser.parse_args()
  if not torch.cuda.is_available():
    parser.error("CUDA is needed to encode the two real branch frames")
  manifest = json.loads(args.manifest.read_text())
  feature_meta = json.loads((args.features / "meta.json").read_text())
  if (feature_meta["manifest_sha256"] != hashlib.sha256(args.manifest.read_bytes()).hexdigest()
      or feature_meta["camera"] != manifest["camera"]
      or feature_meta["clip_frames"] != manifest["clip_frames"]):
    parser.error("feature cache does not match the manifest")
  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
  if (checkpoint["encoder_model"] != feature_meta["model"]
      or checkpoint["pooling"] != feature_meta["pooling"]
      or checkpoint["camera"] != feature_meta["camera"]
      or checkpoint["clip_frames"] != feature_meta["clip_frames"]
      or checkpoint["proprio_columns"] != manifest["proprio_columns"]
      or checkpoint["action_columns"] != manifest["action_columns"]):
    parser.error("checkpoint does not match the manifest/feature cache")
  val = [s for s in manifest["samples"] if s["split"] == "val"]
  if not val:
    parser.error("manifest has no validation samples")
  if args.sample_index is None:
    cfg = load_config(overrides=read_episode(
      (args.manifest.resolve().parent / manifest["episodes_dir"]).resolve(), val[0])[2]["config"])
    low, high = cfg.control.workspace.low[0], cfg.control.workspace.high[0]
    sample_index = max(range(len(val)), key=lambda i: min(val[i]["p"][14] - low,
                                                          high - val[i]["p"][14]))
  else:
    sample_index = args.sample_index
  if not 0 <= sample_index < len(val):
    parser.error(f"--sample-index must be in 0..{len(val) - 1}")
  sample = val[sample_index]
  episodes = (args.manifest.resolve().parent / manifest["episodes_dir"]).resolve()
  folder, rows, meta = read_episode(episodes, sample)
  print(f"validation sample {sample_index}/{len(val) - 1}: "
        f"{sample['episode']} after serial {sample['source']}", flush=True)
  p = np.asarray(sample["p"], dtype=np.float32)
  gripper = float(p[-1])
  actions = {"plus_x": [1.0, 0.0, 0.0, 0.0, gripper],
             "minus_x": [-1.0, 0.0, 0.0, 0.0, gripper]}
  if len(manifest["action_columns"]) != 5:
    parser.error("this contrast expects the five-action yaw-enabled controller")

  latents = np.load(args.features / "latents.npy")
  source_key = f"{sample['episode']}:{sample['source']}"
  z = latents[feature_meta["index"][source_key]].astype(np.float32)
  if (len(z) != checkpoint["z_dim"] or len(p) != checkpoint["p_dim"]
      or len(actions["plus_x"]) != checkpoint["a_dim"]):
    parser.error("checkpoint tensor dimensions do not match the selected sample")
  device = "cuda"
  dynamics = Dynamics(checkpoint["z_dim"], checkpoint["p_dim"],
                      checkpoint["a_dim"], width=checkpoint["width"]).to(device).eval()
  dynamics.load_state_dict(checkpoint["model"])
  processor = AutoVideoProcessor.from_pretrained(feature_meta["model"])
  encoder = AutoModel.from_pretrained(feature_meta["model"],
                                      attn_implementation="sdpa")
  encoder.to(device, dtype=torch.bfloat16).eval()

  branches = {}
  replay_mode = None
  source_state = None
  for name, action in actions.items():
    actual_p, jpeg, p_replay, object_replay, replay_mode, replay_state = replay_branch(
      sample, rows, meta, action, manifest["camera"], replay_mode)
    if source_state is None:
      source_state = replay_state
    elif not np.allclose(source_state, replay_state, rtol=0, atol=1e-8):
      raise ValueError("the two branches did not start from the same simulator/controller state")
    clip = _next_clip(folder, manifest["camera"], sample["source"],
                      manifest["clip_frames"], jpeg)
    actual_z = _encode(encoder, processor, clip)
    predicted_z, predicted_p = _predict(dynamics, checkpoint, z, p, action, device)
    branches[name] = {"action": action, "actual_p": actual_p,
                      "actual_z": actual_z, "predicted_p": predicted_p,
                      "predicted_z": predicted_z, "jpeg": jpeg,
                      "replay_p_max_abs": p_replay,
                      "replay_object_pose_max_abs": object_replay}

  plus, minus = branches["plus_x"], branches["minus_x"]
  matched = {name: _errors(b["predicted_z"], b["predicted_p"],
                           b["actual_z"], b["actual_p"], checkpoint)
             for name, b in branches.items()}
  wrong = {"plus_x": _errors(minus["predicted_z"], minus["predicted_p"],
                             plus["actual_z"], plus["actual_p"], checkpoint),
           "minus_x": _errors(plus["predicted_z"], plus["predicted_p"],
                              minus["actual_z"], minus["actual_p"], checkpoint)}
  effect = {"actual_ee_x_cm": float(100 * (plus["actual_p"][14] - minus["actual_p"][14])),
            "predicted_ee_x_cm": float(100 * (plus["predicted_p"][14]
                                                 - minus["predicted_p"][14]))}
  report = {
    "source": {"episode": sample["episode"], "serial": sample["source"],
               "validation_index": sample_index, "ee_x_m": float(p[14])},
    "manifest": str(args.manifest.resolve()), "checkpoint": str(args.checkpoint.resolve()),
    "replay_mode": replay_mode,
    "replay_max_abs": {name: {"proprio": b["replay_p_max_abs"],
                               "object_pose": b["replay_object_pose_max_abs"]}
                       for name, b in branches.items()},
    "actions": actions,
    "ee_x_cm_from_source": {
      name: {"actual": float(100 * (b["actual_p"][14] - p[14])),
             "predicted": float(100 * (b["predicted_p"][14] - p[14]))}
      for name, b in branches.items()},
    "branch_difference": effect, "matched_action_error": matched,
    "wrong_action_error": wrong,
  }
  args.out.mkdir(parents=True, exist_ok=True)
  path = args.out / f"{sample['episode']}_serial_{sample['source']:04d}"
  path.mkdir(parents=True, exist_ok=True)
  (path / "results.json").write_text(json.dumps(report, indent=2) + "\n")
  np.savez_compressed(path / "states.npz", source_z=z, source_p=p,
                      **{f"{name}_{key}": b[key] for name, b in branches.items()
                         for key in ("actual_z", "actual_p", "predicted_z", "predicted_p")})
  for name, b in branches.items():
    (path / f"{name}.jpg").write_bytes(b["jpeg"])
    moves = report["ee_x_cm_from_source"][name]
    print(f"{name:7s} actual ee x change={moves['actual']:+.2f} cm; "
          f"D predicted={moves['predicted']:+.2f} cm; "
          f"matched z/p MSE={matched[name]['z_mse']:.4f}/{matched[name]['p_mse']:.4f}")
  print(f"real +x versus -x ee difference={effect['actual_ee_x_cm']:+.2f} cm; "
        f"D predicted={effect['predicted_ee_x_cm']:+.2f} cm")
  print(f"mean matched total MSE={np.mean([x['total'] for x in matched.values()]):.4f}; "
        f"wrong-action total MSE={np.mean([x['total'] for x in wrong.values()]):.4f}")
  print(f"saved branch images, states and metrics to {path}")


if __name__ == "__main__":
  main()
