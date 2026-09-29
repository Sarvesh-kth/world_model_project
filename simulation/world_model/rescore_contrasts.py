"""Score a new dynamics checkpoint on previously saved real action branches."""

import argparse
import json
import pathlib

import numpy as np
import torch

from .train_dynamics import model_from_checkpoint


def _predict(model, checkpoint, z, p, action, device):
  z_mean = checkpoint["z_mean"].numpy()
  z_std = checkpoint["z_std"].numpy()
  p_mean = checkpoint["p_mean"].numpy()
  p_std = checkpoint["p_std"].numpy()
  z_in = torch.as_tensor((z - z_mean) / z_std, dtype=torch.float32, device=device)[None]
  p_in = torch.as_tensor((p - p_mean) / p_std, dtype=torch.float32, device=device)[None]
  a_in = torch.as_tensor(action, dtype=torch.float32, device=device)[None]
  with torch.inference_mode():
    predicted_z, predicted_p = model(z_in, p_in, a_in)
  return (predicted_z[0].cpu().numpy() * z_std + z_mean,
          predicted_p[0].cpu().numpy() * p_std + p_mean)


def _errors(predicted_z, predicted_p, actual_z, actual_p, checkpoint):
  z_mse = float(np.square((predicted_z - actual_z) / checkpoint["z_std"].numpy()).mean())
  p_mse = float(np.square((predicted_p - actual_p) / checkpoint["p_std"].numpy()).mean())
  return {"z_mse": z_mse, "p_mse": p_mse, "total": z_mse + p_mse}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", required=True, type=pathlib.Path,
                      help="existing action_contrast directory with real states.npz files")
  parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
  parser.add_argument("--out", required=True, type=pathlib.Path)
  args = parser.parse_args()
  if args.source.resolve() == args.out.resolve():
    parser.error("--out must differ from --source to preserve the real-branch baseline")
  reports = sorted(args.source.glob("*/results.json"))
  if not reports:
    parser.error("--source has no */results.json files")
  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
  first_report = json.loads(reports[0].read_text())
  source_checkpoint = torch.load(first_report["checkpoint"], map_location="cpu",
                                 weights_only=True)
  if any(checkpoint[key] != source_checkpoint[key]
         for key in ("encoder_model", "pooling", "camera", "clip_frames")):
    parser.error("new checkpoint uses a different V-JEPA representation than saved branches")
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = model_from_checkpoint(checkpoint).to(device).eval()
  args.out.mkdir(parents=True, exist_ok=True)
  effect_errors = []
  branch_errors = []
  visual_errors = []
  for report_path in reports:
    old = json.loads(report_path.read_text())
    if old["checkpoint"] != first_report["checkpoint"]:
      parser.error(f"source reports used different checkpoints: {report_path}")
    manifest = json.loads(pathlib.Path(old["manifest"]).read_text())
    if (checkpoint["proprio_columns"] != manifest["proprio_columns"]
        or checkpoint["action_columns"] != manifest["action_columns"]
        or checkpoint["camera"] != manifest["camera"]
        or checkpoint["clip_frames"] != manifest["clip_frames"]):
      parser.error(f"checkpoint differs from source observation contract: {report_path}")
    with np.load(report_path.parent / "states.npz") as states:
      z, p = states["source_z"], states["source_p"]
      if (len(z) != checkpoint["z_dim"] or len(p) != checkpoint["p_dim"]):
        parser.error(f"checkpoint input size differs from {report_path}")
      predicted = {}
      actual = {}
      for name in ("plus_x", "minus_x"):
        action = old["actions"][name]
        if len(action) != checkpoint["a_dim"]:
          parser.error(f"checkpoint action size differs from {report_path}")
        actual[name] = (states[f"{name}_actual_z"], states[f"{name}_actual_p"])
        predicted[name] = _predict(model, checkpoint, z, p, action, device)
    matched = {name: _errors(*predicted[name], *actual[name], checkpoint)
               for name in predicted}
    wrong = {"plus_x": _errors(*predicted["minus_x"], *actual["plus_x"], checkpoint),
             "minus_x": _errors(*predicted["plus_x"], *actual["minus_x"], checkpoint)}
    moves = {name: {"actual": float(100 * (actual[name][1][14] - p[14])),
                    "predicted": float(100 * (predicted[name][1][14] - p[14]))}
             for name in predicted}
    real_effect = float(100 * (actual["plus_x"][1][14] - actual["minus_x"][1][14]))
    if abs(real_effect - old["branch_difference"]["actual_ee_x_cm"]) > 1e-4:
      parser.error(f"saved real branch arrays differ from report: {report_path}")
    predicted_effect = float(100 * (predicted["plus_x"][1][14]
                                    - predicted["minus_x"][1][14]))
    effect_errors.append(abs(predicted_effect - real_effect))
    branch_errors.extend(abs(moves[name]["predicted"] - moves[name]["actual"])
                         for name in predicted)
    visual_errors.extend(matched[name]["z_mse"] for name in predicted)
    report = dict(old)
    report.update(checkpoint=str(args.checkpoint.resolve()),
                  ee_x_cm_from_source=moves,
                  branch_difference={"actual_ee_x_cm": real_effect,
                                     "predicted_ee_x_cm": predicted_effect},
                  matched_action_error=matched, wrong_action_error=wrong)
    dest = args.out / report_path.parent.name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report_path.parent.name}: real x effect={real_effect:+.3f} cm "
          f"predicted={predicted_effect:+.3f} cm")
  print(f"{len(reports)} held-out states; mean effect MAE={np.mean(effect_errors):.3f} cm; "
        f"branch x MAE={np.mean(branch_errors):.3f} cm; "
        f"matched visual z MSE={np.mean(visual_errors):.4f}")
  print(f"saved reports to {args.out}")


if __name__ == "__main__":
  main()
