#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
old_repo="${1:?Pass the existing world_model_project directory}"
cd "$repo_dir"
git lfs install --local
python_base="$old_repo/.venv/bin/python"
if [[ ! -x "$python_base" ]]; then
  python_base="$(command -v python3)"
fi
if [[ ! -x .venv/bin/python ]]; then
  "$python_base" -m venv --system-site-packages .venv
  # Reuse the existing CUDA wheels without editing the old virtual environment.
  "$python_base" - <<'PY' > .venv/source-site-packages.txt
import site
print("\n".join(site.getsitepackages()))
PY
  .venv/bin/python - <<'PY'
import pathlib, site
pathlib.Path(site.getsitepackages()[0], "existing_cluster_runtime.pth").write_text(
    pathlib.Path(".venv/source-site-packages.txt").read_text())
PY
fi
if ! .venv/bin/python -c 'import torch, torchvision; assert torch.cuda.is_available()'; then
  .venv/bin/python -m pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
fi
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
.venv/bin/python - <<'PY'
import torch, torchvision, stable_baselines3, mujoco
from transformers import AutoModel, AutoVideoProcessor
assert stable_baselines3.__version__ == "2.7.1"
assert torch.cuda.is_available(), "No working CUDA; check the cluster GPU allocation"
print("Runtime:", torch.__version__, torchvision.__version__, stable_baselines3.__version__, mujoco.__version__)
print("GPU:", torch.cuda.get_device_name())
PY
.venv/bin/python scripts/prepare_sac_obstacles.py --existing-root "$old_repo"
printf '\nSetup finished. The old checkout and its Python packages were not edited.\n'
