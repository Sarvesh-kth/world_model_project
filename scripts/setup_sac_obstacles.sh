#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONPATH
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
old_repo="${1:?Pass the existing world_model_project directory}"
cd "$repo_dir"
git lfs install --local
python_base="$old_repo/.venv/bin/python"
if [[ ! -x "$python_base" ]]; then
  python_base="$(command -v python3)"
fi
if [[ ! -x .venv/bin/python ]]; then
  "$python_base" -m venv .venv
fi
# Repair the earlier setup too: base Conda TensorFlow is incompatible with the project's NumPy 2.
"$python_base" - <<'PY'
import pathlib
cfg = pathlib.Path(".venv/pyvenv.cfg")
lines = cfg.read_text().splitlines()
cfg.write_text("\n".join("include-system-site-packages = false" if line.startswith("include-system-site-packages")
                        else line for line in lines) + "\n")
PY
# Reuse only the old virtualenv's CUDA packages, not its inherited base Conda packages.
"$python_base" - <<'PY' > .venv/source-site-packages.txt
import sys, sysconfig
if sys.prefix != sys.base_prefix:
    print(sysconfig.get_path("purelib"))
PY
.venv/bin/python - <<'PY'
import pathlib, site
site_dir = pathlib.Path(site.getsitepackages()[0])
source_paths = pathlib.Path(".venv/source-site-packages.txt").read_text()
site_dir.joinpath("existing_cluster_runtime.pth").write_text(source_paths)
# Adding a package directory does not execute its .pth hooks. Restore only our known renderer hook.
if any(pathlib.Path(p, "_world_model_mujoco_osmesa_bootstrap.py").is_file() for p in source_paths.splitlines() if p):
    site_dir.joinpath("zz_cluster_mujoco_osmesa.pth").write_text("import _world_model_mujoco_osmesa_bootstrap\n")
    print("Reusing the original project's OSMesa startup hook")
PY
if ! .venv/bin/python -c 'import torch, torchvision; assert torch.cuda.is_available()'; then
  .venv/bin/python -m pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
fi
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
.venv/bin/python - <<'PY'
import os, pathlib, sys
base_site = pathlib.Path(sys.base_prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
assert all(pathlib.Path(p).resolve() != base_site.resolve() for p in sys.path if p), "Base Conda packages leaked into this environment"
sys.path.insert(0, str(pathlib.Path("simulation").resolve()))
import offscreen
import torch, torchvision, stable_baselines3, mujoco
from transformers import AutoModel, AutoVideoProcessor
assert stable_baselines3.__version__ == "2.7.1"
assert torch.cuda.is_available(), "No working CUDA; check the cluster GPU allocation"
print("Runtime:", torch.__version__, torchvision.__version__, stable_baselines3.__version__, mujoco.__version__)
print("GPU:", torch.cuda.get_device_name())
model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with mujoco.Renderer(model, height=32, width=32) as renderer:
    renderer.update_scene(data)
    frame = renderer.render()
    assert frame.shape == (32, 32, 3), frame.shape
print("Rendering OK:", os.environ.get("MUJOCO_GL"), frame.shape)
PY
.venv/bin/python scripts/prepare_sac_obstacles.py --existing-root "$old_repo" "${@:2}"
printf '\nSetup finished. The old checkout and its Python packages were not edited.\n'
