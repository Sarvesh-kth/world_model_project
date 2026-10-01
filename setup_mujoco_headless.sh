#!/usr/bin/env bash
# Repo-local, rootless MuJoCo headless rendering bootstrap for Ubuntu/Debian.
# Adapted from the course Lab 1 setup_mujoco_headless.sh for this repository.
#
# What it does:
#   1. Downloads libosmesa6 + any missing runtime dependencies with APT.
#      APT is used ONLY as a downloader; nothing is installed system-wide.
#   2. Extracts the .deb files under this repository.
#   3. Installs a tiny .pth bootstrap into the ACTIVE Python virtualenv.
#      On Python startup it selects MUJOCO_GL=osmesa and preloads the local
#      OSMesa shared library, so no LD_LIBRARY_PATH or shell activation hack
#      is required.
#
# Usage:
#   source .venv/bin/activate
#   ./setup_mujoco_headless.sh
#
# Re-running is safe. Delete .mujoco-osmesa and rerun to refresh it.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${MUJOCO_OSMESA_PREFIX:-$SCRIPT_DIR/.mujoco-osmesa}"
APTROOT="$PREFIX/apt"
ROOTFS="$PREFIX/root"
MANIFEST="$PREFIX/preload.txt"
PYTHON_BIN="${PYTHON_BIN:-python}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python not found"
command -v apt-get >/dev/null 2>&1 || die "apt-get not found (this script targets Debian/Ubuntu images)"
command -v dpkg-deb >/dev/null 2>&1 || die "dpkg-deb not found"
command -v ldd >/dev/null 2>&1 || die "ldd not found"

# Deliberately require a venv so we do not modify the user's global Python.
"$PYTHON_BIN" - <<'PY'
import sys
if sys.prefix == sys.base_prefix:
    raise SystemExit(
        "ERROR: activate the project virtualenv first, e.g. "
        "`source .venv/bin/activate`, then rerun this script."
    )
PY

printf 'MuJoCo headless setup\n'
printf '  Python: %s\n' "$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
printf '  Prefix: %s\n' "$PREFIX"

mkdir -p "$PREFIX" "$APTROOT/lists/partial" "$APTROOT/archives/partial" "$ROOTFS"

# If OSMesa is not cached yet, obtain it without sudo.
OSMESA_REAL="$(find "$ROOTFS" -type f -name 'libOSMesa.so.*' 2>/dev/null | head -n 1 || true)"

if [[ -z "$OSMESA_REAL" ]]; then
  printf '\nDownloading OSMesa runtime locally (no sudo, no system install)...\n'

  # A private APT state lets apt-get update/download run as an unprivileged user.
  # We still read /var/lib/dpkg/status so APT only downloads dependencies that
  # are actually missing from the container.
  APT_OPTS=(
    -o "Debug::NoLocking=1"
    -o "APT::Sandbox::User=$(id -un)"
    -o "Dir::State::status=/var/lib/dpkg/status"
    -o "Dir::State::lists=$APTROOT/lists"
    -o "Dir::Cache::archives=$APTROOT/archives"
    -o "APT::Get::List-Cleanup=0"
  )

  # CUDA/Jupyter images often delete /var/lib/apt/lists to save space, so keep
  # our own package indexes under the repo instead.
  apt-get "${APT_OPTS[@]}" update

  # -d / --download-only means dpkg is never invoked to install anything.
  apt-get "${APT_OPTS[@]}" \
    -y -d --no-install-recommends \
    install libosmesa6

  shopt -s nullglob
  DEBS=("$APTROOT"/archives/*.deb)
  shopt -u nullglob
  (( ${#DEBS[@]} > 0 )) || die "APT downloaded no .deb files"

  printf '\nExtracting %d package(s) under %s ...\n' "${#DEBS[@]}" "$ROOTFS"
  for deb in "${DEBS[@]}"; do
    printf '  %s\n' "$(basename "$deb")"
    dpkg-deb -x "$deb" "$ROOTFS"
  done

  OSMESA_REAL="$(find "$ROOTFS" -type f -name 'libOSMesa.so.*' | head -n 1 || true)"
  [[ -n "$OSMESA_REAL" ]] || die "libOSMesa was not found after extraction"
else
  printf '\nUsing cached OSMesa: %s\n' "$OSMESA_REAL"
fi

OSMESA_DIR="$(dirname "$OSMESA_REAL")"
# PyOpenGL first looks for an unversioned libOSMesa.so on some versions.
ln -sfn "$(basename "$OSMESA_REAL")" "$OSMESA_DIR/libOSMesa.so"

# Build a loader path only for diagnostics and for computing the exact set of
# locally-extracted transitive dependencies. We will NOT require this variable
# at runtime.
mapfile -t LOCAL_LIB_DIRS < <(
  find "$ROOTFS" \( -type f -o -type l \) -name '*.so*' -printf '%h\n' |
  sort -u
)
(( ${#LOCAL_LIB_DIRS[@]} > 0 )) || die "no shared-library directories found"

LOCAL_LD_PATH="$(IFS=:; printf '%s' "${LOCAL_LIB_DIRS[*]}")"

printf '\nChecking OSMesa dependencies...\n'
LDD_OUTPUT="$(
  LD_LIBRARY_PATH="$LOCAL_LD_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    ldd "$OSMESA_REAL"
)"
printf '%s\n' "$LDD_OUTPUT"

if grep -q 'not found' <<<"$LDD_OUTPUT"; then
  printf '\n%s\n' "$LDD_OUTPUT" >&2
  die "one or more OSMesa dependencies are still missing"
fi

# Store only dependencies that came from our private extraction tree.
{
  awk -v root="$ROOTFS/" '
    {
      for (i = 1; i <= NF; i++) {
        if (index($i, root) == 1) print $i
      }
    }
  ' <<<"$LDD_OUTPUT" | sort -u
  printf '%s\n' "$OSMESA_REAL"
} | awk '!seen[$0]++' > "$MANIFEST"

printf '\nInstalling startup bootstrap into the active virtualenv...\n'

MUJOCO_OSMESA_ROOT="$ROOTFS" \
MUJOCO_OSMESA_MANIFEST="$MANIFEST" \
"$PYTHON_BIN" - <<'PY'
import os
import site
from pathlib import Path

root = Path(os.environ["MUJOCO_OSMESA_ROOT"]).resolve()
manifest = Path(os.environ["MUJOCO_OSMESA_MANIFEST"]).resolve()

site_dirs = [Path(p) for p in site.getsitepackages()]
if not site_dirs:
    raise SystemExit("Could not locate site-packages for this virtualenv.")
site_dir = site_dirs[0]
site_dir.mkdir(parents=True, exist_ok=True)

module = site_dir / "_world_model_mujoco_osmesa_bootstrap.py"
pth = site_dir / "_world_model_mujoco_osmesa_bootstrap.pth"

module.write_text(
    f'''# Auto-generated by setup_mujoco_headless.sh. Do not edit.
import ctypes
import os
from pathlib import Path

_ROOT = Path({str(root)!r})
_MANIFEST = Path({str(manifest)!r})

# Respect an explicit user choice, otherwise make headless MuJoCo work.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

if os.environ.get("MUJOCO_GL", "").lower() == "osmesa":
    paths = []
    if _MANIFEST.exists():
        paths = [
            Path(line.strip())
            for line in _MANIFEST.read_text().splitlines()
            if line.strip()
        ]

    # Some extracted libraries can depend on other extracted libraries.
    # Repeated passes allow the leaves to load first.
    pending = paths[:]
    for _ in range(max(2, len(pending) + 1)):
        if not pending:
            break
        next_pending = []
        progressed = False
        for lib in pending:
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                progressed = True
            except OSError:
                next_pending.append(lib)
        pending = next_pending
        if not progressed:
            break
''',
    encoding="utf-8",
)
pth.write_text("import _world_model_mujoco_osmesa_bootstrap\n", encoding="utf-8")

print(f"  wrote {pth}")
print(f"  wrote {module}")
PY

cat <<EOF

Done.

This virtualenv now defaults to:
  MUJOCO_GL=osmesa
  PYOPENGL_PLATFORM=osmesa

No conda, sudo, X11, DISPLAY, or persistent LD_LIBRARY_PATH is required.

If a Jupyter kernel using this virtualenv was already running, restart that
kernel once so Python startup processes the new .pth file.
EOF
