import os
import subprocess
import sys

# A MuJoCo window needs an OpenGL context through GLX, not the EGL the camera renders with. On a hybrid
# laptop the NVIDIA GLX stops working whenever the driver and its libraries disagree (a driver update
# without a reboot), while Mesa still opens a window on the other GPU. glx_environment() tries the
# default vendor and then Mesa in a throwaway process and returns the environment that worked, without
# EGL or the NVIDIA PRIME offload variables (they leave the window blank), or None when no window can
# be created at all.

PROBE = "import glfw, sys; sys.exit(0 if glfw.init() and glfw.create_window(8, 8, 'probe', None, None) else 1)"
DROP = ("MUJOCO_GL", "PYOPENGL_PLATFORM", "__NV_PRIME_RENDER_OFFLOAD", "__GLX_VENDOR_LIBRARY_NAME",
        "__EGL_VENDOR_LIBRARY_FILENAMES")


def glx_environment():
  if not os.environ.get("DISPLAY"):
    return None
  base = {k: v for k, v in os.environ.items() if k not in DROP}
  for vendor in (None, "mesa"):
    env = dict(base)
    if vendor:
      env["__GLX_VENDOR_LIBRARY_NAME"] = vendor
    if subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True).returncode == 0:
      return env
  return None


# Make this process able to open a window: apply the working environment, or say why there is none
def use_window_environment():
  env = glx_environment()
  if env is None:
    print("no OpenGL window can be created on this display (see the README on the rendering backend)", flush=True)
    return False
  for k in DROP:
    os.environ.pop(k, None)
  os.environ.update(env)
  return True
