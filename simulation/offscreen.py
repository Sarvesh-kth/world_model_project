import os
import sys

# Import this before mujoco in every script that renders the camera without a window (collectors, encoder,
# SAC training, controllers). It selects EGL for offscreen rendering on Linux, where MuJoCo reads MUJOCO_GL
# when it is first imported. A shell profile that pins the NVIDIA EGL vendor for other tools
# (__EGL_VENDOR_LIBRARY_FILENAMES) breaks headless EGL whenever the driver and its libraries are out of
# sync, so that pin is dropped for this process and EGL picks a vendor that works. The viewer scripts do
# not import this: a window needs GLFW, not EGL.
if sys.platform.startswith("linux"):
  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
  os.environ.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)
