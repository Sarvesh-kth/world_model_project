import os
import sys

# Import this before mujoco in every script that renders the camera without a window. It selects EGL on
# Linux and drops the shell's pinned EGL vendor, which breaks headless rendering when the driver and its
# libraries are out of sync. The viewer scripts do not import this, a window needs GLFW.
if sys.platform.startswith("linux"):
  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
  os.environ.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)
