# Install OpenCV for Python with pip {#tutorial_py_pip_install}

This quick-start shows the **recommended** way for most users to get OpenCV in Python: install from
**PyPI** with `pip`. It also explains virtual environments, platform notes, and common troubleshooting.
If you need OS‑specific alternatives (system packages or source builds), see the OS pages linked
below, but those are **not required** for typical Python use.

@note: OpenCV team maintains **PyPI** packages only. Conda distributions and platform specific builds
are community builds and hardware vendor builds and may differ from the official one.

## Quick start

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2) Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel

# 3) Install OpenCV from PyPI (choose ONE)
pip install opencv-python          # main package (most users)
# or
pip install opencv-contrib-python  # + extra modules (contrib)
# or
pip install opencv-python-headless # no GUI/backends (servers/CI)
# or
pip install opencv-contrib-python-headless # no GUI/backends with extra modules (servers/CI)
```

### Tiny hello‑world

```python
import cv2 as cv
import numpy as np

print("OpenCV:", cv.__version__)
img = np.zeros((120, 400, 3), dtype=np.uint8)
cv.putText(img, "OpenCV OK", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
# If you installed a non-headless build, you can display a window:
# cv.imshow("hello", img); cv.waitKey(0)
# Always safe (headless or not): save to file
cv.imwrite("hello.png", img)
```

## Virtual environments and IDEs

Using a virtual environment keeps project dependencies isolated. Tools that create or activate envs include:

- `venv` (built-in) and `virtualenv`
- Conda environments
- IDEs (VS Code, PyCharm) that may **auto-create and auto-activate** an env per workspace

If imports fail inside an IDE, verify the interpreter selected by the IDE matches the environment
where you installed OpenCV.

## OS notes

- **Linux:** Your default Python may be `python3`. Use `python3 -m venv .venv` and `python3 -m pip ...`.
If you cannot use a virtual env, `pip --user` installs to your home directory: `python3 -m pip install --user opencv-python`.
- **Windows:** Install Python from [python.org] or via `winget install Python.Python.3`. Make sure
**“Add python to PATH”** is enabled or use the **“Open in terminal”** from your IDE, which selects
the right interpreter automatically.
- **macOS:** Use the system `python3` or a managed one (Homebrew or Python.org).
Always prefer a virtual environment.
- **Raspberry Pi / ARM boards:** Prebuilt wheels may not exist for some Pi OS / Python combinations.
See **Troubleshooting** below.

## Choosing a PyPI variant

- `opencv-python`: core OpenCV modules with GUI/backends
- `opencv-contrib-python`: includes **contrib** modules in addition to the core
- `opencv-python-headless`: no GUI/backends (ideal for servers/containers/CI)
- `opencv-contrib-python-headless`: contrib + headless

Install exactly **one** of these per environment.

## Troubleshooting

Please start with opencv-python project [README](https://github.com/opencv/opencv-python/blob/4.x/README.md)

**Pip is trying to build from source**
Symptoms: very long build step, CMake errors, compiler errors.
Fixes:
- Upgrade build tooling: `python -m pip install --upgrade pip setuptools wheel`
- Ensure your Python version is supported by the chosen package.
- If you are on an uncommon platform or Python build, switch to a supported Python or try a different
variant (headless vs non‑headless).

**“No matching distribution found” or “Unsupported wheel”**
- Confirm your Python version (e.g., `python -V`). Choose a wheel that supports that version
(manylinux/macOS/Windows wheels on PyPI target specific Python versions).
- Create a fresh virtual environment with a mainstream Python (e.g., 3.10–3.12 for now) and reinstall.

**Raspberry Pi / ARM**
- Wheels may lag behind new Python/Pi OS releases. Try `opencv-python-headless` first. If
unavailable, consider system packages for camera/GUI pieces, or build from source following
the OS page linked below.

**Import works in terminal but fails in IDE**
- The IDE is using a different interpreter. Select the **same** environment inside your
IDE’s interpreter settings.

## What about system packages or building from source?

For beginners using Python, **PyPI is recommended**. Native distribution packages and full source
builds are better suited to advanced users with platform‑specific needs. You can still find them on
the OS‑specific pages, moved under “Alternatives.”

## See also

- @ref tutorial_py_root
- OS pages: @ref tutorial_py_setup_in_windows, @ref tutorial_py_setup_in_ubuntu, @ref tutorial_py_setup_in_fedora
