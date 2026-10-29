# BOLD / FEAT Viewer

A desktop viewer for 3D/4D NIfTI images, overlay maps and FSL FEAT results.
Python 3.10 or newer is required. Installation automatically installs PySide6,
pyqtgraph, NiBabel and NumPy and creates the `call_viewer` command.
FSL and fmriproc are not required to view existing results.

## Install locally

Extract this repository and open a terminal in its top-level directory.

### Windows / PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\call_viewer.exe
```

No PowerShell activation-policy changes are needed. To pass an image:

```powershell
.\.venv\Scripts\call_viewer.exe "C:\data\analysis.feat"
```

### Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/call_viewer "/data/analysis.feat"
```

Linux requires a graphical desktop or display forwarding and Qt's system
libraries. On a minimal installation, Qt may report a missing platform plugin
or shared library; install the relevant runtime packages for your distribution.
The offscreen setting used in CI is for automated checks, not interactive use.

## Install from GitHub

After uploading this repository to `gjheij/bold-viewer`, install it into your
active Python environment with:

```bash
python -m pip install "git+https://github.com/gjheij/bold-viewer.git"
call_viewer
```

That URL is the proposed repository location; change it if you use another name.
Git must be installed for this form. A source archive installation avoids Git:

```bash
python -m pip install "https://github.com/gjheij/bold-viewer/archive/refs/heads/main.zip"
```

For an isolated desktop-tool installation, if pipx is already installed:

```bash
pipx install "git+https://github.com/gjheij/bold-viewer.git"
```

Use pipx's PATH setup instructions if your shell cannot find the command.

## Usage

```bash
call_viewer
call_viewer "image.nii.gz"
call_viewer "analysis.feat"
call_viewer "image.nii.gz" -o "zstat1.nii.gz" -o "mask.nii.gz"
python -m bold_viewer --help
```

Paths with spaces should be quoted. The command is the same on Windows and
Linux when the installed environment's scripts directory is on PATH.
Use the module command as a fallback with that environment's Python interpreter.

- Open image or FEAT folders from the toolbar.
- Add multiple overlays and adjust their order, visibility, colors and thresholds.
- Browse voxels and volumes and inspect selected time courses.
- Inspect design columns, contrasts, PE coefficients and COPE partial fits.
- Export displayed voxel curves and coefficients to CSV.

Overlays must share the base image's grid and affine; no registration or
resampling is performed. Model curves require matching first-level FEAT data
and coefficient maps. They are in-sample predictions, not prewhitened report
plots. Voxelwise EV designs disable design-based predictions. COPE partial fits
use normalized absolute contrast weights, not the scalar contrast estimate.
See the module docstring for the complete calculation conventions.

## Development and packaging

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m build
```

Builds produce a wheel and source archive in `dist/`. To test the GUI without
a display, set `QT_QPA_PLATFORM=offscreen` before running the tests.
GitHub Actions builds and installs the wheel, checks both launch commands and
runs the tests on Windows and Linux with Python 3.10 and 3.12. Distribution
artifacts can be downloaded from the workflow run. It does not publish to PyPI.

## Put this project on GitHub

Create an empty `bold-viewer` repository in your GitHub account. From the
extracted project directory, run:

```bash
git init -b main
git add .
git commit -m "Package BOLD and FEAT viewer"
git remote add origin https://github.com/gjheij/bold-viewer.git
git push -u origin main
```

Update the remote if using a different account or repository. GitHub Actions
runs automatically on pushes and pull requests. To release a stable version,
update the version in `pyproject.toml` and create a corresponding Git tag;
users can then install a tag by appending `@v0.1.0` to the Git installation URL.

## License and provenance

The viewer is supplied from the attached `call_viewer` script. No new license
has been assigned by this packaging work. Before public distribution, retain
any original copyright/license notices and add a compatible LICENSE file for
code you own or are authorized to distribute.
