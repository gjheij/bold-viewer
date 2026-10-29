# Packaging validation

Validated locally on Linux with Python 3.12:

- Built the source distribution and a wheel from that source distribution.
- Installed the built wheel into an isolated Python environment.
- Passed all four tests against the installed package:
  - Both installed launch commands display help.
  - A NIfTI image with spaces in its filename and an overlay load in the GUI.
  - A synthetic first-level FEAT design produces the expected prediction.
  - A single-column FSL design retains its matrix dimensions.
- GUI checks used Qt's offscreen platform.
- Packaged viewer.py matches the supplied call_viewer script byte for byte.

Windows, Python 3.10, and interactive desktop behavior have not been tested
locally. The included GitHub Actions workflow checks Windows and Linux with
Python 3.10 and 3.12 after the repository is uploaded.

This archive prepares the repository; it does not create or publish a GitHub
repository or a PyPI release. See README.md for upload instructions.
