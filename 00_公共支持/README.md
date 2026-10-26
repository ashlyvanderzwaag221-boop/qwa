# Shared support

`project_paths.py` defines repository-relative paths; `pipeline.py` provides common labels, feature configuration, MFCC extraction and legacy helpers. The current training entry is `../02_模型训练/experiment.py`.

Use `requirements-lock.txt` for the recorded Python 3.12 environment. `requirements-report.txt` adds PDF generation dependencies. `runtime.ps1` prepares the local environment for the three PowerShell stage wrappers.

See the [root README](../README.md) for installation and reproducible commands. Environments, caches and raw audio are generated locally.
