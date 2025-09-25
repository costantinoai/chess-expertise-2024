GLM — Overview

Purpose
- First- and second-level GLM analysis (MATLAB) and Python plotting/summaries.

Inputs
- Preprocessed data and first-level SPM outputs under `data/BIDS/derivatives/` (see `config.m`/script headers).
- ROI metadata via TSV under `rois/sets/<set>/region_info.tsv` (loaded in Python helpers).

Entry Points
- MATLAB: `matlab_helpers/run_subject_glm.m`, `TwoSample_SecondLevel_MultiContrast.m`.
- Python (run from IDE):
  - `glm/roi_ttests_univ.py` — Univariate second-level ROI summaries
  - `glm/roi_ttests_rsa.py`  — RSA searchlight ROI summaries
  - `glm/roi_ttests_all.py`  — Deprecated aggregator calling both (kept for backward compatibility)

Shared Utilities
- Plotting is centralized:
  - Brain maps and surface grids: `common/brain_plotting.py`, `common/surface_plotting.py`.
  - Bar/summary plots: `common/plotting_utils.py`.
- ROI metadata loader: `rois/meta.py` (no ROIManager; use TSV+NIfTI labels).
- Logging/run utils: `common/logging_utils.py`, `common/common_utils.py`.

Outputs
- All figures/tables saved under `glm/results/<YYYYMMDD-HHMMSS>_<shortname>/`.
- Scripts write a log file in the same folder and copy the entry script for provenance.

Notes
- Avoid absolute paths; pass inputs/outputs via parameters or small configs.
- Ensure reproducibility by setting seeds (where applicable) and logging key parameters.
