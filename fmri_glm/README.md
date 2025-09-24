FMRI GLM — Overview

Purpose
- First- and second-level GLM analyses in SPM (MATLAB) and summary/visualization scripts in Python.

Inputs (under `data/`)
- `data/BIDS/` — raw BIDS dataset
- `data/BIDS/derivatives/fmriprep/` — fMRIPrep outputs
- `rois/` — ROI resources used by Python summaries, if needed

Key scripts
- MATLAB
  - `spm_glm_runallsubjects.m`: runs subject-level GLM for selected tasks; uses `fullfile` and assumes `data/` tree
  - `WithinGroup_SecondLevel_MultiContrast.m`: within-group second-level analysis
  - `TwoSample_SecondLevel_MultiContrast.m`: two-sample group comparison
  - `RSA_TwoSample_SecondLevel_MultiContrast.m`: second-level for RSA searchlight maps
  - `matlab_helpers/glm/*.m`: helper functions (events conversion, confounds, smoothing, etc.)
- Python
  - `roi_ttests_all.py`: summarizes ROI stats and generates surface/glass visualizations and LaTeX tables

Outputs
- `data/BIDS/derivatives/fmriprep-SPM_*` — subject and second-level GLM outputs
- `fmri_glm/results/<YYYYMMDD-HHMMSS>_secondlevel-rois/` — figures/tables from Python summaries

How to run (IDE)
- Open MATLAB, edit the top “Paths” section in `spm_glm_runallsubjects.m` if needed (defaults to `data/`), then run the script.
- For Python summaries, open `roi_ttests_all.py`, ensure `config.py` points to the correct data roots, and run.
