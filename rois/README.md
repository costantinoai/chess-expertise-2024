Chess ROIs — Overview
ROIs — Overview

Purpose
- ROI definitions and helper scripts. Includes generated ROI NIfTIs and TSVs with region info.

Layout
- `rois/data/` — reference resources (e.g., HCP-MMP1 CSV/TSV, template NIfTIs)
- `rois/results/<roi_set>/` — generated ROI NIfTIs and `region_info.tsv`

Usage
- Python scripts expect ROI annotations under `rois/results/<roi_set>/region_info.tsv` and the corresponding `.nii` in the same folder.
- MATLAB scripts that build ROIs should use `fullfile` to read/write within this folder.
Purpose
- ROI definitions, lookup tables, and ROI-specific utilities for analyses.

Contents
- ROI atlases/tables and any helpers to map ROI IDs to names/hemispheres/groups.

How To Use
- Import ROI helpers from `modules/` (if present) in other analyses.
- Scripts that derive ROI artefacts should save to `results/<YYYYMMDD-HHMMSS>_rois_<shortname>/`.

Results
- Timestamped subfolders with generated ROI artefacts.
