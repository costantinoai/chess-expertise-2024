Chess Dataset Visualisation — Overview

Purpose
- Visualise the chess stimulus set and generate descriptive plots and RDMs.

Inputs
- Stimulus metadata (CSV/Excel) and derived labels as referenced by scripts.

How To Run
- Execute the Python scripts in this folder to generate figures:
  1) Setup output directory with run ID
  2) Load dataset and labels
  3) Generate plots (e.g., stimulus overviews, RDMs)
  4) Save under `results/<YYYYMMDD-HHMMSS>_dataset-vis_<shortname>/`

Results
- All artefacts (PNGs, CSVs) saved under `results/` with timestamped names.

Reuse
- Put shared plotting or loading code in `modules/` and import from scripts.

