Chess Behavioural — Overview

Purpose
- Analyse behavioural data collected alongside fMRI to compare experts vs novices.

Inputs
- CSVs in `data/` (e.g., `dataset.csv`, `participants.xlsx`).
- Any additional task‑specific inputs referenced by scripts.

How To Run
- Use the Python scripts in this folder to reproduce behavioural results and figures. Typical pattern:
  - Setup logging and run ID
  - Load datasets via `modules/helpers.py` utilities
  - Compute statistics and generate figures
  - Save outputs under `results/<YYYYMMDD-HHMMSS>_behavioural/`

Results
- Figures/tables are written under `results/`. Each producing script must create a timestamped subfolder named `<YYYYMMDD-HHMMSS>_behavioural_<shortname>`.

Reuse
- Shared utilities live in `modules/` (e.g., logging, data loading, CI computation). Prefer importing these helpers over duplicating logic.

