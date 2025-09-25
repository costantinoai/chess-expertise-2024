Chess Behavioural — Overview

Purpose
- Analyse behavioural data collected alongside fMRI to compare experts vs novices.

Inputs
- CSVs in `data/` (e.g., `dataset.csv`, `participants.xlsx`).
- Any additional task‑specific inputs referenced by scripts.

How To Run
- Preferred entry points (run from repo root):
  - `run_behavioural_main.py` – main behavioural RDM analysis
  - `run_behavioural_intercorr.py` – between-group/inter-correlation
  - `run_behavioural_splithalf.py` – split-half reliability
- You can also run the scripts in this folder. Typical pattern:
  - Setup logging and run ID
  - Load datasets via `modules/helpers.py` utilities
  - Compute statistics and generate figures
  - Save outputs under `results/<YYYYMMDD-HHMMSS>_behavioural/`
 - Prefer shared utilities for plotting/statistics when possible:
   - Stats: `common/stats_utils.py`
   - Bar plots: `common/plotting_utils.py`

Results
- Figures/tables are written under `results/`. Each producing script must create a timestamped subfolder named `<YYYYMMDD-HHMMSS>_behavioural_<shortname>`.

Reuse
- Shared utilities live in `modules/` (e.g., logging, data loading, CI computation). Prefer importing these helpers over duplicating logic.
