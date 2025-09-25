Chess Neurosynth — Overview

Purpose
- Meta-analytic lookups and term-based analyses using Neurosynth or related tools.

Inputs
- Group maps under `data/`:
  - RSA searchlight: `data/BIDS/derivatives/rsa_searchlight/<model>/`
  - Univariate GLM: `data/BIDS/derivatives/fmriprep-SPM_*/GLM/2ndLevel_*`

How To Run
- Entry scripts:
  - `rsa_neurosynth.py` — RSA-based term associations on group maps.
  - `univariate_neurosynth.py` — Univariate term associations on group maps.
- Save outputs under `results/<YYYYMMDD-HHMMSS>_neurosynth_<shortname>/`.
 - Plotting is centralized:
   - Brain maps: `common/brain_plotting.py`
   - Correlations/differences: `common/stats_plotting.py`

Running
- Preferred entry points (run from repo root):
  - `run_neurosynth_univ.py` – univariate Neurosynth correlation analysis
  - `run_neurosynth_rsa.py` – RSA→Neurosynth term correlation analysis

Results
- Figures/tables are written to timestamped subfolders under `results/`.

Reuse
- Factor out shared helpers and IO into `modules/` if multiple scripts share logic.
