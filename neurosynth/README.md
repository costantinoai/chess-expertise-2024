Chess Neurosynth — Overview

Purpose
- Meta-analytic lookups and term-based analyses using Neurosynth or related tools.

Inputs
- Terms/seed ROIs and any required meta-analytic resources.

How To Run
- Entry scripts:
  - `rsa_neurosynth.py` — RSA-based term associations on group maps.
  - `univariate_neurosynth.py` — Univariate term associations on group maps.
- Save outputs under `results/<YYYYMMDD-HHMMSS>_neurosynth_<shortname>/`.

Results
- Figures/tables are written to timestamped subfolders under `results/`.

Reuse
- Factor out shared helpers and IO into `modules/` if multiple scripts share logic.
