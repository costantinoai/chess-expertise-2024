Manifold (PR) — Overview

Purpose
- Participation Ratio (PR) analyses for the manuscript (main and supplementary).

How To Run (from IDE)
- participation_ratio.py: computes PR per ROI, tests group differences (Welch + FDR), saves consolidated CSV and logs.
- plot_pr_roi_size.py: plots PR vs ROI size and a simple 2D PCA classification figure.

Configuration
- Paths and subjects are defined centrally in `config.py` (GLM_BASE_PATH, ATLAS_CORTICES, EXPERTS/NONEXPERTS).
- Figures and logs are written under `manifold/results/<YYYYMMDD-HHMMSS>_*`.

Style
- Plot styles and ROI meta-data are centralized in `meta.py` and `viz_utils.py`.

