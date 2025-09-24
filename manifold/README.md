Chess Manifold MFT — Overview

Purpose
- Manifold analyses (e.g., geometry/metrics) of stimulus representations.

Inputs
- Analysis-ready embeddings or metrics; see scripts for exact inputs.

How To Run
- Example entry point: `python main.py` (check script args with `-h`).
- Workflow pattern:
  1) Set output directory with a run ID
  2) Load inputs
  3) Compute manifold metrics/plots
  4) Save under `manifold_results/<YYYYMMDD-HHMMSS>_manifoldmft_<shortname>/` or `results/`

Results
- Plots and tables saved under the analysis results folder with timestamped names.

Reuse
- Move shared utilities into a `modules/` subfolder for reuse.

