Chess DNN — Overview

Purpose
- Deep neural network analyses comparing model responses to human/brain data.

Inputs
- Model response CSVs under `data/` and any additional resources referenced by scripts.

How To Run
- Run Python scripts in this folder. General workflow:
  1) Configure inputs/outputs (CLI flags)
  2) Load datasets and model responses
  3) Compute metrics/visualisations
  4) Save under `results/<YYYYMMDD-HHMMSS>_dnn_<shortname>/`

Results
- Timestamped subfolders under `results/` contain figures and tables.

Reuse
- Implement reusable logic in `modules/` and call it from entry scripts.

