# chess-expertise-2024

Analysis code for the Chess Expertise study. The repository contains multiple analysis folders (Python and MATLAB). Each folder is a main analysis; modules provide reusable functions, and main scripts orchestrate workflows to produce reported results and figures.

Analyses Overview
- One folder per analysis (e.g., `chess-mvpa/`, `chess-glm/`, `chess-behavioural/`, `chess-dataset-vis/`, `chess-neurosynth/`, `chess-manifoldMFT/`).
- For a high‑level inventory of analyses and reported figures, see `manuscript/main.tex` and `manuscript/supplementary.tex`.

Repository Structure
- `chess-glm/` – MATLAB utilities and scripts for first/second level GLM.
- `chess-mvpa/` – Python MVPA and RSA code with plotting utilities.
- `chess-behavioural/` – Behavioural analysis scripts and helper modules.
- `chess-dataset-vis/` – Tools for visualising the chess stimulus set.
- `chess-neurosynth/` – Scripts for meta‑analytic lookups using Neurosynth.
- `chess-manifoldMFT/` – Manifold/PR analysis workflows and outputs.
- `chess-rois/` – ROI definitions and utilities.
- `common_utils.py` – Shared utilities (logging tee, seeding, run IDs, script copy).
- `misc/` – Example JSON sidecars for the BIDS dataset.
- `local/` – Optional local configuration (not tracked by default).
- `tests/` – Tests for shared utilities (if present/extended).

Install
1) Python dependencies
```bash
pip install -r requirements.txt
```
2) Optional editable install for imports
```bash
pip install -e .
```

MATLAB Dependencies
- GLM and MVPA MATLAB scripts require SPM12 and (for MVPA/RSA) CoSMoMVPA. Ensure toolboxes are on the MATLAB path.

Dataset
- The project expects a BIDS‑compliant dataset, typically under `/data/projects/chess/data`:
  - `BIDS/` — raw data
  - `derivatives/` — outputs from fMRIPrep, FastSurfer, etc.
- Avoid hard‑coded absolute paths in code; pass paths via CLI flags or configs.

Running Analyses
- Each analysis folder has one or more entry scripts you can run independently.
  - GLM example: `chess-glm/run_subject_glm.m`
  - MVPA/RSA examples: Python scripts in `chess-mvpa/` (see module docstrings and `API.md`).
  - Behavioural examples: scripts in `chess-behavioural/` using `modules/` helpers.
- Typical script pattern:
  1) Setup logging and a run ID
  2) Load inputs
  3) Compute/fit
  4) Save figures/tables under the analysis `results/` directory

Conventions (Important)
- Reuse over rewrite: shared logic belongs in `modules/` or a shared utility. Do not duplicate code across analyses.
- Modules contain functions; main scripts orchestrate workflows and handle I/O.
- Comment generously for non‑experts: explain what and why, not just how.
- Ensure reproducibility: set seeds, log parameters, and save a copy of the calling script with outputs.
- See `AGENTS.md` for the full contributor guide and expectations.

API Reference
- A consolidated overview of reusable functions is in `API.md`.

License
- MIT License (see `LICENSE`).
