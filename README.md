# chess-expertise-2024

Analysis code for the Chess Expertise study. The repository contains multiple analysis folders (Python and MATLAB). Each folder is a main analysis; modules provide reusable functions, and main scripts orchestrate workflows to produce reported results and figures.

Analyses Overview
- One folder per analysis (e.g., `mvpa/`, `fmri_glm/`, `behavioural/`, `neurosynth/`, `manifold/`).
- For a high‑level inventory of analyses and reported figures, see `manuscript/main.tex` and `manuscript/supplementary.tex`.

Repository Structure
- `fmri_glm/` – MATLAB utilities and scripts for first/second level GLM.
- `mvpa/` – Python MVPA and RSA code with plotting utilities.
- `behavioural/` – Behavioural analysis scripts and helper modules.
- `neurosynth/` – Scripts for meta‑analytic lookups using Neurosynth.
- `manifold/` – Manifold/PR analysis workflows and outputs.
- `rois/` – ROI definitions and utilities.
- `common_utils.py` – Shared utilities (logging tee, seeding, run IDs, script copy).
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
- GLM example: `fmri_glm/run_subject_glm.m`
- MVPA/RSA examples: Python scripts in `mvpa/` (see module docstrings and `API.md`).
- Behavioural examples: scripts in `behavioural/` using `modules/` helpers.
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

Maintenance notes
- Legacy and machine‑specific folders (`old/`, `misc/`, `local/`) have been removed to keep the repo focused on manuscript analyses. Dataset visualisation scripts were removed; see `manuscript/write/figures/` for figures and `docs/ANALYSES.md` for the inventory.

License
- MIT License (see `LICENSE`).
