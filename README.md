# chess-expertise-2024

Analysis code for the Chess Expertise study. The repository contains multiple analysis folders (Python and MATLAB). Each folder is a main analysis; modules provide reusable functions, and main scripts orchestrate workflows to produce reported results and figures.

Analyses Overview
- One folder per analysis (e.g., `mvpa/`, `glm/`, `behavioural/`, `neurosynth/`, `manifold/`).
- For a high‑level inventory of analyses and reported figures, see `docs/ANALYSES.md` (inventory) and `docs/manuscript/README.md`.

Repository Structure
- `glm/` – MATLAB utilities and scripts for first/second level GLM; Python summaries.
- `mvpa/` – Python MVPA and RSA code with plotting utilities.
- `behavioural/` – Behavioural analysis scripts and helper modules.
- `neurosynth/` – Scripts for meta‑analytic lookups using Neurosynth.
- `manifold/` – Manifold/PR analysis workflows and outputs.
- `rois/` – ROI definitions and utilities.
- Root runners (IDE-friendly): `run_glm_univ.py`, `run_glm_rsa.py`, `run_neurosynth_univ.py`, `run_neurosynth_rsa.py`, `run_behavioural_main.py`, `run_behavioural_intercorr.py`, `run_behavioural_splithalf.py`, `run_mvpa_ttests.py`, `run_pr_participation_ratio.py`.
- `common/common_utils.py` – Shared utilities (logging tee, seeding, run IDs, script copy).
- `common/logging_utils.py` – Console and file logging setup.
 - `common/stats_utils.py` – Statistical helpers (t-tests, CIs, FDR, correlations).
 - `common/stats_plotting.py` – Shared correlation/difference plotting for Neurosynth.
 - `common/brain_plotting.py` – Shared glass/surface map plotting helpers.
 - `common/plotting_utils.py` – Shared bar plotting (MVPA group plots, etc.).
 - `common/surface_plotting.py` – Shared fsaverage overlays (lateral/medial/ventral + dorsal grid) with contours.
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
- Place data under a single `data/` folder at the repository root for clarity:
  - `data/BIDS/` — raw BIDS dataset
  - `data/BIDS/derivatives/` — outputs from fMRIPrep, FastSurfer, GLM, RSA searchlight, MVPA
  - `data/misc/` — templates/LUTs (e.g., Glasser atlases, color tables)
  - `data/temp/` — temporary folders used by some scripts (created if missing)

Paths
- Python scripts read paths from `config.py` (overridable via env vars). MATLAB scripts define a small set of variables at the top and use `fullfile`/`filesep` to avoid OS-specific separators.
- Avoid hard‑coded absolute paths in code; pass paths via CLI flags or configs.
- ROI metadata is centralized under `rois/sets/<set>/region_info.tsv` and loaded via `rois/meta.py`.

Running Analyses
- Use the root runner scripts from your IDE (no CLI needed). Configure inputs via `config.py` or environment variables.
- You can also run analysis-local scripts inside each analysis folder if you prefer.
- GLM example: `glm/matlab_helpers/run_subject_glm.m`
- MVPA/RSA examples: Python scripts in `mvpa/` (see module docstrings and `API.md`). MVPA plotting uses shared utilities in `common/plotting_utils.py` and `common/surface_plotting.py`.
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
ROI Metadata and Overlays
- ROIManager is deprecated. Use NIfTI label images plus TSV metadata under `rois/sets/*`.
- Use `rois/meta.get_roi_info(set_name)` to access names/colors/fs_name/cortex and `rois/meta.validate_roi_tsv(...)` to check TSVs.
- Surface overlays and contours are produced via `common/surface_plotting.plot_left_fsaverage_grid`.
- See `AGENTS.md` for the full contributor guide and expectations.

API Reference
- A consolidated overview of reusable functions is in `API.md`.

Maintenance notes
- Legacy and machine‑specific folders (`old/`, `misc/`, `local/`) have been removed to keep the repo focused on manuscript analyses. See `docs/ANALYSES.md` for the up-to-date inventory of analyses and figures.

License
- MIT License (see `LICENSE`).
