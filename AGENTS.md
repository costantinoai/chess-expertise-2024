Chess Expertise 2024 — Agents Guide

Scope: This file applies to the entire repository and all subfolders. Follow these practices for any new code or edits.

Goals
- Reuse over rewrite: factor out shared logic into modules and import it where needed.
- Modular, readable, and maintainable code with clear separation between reusable functions (modules) and orchestration (main scripts).
- Reproducible analyses that are easy to re‑run and verify.
- Academic sharing: extensive comments and docstrings explaining both what and why, suitable for non‑expert readers.
 - Every top‑level analysis folder must include a README that lists its analyses, explains how to run them, and how to obtain the figures/tables reported in the paper.
 - Any script that produces artefacts (figures, tables, files) must create an output directory named `<YYYYMMDD-HHMMSS>_<short-analysis-name>` under the analysis’ `results/` (or equivalent) folder.

Repository Layout
- One analysis per top‑level folder (e.g., `mvpa/`, `fmri_glm/`, `behavioural/`, `neurosynth/`, `manifold/`).
- Inside an analysis folder:
  - `modules/`: reusable functions and small utilities.
  - main scripts in the folder root: run the workflow to produce results and figures.
  - `results/` or analysis‑specific output folders contain generated artifacts.
- Global helpers live at the repository root (e.g., `logging_utils.py`, `common_utils.py`, `viz_utils.py`).
- For an overview of all analyses and reported figures, consult `manuscript/main.tex` and `manuscript/supplementary.tex`.
 - Ensure each top‑level folder has a `README.md` with: purpose, required inputs, how to run, expected outputs, and where results are saved.

Design Principles
- DRY (Don’t Repeat Yourself):
  - If code is used by more than one script, move it to a module and import it.
  - Prefer a single source of truth for constants, paths, and plotting styles.
  - If similar helpers exist in multiple analyses, consolidate into a shared module (e.g., a new `common/` package or an existing top‑level utility) and migrate callers incrementally.
- Separation of concerns:
  - Modules: pure(ish) functions with minimal side effects, typed arguments, and documented returns.
  - Main scripts: parse inputs, call module functions, handle I/O, logging, and figure saving.
- Readability and maintainability:
  - Use descriptive names; avoid single‑letter variables.
  - Keep functions focused and short; prefer composition over long procedural blocks.
  - Add docstrings (what and why) and inline comments for non‑obvious steps.
- Reproducibility:
  - Centralize random seeds; expose a `--seed` or config option and set numpy/random seeds once.
  - Every artefact‑producing script must:
    - Create an output directory named `<YYYYMMDD-HHMMSS>_<short-analysis-name>`
    - Configure logging to both console and file (use `logging_utils.setup_logging(log_file=...)` or add a file handler); do not use `print()`.
    - Save a copy of the current script in the output folder (`save_script_to_file(...)`).
  - Log key parameters, versions, and run IDs; write logs to the output directory.
- Config and paths:
  - Do not hard‑code absolute paths. Accept input/output paths via CLI flags or a small config file.
  - Use `Path`/`os.path` joins and validate that inputs exist.
- Results and figures:
  - Write all outputs under the analysis folder’s results directory (or a path provided by the user).
  - Prefer deterministic file names using a run ID (timestamp) to avoid overwriting.

Python Conventions
- Style: PEP 8, PEP 257 (docstrings), and type hints where practical.
- Imports: standard library → third‑party → local modules; avoid wildcard imports.
- Logging: use `logging` and the shared `setup_logging` helper (`logging_utils.setup_logging`).
- Testing: when feasible, add small tests for new utility functions under `tests/` or an analysis‑local `tests/` dir.
- CLI: prefer `argparse` for scripts that need parameters; document defaults and expected inputs.

MATLAB Conventions
- Use function files for reusable logic; scripts only as thin entry points.
- Add help text (first comment block) describing inputs/outputs and the rationale.
- Avoid global state; pass parameters explicitly.

Adding a New Analysis
1) Create a new top‑level folder (e.g., `chess-newanalysis/`).
2) Add a `modules/` subfolder for reusable functions.
3) Write one or more main scripts in the analysis root that orchestrate the workflow and produce figures.
4) Emit outputs to a `results/` subtree using a run ID.
5) Reuse existing helpers (logging, seeding, plotting) rather than duplicating code.
6) Document the workflow in the analysis README (inputs, steps, outputs) and reference `AGENTS.md`.

API Generation
- To generate an up‑to‑date API reference from the source tree, run:
  - `python scripts/generate_api.py --output API.md`
- The script scans Python modules across analysis folders and extracts top‑level functions/classes and their docstrings; it also lists MATLAB files by path for orientation.
- Keep `API.md` in sync when adding, renaming, or removing public functions.

Review Checklist (Before Merging)
- No duplicated logic across analyses; shared code extracted.
- Clear docstrings and inline comments (what and why) suitable for non‑experts.
- No absolute paths; parameters or config control inputs/outputs.
- Logging and run ID set; script saved alongside outputs where applicable.
- Figures and tables saved under the analysis results directory with informative names.

Where to Look
- High‑level overview: `README.md` and `manuscript/*.tex`.
- Shared logging: `logging_utils.py` (root) and analysis‑local logging utilities.
- Shared helpers (DRY): `common_utils.py` (OutputLogger, seeds, run IDs, script copy, output dirs).
- MVPA utilities: `chess-mvpa/modules/`.
- Behavioural utilities: `chess-behavioural/modules/`.
- GLM MATLAB utilities: `chess-glm/matlab_helpers/` and related scripts.
