Chess Expertise 2024 — Cleanup Branch Scratchpad

Branch
- Name: cleanup-20250924-191124
- Purpose: Trim repo to only manuscript-reported analyses, enforce DRY, simplify structure, centralize config/meta/logging, and prepare for clean academic sharing.

Scope
- KEEP: fmri_glm, mvpa, behavioural, neurosynth, manifold (PR), rois
- DROP: dataset-vis, DNN manifold, machine-specific scripts, unneeded utils
- Reference map: docs/ANALYSES.md (authoritative inventory), AGENTS.md (conventions), API.md (functions overview)

High-level Plan
1) Confirm analyses inventory (done) and KEEP/DROP list (done)
2) Prune DROP folders (done)
3) DRY pass: centralize run/logging/colormap/meta/config (done)
4) Update packaging, ignores, and docs to reflect new structure (done)
5) Document inputs/outputs per analysis; regenerate API (done)

Todo Checklist
- [x] Prune folders not in manuscript / clean utils
  - [x] Remove dataset-vis, local/misc, old/*, unused utils
- [x] DRY / Refactors
  - [x] `common_utils.py` (run id, output dir, script copy, seeding)
  - [x] `logging_utils.setup_logging` in all entry scripts
  - [x] `viz_utils.make_brain_cmap` for maps
  - [x] `meta.py` for ROI names/colors and plot styles
  - [x] `config.py` for data roots/atlases/subjects with env overrides
  - [x] Sweep absolute paths; enforce os.path.join/Path
- [x] Packaging / Config
  - [x] Update pyproject.toml packages to mvpa, fmri_glm, behavioural, neurosynth, manifold, rois
  - [x] Update .gitignore to match new structure
- [x] Documentation
  - [x] Update per-analysis READMEs (fmri_glm, mvpa, neurosynth, manifold, rois)
  - [x] Update root README (data layout and paths)
  - [x] Update AGENTS.md and docs/ANALYSES.md
- [x] API
  - [x] Generator scans analysis dirs and excludes results
  - [x] Regenerate API.md
- [ ] Verification
  - [ ] Run representative scripts from IDE to confirm outputs and logs
  - [ ] Validate env overrides on a second machine

Risks & Notes
- Ensure `data/` layout is consistent on your machine; override via env as needed (see `config.py`).
- MATLAB requires SPM/CoSMoMVPA on path; adjust scripts accordingly.

Commands (for later execution)
- Remove folders:
  - git rm -r chess-connectivity chess-representational-conn chess-results-vis chess-dnn
- Regenerate API:
  - python scripts/generate_api.py --output API.md
- Quick reference scan:
  - rg "chess-(connectivity|representational-conn|results-vis|dnn)" -n

Decision Log
- 2025-09-24: Agreed KEEP/DROP list per manuscript; added docs/ANALYSES.md as authoritative inventory.

Next Steps — Central Stats + Script Cleanup

Goal: centralize all statistical helpers under `common/stats_utils.py`, move functions out of entry scripts into analysis-level modules, and keep scripts as thin runners. Then regenerate the API and update docs.

Phase 1 — Neurosynth extraction (scripts → modules)
- [ ] Create `neurosynth/modules/univariate_main.py` implementing the current end-to-end workflow (formerly in `neurosynth/univariate_neurosynth.py`).
- [ ] In `neurosynth/univariate_neurosynth.py`, remove function definitions and import:
  - plotting + utilities from `neurosynth/modules/univariate_utils.py`
  - stats from `neurosynth/modules/stats_utils.py` and `common/stats_utils.py`
  - call `univariate_main.main()`; keep only run orchestration
- [ ] Do the same for RSA:
  - Add `neurosynth/modules/rsa_main.py`
  - Trim `neurosynth/rsa_neurosynth.py` to a thin runner importing and invoking `rsa_main.main()`

Phase 2 — Adopt central stats wrappers
- [x] Replace direct uses of `statsmodels.multipletests`/`fdrcorrection` and `pingouin.multicomp` with `common.stats_utils.fdr_correction` across analyses (manifold, neurosynth, mvpa/manifold).
- [ ] Replace ad-hoc one-sample/independent t-test code with `one_sample_ttest` and `independent_ttest` where possible.
- [x] Use `pearson_corr_bootstrap` and `corr_diff_bootstrap` wrappers where bootstrap correlations are computed (neurosynth, behavioural).

Phase 3 — MVPA scripts restructure
- [ ] Move helper functions from `mvpa/mvpa_barplot*.py`, `mvpa/print_mvpa_stats.py`, etc., into `mvpa/modules/` (e.g., `barplot_utils.py`, `report_utils.py`).
- [ ] Ensure entry scripts only orchestrate: setup run_id/output dir/logging, parse args, call module functions, save artefacts.

Phase 4 — fMRI GLM restructure
- [x] Extract helpers into `glm/modules/glm_utils.py` (FDR, Welch test wrapper, CI, LaTeX export).
- [ ] Add `glm/modules/roi_stats.py` with `run_analysis()`; make any CLI script a thin wrapper calling it (pending script consolidation).

Phase 5 — Behavioural cleanup
- [ ] Move reusable functions from `behavioural/bh_fmri.py` and `behavioural/bh_fmri_intercorr.py` into `behavioural/modules/`.
- [ ] Replace `calculate_mean_and_ci` with `common.stats_utils.mean_ci_t` where applicable; standardize correlations via common wrappers.

Phase 6 — Consistency and hygiene
- [ ] Ensure every artefact-producing script:
  - creates `<YYYYMMDD-HHMMSS>_<shortname>` output dir under `results/`
  - uses `common.logging_utils.setup_logging(log_file=...)` (no print)
  - calls `common.common_utils.save_script_to_file(output_dir)`
- [ ] Sweep for remaining absolute paths; replace with `config.py`/env or parameters.

Phase 7 — API + Docs
- [ ] Run `python scripts/generate_api.py --output API.md` after refactors.
- [ ] Update per-analysis READMEs if entrypoint names changed.
- [ ] Note progress here and in commit messages.

Reference
- Conventions: `AGENTS.md` (this plan is referenced there)
- Shared utilities: `common/` (logging, stats, run utils)
- API overview: `API.md`

Next Steps — DRY/Centralization Pass (2025‑09‑25)

Goals
- Eliminate duplication across modules; use shared helpers.
- Keep entry scripts as thin runners with argparse + logging + run IDs.

Todo Checklist
- [x] Neurosynth plotting: delegate correlations/differences to `common.stats_plotting` (done)
- [x] Remove ROIManager Python module and references; migrate overlays to NIfTI+TSV metadata (done)
- [x] mvpa_surfplot_hcp: remove local `load_glasser_surf`; use centralized overlay builder; add contours from FS annot (done)
- [x] MVPA bar plots: switch callers to `common.plotting_utils.plot_mvpa_barplot` (done)
- [x] MVPA bar plots: remove local implementation from `mvpa/modules/plotting_helpers.py` (overridden by thin wrapper)
- [x] Consolidate surface plotting helpers under a shared utility (fsaverage meshes, curvature, contours)
- [x] ROI TSVs: complete fs_name/cortex/color across sets and validate name↔annot matches (populated from HCP CSV)
- [x] Add `rois.meta.validate_roi_tsv(...)` and call in consumers
- [x] Optimize overlay mapping in `mvpa/modules/surf_helpers.build_significance_overlay` (precompute name→index, warn on misses)
- [x] Update READMEs to mention shared plotting modules and ROI TSV ingestion
- [x] Regenerate API after above
- [x] Update all repository README/MD files to reflect: centralized plotting (common/*), ROI TSV ingestion replacing ROIManager, surface plotting helper usage, and run/output conventions.

Next Items — Priority Queue (2025‑09‑25)

- [x] Neurosynth plotters: verify `neurosynth/modules/plot_utils.py` fully delegates (`plot_correlations`, `plot_difference`) and remove any legacy/duplicate plotting code.
- [x] MVPA plotting helpers: remove the local `plot_mvpa_barplot` implementation and keep the thin wrapper to `common.plotting_utils.plot_mvpa_barplot`; clean up duplicate definitions and update docstrings.
- [x] Surf helpers docs: update `mvpa/modules/surf_helpers.py` and `mvpa/modules/plotting_helpers.py` docstrings to reflect NIfTI+TSV overlays (no ROIManager/LUT mentions).
- [x] ROI creation code: delete or move `rois/make_glasser_rois.py` to legacy; update `rois/README.md` to state ROI sets are provided as NIfTI + TSV metadata only.
- [ ] ROI TSVs completeness: verify both sets are fully populated (id, name, pretty_name, fs_name, cortex, color, cortex_id/order); patch if missing.
- [ ] ROI TSV validation in consumers: call `rois.meta.validate_roi_tsv(...)` in callers and warn on mismatches; precompute FS name→index maps to speed overlay mapping.
- [ ] READMEs refresh: mvpa, neurosynth, glm READMEs should mention shared plotting modules (common/brain_plotting.py, common/stats_plotting.py, common/plotting_utils.py) and ROI TSV ingestion replacing ROIManager.
- [x] Regenerate API: `python scripts/generate_api.py --output API.md`; commit the updated `API.md`.
- [x] Repository MD sweep: ensure each top‑level analysis README has inputs, how to run, where outputs save, and references the timestamped results folder convention.
- [ ] Optional tests: add small tests for `validate_roi_tsv` and shared plotting functions to catch regressions.

In‑Progress / Next
- Next: Optional small tests for ROI TSVs and plotting (if desired).

Completed Cleanup
- [x] Trimmed large commented legacy code blocks from Python modules to keep the codebase lean (e.g., neurosynth/modules/stats_utils.py).
 - [x] Behavioural: removed absolute paths; use `config.py` for participants/categories; added logging + script copy to outputs.
 - [x] GLM: removed duplicate local helpers from `glm/roi_ttests_all.py`; use `glm/modules/*` and shared logging.
 - [x] MVPA: docstrings updated to reference `fdr_correction()`.
 - [x] Manifold: fixed missing nibabel import in `plot_pr_roi_size.py`.

 Notes
 - Analyses are run from IDE (no CLI). Configure inputs via `config.py` or environment variables.
