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
- [ ] Replace direct uses of `statsmodels.multipletests`/`fdrcorrection` and `pingouin.multicomp` with `common.stats_utils.fdr_correction` across analyses.
- [ ] Replace ad-hoc one-sample/independent t-test code with `one_sample_ttest` and `independent_ttest` where possible.
- [ ] Use `pearson_corr_bootstrap` and `corr_diff_bootstrap` wrappers where bootstrap correlations are computed.

Phase 3 — MVPA scripts restructure
- [ ] Move helper functions from `mvpa/mvpa_barplot*.py`, `mvpa/print_mvpa_stats.py`, etc., into `mvpa/modules/` (e.g., `barplot_utils.py`, `report_utils.py`).
- [ ] Ensure entry scripts only orchestrate: setup run_id/output dir/logging, parse args, call module functions, save artefacts.

Phase 4 — fMRI GLM restructure
- [ ] Extract helpers from `fmri_glm/roi_ttests_all.py` into `fmri_glm/modules/glm_utils.py` (e.g., `export_diff_stats_to_latex`, `compute_confidence_intervals`, plotting helpers).
- [ ] Keep a module `fmri_glm/modules/roi_stats.py` with `run_analysis()`; make any CLI script a thin wrapper calling it.

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
