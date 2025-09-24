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
