Chess Expertise 2024 — Cleanup Branch Scratchpad

Branch
- Name: cleanup-20250924-191124
- Purpose: Trim repo to only manuscript-reported analyses, enforce DRY, simplify structure, and prepare for clean academic sharing.

Scope
- KEEP: chess-glm, chess-mvpa, chess-behavioural, chess-dataset-vis, chess-neurosynth, chess-manifoldMFT, chess-rois
- DROP (after double-check): chess-connectivity, chess-representational-conn, chess-results-vis, chess-dnn
- Reference map: docs/ANALYSES.md (authoritative inventory), AGENTS.md (conventions), API.md (functions overview)

High-level Plan
1) Confirm analyses inventory (done) and KEEP/DROP list (done)
2) Prune DROP folders (git rm -r) on cleanup branch
3) DRY pass: consolidate shared utilities and remove duplicates
4) Update packaging and docs to reflect pruned set
5) Verify scripts run/entry points documented; regenerate API and READMEs

Todo Checklist
- [x] Prune folders not in manuscript
  - [x] Remove: chess-connectivity/
  - [x] Remove: chess-representational-conn/
  - [x] Remove: chess-results-vis/
  - [x] Remove: chess-dnn/
  - [x] rg sanity check for any remaining references to removed folders (only references remain in docs and archival result scripts)
- [ ] DRY / Refactors
  - [ ] Unify OutputLogger implementations (prefer one in a shared module; e.g., reuse chess-mvpa/modules/helpers or move a generic version to root/common and import elsewhere)
  - [ ] Centralize seeding (set_rnd_seed), create_run_id, save_script_to_file in a shared utility
  - [ ] Standardize logging via logging_utils.setup_logging; ensure all main scripts call it
  - [ ] Enforce timestamped output dir convention `<YYYYMMDD-HHMMSS>_<shortname>` across scripts
  - [ ] Audit absolute/hard-coded paths; replace with CLI flags/config
  - [ ] Ensure modules only contain functions; main scripts orchestrate workflows
- [ ] Packaging / Config
  - [ ] Update pyproject.toml package discovery to remove dropped packages (chess_dnn, etc.)
  - [ ] Review requirements.txt and remove deps only used by dropped code (keep if shared)
  - [ ] Ensure tests/ still valid (or trim if they belong to removed code)
- [ ] Documentation
  - [ ] Update folder READMEs to reflect trimmed scope and provide run instructions
  - [x] Regenerate API.md via `python scripts/generate_api.py --output API.md`
  - [ ] Update README.md (root) “Repository Structure” to list only kept analyses
  - [ ] Update AGENTS.md if any convention adjustments emerge during DRY
- [ ] Verification
  - [ ] Run `rg` to find imports of removed modules
  - [ ] Spot-run key entry scripts with `-h` to ensure CLI works
  - [ ] Check figures/tables save to timestamped result folders

Risks & Notes
- Removing folders may break imports if code references them implicitly; mitigate with rg search and targeted fixes.
- pyproject.toml currently maps packages to hyphenated dirs (e.g., chess_mvpa = "chess-mvpa"); update includes accordingly.
- Requirements pruning is optional; prefer conservative retention initially, then prune after confirming no usages.

Commands (for later execution)
- Remove folders:
  - git rm -r chess-connectivity chess-representational-conn chess-results-vis chess-dnn
- Regenerate API:
  - python scripts/generate_api.py --output API.md
- Quick reference scan:
  - rg "chess-(connectivity|representational-conn|results-vis|dnn)" -n

Decision Log
- 2025-09-24: Agreed KEEP/DROP list per manuscript; added docs/ANALYSES.md as authoritative inventory.
