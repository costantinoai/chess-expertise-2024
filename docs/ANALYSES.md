Chess Expertise 2024 — Analyses Inventory

Purpose
- Authoritative list of analyses reported in the manuscript, with pointers to code locations and expected outputs. Also tracks candidates for removal to keep the repo aligned with the paper.

How to use
- Run entry scripts from your IDE (no CLI required). Configure inputs via `config.py` or environment variables.
- Keep this file updated when adding/removing analyses or regenerating figures/tables.
- Use it as the source of truth when pruning folders or regenerating the API.

Reported Analyses (from manuscript)
-- Stimulus Set Overview
  - Manuscript: Supplementary section “Stimulus Set Details”; figure `stimuli_with_tags.png` (supplementary figure) and table `stimuli`.
  - Code: removed from this repo; figures are produced externally and stored under `manuscript/write/figures/`.
  - Outputs: n/a

- Behavioural Similarity (Human RDMs)
  - Manuscript: Supplementary section “Split-half Reliability of Behavioral RDMs”; table `bh_rdm_splithalf`.
  - Code: `behavioural/bh_fmri_splithalf.py` (split-half); `behavioural/bh_fmri.py` (behavioural RSA); `behavioural/bh_fmri_intercorr.py` (between-group/inter-corr if used).
  - Outputs: `behavioural/results/<RUN>_behavioural_*/*`

- fMRI Univariate GLM
  - Manuscript: Main and Supplementary (e.g., All>Rest; Checkmate>Non‑checkmate); ROI maps (supplementary `roi_maps_univ`).
  - Code: `glm/WithinGroup_SecondLevel_MultiContrast.m`, `glm/TwoSample_SecondLevel_MultiContrast.m`, `glm/contrasts_only.m`.
  - Python runners (IDE): `glm/roi_ttests_univ.py` (univariate), `glm/roi_ttests_rsa.py` (RSA). Legacy `glm/roi_ttests_all.py` aggregates both.
  - Outputs: `glm/results/<RUN>_secondlevel-rois/*` (surface/glass PNGs, HTMLs, TSVs)

- MVPA Decoding (ROI/group level)
  - Manuscript: Main text and Supplementary tables `decoding_main_dims`, `mvpa_extended_dimensions` (decoding of Checkmate, Strategy, Visual Similarity; extended dimensions on checkmate subset).
  - Code: ROI decoding and stats in `mvpa/` (e.g., `mvpa_ttest.py`); plotting via root runners `run_mvpa_barplots.py` and `run_mvpa_surfplots.py`; ROI utilities in `mvpa/modules/*`.
  - Outputs: `mvpa/results/<RUN>_mvpa_*/*` (barplots, pickles, TSVs)

- Representational Similarity Analysis (RSA)
  - Manuscript: Whole‑brain searchlight + ROI summaries; main ROI figure (Fig. “rsa_rois” in text); supplementary tables `rsa_main_dims`, `roi_maps_rsa`.
  - Code: Searchlight in `mvpa/rsa_searchlight_final.m` and/or `glm/RSA_TwoSample_SecondLevel_MultiContrast.m` (SPM); ROI RSA summaries via `mvpa/*` and `mvpa/modules/*`.
  - Outputs: searchlight maps (HTML/PNG) and ROI summaries in `glm/results/*`; ROI figures in `mvpa/results/*`

- Manifold Dimensionality (Participation Ratio, PR)
  - Manuscript: Main PR results figure (`pr_classification.png`); supplementary PR vs ROI size figure (`pr_supp.png`) and table `pr_ttest`.
  - Code: `manifold/participation_ratio.py`, `manifold/plot_pr_roi_size.py`.
  - Outputs: PR figures/tables in `manifold/results/*`

- RDM Orthogonality (Correlation and Variance Partitioning)
  - Manuscript: Supplementary section “Orthogonality Across RDMs” with correlation and variance partitioning panels.
  - Code: removed from this repo; not part of the final analyses set.
  - Outputs: n/a

- Neurosynth Meta‑Analysis
  - Manuscript: Methods/results reporting term‑map associations with expert/novice maps; supplementary tables `neurosynth_rsa`, `neurosynth_univ` and figure `terms_flat.png`.
  - Code: `neurosynth/rsa_neurosynth.py`, `neurosynth/univariate_neurosynth.py`.
  - Outputs: `neurosynth/results/<RUN>_neurosynth_*/*`

- Eye‑Movement Decoding (Supplementary)
  - Manuscript: Supplementary section “Groups cannot be inferred from estimated eye‑movements” (`et_decoding` figure/table).
  - Code: `mvpa/utils/svm_class_xy.py` (gaze x,y) and `mvpa/utils/svm_class_displacement.py` (distance from screen center). Task-vs-rest is out of scope.
  - Outputs: `results/<RUN>_et-mvpa/*` with plots and logs

- Sample Balance: Number of fMRI Runs per Group (Supplementary)
  - Manuscript: Supplementary control analysis showing no difference in runs across groups.
  - Code: `mvpa/test_number_of_runs.py`
  - Outputs: text and LaTeX table under `mvpa/results/<RUN>_runs-per-group/*`

Dependencies & Shared Resources
- ROI definitions/utilities: `rois/*` (TSV-driven sets via `rois/meta.py`)
- Shared helpers: `common/common_utils.py`, `common/logging_utils.py`, `common/stats_utils.py`
- Shared plotting: `common/brain_plotting.py`, `common/stats_plotting.py`, `common/plotting_utils.py`, `common/surface_plotting.py`
- MVPA helpers: `mvpa/modules/*` (overlays via TSV + FreeSurfer annotations)
- Behavioural helpers: `behavioural/modules/*`

Removed or Out of Scope
- dataset-vis and similar demo/utility repos are removed from this repository.
- machine‑specific or acquisition helpers are not part of analyses.

Notes
- Items marked “keep only if used …” are uncertain; confirm linkage to specific supplementary panels before removal.
- Before any deletion, create a timestamped backup branch and regenerate `API.md`.

KEEP list (in scope for the paper)
- `glm`, `mvpa`, `behavioural`, `neurosynth`, `manifold`, `rois`, plus shared utilities under `common/`.

Next actions
1) Validate each “Candidates For Removal” item against the manuscript figures/tables; move confirmed items to a removal list.
2) Delete confirmed extras on the cleanup branch, then regenerate `API.md` and update folder READMEs.
3) Re‑scan for duplicates/non‑DRY code and centralize helpers where possible.
