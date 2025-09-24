Chess Expertise 2024 — Analyses Inventory

Purpose
- Authoritative list of analyses reported in the manuscript, with pointers to code locations and expected outputs. Also tracks candidates for removal to keep the repo aligned with the paper.

How to use
- Keep this file updated when adding/removing analyses or regenerating figures/tables.
- Use it as the source of truth when pruning folders or regenerating the API.

Reported Analyses (from manuscript)
-- Stimulus Set Overview
  - Manuscript: Supplementary section “Stimulus Set Details”; figure `stimuli_with_tags.png` (supplementary figure) and table `stimuli`.
  - Code: removed from this repo; figures are produced externally and stored under `manuscript/write/figures/`.
  - Outputs: n/a

- Behavioural Similarity (Human RDMs)
  - Manuscript: Supplementary section “Split-half Reliability of Behavioral RDMs”; table `bh_rdm_splithalf`.
  - Code: `chess-behavioural/bh_fmri_splithalf.py` (split-half); `chess-behavioural/bh_fmri.py` (behavioural RSA); `chess-behavioural/bh_fmri_intercorr.py` (between-group/inter-corr if used).
  - Outputs: `chess-behavioural/results/<RUN>_behavioural_*/*`

- fMRI Univariate GLM
  - Manuscript: Main and Supplementary (e.g., All>Rest; Checkmate>Non‑checkmate); ROI maps (supplementary `roi_maps_univ`).
  - Code: `fmri_glm/WithinGroup_SecondLevel_MultiContrast.m`, `fmri_glm/TwoSample_SecondLevel_MultiContrast.m`, `fmri_glm/contrasts_only.m`, ROI aggregation in `fmri_glm/roi_ttests_all.py`.
  - Outputs: `fmri_glm/results/<RUN>_secondlevel-rois/*` (surface/glass PNGs, HTMLs, TSVs)

- MVPA Decoding (ROI/group level)
  - Manuscript: Main text and Supplementary tables `decoding_main_dims`, `mvpa_extended_dimensions` (decoding of Checkmate, Strategy, Visual Similarity; extended dimensions on checkmate subset).
  - Code: ROI decoding and stats in `mvpa/` (e.g., `mvpa_barplot.py`, `mvpa_barplot_comparison.py`, `mvpa_ttest.py`); ROI management/plotting in `mvpa/modules/*`.
  - Outputs: `mvpa/results/<RUN>_mvpa_*/*` (barplots, pickles, TSVs)

- Representational Similarity Analysis (RSA)
  - Manuscript: Whole‑brain searchlight + ROI summaries; main ROI figure (Fig. “rsa_rois” in text); supplementary tables `rsa_main_dims`, `roi_maps_rsa`.
  - Code: Searchlight in `chess-mvpa/rsa_searchlight_final.m` and/or `chess-glm/RSA_TwoSample_SecondLevel_MultiContrast.m` (SPM); ROI RSA summaries via `chess-mvpa/*` and `chess-mvpa/modules/*`.
  - Outputs: searchlight maps (HTML/PNG) and ROI summaries in `chess-glm/results/*`; ROI figures in `chess-mvpa/results/*`

- Manifold Dimensionality (Participation Ratio, PR)
  - Manuscript: Main PR results figure (`pr_classification.png`); supplementary PR vs ROI size figure (`pr_supp.png`) and table `pr_ttest`.
  - Code: `manifold/participation_ratio.py`, `manifold/plot_pr_roi_size.py`.
  - Outputs: PR figures/tables in `manifold/results/*`

- RDM Orthogonality (Correlation and Variance Partitioning)
  - Manuscript: Supplementary section “Orthogonality Across RDMs” with correlation and variance partitioning panels.
  - Code: `chess-dataset-vis/rdms_intercorr.py` (pairwise and partial correlations; variance partitioning; LaTeX table snippets in logs) and `chess-dataset-vis/rdms_intercorr_checkmate_boards.py` (checkmate-only variant used in supplementary panels).
  - Outputs: figures under `chess-dataset-vis/results/<RUN>_*/*.png` (or logs with LaTeX table content)

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
- ROI definitions/utilities: `chess-rois/*`
- Shared helpers: `common_utils.py`, `logging_utils.py`
- MVPA helpers: `chess-mvpa/modules/*`
- Behavioural helpers: `chess-behavioural/modules/*`

 Candidates For Removal (not referenced in manuscript analyses)
- chess-dataset-vis
  - `chess-dataset-vis/barplot-manim.py` (demo/animation)
  - `chess-dataset-vis/plot_rdms.py` (generic RDM plotting; not cited)
  - `chess-dataset-vis/visualize_manifold.py` (manifold demo; not in paper)

- chess-behavioural
  - `chess-behavioural/bh_familiarization_data_cleaning.py` (familiarization, not reported)
  - `chess-behavioural/bh_familiarization_data_plotting.py` (familiarization, not reported)

- chess-mvpa (top‑level and utils)
  - `chess-mvpa/manifold.py` (separate manifold script; paper uses `chess-manifoldMFT`)
  - `chess-mvpa/utils/get_philips_MB_slicetiming.py` (acquisition helper; not an analysis)
  - `chess-mvpa/utils/anon_nii_filename.py` (utility; not an analysis)
  - `chess-mvpa/utils/svm_class_taskvsrest.py` (task vs rest; not reported)
  - `chess-mvpa/utils/sanitize_json.py` (utility; not an analysis)

- chess-manifoldMFT
  - `chess-manifoldMFT/main_noavg.py` (alternative entrypoint; keep only if used for a reported figure)

- misc and local
  - `misc/*` (BIDS converters and templates; not part of analyses)
  - `local/*` (machine‑specific configuration; not part of analyses)

Notes
- Items marked “keep only if used …” are uncertain; confirm linkage to specific supplementary panels before removal.
- Before any deletion, create a timestamped backup branch and regenerate `API.md`.

KEEP list (in scope for the paper)
- `fmri_glm`, `mvpa`, `behavioural`, `neurosynth`, `manifold`, `rois`, plus root utilities (`common_utils.py`, `logging_utils.py`).

Next actions
1) Validate each “Candidates For Removal” item against the manuscript figures/tables; move confirmed items to a removal list.
2) Delete confirmed extras on the cleanup branch, then regenerate `API.md` and update folder READMEs.
3) Re‑scan for duplicates/non‑DRY code and centralize helpers where possible.
