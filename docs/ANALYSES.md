Chess Expertise 2024 — Analyses Inventory

Purpose
- Authoritative list of analyses reported in the manuscript, with pointers to where they live in this repository and where they appear in the paper.

How to use
- Keep this file updated when adding/removing analyses.
- Use it as a reference when pruning folders or regenerating API docs.

1) Stimulus Set Overview (Dataset Visualisation)
- Manuscript: Supplementary Material (Stimulus Set Details; Supp. Fig. stimuli)
- Folder: `chess-dataset-vis/`
- Outputs: `results/<RUN>_dataset-vis_*/*.png`

2) Behavioural Similarity (Human RDMs)
- Manuscript: Supplementary tables (split‑half reliability; `bh_rdm_splithalf`)
- Folder: `chess-behavioural/`
- Outputs: `results/<RUN>_behavioural_*/*`

3) fMRI Univariate GLM
- Manuscript: Main and supplement (e.g., All>Rest; Checkmate>Non‑checkmate; ROI maps)
- Folder: `chess-glm/` (MATLAB; SPM12)
- Selected outputs: `results/<RUN>_secondlevel-rois/*` (surface/glass PNGs, HTMLs, tables)

4) MVPA Decoding (ROI/group level)
- Manuscript: Main text and tables (decoding of Checkmate, Strategy, Visual Similarity; extended dims)
- Folder: `chess-mvpa/` (Python)
- Outputs: `results/<RUN>_mvpa_*/*.png`, barplots and logs

5) Representational Similarity Analysis (RSA)
- Manuscript: Whole‑brain searchlight and ROI summaries; group differences; tables `rsa_main_dims`, `roi_maps_rsa`
- Folders: `chess-mvpa/` (ROI RSA) and `chess-glm/` (searchlight RSA outputs)
- Outputs: searchlight maps (HTML/PNG) and tables in GLM `results/`; ROI figures in MVPA `results/`

6) Manifold Dimensionality (Participation Ratio, PR)
- Manuscript: PR section (main) + supplementary (PR vs ROI size; `pr_ttest`)
- Folders: `chess-mvpa/` (PR utilities) and `chess-manifoldMFT/` (manifold workflows)
- Outputs: PR figures/tables in respective `results/` or `manifold_results/`

7) Neurosynth Meta‑Analysis
- Manuscript: Methods/results describing term‑map associations with expert/novice maps; tables `neurosynth_rsa`, `neurosynth_univ`; supplementary maps
- Folder: `chess-neurosynth/`
- Outputs: figures/tables under `results/<RUN>_neurosynth_*`

8) Eye‑Movement Decoding (Supplementary)
- Manuscript: Supplementary section “Groups cannot be inferred from estimated eye‑movements” (`et_decoding`)
- Folder: estimated gaze decoding (uses BidsMReye outputs; integrated with fMRI analyses)
- Outputs: supplementary figure/table under analysis `results/`

Non‑manuscript or out‑of‑scope folders (candidates to remove after confirmation)
- `chess-connectivity/`: connectivity analyses not referenced in the manuscript.
- `chess-representational-conn/`: representational connectivity not referenced.
- `chess-results-vis/`: cross‑analysis visualization utilities not used in final figures.
- `chess-dnn/`: DNN comparisons not explicitly reported.
  - Note: retain if any figure/table in manuscript depends on these.

Dependencies & Shared Resources
- ROI definitions/utilities: `chess-rois/` (keep)
- Logging utilities: `logging_utils.py` (root) and analysis‑local variants
- MVPA helpers: `chess-mvpa/modules/`
- Behavioural helpers: `chess-behavioural/modules/`

Next actions (cleanup branch)
1) Confirm KEEP list: `chess-glm`, `chess-mvpa`, `chess-behavioural`, `chess-dataset-vis`, `chess-neurosynth`, `chess-manifoldMFT`, `chess-rois`.
2) Confirm DROP list: `chess-connectivity`, `chess-representational-conn`, `chess-results-vis`, `chess-dnn` (unless supporting figures).
3) After confirmation, remove DROP folders and update imports/paths accordingly.
4) Regenerate `API.md` and update READMEs (folder and root) to reflect the trimmed set.
5) DRY pass: consolidate duplicate helpers (e.g., OutputLogger) into shared utilities.

