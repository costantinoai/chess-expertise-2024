MVPA — Overview

Purpose
- ROI-based MVPA/RSA summaries and plotting utilities. Eyetracking decoding (supplementary) lives in `utils/svm_class_*.py`.

Inputs (under `data/`)
- MVPA results: `data/BIDS/derivatives/mvpa/<RUN>_*/` containing group pickles (`svm/*group/*.pkl`, `rsa_corr/*group/*.pkl`)
- ROI annotations: TSV + atlas: `rois/sets/<roi_set>/region_info.tsv` and the atlas labels (FreeSurfer HCP-MMP1) referenced by `config.py`

Key scripts
  (Barplots and surface plots are now orchestrated via root runners.)
- `mvpa_barplot_comparison.py`: compare across analyses (e.g., SVM vs RSA)
- `mvpa_ttest.py`: group tests and exports
- `modules/*`: plotting helpers and surface overlays (ROIManager deprecated).
- Eyetracking (supplementary): `utils/svm_class_xy.py`, `utils/svm_class_displacement.py`

How to run (IDE)
- Ensure `config.py` has the correct `MVPA_RESULTS_ROOT` (or set `CHESS_MVPA_RESULTS_DIR*` env vars).
- Use shared plotting utilities:
  - Bar plots: `common/plotting_utils.plot_mvpa_barplot`

Running
- Preferred entry points (run from repo root):
  - `run_mvpa_ttests.py` — MVPA group t-tests (saves pickled group stats)
  - `run_mvpa_barplots.py` — generates barplots from group stats
  - `run_mvpa_surfplots.py` — generates fsaverage surface overlays for significant ROIs
  - Surface overlays: `common/surface_plotting.plot_left_fsaverage_grid`
- ROI metadata is loaded via `rois/meta.get_roi_info('<set>')` from TSV; no ROIManager.
- Run the target script and check `results/` or the specified output directory for figures and logs.
