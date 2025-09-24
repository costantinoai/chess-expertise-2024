MVPA — Overview

Purpose
- ROI-based MVPA/RSA summaries and plotting utilities. Eyetracking decoding (supplementary) lives in `utils/svm_class_*.py`.

Inputs (under `data/`)
- MVPA results: `data/BIDS/derivatives/mvpa/<RUN>_*/` containing group pickles (`svm/*group/*.pkl`, `rsa_corr/*group/*.pkl`)
- ROI annotations: `rois/results/<roi_set>/region_info.tsv` and corresponding ROI NIfTI in the same folder

Key scripts
- `mvpa_barplot.py`: build barplots from group-level pickled results for multiple ROI sets
- `mvpa_barplot_comparison.py`: compare across analyses (e.g., SVM vs RSA)
- `mvpa_ttest.py`: group tests and exports
- `modules/*`: ROI manager, plotting helpers, surf overlays
- Eyetracking (supplementary): `utils/svm_class_xy.py`, `utils/svm_class_displacement.py`

How to run (IDE)
- Ensure `config.py` has the correct `MVPA_RESULTS_ROOT` (or set `CHESS_MVPA_RESULTS_DIR*` env vars).
- Run the target script and check `results/` or the specified output directory for figures and logs.
