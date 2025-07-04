# chess-manifoldMFT

This module implements the manifold capacity analysis for the Chess Expertise project. It wraps the methods of [Chung et al., 2018](https://doi.org/10.1103/PhysRevX.8.031003) to assess the geometry of neural response manifolds derived from fMRI beta images. The code compares ROI-wise metrics between expert and non-expert chess players.

For each subject we load the beta volumes from all runs of a first level SPM
model. Beta images are organised per condition and have shape
`(n_runs, X, Y, Z)` after loading. Voxels belonging to the selected ROI are
extracted and arranged as `(n_runs, n_voxels)` arrays. After discarding voxels
containing `NaN` values or zero variance, the data are z-scored across runs so
that each run becomes an observation and voxels correspond to features. Each
condition therefore yields a matrix of shape `(n_features, n_runs)` that defines
a single response manifold for that ROI.

## Data inputs

- **SPM GLM outputs**: beta images from a first level GLM are expected under `BASE_GLM_PATH` defined in `modules/__init__.py`.
- **Glasser atlas**: ROIs are taken from `ATLAS_FILE` (also defined in `modules/__init__.py`).
- **Subject groups**: Lists of expert and non‑expert participant IDs are specified in the same file.

## Processing steps

1. **Load atlas and select ROIs** – `load_atlas()` returns the atlas data array and unique ROI labels. `main.py` selects ROIs `[1, 2, 21, 22]` for the example analysis.
2. **Iterate over subjects** – `run_group()` applies `process_subject()` to each subject in the expert and non‑expert lists using joblib for optional parallelism.
3. **Extract beta values** – `load_all_betas()` loads all beta images per condition and run for a subject. The helper `get_spm_betas()` parses the `SPM.mat` design matrix to locate beta file paths.
4. **Build manifolds per ROI** – Each ROI-specific beta stack has shape
   `(n_runs, n_voxels)`. Voxels containing `NaN` values or zero variance are
   discarded across **all** conditions so that the manifolds share a common
   feature dimension. The remaining data are z‑scored across runs using
   `StandardScaler` and then transposed to `(n_features, n_runs)` arrays. Each
   of these arrays represents a single condition/manifold for that ROI.
5. **Manifold metrics** – `compute_manifold()` invokes
   `manifold_analysis_corr` from the `mftma` package with the cleaned data. The
   analysis samples 200 Gaussian vectors per manifold and returns three
   quantities for each: capacity ``a_M`` (how many random dichotomies can be
   linearly separated with margin ``kappa``), anchor radius ``R_M`` and anchor
   dimension ``D_M``. We take the harmonic mean of the capacities across
   conditions and the arithmetic mean of the radii and dimensions to obtain a
   single value per ROI and subject.
6. **Subject‑level plots** – `plot_subject_roi()` saves a scatter plot of dimension versus radius for each subject/ROI pair.
7. **Group statistics** – Arrays of shape ``(n_subjects, n_rois)`` are obtained
   for capacity, radius and dimension. For each ROI we perform an independent
   two‑sample t‑test comparing experts and non‑experts. ``fdr_ttest`` applies the
   Benjamini–Hochberg procedure across ROIs to control the false discovery rate
   at ``α=0.05`` and writes the resulting ``t`` values and corrected ``p`` values
   to ``stats_*.csv`` files.
8. **Result aggregation and plotting** – Mean metrics for each group/ROI are
   stored in ``group_means.csv``. ``plot_group_comparison`` draws box plots of
   the subject distributions per ROI, while ``plot_group_heatmap`` displays the
   expert versus non‑expert means in a heatmap. All figures and CSV files are
   saved under ``manifold_results``.

## Running the analysis

Execute `python main.py` from within `chess-manifoldMFT`. The script saves outputs to the `manifold_results` directory.

## Dependencies

- `mftma` library providing `manifold_analysis_corr` (imported in `analysis_utils.py`).
- `nibabel`, `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, and `seaborn`.

The example notebook `MFTMA_VGG16_example.ipynb` illustrates usage of `manifold_analysis_corr` on neural network activations.
