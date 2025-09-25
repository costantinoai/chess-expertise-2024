ROIs — Overview

Purpose
- Provide ROI definitions and metadata used across analyses (GLM, MVPA, RSA, PR).

Contents
- ROI label files and region metadata (e.g., `region_info.tsv`).
- ROI NIfTI volumes or surface label files matching analysis needs.
- Centralized metadata loader in `rois/meta.py`.
- Validator: `rois/meta.validate_roi_tsv(tsv_path, valid_fs_names=None)` checks required columns, color formats, and optional FreeSurfer name matches.
 - Source CSV: `rois/data/HCP-MMP1_UniqueRegionList.csv` used to populate TSVs (per-hemisphere regions with cortex groups).
 - Legacy: historical ROI-construction scripts are moved under `rois/legacy/` and are not used by the pipeline.

Expected Layout
- `rois/sets/glasser_cortex_bilateral/region_info.tsv`
- `rois/sets/glasser_regions_bilateral/region_info.tsv`
- `rois/atlases/` — atlas NIfTI/label files (e.g., Glasser HCP-MMP1) as needed by scripts.

TSV Columns (regions)
- `ROI_idx` (int): sequential ID for the ROI row.
- `region_name` (str): display name (includes hemisphere suffix for clarity).
- `pretty_name` (str): same as region_name (reserved for future alt naming).
- `color` (hex): color, derived from its cortex group.
- `hemisphere` (str): `left` or `right`.
- `fs_name` (str): exact FreeSurfer annotation name (e.g., `L_V1_ROI`).
- `cortex` (str): cortex group name (one of the 7 groups).
- `cortex_id` (int): links to the corresponding row in `glasser_cortex_bilateral/region_info.tsv`.
- `order` (int): order of the ROI within its cortex group (useful for plotting).

TSV Columns (cortex)
- `ROI_idx` (int): cortex group ID (1..7).
- `region_name` (str): one of [Early Visual, Intermediate Visual, Sensorimotor, Auditory, Temporal, Posterior, Anterior].
- `color` (hex): canonical palette color.
- `hemisphere` (str): `bilateral`.
- `order` (int): group order for plotting.

Use in Analyses
- GLM summaries (`glm/roi_ttests_all.py`) use `rois/meta.py` for region names and colors.
- MVPA (`mvpa/`) can point to `rois/sets/<set>/region_info.tsv` for annotations.
- Manifold (`manifold/`) may subset/label ROIs by this metadata for PR reporting.
 - Surface overlays use FreeSurfer annotations plus TSV `fs_name` values; ensure TSV fs_name matches `.annot` names exactly (e.g., `L_V1_ROI`).

Notes
- Keep ROI resources versioned and documented to ensure reproducibility.
- Large atlas files may be referenced from `data/` if not committed; update `config.py` accordingly.
 - Python ROIManager has been removed; rely on TSV + atlas labels for metadata and overlays.
 - ROI construction code is deprecated; use provided NIfTI label images + TSV metadata only.
