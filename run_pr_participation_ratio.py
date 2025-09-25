#!/usr/bin/env python3
"""Run PR (participation ratio) analysis from repo root.

Loads subject betas per ROI, computes PR per ROI per subject, performs
Welch tests with FDR across ROIs and saves consolidated CSV.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from joblib import Parallel, delayed

from common.common_utils import create_run_id, save_script_to_file
from common.logging_utils import setup_logging
from manifold.modules.pr_core import (
    Config,
    process_subject,
    per_roi_welch_and_fdr,
    consolidate_results,
)

RUN_ID = create_run_id()
OUT_ROOT = Path("results") / f"{RUN_ID}_participation_ratio"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
setup_logging(); setup_logging(log_file=str(OUT_ROOT / "run.log"))
save_script_to_file(OUT_ROOT)

cfg = Config()
atlas_img = nib.load(cfg.atlas_file.as_posix())
atlas_data = atlas_img.get_fdata().astype(int)
roi_labels = np.unique(atlas_data)
roi_labels = roi_labels[roi_labels != 0]

def _run(sub_ids):
    if cfg.use_parallel:
        return np.array(Parallel(n_jobs=cfg.n_jobs, verbose=5)(
            delayed(process_subject)(sid, atlas_data, roi_labels, cfg) for sid in sub_ids
        ))
    return np.array([process_subject(sid, atlas_data, roi_labels, cfg) for sid in sub_ids])

expert_pr = _run(cfg.expert_subjects)
novice_pr = _run(cfg.nonexpert_subjects)
stats_df = per_roi_welch_and_fdr(expert_pr, novice_pr, roi_labels, cfg.alpha_fdr)
results_df = consolidate_results(expert_pr, novice_pr, roi_labels, stats_df, cfg.roi_name_map)

csv_out = OUT_ROOT / "roi_pr_results_consolidated.csv"
results_df.to_csv(csv_out, index=False)
