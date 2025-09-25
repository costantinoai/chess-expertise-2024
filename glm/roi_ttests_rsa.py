#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Group ROI stats — RSA searchlight (Experts vs Novices).

Thin IDE-runner that orchestrates ROI-level summaries for RSA maps.
Figures and tables are written under a timestamped results folder.
"""
import os
from common.common_utils import create_run_id, save_script_to_file
from common.logging_utils import setup_logging
from config import ATLAS_BILATERAL, DERIVATIVES_PATH
from glm.modules.roi_stats import run_analysis


if __name__ == "__main__":
    run_id = create_run_id()
    OUTPATH = os.path.join("results", f"{run_id}_secondlevel-rois-rsa")
    os.makedirs(OUTPATH, exist_ok=True)
    setup_logging()
    setup_logging(log_file=os.path.join(OUTPATH, "glm_roi_ttests_rsa.log"))
    save_script_to_file(OUTPATH)

    RSA_DATA_DIR = os.path.join(str(DERIVATIVES_PATH), "rsa_searchlight")
    CONTRASTS = {
        "checkmate.nii.gz": "Checkmate",
        "strategy.nii.gz": "Strategy",
        "visualSimilarity.nii.gz": "Visual Similarity",
    }

    run_analysis(
        data_dir=RSA_DATA_DIR,
        contrasts=CONTRASTS,
        atlas_path=str(ATLAS_BILATERAL),
        mode="rsa",
        out_dir=OUTPATH,
    )

