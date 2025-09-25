#!/usr/bin/env python3
"""Run GLM univariate ROI summaries from root."""

import os
from common.common_utils import create_run_id, save_script_to_file
from common.logging_utils import setup_logging
from config import ATLAS_BILATERAL, GLM_BASE_PATH
from glm.modules.roi_stats import run_analysis

if __name__ == "__main__":
    run_id = create_run_id()
    OUTPATH = os.path.join("results", f"{run_id}_secondlevel-rois-univ")
    os.makedirs(OUTPATH, exist_ok=True)
    setup_logging(); setup_logging(log_file=os.path.join(OUTPATH, "glm_roi_ttests_univ.log"))
    save_script_to_file(OUTPATH)

    CONTRASTS = {
        "con_0001.nii": "Checkmate > Non-checkmate",
        "con_0002.nii": "All > Rest",
    }
    run_analysis(
        data_dir=str(GLM_BASE_PATH),
        contrasts=CONTRASTS,
        atlas_path=str(ATLAS_BILATERAL),
        mode="univ",
        out_dir=OUTPATH,
    )
