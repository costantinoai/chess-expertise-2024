#!/usr/bin/env python3
"""RSA → Neurosynth term correlation analysis (root runner).

Orchestrates group GLM on subject searchlight maps, plotting contrasts,
and correlating positive/negative z-maps with term maps.
"""

import os
import logging
from pathlib import Path
import numpy as np
from nilearn.glm.second_level import SecondLevelModel

from common.common_utils import create_run_id, create_output_directory, save_script_to_file, add_file_logger
from common.niimg_utils import find_nifti_files, fisher_z_maps, load_and_mask_imgs, build_design_matrix
from neurosynth.modules.io_utils import load_term_maps
from neurosynth.modules.stats_utils import compute_all_zmap_correlations, generate_latex_multicolumn_table, get_brain_mask
from common.brain_plotting import plot_surface_map_flat, plot_map
from common.stats_plotting import plot_correlations, plot_difference
from config import DERIVATIVES_PATH, EXPERTS, NONEXPERTS

PATTERNS = [
    "searchlight_checkmate",
    "searchlight_strategy",
    "searchlight_visualSimilarity",
]
PATTERN_CLEAN = {
    "searchlight_checkmate": "Checkmate | RSA searchlight",
    "searchlight_strategy": "Strategy | RSA searchlight",
    "searchlight_visualSimilarity": "Visual Similarity | RSA searchlight",
}

DATA_DIR = str(Path(DERIVATIVES_PATH) / "rsa_searchlight")
TERM_DIR = str(Path("neurosynth") / "data" / "terms")
RUN_ID = create_run_id()
OUT_ROOT = os.path.join("results", f"{RUN_ID}_neurosynth-rsa-searchlight")
create_output_directory(OUT_ROOT)
save_script_to_file(OUT_ROOT)
add_file_logger(os.path.join(OUT_ROOT, "console.log"))
logger = logging.getLogger(__name__)
logger.info("Results directory: %s", OUT_ROOT)

term_maps = load_term_maps(TERM_DIR)
all_pos, all_neg, all_diff = {}, {}, {}

def _masked(imgs_list):
    return load_and_mask_imgs(imgs_list, mask_func=get_brain_mask)

for pat in PATTERNS:
    pat_name = PATTERN_CLEAN[pat]
    logger.info("--- Pattern: %s ---", pat_name)

    r_files = find_nifti_files(DATA_DIR, pattern=pat)
    exp_files = [f for sid in EXPERTS for f in r_files if f"sub-{sid}" in os.path.basename(f)]
    nov_files = [f for sid in NONEXPERTS for f in r_files if f"sub-{sid}" in os.path.basename(f)]

    z_exp, _ = _masked(fisher_z_maps(exp_files))
    z_nov, _ = _masked(fisher_z_maps(nov_files))
    z_all = z_exp + z_nov

    design = build_design_matrix(len(z_exp), len(z_nov))
    slm = SecondLevelModel(smoothing_fwhm=None, n_jobs=-1).fit(z_all, design_matrix=design)
    con_img = slm.compute_contrast("group", output_type="z_score")
    z_data = con_img.get_fdata()
    con_img_path = os.path.join(OUT_ROOT, f"{pat_name}_zmap_experts_gt_novices.nii.gz")
    con_img.to_filename(con_img_path)

    plot_map(z_data, con_img, title=f"{pat_name} | Experts>Novices", outpath=os.path.join(OUT_ROOT, f"{pat_name}_glass_experts_gt_novices.png"), thresh=1e-5)
    plot_surface_map_flat(con_img, title=f"{pat_name} | Experts>Novices", threshold=1e-5, output_file=os.path.join(OUT_ROOT, f"{pat_name}_surface_flat_experts_gt_novices.png"))

    z_pos = np.where(z_data > 0, z_data, 0)
    z_neg = np.where(z_data < 0, -z_data, 0)
    df_diff = compute_all_zmap_correlations(z_pos, z_neg, term_maps, con_img, n_boot=10000, fdr_alpha=0.05, ci_alpha=0.05, n_jobs=-1)

    key = pat_name.split()[0].lower()
    df_pos = df_diff[["term", "r_pos"]].rename(columns={"r_pos": "r"})
    df_neg = df_diff[["term", "r_neg"]].rename(columns={"r_neg": "r"})
    all_pos[key] = df_pos
    all_neg[key] = df_neg
    all_diff[key] = df_diff

    df_pos.to_csv(os.path.join(OUT_ROOT, f"{pat_name}_term_corr_positive.csv"), index=False)
    df_neg.to_csv(os.path.join(OUT_ROOT, f"{pat_name}_term_corr_negative.csv"), index=False)
    df_diff.to_csv(os.path.join(OUT_ROOT, f"{pat_name}_term_corr_difference.csv"), index=False)
    plot_correlations(df_pos, df_neg, df_diff, run_id=pat_name, out_fig=os.path.join(OUT_ROOT, f"{pat_name}_term_correlations.png"))
    plot_difference(df_diff, run_id=pat_name, out_fig=os.path.join(OUT_ROOT, f"{pat_name}_term_correlation_differences.png"))

generate_latex_multicolumn_table(all_diff, os.path.join(OUT_ROOT, "rsa_searchlight_diff.tex"), "diff",
                                 caption="RSA searchlight results for expert–novice difference in correlation with term maps.",
                                 label="tab:rsa_searchlight_diff")
generate_latex_multicolumn_table(all_pos, os.path.join(OUT_ROOT, "rsa_searchlight_pos.tex"), "pos",
                                 caption="RSA searchlight results for positive z-maps (experts only).",
                                 label="tab:rsa_searchlight_expert_pos")
generate_latex_multicolumn_table(all_neg, os.path.join(OUT_ROOT, "rsa_searchlight_neg.tex"), "neg",
                                 caption="RSA searchlight results for negative z-maps (novices > experts).",
                                 label="tab:rsa_searchlight_expert_neg")
logger.info("Done. Results: %s", OUT_ROOT)
