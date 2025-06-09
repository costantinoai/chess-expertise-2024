#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 22:19:02 2025

@author: costantino_ai
"""

import os
import logging
from glob import glob

from modules.config          import LEVELS_MAPS
from modules.io_utils        import load_term_maps, load_nifti
from modules.stats_utils     import split_and_convert_t_to_z, compute_all_zmap_correlations, save_latex_correlation_tables
from modules.plot_utils      import plot_term_maps, plot_map, plot_correlations, plot_difference

logging.basicConfig(format="[%(levelname)s %(asctime)s] %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    term_dir    = 'data/terms'
    brain_dir   = 'data/smooth4'
    result_root = 'results/smooth4-new'
    os.makedirs(result_root, exist_ok=True)

    term_maps = load_term_maps(term_dir)
    plot_term_maps(term_maps, os.path.join(result_root, 'term_maps'))

    for filepath in glob(os.path.join(brain_dir, '*.nii')):
        run_id = os.path.splitext(os.path.basename(filepath))[0]
        parts = run_id.split('_')
        level1 = LEVELS_MAPS[parts[1]]
        level2 = LEVELS_MAPS[parts[2]]
        subtitle = f"{level1} | {level2}"
        out_dir = os.path.join(result_root, run_id)
        os.makedirs(out_dir, exist_ok=True)

        logger.info(f"Processing: {run_id}")
        ref_img = load_nifti(filepath)
        tmap    = ref_img.get_fdata()
        dof     = 38 if "exp>" in filepath else 19

        # Convert to z-maps
        z_pos, z_neg = split_and_convert_t_to_z(tmap, dof)

        # Glass brain plots
        plot_map(tmap, ref_img,
                  f"{run_id}: T-map",
                  os.path.join(out_dir, f"tmap_{run_id}.png"))
        plot_map(z_pos, ref_img,
                  f"{run_id}: Positive z-map",
                  os.path.join(out_dir, f"zpos_{run_id}.png"))
        plot_map(z_neg, ref_img,
                  f"{run_id}: Negative z-map",
                  os.path.join(out_dir, f"zneg_{run_id}.png"))

        # Compute all correlations and differences
        df_pos, df_neg, df_diff = compute_all_zmap_correlations(
            z_pos, z_neg, term_maps, ref_img,
            n_boot=10000, fdr_alpha=0.05, ci_alpha=0.05
        )

        # Plot paired correlations
        corr_png = os.path.join(out_dir, f"correlations_{run_id}.png")
        corr_csv = os.path.join(out_dir, f"correlations_{run_id}.csv")
        plot_correlations(df_pos, df_neg, df_diff, corr_png, corr_csv, subtitle)

        # Plot differences
        diff_png = os.path.join(out_dir, f"diff_correlations_{run_id}.png")
        plot_difference(df_diff, diff_png, subtitle)

        # Save LaTeX tables
        save_latex_correlation_tables(
            df_pos, df_neg, df_diff,
            run_id=run_id,
            out_dir=os.path.join(out_dir, "latex_tables")
        )

    logger.info("All analyses complete.")

if __name__ == '__main__':
    main()
