#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main entry point for the Chess-Neurosynth analysis pipeline.

The script iterates over all statistical maps in ``brain_dir`` and for each map
performs the following high level steps:

1. Convert the T-map to one-tailed positive/negative Z-maps.
2. Plot the raw T-map and the derived Z-maps for visual inspection.
3. Correlate each Z-map with a set of Neurosynth term maps using Pearson
   correlation.  Bootstrapping is used to obtain confidence intervals and to
   assess the significance of the difference between the two correlations.
4. Produce bar plots summarising the correlations and their differences.
5. Save LaTeX tables with the numeric results.

All outputs are written under ``result_root`` organised by run identifier.
"""

import os
import logging
from glob import glob

from modules.run_utils import (
    create_run_id,
    create_output_directory,
    save_script_to_file,
    OutputLogger,
)
from modules.config import setup_logging

from modules.config          import LEVELS_MAPS
from modules.io_utils        import load_term_maps, load_nifti
from modules.stats_utils     import split_and_convert_t_to_z, compute_all_zmap_correlations, save_latex_correlation_tables
from modules.plot_utils      import plot_term_maps, plot_map, plot_correlations, plot_difference

# Configure a consistent logger
setup_logging()
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 1) Load meta-analytic term maps.  These serve as the reference
#    "Neurosynth" activation patterns that we correlate each brain map with.
# ------------------------------------------------------------------
term_dir    = 'data/terms'
brain_dir   = 'data/smooth4'
result_root = os.path.join('results', f"{create_run_id()}_neurosynth-full")
create_output_directory(result_root)
save_script_to_file(result_root)
out_text_file = os.path.join(result_root, 'console.log')

term_maps = load_term_maps(term_dir)
# plot_term_maps(term_maps, os.path.join(result_root, 'term_maps'))

# ------------------------------------------------------------------
# 2) Iterate over all statistical maps to correlate.
# ------------------------------------------------------------------
with OutputLogger(True, out_text_file):
    for filepath in glob(os.path.join(brain_dir, '*.nii')):
        run_id = os.path.splitext(os.path.basename(filepath))[0]
        parts = run_id.split('_')
        level1 = LEVELS_MAPS[parts[1]]
        level2 = LEVELS_MAPS[parts[2]]
        subtitle = f"{level1} | {level2}"
        out_dir = os.path.join(result_root, run_id).replace(">", "-gt-")
        create_output_directory(out_dir)

        logger.info(f"Processing: {run_id}")
        # Load the image and degrees of freedom (depends on the contrast).
        ref_img = load_nifti(filepath)
        tmap    = ref_img.get_fdata()
        dof     = 38 if "exp>" in filepath else 19

    # --------------------------------------------------------------
    # 2a) Convert the T-map into one-tailed Z-maps.  Positive values
    #     indicate expert > novice effects, negative values the opposite.
    # --------------------------------------------------------------
        z_pos, z_neg = split_and_convert_t_to_z(tmap, dof)

    # --------------------------------------------------------------
    # 2b) Visualise the maps so that we can spot any obvious issues.
    # --------------------------------------------------------------
        plot_map(tmap, ref_img,
                 f"{run_id}: T-map",
                 os.path.join(out_dir, f"tmap_{run_id}.png"))
        plot_map(z_pos, ref_img,
                 f"{run_id}: Positive z-map",
                 os.path.join(out_dir, f"zpos_{run_id}.png"))
        plot_map(z_neg, ref_img,
                 f"{run_id}: Negative z-map",
                 os.path.join(out_dir, f"zneg_{run_id}.png"))

    # --------------------------------------------------------------
    # 2c) Correlate the Z-maps with each term map.  The helper function
    #     returns DataFrames with bootstrap CIs and FDR-corrected p-values
    #     for both individual correlations and their difference.
    # --------------------------------------------------------------
        df_pos, df_neg, df_diff = compute_all_zmap_correlations(
            z_pos, z_neg, term_maps, ref_img,
            n_boot=1000, fdr_alpha=0.05, ci_alpha=0.05,
            n_jobs=-1
        )

    # 2d) Produce bar plots for the individual correlations and their
    #     difference.
        corr_png = os.path.join(out_dir, f"correlations_{run_id}.png")
        corr_csv = os.path.join(out_dir, f"correlations_{run_id}.csv")
        plot_correlations(df_pos, df_neg, df_diff, corr_png, corr_csv, subtitle)

    # Additional visualisation focusing solely on the differences.
        diff_png = os.path.join(out_dir, f"diff_correlations_{run_id}.png")
        plot_difference(df_diff, diff_png, subtitle)

    # Export nicely formatted LaTeX tables so the numbers can be directly
    # included in the manuscript or supplementary materials.
        save_latex_correlation_tables(
            df_pos, df_neg, df_diff,
            run_id=run_id,
            out_dir=os.path.join(out_dir, "latex_tables")
        )

logger.info("All analyses complete.")
