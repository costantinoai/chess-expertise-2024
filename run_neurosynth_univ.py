#!/usr/bin/env python3
"""Univariate Neurosynth correlation analysis (root runner).

Orchestrates loading term maps and brain maps, masking, plotting, and
term correlations. Results saved under results/<RUN>_neurosynth-full.
"""

import os
import logging
from glob import glob
import numpy as np

from common.common_utils import create_run_id, create_output_directory, save_script_to_file, add_file_logger
from neurosynth.modules.io_utils import load_term_maps
from neurosynth.modules.univariate_utils import (
    plot_surface_map,
    plot_surface_map_flat,
    t_to_two_tailed_z,
)
from common.niimg_utils import load_and_mask_imgs
from neurosynth.modules.stats_utils import compute_all_zmap_correlations, save_latex_correlation_tables
from neurosynth.modules.plot_utils import plot_correlations, plot_difference, plot_map
from neurosynth.modules.config import BRAIN_CMAP

# Config (can adjust as needed)
TERM_DIR = 'neurosynth/data/terms'
BRAIN_DIR = 'neurosynth/data/smooth4'
MIN_VOX = 1e-5
DOF = 38

RUN_ID = create_run_id()
OUT_ROOT = os.path.join('results', f"{RUN_ID}_neurosynth-full")
create_output_directory(OUT_ROOT)
save_script_to_file(OUT_ROOT)
add_file_logger(os.path.join(OUT_ROOT, 'console.log'), level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("=== Starting Neurosynth Correlation Analysis ===")

term_maps = load_term_maps(TERM_DIR)

for filepath in sorted(glob(os.path.join(BRAIN_DIR, '*.nii'))):
    base = os.path.splitext(os.path.basename(filepath))[0]
    out_dir = os.path.join(OUT_ROOT, base)
    create_output_directory(out_dir)
    logger.info("Processing: %s", base)

    masked_imgs, brain_mask = load_and_mask_imgs([filepath])
    ref_img = masked_imgs[0]
    tmap = ref_img.get_fdata()
    z_data = t_to_two_tailed_z(tmap, dof=DOF)

    plot_map(
        z_data,
        ref_img,
        title=base,
        outpath=os.path.join(OUT_ROOT, f'{base}_glass.png'),
        thresh=MIN_VOX,
    )
    plot_surface_map(ref_img, title=base, threshold=MIN_VOX, output_file=os.path.join(OUT_ROOT, f'{base}_surface.png'), cmap=BRAIN_CMAP)
    plot_surface_map_flat(ref_img, title=base, threshold=MIN_VOX, output_file=os.path.join(OUT_ROOT, f'{base}_surface_flat.png'), cmap=BRAIN_CMAP)

    z_pos = np.where(z_data > 0, z_data, 0)
    z_neg = np.where(z_data < 0, -z_data, 0)

    df_pos, df_neg, df_diff = compute_all_zmap_correlations(z_pos, z_neg, term_maps, ref_img=ref_img)
    df_pos.to_csv(os.path.join(OUT_ROOT, f'{base}_term_corr_positive.csv'), index=False)
    df_neg.to_csv(os.path.join(OUT_ROOT, f'{base}_term_corr_negative.csv'), index=False)
    df_diff.to_csv(os.path.join(OUT_ROOT, f'{base}_term_corr_difference.csv'), index=False)
    save_latex_correlation_tables(df_pos, df_neg, df_diff, out_dir=OUT_ROOT, run_id=base)

    plot_correlations(df_pos, df_neg, df_diff, run_id=base, out_fig=os.path.join(OUT_ROOT, f'{base}_term_correlations.png'))
    plot_difference(df_diff, run_id=base, out_fig=os.path.join(OUT_ROOT, f'{base}_term_correlation_differences.png'))

logger.info("All analyses complete. Output: %s", OUT_ROOT)

