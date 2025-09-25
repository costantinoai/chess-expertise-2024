#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Univariate Neurosynth analysis main orchestration.

This module implements the end-to-end workflow formerly embedded in
`neurosynth/univariate_neurosynth.py`. Scripts should import and call `main()`.
"""
from __future__ import annotations

import os
import logging
import numpy as np
from glob import glob


from modules.io_utils import load_term_maps
from modules.stats_utils import (
    save_latex_correlation_tables,
)
from modules.plot_utils import plot_correlations, plot_difference, plot_map
from modules.config import BRAIN_CMAP

from common.common_utils import (
    create_run_id,
    create_output_directory,
    save_script_to_file,
    add_file_logger,
)

from neurosynth.modules.univariate_utils import (
    plot_surface_map,
    plot_surface_map_flat,
    t_to_two_tailed_z,
)
from common.niimg_utils import load_and_mask_imgs

logger = logging.getLogger(__name__)


def main(
    term_dir: str = 'data/terms',
    brain_dir: str = 'data/smooth4',
    dof: int = 38,
    min_voxel_value: float = 1e-5,
) -> None:
    run_id = create_run_id()
    result_root = os.path.join('results', f"{run_id}_neurosynth-full")
    create_output_directory(result_root)
    save_script_to_file(result_root)
    add_file_logger(os.path.join(result_root, 'console.log'), level=logging.INFO)
    logger.info("=== Starting Neurosynth Correlation Analysis ===")

    term_maps = load_term_maps(term_dir)

    # Iterate over brain maps (.nii preferred here; adapt as needed)
    for filepath in sorted(glob(os.path.join(brain_dir, '*.nii'))):
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_dir = os.path.join(result_root, base)
        create_output_directory(out_dir)
        logger.info("Processing: %s", base)

        # Load and mask image; use the first (and only) image for reference
        masked_imgs, brain_mask = load_and_mask_imgs([filepath])
        ref_img = masked_imgs[0]
        tmap = ref_img.get_fdata()
        z_data = t_to_two_tailed_z(tmap, dof=dof)

        # Glass brain
        plot_map(
            z_data,
            ref_img,
            title=base,
            outpath=os.path.join(result_root, f'{base}_glass.png'),
            thresh=min_voxel_value,
        )

        # Surface plots
        plot_surface_map(
            ref_img,
            title=base,
            threshold=min_voxel_value,
            output_file=os.path.join(result_root, f'{base}_surface.png'),
            cmap=BRAIN_CMAP,
        )
        plot_surface_map_flat(
            ref_img,
            title=base,
            threshold=min_voxel_value,
            output_file=os.path.join(result_root, f'{base}_surface_flat.png'),
            cmap=BRAIN_CMAP,
        )

        # Split z into positive and negative (flip neg to positive magnitude)
        z_pos = np.where(z_data > 0, z_data, 0)
        z_neg = np.where(z_data < 0, -z_data, 0)

        # Correlations (point-estimates only for speed; swap to bootstrap if desired)
        from modules.stats_utils import compute_all_zmap_correlations
        df_pos, df_neg, df_diff = compute_all_zmap_correlations(
            z_pos, z_neg, term_maps, ref_img=ref_img,
        )

        # Save correlation data and plots
        df_pos.to_csv(os.path.join(result_root, f'{base}_term_corr_positive.csv'), index=False)
        df_neg.to_csv(os.path.join(result_root, f'{base}_term_corr_negative.csv'), index=False)
        df_diff.to_csv(os.path.join(result_root, f'{base}_term_corr_difference.csv'), index=False)
        save_latex_correlation_tables(df_pos, df_neg, df_diff, out_dir=result_root, run_id=base)

        plot_correlations(
            df_pos, df_neg, df_diff,
            run_id=base,
            out_fig=os.path.join(result_root, f'{base}_term_correlations.png')
        )
        plot_difference(
            df_diff,
            run_id=base,
            out_fig=os.path.join(result_root, f'{base}_term_correlation_differences.png')
        )

    logger.info("All analyses complete. Output: %s", result_root)


if __name__ == "__main__":  # pragma: no cover
    main()
