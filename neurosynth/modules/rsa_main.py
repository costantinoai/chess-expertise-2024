#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Group-level RSA → Neurosynth term correlation (modular).

Thin orchestration that:
- finds subject searchlight maps per pattern
- builds a second-level GLM (Experts vs Novices)
- saves the contrast z-map and plots (glass + flat surface)
- correlates directional z-maps (pos/neg) with term maps
- writes figures + CSV + LaTeX tables

Relies on existing helpers from `neurosynth/modules` and `common/`.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
 # no direct nilearn.image imports needed here
from nilearn.glm.second_level import SecondLevelModel

from neurosynth.modules.io_utils import load_term_maps
from common.brain_plotting import plot_surface_map_flat, plot_map
from common.stats_plotting import plot_correlations, plot_difference
from neurosynth.modules.stats_utils import compute_all_zmap_correlations
from common.common_utils import create_run_id, create_output_directory, save_script_to_file, add_file_logger
from config import DERIVATIVES_PATH, EXPERTS, NONEXPERTS
from common.niimg_utils import find_nifti_files, fisher_z_maps, load_and_mask_imgs, build_design_matrix


logger = logging.getLogger(__name__)


def _split_by_group(files: Iterable[str], expert_ids: Iterable[str], novice_ids: Iterable[str]) -> Tuple[list[str], list[str]]:
    exp, nov = [], []
    for sid in expert_ids:
        exp += [f for f in files if f"sub-{sid}" in os.path.basename(f)]
    for sid in novice_ids:
        nov += [f for f in files if f"sub-{sid}" in os.path.basename(f)]
    return exp, nov


def _masked(imgs_list: list):
    # Use neurosynth brain mask function for consistency
    from neurosynth.modules.stats_utils import get_brain_mask
    return load_and_mask_imgs(imgs_list, mask_func=get_brain_mask)


def main(
    data_dir: str | Path | None = None,
    term_dir: str | Path | None = None,
    results_root: str | Path = "results",
    smoothing_mm: float | None = None,
    min_voxel_value: float = 1e-5,
) -> None:
    logging.basicConfig(level=logging.INFO)

    data_dir = str(data_dir or (Path(DERIVATIVES_PATH) / "rsa_searchlight"))
    term_dir = str(term_dir or (Path(__file__).resolve().parents[1] / "data" / "terms"))

    patterns = [
        "searchlight_checkmate",
        "searchlight_strategy",
        "searchlight_visualSimilarity",
    ]
    pattern_clean = {
        "searchlight_checkmate": "Checkmate | RSA searchlight",
        "searchlight_strategy": "Strategy | RSA searchlight",
        "searchlight_visualSimilarity": "Visual Similarity | RSA searchlight",
    }

    run_id = create_run_id()
    results_dir = os.path.join(str(results_root), f"{run_id}_neurosynth-rsa-searchlight")
    create_output_directory(results_dir)
    save_script_to_file(results_dir)
    add_file_logger(os.path.join(results_dir, "console.log"))

    logger.info("Results directory: %s", results_dir)
    logger.info("Loading term maps from: %s", term_dir)
    term_maps = load_term_maps(term_dir)

    all_pos: dict[str, pd.DataFrame] = {}
    all_neg: dict[str, pd.DataFrame] = {}
    all_diff: dict[str, pd.DataFrame] = {}

    for pat in patterns:
        pat_name = pattern_clean[pat]
        logger.info("--- Pattern: %s ---", pat_name)

        r_files = find_nifti_files(data_dir, pattern=pat)
        exp_files, nov_files = _split_by_group(r_files, EXPERTS, NONEXPERTS)
        z_exp, _ = _masked(fisher_z_maps(exp_files))
        z_nov, _ = _masked(fisher_z_maps(nov_files))
        z_all = z_exp + z_nov

        design = build_design_matrix(len(z_exp), len(z_nov))
        slm = SecondLevelModel(smoothing_fwhm=smoothing_mm, n_jobs=-1).fit(z_all, design_matrix=design)

        con_img = slm.compute_contrast("group", output_type="z_score")
        z_data = con_img.get_fdata()
        con_img_path = os.path.join(results_dir, f"{pat_name}_zmap_experts_gt_novices.nii.gz")
        con_img.to_filename(con_img_path)

        plot_map(z_data, con_img, title=f"{pat_name} | Experts>Novices", outpath=os.path.join(results_dir, f"{pat_name}_glass_experts_gt_novices.png"), thresh=min_voxel_value)
        plot_surface_map_flat(con_img, title=f"{pat_name} | Experts>Novices", threshold=min_voxel_value, output_file=os.path.join(results_dir, f"{pat_name}_surface_flat_experts_gt_novices.png"))

        z_pos = np.where(z_data > 0, z_data, 0)
        z_neg = np.where(z_data < 0, -z_data, 0)

        df_diff = compute_all_zmap_correlations(
            z_pos, z_neg, term_maps, con_img,
            n_boot=10000, fdr_alpha=0.05, ci_alpha=0.05, n_jobs=-1
        )

        key = pat_name.split()[0].lower()
        df_pos = df_diff[["term", "r_pos"]].rename(columns={"r_pos": "r"})
        df_neg = df_diff[["term", "r_neg"]].rename(columns={"r_neg": "r"})

        all_pos[key] = df_pos
        all_neg[key] = df_neg
        all_diff[key] = df_diff

        df_pos.to_csv(os.path.join(results_dir, f"{pat_name}_term_corr_positive.csv"), index=False)
        df_neg.to_csv(os.path.join(results_dir, f"{pat_name}_term_corr_negative.csv"), index=False)
        df_diff.to_csv(os.path.join(results_dir, f"{pat_name}_term_corr_difference.csv"), index=False)

        plot_correlations(df_pos, df_neg, df_diff, run_id=pat_name, out_fig=os.path.join(results_dir, f"{pat_name}_term_correlations.png"))
        plot_difference(df_diff, run_id=pat_name, out_fig=os.path.join(results_dir, f"{pat_name}_term_correlation_differences.png"))

    # Multicolumn LaTeX tables per regressor collection
    from neurosynth.modules.stats_utils import generate_latex_multicolumn_table
    generate_latex_multicolumn_table(all_diff, os.path.join(results_dir, "rsa_searchlight_diff.tex"), "diff",
                                     caption="RSA searchlight results for expert–novice difference in correlation with term maps.",
                                     label="tab:rsa_searchlight_diff")
    generate_latex_multicolumn_table(all_pos, os.path.join(results_dir, "rsa_searchlight_pos.tex"), "pos",
                                     caption="RSA searchlight results for positive z-maps (experts only).",
                                     label="tab:rsa_searchlight_expert_pos")
    generate_latex_multicolumn_table(all_neg, os.path.join(results_dir, "rsa_searchlight_neg.tex"), "neg",
                                     caption="RSA searchlight results for negative z-maps (novices > experts).",
                                     label="tab:rsa_searchlight_expert_neg")

    logger.info("Done. Results: %s", results_dir)


if __name__ == "__main__":
    main()
