#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run representational connectivity analysis for all subjects."""

import os
import sys
import numpy as np
sys.path.append('/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-mvpa')

from modules import (
    EXPERT_SUBJECTS,
    NONEXPERT_SUBJECTS,
)
from modules.helpers import (
    create_run_id,
    save_script_to_file,
    OutputLogger,
)
#from logging_utils import setup_logging

from rc_analysis import (
    compute_subject_matrix,
    fisher_z,
    group_statistics,
    group_difference,
    ATLAS_FILE,
)
from plotting import plot_connectivity_matrix

#logger = setup_logging()

def create_output_directory(directory_path):
    """
    Creates an output directory at the specified path.

    Parameters:
    - directory_path (str): The path where the output directory will be created.

    The function attempts to create a directory at the given path.
    It logs the process, indicating whether the directory creation was successful or if any error occurred.
    If the directory already exists, it will not be created again, and this will also be logged.
    """
    # Log the attempt to create the output directory

    # Check if directory already exists to avoid overwriting
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)
        # Log the successful creation

def run_representational_connectivity(out_root: str | None = None,
                                       atlas_path: str = ATLAS_FILE) -> None:
    """Compute subject- and group-level representational connectivity."""
    if out_root is None:
        out_root = os.path.join(
            "results",
            f"{create_run_id()}_representational-connectivity",
        )
    create_output_directory(out_root)
    save_script_to_file(out_root)
    log_file = os.path.join(out_root, "console.log")

    subjects = list(EXPERT_SUBJECTS) + list(NONEXPERT_SUBJECTS)
    subj_dir = os.path.join(out_root, "subjects")
    os.makedirs(subj_dir, exist_ok=True)

    subject_mats: dict[str, np.ndarray] = {}

    with OutputLogger(True, log_file):
        for sub in subjects:
            mat, _ = compute_subject_matrix(sub, atlas_path)
            subject_mats[sub] = mat
            np.save(os.path.join(subj_dir, f"sub-{sub}_matrix.npy"), mat)
            plot_connectivity_matrix(
                mat,
                title=f"Subject {sub}",
                out_path=os.path.join(subj_dir, f"sub-{sub}_matrix.png"),
            )
            # logger.info("Finished subject %s", sub)

        # Convert r to z
        expert_z = [fisher_z(subject_mats[s]) for s in EXPERT_SUBJECTS]
        novice_z = [fisher_z(subject_mats[s]) for s in NONEXPERT_SUBJECTS]

        # Group stats
        exp_mean, exp_t, exp_p, exp_p_fdr, exp_sig = group_statistics(expert_z)
        nov_mean, nov_t, nov_p, nov_p_fdr, nov_sig = group_statistics(novice_z)
        diff_mean, diff_t, diff_p, diff_p_fdr, diff_sig = group_difference(
            expert_z, novice_z
        )

        group_dir = os.path.join(out_root, "group")
        os.makedirs(group_dir, exist_ok=True)

        np.savez(
            os.path.join(group_dir, "experts.npz"),
            mean=exp_mean,
            t=exp_t,
            p=exp_p,
            p_fdr=exp_p_fdr,
            sig=exp_sig,
        )
        np.savez(
            os.path.join(group_dir, "novices.npz"),
            mean=nov_mean,
            t=nov_t,
            p=nov_p,
            p_fdr=nov_p_fdr,
            sig=nov_sig,
        )
        np.savez(
            os.path.join(group_dir, "difference.npz"),
            mean=diff_mean,
            t=diff_t,
            p=diff_p,
            p_fdr=diff_p_fdr,
            sig=diff_sig,
        )

        # Plots
        plot_connectivity_matrix(
            exp_mean,
            title="Experts",
            out_path=os.path.join(group_dir, "experts.png"),
            mask=exp_sig,
        )
        plot_connectivity_matrix(
            nov_mean,
            title="Novices",
            out_path=os.path.join(group_dir, "novices.png"),
            mask=nov_sig,
        )
        plot_connectivity_matrix(
            diff_mean,
            title="Experts - Novices",
            out_path=os.path.join(group_dir, "difference.png"),
            mask=diff_sig,
        )

        # logger.info("Representational connectivity analysis complete.")


if __name__ == "__main__":
    run_representational_connectivity()
