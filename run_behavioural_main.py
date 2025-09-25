#!/usr/bin/env python3
"""Run the behavioural RDM main analysis (Experts vs. Novices) from repo root.

All orchestration happens here; behavioural/bh_fmri.py contains functions only.
"""

import os
import multiprocessing
import pandas as pd
import logging

from common.common_utils import create_run_id, save_script_to_file
from common.logging_utils import setup_logging
from config import SOURCEDATA_PATH, PARTICIPANTS_XLSX, CATEGORIES_XLSX
import behavioural.bh_fmri as bm

# Set up output directory and logging
RUN_ID = create_run_id()
OUT_ROOT = os.path.join("results", f"{RUN_ID}_bh_fmri")
os.makedirs(OUT_ROOT, exist_ok=True)
setup_logging()
setup_logging(log_file=os.path.join(OUT_ROOT, "bh_fmri.log"))
save_script_to_file(OUT_ROOT)
bm.set_output_dir(OUT_ROOT)

# Load participants
participants_list, (num_exp, num_non) = bm.load_participants(
    participants_xlsx_path=str(PARTICIPANTS_XLSX),
    sourcedata_root=str(SOURCEDATA_PATH),
)
logging.info("Number of Experts: %s | Number of Non-Experts: %s", num_exp, num_non)

trial_columns = [
    "sub_id", "run", "run_trial_n", "stim_id",
    "stim_onset_real", "response", "stim_onset_expected",
    "button_mapping",
]

experts_df = pd.DataFrame([], columns=trial_columns)
novices_df = pd.DataFrame([], columns=trial_columns)
experts_rdms, novices_rdms = [], []

sourcedata_root = str(SOURCEDATA_PATH)

if bm.MULTIPROCESS:
    n_cpu = max(1, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(processes=n_cpu) as pool:
        args = [
            (sub_id, is_expert, sourcedata_root, trial_columns)
            for sub_id, is_expert in participants_list
        ]
        results = pool.map(bm.worker_process, args)
    for sub_id, single_df, is_expert, rdm_ind, dsm_ind in results:
        if single_df is None:
            continue
        if is_expert:
            experts_rdms.append(rdm_ind)
            experts_df = pd.concat([experts_df, single_df], ignore_index=True)
        else:
            novices_rdms.append(rdm_ind)
            novices_df = pd.concat([novices_df, single_df], ignore_index=True)
else:
    for sub_id, is_expert in participants_list:
        logging.info("Processing participant: %s | Expert: %s", sub_id, is_expert)
        single_df = bm.process_single_participant(
            sub_id=sub_id,
            is_expert=is_expert,
            sourcedata_root=sourcedata_root,
            columns=trial_columns,
        )
        if single_df is None:
            continue
        pairwise_df = bm.create_pairwise_df(single_df)
        rdm_ind = bm.compute_symmetric_rdm(pairwise_df, "Individual", do_plot=False)
        dsm_ind = bm.compute_directional_dsm(pairwise_df, "Individual", do_plot=False)
        if is_expert:
            experts_rdms.append(rdm_ind)
            experts_df = pd.concat([experts_df, single_df], ignore_index=True)
        else:
            novices_rdms.append(rdm_ind)
            novices_df = pd.concat([novices_df, single_df], ignore_index=True)

# Group analyses
category_df = bm.load_stimulus_categories(cat_path=str(CATEGORIES_XLSX))
expert_rdm, expert_corrs, expert_cols = bm.analyze_group(
    df_group=experts_df, expertise_label="Experts", df_cat=category_df
)
novice_rdm, novice_corrs, novice_cols = bm.analyze_group(
    df_group=novices_df, expertise_label="Novices", df_cat=category_df
)

if expert_cols != novice_cols:
    raise ValueError("Mismatch in column labels between expert and novice results!")

bm.plot_model_behavior_correlations(
    expert_results=expert_corrs, novice_results=novice_corrs, column_labels=expert_cols
)

pretty_labels = ["Visual Similarity", "Strategy", "Checkmate"]
stats_df = pd.DataFrame({
    "Dimension": pretty_labels,
    "r_Experts": [f"{tup[1]:.3f}" for tup in expert_corrs],
    "95% CI Experts": [f"[{tup[3]:.3f}, {tup[4]:.3f}]" for tup in expert_corrs],
    "r_Novices": [f"{tup[1]:.3f}" for tup in novice_corrs],
    "95% CI Novices": [f"[{tup[3]:.3f}, {tup[4]:.3f}]" for tup in novice_corrs],
})
logging.info("\n=== Behavioral RDM Correlations (Experts vs. Novices) ===\n%s", stats_df.to_string(index=False))

bm.plot_shared_colorbar(bm.CUSTOM_CMAP, vmin=-18, vmax=18)
