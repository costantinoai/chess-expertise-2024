#!/usr/bin/env python3
"""Run behavioural between-group/inter-correlation from root.

Orchestrates loading, per-subject processing, reliability computation,
and bar plotting using functions from behavioural.bh_fmri_intercorr.
"""

import os
import logging
import multiprocessing
import numpy as np
import pandas as pd

from common.common_utils import create_run_id, save_script_to_file
from common.logging_utils import setup_logging
from config import SOURCEDATA_PATH, PARTICIPANTS_XLSX
import behavioural.bh_fmri_intercorr as bi

RUN_ID = create_run_id()
OUT_ROOT = os.path.join("results", f"{RUN_ID}_bh_fmri_intercorr")
os.makedirs(OUT_ROOT, exist_ok=True)
setup_logging()
setup_logging(log_file=os.path.join(OUT_ROOT, "bh_fmri_intercorr.log"))
save_script_to_file(OUT_ROOT)

sourcedata_root = str(SOURCEDATA_PATH)
participants_list, (num_exp, num_non) = bi.load_participants(
    participants_xlsx_path=str(PARTICIPANTS_XLSX),
    sourcedata_root=sourcedata_root,
)
logging.info("Number of Experts: %s | Number of Non-Experts: %s", num_exp, num_non)

trial_columns = [
    "sub_id", "run", "run_trial_n", "stim_id",
    "stim_onset_real", "response", "stim_onset_expected",
    "button_mapping",
]

experts_df = pd.DataFrame([], columns=trial_columns)
novices_df = pd.DataFrame([], columns=trial_columns)
rdms_dict = {"Experts": {}, "Novices": {}}

n_cpu = max(1, multiprocessing.cpu_count() - 1)
logging.info(f"Using {n_cpu} CPU cores for parallel processing.")
args = [(sub_id, is_expert, sourcedata_root, trial_columns) for sub_id, is_expert in participants_list]
with multiprocessing.Pool(processes=n_cpu) as pool:
    results = pool.map(bi.worker_process, args)

logging.info("Processing subject data...")
for sub_id, single_df, is_expert, rdm_ind, dsm_ind in results:
    if rdm_ind is None or single_df is None:
        continue
    if is_expert:
        experts_df = pd.concat([experts_df, single_df], ignore_index=True)
        rdms_dict["Experts"][sub_id] = rdm_ind
    else:
        novices_df = pd.concat([novices_df, single_df], ignore_index=True)
        rdms_dict["Novices"][sub_id] = rdm_ind

trial_df_dict = {
    "Experts": {sid: experts_df[experts_df["sub_id"] == sid] for sid in experts_df["sub_id"].unique()},
    "Novices": {sid: novices_df[novices_df["sub_id"] == sid] for sid in novices_df["sub_id"].unique()},
}

results = bi.compute_reliabilities_from_trials(trial_df_dict)

summary = {}
for group in ["Experts", "Novices"]:
    summary[group] = {}
    for typ in ["within", "between"]:
        data = pd.Series(results[group][typ])
        ttest = bi.pg.ttest(data, 0, alternative='two-sided')
        row = ttest.iloc[0]
        summary[group][typ] = {
            "mean": data.mean(),
            "ci95%": (row['CI95%'][0], row['CI95%'][1]),
            "p": row['p-val']
        }

p_vals_between = {}
for typ in ["within", "between"]:
    data_exp = pd.Series(results["Experts"][typ])
    data_nov = pd.Series(results["Novices"][typ])
    ttest = bi.pg.ttest(data_exp, data_nov, paired=False, alternative='two-sided')
    row = ttest.iloc[0]
    p_vals_between[typ] = {
        "p": row["p-val"],
        "mean_diff": data_exp.mean() - data_nov.mean(),
        "ci95%": (row['CI95%'][0], row['CI95%'][1])
    }

labels = ["Within", "Between"]
x = np.arange(len(labels))
width = 0.35
means_exp = [summary["Experts"]["within"]["mean"], summary["Experts"]["between"]["mean"]]
means_nov = [summary["Novices"]["within"]["mean"], summary["Novices"]["between"]["mean"]]
ci_exp = np.array([[summary["Experts"][k]["mean"] - summary["Experts"][k]["ci95%"][0], summary["Experts"][k]["ci95%"][1] - summary["Experts"][k]["mean"]] for k in ["within", "between"]]).T
ci_nov = np.array([[summary["Novices"][k]["mean"] - summary["Novices"][k]["ci95%"][0], summary["Novices"][k]["ci95%"][1] - summary["Novices"][k]["mean"]] for k in ["within", "between"]]).T

plot_data = {
    "x": x,
    "width": width,
    "means_exp": means_exp,
    "means_nov": means_nov,
    "ci_exp": ci_exp,
    "ci_nov": ci_nov,
    "summary": summary,
    "p_vals_between": p_vals_between,
}

bi.plot_reliability_bars(plot_data, run_id=RUN_ID)
