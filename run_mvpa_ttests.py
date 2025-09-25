#!/usr/bin/env python3
"""Run MVPA group t-tests from repo root."""

import os
import glob
import pickle
import logging
import pandas as pd
from natsort import natsorted

from common.common_utils import create_run_id, save_script_to_file
from common.logging_utils import setup_logging
from config import MVPA_RESULTS_ROOT
from mvpa.mvpa_ttest import load_mvpa_subject, run_stats_on_df

RUN_ID = create_run_id()
OUT_ROOT = os.path.join("results", f"{RUN_ID}_mvpa_ttests")
os.makedirs(OUT_ROOT, exist_ok=True)
setup_logging(); setup_logging(log_file=os.path.join(OUT_ROOT, "mvpa_ttests.log"))
save_script_to_file(OUT_ROOT)

def _candidate_dirs(root: str, subdir: str) -> list[str]:
    patterns = [os.path.join(root, subdir), os.path.join(root, "*", subdir)]
    dirs = []
    for pat in patterns:
        dirs.extend(glob.glob(pat))
    return natsorted([d for d in dirs if os.path.isdir(d)])

for mvpa_root in [str(MVPA_RESULTS_ROOT)]:
    for mvpa_subdir in ("svm", "rsa_corr"):
        for mvpa_dir in _candidate_dirs(mvpa_root, mvpa_subdir):
            subject_files = natsorted(glob.glob(os.path.join(mvpa_dir, "*/*.tsv")))
            if not subject_files:
                logging.warning("No subject TSV files under %s", mvpa_dir)
                continue

            df = pd.concat([load_mvpa_subject(f) for f in subject_files], ignore_index=True)
            if "rsa" in mvpa_subdir and "layer" in df.columns:
                df = df.rename(columns={"layer": "target"})
                df = df[~df["target"].isin(["stimuli", "stimuli_half"])]

            results = run_stats_on_df(df, method=mvpa_subdir)
            results_out = os.path.join(mvpa_dir, f"{create_run_id()}_group")
            os.makedirs(results_out, exist_ok=True)
            with open(os.path.join(results_out, "ttest_group_results.pkl"), "wb") as f:
                pickle.dump(results, f)
            logging.info("Saved group t-test results under: %s", results_out)
