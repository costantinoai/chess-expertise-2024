import os
import glob
import numpy as np
import pickle
import pandas as pd

# ==== CONFIGURATION ====
# analyses = ["svm"]
analyses = ["rsa_corr"]
MVPA_RESULTS_PATHS = [
    (
        "/data/projects/chess/data/BIDS/derivatives/mvpa/20250402-191833_glasser_cortices_bilateral",
        "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-rois/results/glasser_cortex_bilateral",
    ),
]
regressor_mapping = {
    # "checkmate": "Checkmate vs. Non-checkmate boards",
    # "stimuli_half": "Pairwise checkmate boards",
    # "stimuli": "Pairwise all boards",
    "motif_half": "Motifs (Checkmate boards only)",
    "check_n_half": "Moves to Checkmate",
    # "side_half": "King Position (L-R)",
    "categories_half": "Strategy (Checkmate boards)",
    # "categories": "Strategy (All boards)",
    # "visualStimuli": "Visual Similarity",
    "total_pieces_half": "Total Pieces (Checkmate)",
    "legal_moves_half": "Legal Moves (Checkmate)",
    # "total_pieces": "Total Pieces",
    # "legal_moves": "Legal Moves",
    # "difficulty_half": "Board Difficulty",
    # "first_piece_half": "First Piece to Move",
    # "checkmate_piece_half": "Checkmate Piece",
}

# ==== FUNCTIONS ====

def find_single_file(directory, pattern):
    matches = glob.glob(os.path.join(directory, pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected 1 file matching {pattern} in {directory}, found {len(matches)}.")
    return matches[0]

def load_results(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def extract_stats(result_dict, regressor):
    stats = result_dict["experts_vs_nonexperts"][regressor]
    rows = []
    for roi, data in stats.items():
        mean_diff = data["mean_diff"]
        ci_low, ci_high = data["CI95"]
        p_val = data["p_uncorrected"]
        p_fdr = data["p_fdr"]
        reject = data["fdr_reject"]

        rows.append({
            "roi": roi,
            "mean_diff": mean_diff,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_val": p_val,
            "p_fdr": p_fdr,
            "sig_fdr": reject,
        })
    return pd.DataFrame(rows)

def print_latex_table(df, analysis, regressor, label):
    logging.info("\n\\textbf{%s (%s)}\\\\", label, analysis.upper())
    logging.info("\\begin{tabular}{lrrrrr}")
    logging.info("ROI & $M$ & 95\\%% CI & $p$ & $p_\\text{FDR}$ & Sig. \\\\")
    logging.info("\\hline")
    for _, row in df.iterrows():
        roi = row["roi"]
        m = row["mean_diff"]
        ci = f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        p = f"{row['p_val']:.3g}"
        pfdr = f"{row['p_fdr']:.3g}"
        sig = "*" if row["sig_fdr"] else ""
        logging.info("%s & %.3f & %s & %s & %s & %s \\\\", roi, m, ci, p, pfdr, sig)
    logging.info("\\end{tabular}\n")

# ==== MAIN LOOP ====
for analysis in analyses:
    for result_path, roi_annot_path in MVPA_RESULTS_PATHS:
        try:
            group_dir = os.path.join(result_path, analysis, "*group")
            pickle_path = find_single_file(group_dir, "*.pkl")
            results = load_results(pickle_path)
            regressors = sorted(set(r for c in results.values() for r in c.keys()))
            for reg in regressors:
                if reg not in regressor_mapping.keys():
                    continue
                df = extract_stats(results, reg)
                label = regressor_mapping.get(reg, reg)
                print_latex_table(df, analysis, reg, label)
        except Exception as e:
            logging.error("Failed to process %s / %s: %s", analysis, reg, e)
import logging
