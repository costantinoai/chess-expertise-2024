#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:11:15 2025

@author: costantino_ai
"""

# participation_ratio_analysis.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load saved activations and compute Participation Ratio (PR) per layer.
"""
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from natsort import natsorted
import sys
sys.path.insert(0, "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/AlphazeroChess")

from modules.utils.helper_funcs import (
    create_run_id,
    save_script_to_file,
    create_output_directory
)

# Directory with pickles
ACTIVATIONS_DIR = "results/20250419-190918_extract-net-activations-alphavile_dataset-fmri"
OUTPUT_DIR      = f"results/{create_run_id()}_pr-alphavile"  # save plots alongside the pickles
create_output_directory(OUTPUT_DIR)
save_script_to_file(OUTPUT_DIR)


def participation_ratio(activations: np.ndarray) -> float:
    pca = PCA(n_components=None)
    pca.fit(activations)
    eig = pca.explained_variance_
    num = (eig.sum())**2
    den = (eig**2).sum()
    return num/den if den != 0.0 else 0.0

def main():
    # Load activations
    acts = {}
    for tag in ["trained","untrained"]:
        p = os.path.join(ACTIVATIONS_DIR, f"activations_model-{tag}_seed-0.pkl")
        with open(p,"rb") as f:
            acts[tag] = pickle.load(f)

    # Compute PR per layer
    pr_res = {"trained":{}, "untrained":{}}
    for tag, adict in acts.items():
        for layer, data in adict.items():
            X = np.vstack([v["activation"].ravel() for v in data.values()])
            pr_res[tag][layer] = participation_ratio(X)

    layers = natsorted(pr_res["trained"].keys())
    tr_vals = [pr_res["trained"][l] for l in layers]
    un_vals= [pr_res["untrained"][l] for l in layers]
    diff   = np.array(tr_vals) - np.array(un_vals)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(30,10))
    ax1.plot(layers, tr_vals,   marker='o', label="Trained")
    ax1.plot(layers, un_vals,  marker='o', label="Untrained")
    ax1.set_title("Participation Ratio by Layer")
    ax1.set_ylabel("PR"); ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', ls='--', alpha=0.5); ax1.legend()

    ax2.bar(layers, diff, color="purple", edgecolor='black', linewidth=0.5)
    ax2.axhline(0, ls='--', color='gray')
    ax2.set_title("ΔPR (Trained − Untrained)")
    ax2.set_ylabel("ΔParticipation Ratio"); ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', ls='--', alpha=0.5)

    fig.tight_layout(); fig.subplots_adjust(top=0.88)
    fig.savefig(os.path.join(OUTPUT_DIR, "participation_ratio_comparison.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
