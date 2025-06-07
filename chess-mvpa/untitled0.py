#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 09:59:21 2025

@author: costantino_ai
"""

import os
import numpy as np
import nibabel as nib
import joblib
from net2brain.feature_extraction import FeatureExtractor
from net2brain.rdm_creation import RDMCreator
from net2brain.rdm_creation import LayerRDM
from net2brain.evaluations.rsa import RSA
from net2brain.evaluations.plotting import Plotting
from net2brain.evaluations.encoding import Ridge_Encoding

# --------------------------- CONFIGURATION ---------------------------

# Beta maps for experts and non-experts
BETA_PATHS = {
    "expert": "/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-4_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM/2ndLevel_Experts_con_0002/spmT_0001.nii",
    "nonexpert": "/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-4_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM/2ndLevel_NonExperts_con_0002/spmT_0001.nii"
}

# Glasser atlas path
ATLAS_PATH = "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-rois/results/glasser_cortex_bilateral/glasser_cortex_bilateral.nii"

# DNN activation paths
DNN_ACTIVATIONS = {
    "trained": "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-dnn/results/20250419-190918_extract-net-activations-alphavile_dataset-fmri/activations_model-trained_seed-0.pkl",
    "untrained": "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-dnn/results/20250419-190918_extract-net-activations-alphavile_dataset-fmri/activations_model-untrained_seed-0.pkl"
}

# Path to raw stimulus images
STIMULI_PATH = "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-dataset-vis/data/stimuli"

OUTPUT_DIR = "results/encoding_results"
FEATURE_DIR = os.path.join(OUTPUT_DIR, "features")
ROI_DIR = os.path.join(OUTPUT_DIR, "rois")
MODEL_RDM_DIR = os.path.join(OUTPUT_DIR, "model_rdms")
BRAIN_RDM_DIR = os.path.join(OUTPUT_DIR, "brain_rdms")
ALEXNET_FEAT_DIR = os.path.join(FEATURE_DIR, "alexnet")
RSA_RESULT_DIR = os.path.join(OUTPUT_DIR, "rsa_results")
MIN_VOXELS = 10



# ------------------ EXTRACTION FUNCTIONS ------------------

def load_parcellated_beta(beta_path, atlas_path):
    beta = nib.load(beta_path).get_fdata()
    atlas = nib.load(atlas_path).get_fdata()
    rois = np.unique(atlas)[1:]
    return {int(r): beta[atlas == r].flatten() for r in rois}


def save_rois_as_npz(roi_betas, out_dir, group):
    os.makedirs(out_dir, exist_ok=True)
    for roi, vals in roi_betas.items():
        if len(vals) >= MIN_VOXELS:
            np.savez_compressed(os.path.join(out_dir, f"ROI-{roi}_{group}.npz"), vals)


def extract_and_save_alexnet(stimuli_path, out_dir, device='cpu'):
    fx = FeatureExtractor(model='AlexNet', netset='Standard', device=device)
    os.makedirs(out_dir, exist_ok=True)
    fx.extract(data_path=stimuli_path, save_path=out_dir, consolidate_per_layer=True)


# def save_dnn_features_as_npz(dnn_acts_by_stim, out_dir, model_name):
#     """
#     Save activations in Net2Brain-compatible format:
#     - One .npz file per layer
#     - Inside: one array per stimulus, key = stimulus name

#     Parameters:
#     - dnn_acts_by_stim: dict[stimulus] = {layer: activation}
#     """
#     from collections import defaultdict

#     # Reorganize: layer_name → {stimulus_name: activation}
#     layer_dict = defaultdict(dict)
#     for stim_name, layer_acts in dnn_acts_by_stim.items():
#         for layer, vec in layer_acts.items():
#             layer_dict[layer][stim_name] = np.asarray(vec).flatten().reshape(1, -1)


#     # Save to .npz files: each layer gets its own file
#     model_dir = os.path.join(out_dir, model_name)
#     os.makedirs(model_dir, exist_ok=True)

#     for layer_name, stim_data in layer_dict.items():
#         out_path = os.path.join(model_dir, f"{layer_name}.npz")
#         np.savez_compressed(out_path, **stim_data)

def save_dnn_features_as_npz(dnn_acts_by_stim, out_dir, model_name):
    """
    Save DNN activations in Net2Brain-compatible format:
    - One .npz file per stimulus
    - Inside: one array per layer, key = layer name

    Parameters:
    - dnn_acts_by_stim: dict[stimulus] = {layer: activation}
    """
    model_dir = os.path.join(out_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    for stim_name in sorted(dnn_acts_by_stim.keys()):
        layer_acts = dnn_acts_by_stim[stim_name]
        save_dict = {
            layer: np.asarray(vec).flatten().reshape(1, -1)
            for layer, vec in layer_acts.items()
        }
        out_path = os.path.join(model_dir, f"{stim_name}.npz")
        np.savez_compressed(out_path, **save_dict)


def load_dnn_by_image(path):
    data = joblib.load(path)
    layers = list(data.keys())
    stim_ids = sorted(data[layers[0]].keys())
    from collections import OrderedDict
    result = OrderedDict()
    for i in stim_ids:
        name = data[layers[0]][i].get("stim_name", f"stim_{i:03d}")
        result[name] = {layer: np.array(data[layer][i]["activation"]).reshape(-1) for layer in layers}
    return result

# ------------------ RDM CREATION ------------------

def create_and_save_model_rdms(feat_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    creator = RDMCreator(verbose=True, device='cpu')
    creator.create_rdms(feature_path=feat_dir, save_path=out_dir)


def create_and_save_brain_rdms(roi_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    creator = RDMCreator(verbose=True, device='cpu')
    creator.create_rdms(feature_path=roi_dir, save_path=out_dir)
    # creator.create_rdms(feature_path=roi_dir)


# ------------------ RSA & PLOTTING ------------------

def run_rsa_evaluation(model_rdm_dir, brain_rdm_dir, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    rsa = RSA(model_rdm_dir, brain_rdm_dir, save_path=save_dir, model_name=model_name)
    df = rsa.evaluate()
    return rsa, df


def plot_rsa(df_list, save_dir, metric="R2"):
    plotter = Plotting(df_list)
    results = plotter.plot_all_layers(metric=metric)
    return results

# ------------------ MAIN PIPELINE ------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)

print("Extracting brain ROIs...")
for group in ["expert", "nonexpert"]:
    betas = load_parcellated_beta(BETA_PATHS[group], ATLAS_PATH)
    save_rois_as_npz(betas, os.path.join(ROI_DIR, group), group)

print("Extracting AlexNet features...")
extract_and_save_alexnet(STIMULI_PATH, ALEXNET_FEAT_DIR)

print("Reformatting DNN features...")
trained = load_dnn_by_image(DNN_ACTIVATIONS["trained"])
untrained = load_dnn_by_image(DNN_ACTIVATIONS["untrained"])
save_dnn_features_as_npz(trained, FEATURE_DIR, "trained")
save_dnn_features_as_npz(untrained, FEATURE_DIR, "untrained")

print("Creating model RDMs...")
for model in ["alexnet", "trained", "untrained"]:
    create_and_save_model_rdms(os.path.join(FEATURE_DIR, model),
                               os.path.join(MODEL_RDM_DIR, model))

print("Creating brain RDMs...")
for group in ["expert", "nonexpert"]:
    create_and_save_brain_rdms(os.path.join(ROI_DIR, group),
                               os.path.join(BRAIN_RDM_DIR, group))

print("Running RSA evaluations...")
results = []
for model in ["alexnet", "trained", "untrained"]:
    for group in ["expert", "nonexpert"]:
        print(f"RSA: {model} x {group}")
        model_rdm_path = os.path.join(MODEL_RDM_DIR, model)
        brain_rdm_path = os.path.join(BRAIN_RDM_DIR, group)
        rsa, df = run_rsa_evaluation(model_rdm_path, brain_rdm_path, model, RSA_RESULT_DIR)
        results.append(df)

print("Plotting results...")
plot_rsa(results, RSA_RESULT_DIR, metric="R2")

print("✅ Pipeline complete.")
