#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:04:41 2025

@author: costantino_ai
"""

# extract_activations.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract and save layer‐wise activations from trained and untrained AlphaVile‐Large models.
"""
import os
import sys
from copy import deepcopy

from torch.optim import SGD

sys.path.insert(0, "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/AlphazeroChess")
from modules import logging
from modules.net import device
from modules.analysis.extract_activations_funcs import probe_network_with_stimuli_alphavile
from modules.net.dataset_funcs import get_dataloader_from_csv
from modules.net.helper_funcs import get_last_level_layer_names, initialize_weights
from modules.utils.helper_funcs import (
    env_check,
    print_dict,
    set_random_seeds,
    create_run_id,
    save_ordered_dict,
    OutputLogger,
    save_script_to_file,
    create_output_directory
)

def load_crazyhouse_model(weights_path: str, device_id: int = 0):
    """
    Load untrained and trained AlphaVile‐Large models for Crazyhouse.
    """
    from pathlib import Path
    # Insert the CrazyAra codebase path
    sys.path.insert(0, '/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/CrazyAra')
    from DeepCrazyhouse.src.training.train_cli_util import create_pytorch_model, TrainConfig
    from DeepCrazyhouse.src.training.trainer_agent_pytorch import load_torch_state

    train_config = TrainConfig()
    train_config.model_type = 'alphavile-large'
    train_config.tar_file   = str(weights_path)
    train_config.device_id  = device_id

    input_shape = (52, 8, 8)
    untrained_model = create_pytorch_model(input_shape, train_config)
    initialize_weights(untrained_model)
    untrained_model = untrained_model.to(f"cuda:{device_id}" if device_id>=0 else "cpu")
    untrained_model.eval()

    trained_model = deepcopy(untrained_model)
    load_torch_state(
        trained_model,
        SGD(trained_model.parameters(), lr=train_config.max_lr),
        Path(train_config.tar_file),
        train_config.device_id
    )
    trained_model.eval()
    return untrained_model, trained_model

def main():
    env_check()
    run_id = "results/" + create_run_id() + "_extract-net-activations-alphavile_dataset-fmri"
    params = {
        "save_logs": True,
        "run_id": run_id,
        "seeds_num": 1,
        "selected_layers": [],
        "load_models": {
            "weights_path": "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/CrazyAra/weights/AlphaVile-Large-(lichess-puzzles)/model-0.31201-0.886-0085.tar",
            "device_id": 0
        },
        "get_dataloader_from_csv": {
        "csv_file_path": "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/AlphazeroChess/datasets/fmri_dataset/dataset.csv",
            "batch_size": 100,
            "shuffle": False,
            "num_workers": 8,
            "pin_memory": True,
        },
        "probe_network_with_stimuli": {
            "device": device,
            "PCA_components": None
        },
    }

    out_dir      = params["run_id"]
    out_text_file = os.path.join(out_dir, "output_log.txt")

    if params["save_logs"]:
        create_output_directory(out_dir)
        save_script_to_file(out_dir)
        logging.info("Output folder created and script file saved")

    activation_paths = {}
    with OutputLogger(params["save_logs"], out_text_file):
        print_dict(params)
        logging.info(f"Starting processing with {params['seeds_num']} seeds.")
        for seed in range(params["seeds_num"]):
            set_random_seeds(seed)
            logging.info("Loading trained/untrained models...")
            untrained_model, trained_model = load_crazyhouse_model(**params["load_models"])
            data_loader = get_dataloader_from_csv(**params["get_dataloader_from_csv"])
            logging.info("DataLoader loaded successfully.")

            for model, tag in [(trained_model, "trained"), (untrained_model, "untrained")]:
                layers = get_last_level_layer_names(
                    model,
                    ["_Stem", "_BottlekneckResidualBlock", "NTB", "_ValueHead"]
                )
                if params["selected_layers"]:
                    layers = [l for l in layers if l in params["selected_layers"]]

                logging.debug(f"Extracting activations from {tag}...")
                acts = probe_network_with_stimuli_alphavile(
                    model, layers, data_loader, **params["probe_network_with_stimuli"]
                )

                if params["save_logs"]:
                    pkl_path = os.path.join(out_dir, f"activations_model-{tag}_seed-{seed}.pkl")
                    activation_paths[tag] = pkl_path
                    save_ordered_dict(acts, pkl_path)
                    logging.info(f"Saved activations for {tag} at {pkl_path}")

if __name__ == "__main__":
    main()
