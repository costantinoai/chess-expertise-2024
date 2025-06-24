# chess-expertise-2024

This repository contains the analysis code used in the Chess Expertise study. Code is divided across several submodules written in Python and MATLAB.

## Repository structure

- `chess-glm/` – MATLAB utilities and helper scripts for first and second level GLM analysis.
- `chess-mvpa/` – Python MVPA and RSA analysis code with general plotting utilities.
- `chess-behavioural/` – Behavioural analysis scripts and helper modules.
- `chess-dataset-vis/` – Tools for visualising the chess stimulus set.
- `chess-neurosynth/` – Scripts for meta-analytic lookups using Neurosynth.
- `chess-representational-conn/` – Representational connectivity analysis tools.
- `misc/` – Example JSON sidecars for the BIDS dataset.
- `notebooks/` – Jupyter notebooks demonstrating selected analyses.

## Installation

1. Install the Python dependencies

```bash
pip install -r requirements.txt
```

2. (Optional) Install the package locally for easier imports

```bash
pip install -e .
```

### MATLAB dependencies

The GLM and MVPA MATLAB scripts require [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) and [CoSMoMVPA](https://www.cosmomvpa.org/). Ensure both toolboxes are on your MATLAB path before running the analysis.

## Dataset

The project expects a BIDS compliant dataset organised under `/data/projects/chess/data`. Inside this folder the `BIDS/` directory should contain the raw data and `derivatives/` should contain outputs from fMRIprep and FastSurfer. Example JSON sidecars are provided in `misc/`.

## Running the analysis

Each submodule contains scripts that can be executed independently. Refer to the README in the corresponding folder for specific usage examples. Typical entry points include `chess-glm/run_subject_glm.m` for first level models and the Python scripts in `chess-mvpa/` for MVPA and RSA.

## License

This project is licensed under the MIT License (see `LICENSE`).
