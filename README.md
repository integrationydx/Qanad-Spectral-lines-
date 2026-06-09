# Spectral Error Indicators for Neural PDE Solvers

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview
Physics-Informed Neural Networks (PINNs) often produce solutions that appear globally accurate while concealing large local errors. This repository implements a novel, post-hoc spectral error indicator framework that analyzes intermediate PINN training snapshots using Dynamic Mode Decomposition (DMD). 

By extracting spectral features (modal energy distribution, eigenvalue drift, and spectral entropy) from training epoch sequences, we train a lightweight regression model to map these features to local PDE solution errors. This provides a solver-agnostic mechanism to flag high-error spatial regions and drive adaptive collocation refinement—without requiring access to ground-truth data at inference time.

## Key Features
* **Unsupervised Feature Extraction:** Utilizes DMD to track spatial modes and eigenvalue trajectories across training epochs.
* **Spatial Error Regression:** Maps DMD-derived spectral signatures to local absolute errors using gradient boosted trees (XGBoost) or lightweight CNNs.
* **Adaptive Refinement:** Identifies high-error hotspots dynamically to guide targeted collocation point placement.
* **Solver-Agnostic:** Operates entirely on snapshot matrices; independent of the underlying PINN architecture.

## Repository Structure

├── data/                   # Generated datasets and true solutions
│   ├── burgers/            # 1D/2D Burgers' Equation data
│   ├── allen_cahn/         # Allen-Cahn Equation phase-field data
│   └── navier_stokes/      # 2D Cylinder Wake CFD data
├── notebooks/              # Jupyter notebooks for EDA and visualization
├── src/                    # Source code for the pipeline
│   ├── pinn/               # Baseline PINN implementations
│   ├── dmd/                # Exact and Standard DMD extraction modules
│   ├── models/             # Spatial error regression heads (XGBoost, CNN)
│   └── utils/              # Metrics, plotting, and data loaders
├── scripts/                # Execution scripts for training and evaluation
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation


## Installation

1. Clone the repository:
   git clone https://github.com/yourusername/spectral-error-indicators.git
   cd spectral-error-indicators

2. Create a virtual environment and activate it:
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required dependencies:
   pip install -r requirements.txt

## Quick Start

1. Generate standard PINN training snapshots (e.g., for Burgers' Equation):
   python scripts/train_pinn.py --equation burgers --epochs 5000 --snapshot_freq 100

2. Run the DMD feature extraction pipeline on the saved snapshots:
   python scripts/extract_dmd_features.py --input data/burgers/snapshots.npy

3. Train the spatial error regression head:
   python scripts/train_error_model.py --model xgboost --features data/burgers/dmd_features.csv

4. Evaluate and visualize the predicted spatial error maps:
   python scripts/evaluate.py --model_weights saved_models/xgb_burgers.pkl

## Validation Datasets
The model is validated against three progressively complex benchmarks:
1. **Burgers' Equation (1D/2D):** Primary validation environment with known analytical solutions (via Cole-Hopf transformation) to verify shock front error detection.
2. **Allen-Cahn Equation:** A notoriously stiff nonlinear problem to test phase-interface resolution.
3. **Navier-Stokes (Cylinder Wake):** To test generalization to vector-valued, multi-physics flow regimes.

## Author
**Aditya Alur** PES University, EC Campus  

## License
This project is licensed under the MIT License - see the LICENSE file for details.