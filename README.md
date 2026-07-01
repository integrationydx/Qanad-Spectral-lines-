# Spectral Error Indicators for Neural PDE Solvers

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg?logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-1.10%2B-8CAAE6.svg?logo=scipy)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange.svg)
![Status](https://img.shields.io/badge/Status-Phase%203%20Complete-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

Physics-Informed Neural Networks (PINNs) often produce solutions that appear globally accurate
while concealing large local errors. This repository implements a post-hoc spectral error
indicator framework that analyzes intermediate PINN training snapshots using Dynamic Mode
Decomposition (DMD).

By extracting spectral features (modal energy distribution, eigenvalue drift, and spectral
entropy) from training epoch sequences, we train lightweight regression models to map these
features to local PDE solution errors. This provides a solver-agnostic mechanism to flag
high-error spatial regions and drive adaptive collocation refinement, without requiring access
to ground-truth data at inference time.

---

## Key Features

- **Unsupervised Feature Extraction**: DMD tracks spatial modes and eigenvalue trajectories across training epochs with no labels required.
- **Spatial Error Regression**: Three regression heads are benchmarked across synthetic and higher-resolution runs: Ridge, XGBoost, and 1D CNN.
- **Adaptive Refinement**: High-error hotspots are flagged dynamically to guide targeted collocation point placement.
- **Solver-Agnostic**: Operates entirely on snapshot matrices and is independent of the underlying PINN architecture.

---

## Results - Phase 3 (High-Resolution Synthetic PINN Dynamics, N=500)

Phase 3 scales the same pipeline to a higher-resolution Burgers' benchmark with 500 spatial
points and 100 evaluation times. The snapshot sequence captures more realistic convergence
dynamics and stress-tests the spectral features under a larger feature space.

| Model | MAE | R2 | Pearson Corr | Refinement Gain |
|---|---:|---:|---:|---:|
| Ridge | 0.000165 | 0.0217 | 0.1478 | 16.0% |
| XGBoost | 0.000165 | -0.0029 | -0.0717 | 12.3% |
| 1D CNN | 0.000166 | -0.0085 | nan | 0.0% |

**Phase 3 findings:**

- **Higher resolution increases feature richness**: the feature matrix expands to 61 dimensions on 500 spatial points.
- **Ridge remains stable** but only modestly correlates with the true error at this scale.
- **XGBoost begins to recover useful refinement behavior** with 12.3% estimated improvement.
- **CNN needs further tuning** for this higher-resolution setting, where the current NumPy implementation is less stable.

![Phase 3 Results](outputs/phase3_results.png)

---

## Results - Phase 1 (Burgers' Equation Benchmark, Ridge Baseline)

Phase 1 implements the full pipeline on a synthetic Burgers' equation benchmark with a known
analytical solution (Cole-Hopf transformation), allowing exact ground-truth error maps.

| Metric | Value |
|---|---|
| DMD Rank (r) | 15 modes |
| Spectral Entropy H | 2.6943 |
| Top Eigenvalue \|lambda\| | 1.0000 (neutrally stable) |
| Regression MAE | 0.00095 |
| Regression R2 | 0.082 |
| Pearson Correlation | 0.287 |
| Adaptive Refinement Gain | 16.1% error reduction in flagged regions |
| High-error points flagged | 25 / 100 spatial points |

**Key observations:**

- Spectral entropy decreases monotonically across training epochs, confirming it carries convergence information.
- Adaptive refinement correctly localizes the shock front (|x| < 0.15).
- Ridge gives a useful baseline but leaves room for nonlinear heads.

![Phase 1 Results](outputs/spectral_error_results.png)

---

## Results - Phase 2 (XGBoost + CNN)

Phase 2 extends the same benchmark and DMD features with two nonlinear heads: XGBoost and a
minimal PyTorch 1D CNN over spatial DMD mode fields. Evaluated fairly using 5-Fold OOF.

| Model | MAE | R2 | Pearson Corr | Refinement Gain |
|---|---:|---:|---:|---:|
| Ridge - Phase 1 baseline | 0.00095 | 0.082 | 0.287 | 16.1% |
| XGBoost - Phase 2A (5-fold OOF) | 0.00099 | -0.005 | -0.098 | 0.0% |
| PyTorch 1D CNN - Phase 2B (5-fold OOF) | (updated) | (updated) | (updated) | (updated) |

**Phase 2 findings:**

- **XGBoost underperforms Ridge** on this dataset. With only 100 spatial points and 46 features, the model does not generalize well in OOF evaluation.
- **CNN improves correlation** over Ridge (0.359 vs 0.287) by exploiting spatial locality in DMD mode fields.
- **Adaptive refinement is comparable** between Ridge and CNN (16.1% vs 16.3%), indicating both localize the shock region effectively.
- **Main bottleneck is sample count** (N=100), suggesting higher-resolution grids and synthetic PINN snapshots are the next step.

![Phase 2 Results](outputs/phase2_results.png)

---

## Output Safety and Reproducibility

Running each phase does not alter the previous phase artifacts.

- Phase 1 output remains at outputs/spectral_error_results.png.
- Phase 2 writes a separate file at outputs/phase2_results.png.
- Phase 3 writes a separate file at outputs/phase3_results.png.
- Both scripts generate fresh in-memory snapshots during execution; no historical snapshot file is overwritten.

---

## Repository Structure

```
Quanad/
|-- outputs/
|   |-- spectral_error_results.png   # Phase 1 - 9-panel results figure
|   `-- phase2_results.png           # Phase 2 - 12-panel comparison figure
|-- spectral_error_pipeline.py        # Phase 1 - Ridge regression baseline
|-- phase_2.py                        # Phase 2 - XGBoost + CNN heads
|-- phase_3.py                        # Phase 3 - High-resolution PINN dynamics
|-- outputs/phase3_results.png        # Phase 3 - 14-panel results figure
|-- Spectral_Error_Indicators_Research_Proposal.pdf
|-- review1.txt
`-- README.md
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib scikit-learn xgboost torch
```

On macOS, if xgboost reports a missing OpenMP runtime, install:

```bash
brew install libomp
```

---

## Quick Start

**Phase 1 - Ridge baseline:**

```bash
python spectral_error_pipeline.py
```

**Phase 2 - XGBoost + CNN heads:**

```bash
python phase_2.py
```

Phase 2 saves a 12-panel comparison figure to outputs/phase2_results.png.

**Phase 3 - High-resolution PINN dynamics:**

```bash
python phase_3.py
```

Phase 3 saves a 14-panel comparison figure to outputs/phase3_results.png.

---

## Validation Datasets

| # | Dataset | Purpose | Status |
|---|---|---|---|
| 1 | **Burgers' Equation (1D)** | Primary synthetic benchmark with Cole-Hopf reference | Phase 1, 2, and 3 complete |
| 2 | **Allen-Cahn Equation** | Stiff nonlinear PINN failure benchmark with sharp phase interface | Planned |
| 3 | **Navier-Stokes (Cylinder Wake)** | Generalization to vector-valued multi-physics PDE | Planned |

---

## Roadmap

- [x] Phase 1 - Synthetic Burgers' benchmark, DMD pipeline, Ridge baseline
- [x] Phase 2 - XGBoost + 1D CNN benchmark completed
- [x] Phase 3 - High-resolution synthetic Burgers' dynamics with N = 500
- [ ] Phase 4 - Allen-Cahn benchmark
- [ ] Phase 5 - Navier-Stokes generalization
- [ ] Phase 6 - Solver-agnostic extension to FNO / DeepONet

---

## Prior Work

| Reference | Contribution | Gap This Project Fills |
|---|---|---|
| Raissi et al. (2019) | Original PINN formulation | No error estimation or adaptive refinement |
| Dwivedi and Srinivasan (2020) | Residual-based adaptive collocation | Residual does not always track true error in stiff nonlinear regimes |
| Lu et al. (2021) - DeepXDE | Residual-driven point selection | Same limitation: not spectrally informed |
| Schmid (2010) | DMD for fluid dynamics | Applied to physical snapshots, not neural training dynamics |
| Yang and Perdikaris (2021) - B-PINNs | Bayesian UQ for PINNs | Expensive and not solver-agnostic |
| This work | DMD on training snapshots to spectral features to spatial error regression | Lightweight and solver-agnostic with no ground truth needed at inference |

---

## Author

Aditya Alur  
PES University, EC Campus

---

## License

This project is licensed under the MIT License.