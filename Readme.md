# [FSM: Learnable Fuzzy Spectral Mamba for Uncertainty-Aware Hyperspectral Image Classification](https://doi.org/10.1109/LGRS.2026.3687386)

[![IEEE GRSL](https://img.shields.io/badge/IEEE%20GRSL-2026-blue)](https://doi.org/10.1109/LGRS.2026.3687386)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Learnable Fuzzy Spectral Mamba for Uncertainty Aware Hyperspectral Image Classification**  
> Tanishq Rachamalla\*, Aryan Das\*, Swalpa Kumar Roy, Antonio Plaza  
> (\* Equal contribution)  
> *IEEE Geoscience and Remote Sensing Letters*, Vol. 23, 2026  
> DOI: [10.1109/LGRS.2026.3687386](https://doi.org/10.1109/LGRS.2026.3687386)

---

## 📌 About

This repository contains the official model implementation of **Fuzzy Spectral Mamba (FSM)** — a hierarchical architecture for uncertainty-aware hyperspectral image classification that integrates learnable Gaussian Membership (GM) functions with Mamba-based sequence modeling.

> 📝 Training scripts, data loaders, and configs are not included in this release. The full pipeline may be released in a future update.

---

## 🏗️ Architecture

FSM is a three-stage hierarchical model. Each stage contains:

1. **Fuzzy Frequency Global Enhancement (FFGE)** — 2D FFT-based global spatio-spectral embedding with learnable GM and complex-valued grouped projections.
2. **Dual Dynamics** — Parallel branches:
   - **Spatial Mamba Branch**: Captures long-range spatial dependencies via selective state space modeling.
   - **Spectral Mamba Branch**: Grouped spectral Mamba with GM-gated channel recalibration for uncertainty-aware spectral modeling.
3. **Fuzzy Fusion** — Adaptive learnable weighting of spatial and spectral features with a stage-wise residual.

A **Multiscale Feature Aggregator** combines outputs from all three stages with softmax-normalized learnable weights, followed by GAP and a classification head.

<img width="4494" height="2997" alt="FSM Architecture" src="https://github.com/user-attachments/assets/5fd63595-c1b7-4d1d-ba88-41ea504b07b7" />

---

## 📦 Installation

```bash
pip install torch torchvision
pip install mamba-ssm
pip install einops
```

> ⚠️ `mamba-ssm` requires a CUDA-capable GPU (compute capability ≥ 7.0). CPU-only environments are not supported.

---

## 🚀 Usage

```python
from fsm import FuzzySpectralMamba

model = FuzzySpectralMamba(
    in_channels=274,           # number of spectral bands
    patch_size=11,             # spatial patch size
    num_classes=16,            # number of land-cover classes
    embed_dims=, # per-stage embedding dimensions
    depths=,          # Mamba blocks per stage[1]
    n_groups=,      # spectral groups per stage
    state_dim=16,              # Mamba state dimension
    dropout=0.1
)

import torch
x = torch.randn(4, 274, 11, 11)  # [B, C, H, W]
logits = model(x)                 # [B, num_classes]
print(logits.shape)               # torch.Size()
```

### Dataset-specific configs

| Dataset | `in_channels` | `num_classes` | `depths` | `n_groups` |
|---|---:|---:|---|---|
| WHU-Hi-HanChuan | 274 | 16 | [2, 1, 2] | [32, 16, 8] |
| WHU-Hi-HongHu | 270 | 22 | [2, 1, 2] | [32, 16, 8] |
| NiliFossae | 425 | 9 | [2, 2, 2] | [32, 16, 8] |

---

## 📊 Results

Results are reported as mean ± standard deviation over 10 runs using 30 training samples per class.

| Method | HanChuan OA | HanChuan AA | HanChuan κ | HongHu OA | HongHu AA | HongHu κ | NiliFossae OA | NiliFossae AA | NiliFossae κ |
|---|---|---|---|---|---|---|---|---|---|
| S3ANet | 86.04±2.25 | 82.29±1.99 | 83.77±2.59 | 88.47±0.86 | 86.68±0.36 | 85.63±1.03 | 92.44±2.29 | 91.14±6.17 | 90.71±3.20 |
| SACNet | 86.23±0.84 | 83.29±1.34 | 84.01±0.98 | 87.78±0.31 | 87.04±0.28 | 84.82±0.38 | 95.40±1.17 | 94.40±0.37 | 94.13±1.48 |
| SSFTTNet | 87.55±0.66 | 86.78±1.34 | 85.53±0.77 | 88.60±1.22 | 89.71±0.90 | 85.88±1.44 | 96.61±1.74 | 96.04±1.00 | 95.68±2.21 |
| GAHT | 85.20±1.76 | 84.21±0.73 | 82.82±1.98 | 90.79±3.83 | 90.02±3.09 | 88.43±4.58 | 88.27±0.69 | 89.16±0.20 | 85.15±0.83 |
| SpectralFormer | 88.77±0.05 | 87.17±0.19 | 86.93±0.06 | 90.17±0.32 | 89.97±0.08 | 87.72±0.38 | 96.77±1.03 | 95.66±0.55 | 95.88±1.30 |
| MASSFormer | 89.72±1.19 | 88.03±1.19 | 88.04±1.37 | 90.43±0.54 | 91.02±0.33 | 88.10±0.62 | 97.71±0.31 | 96.67±0.59 | 97.08±0.39 |
| MambaHSI | 87.26±0.30 | 85.19±0.40 | 85.19±0.34 | 90.94±0.41 | 90.32±0.21 | 88.66±0.49 | 95.22±1.79 | 94.33±1.50 | 93.91±2.26 |
| MambaHSI+ | 89.78±2.52 | 87.81±1.70 | 88.10±2.89 | 91.32±0.15 | 90.94±0.21 | 89.07±0.17 | 96.82±1.06 | 96.13±0.64 | 95.94±1.35 |
| SSMamba | 82.71±1.36 | 80.08±2.40 | 79.93±1.58 | 89.76±2.77 | 89.10±2.68 | 87.20±3.34 | 93.41±2.73 | 93.08±2.40 | 91.60±3.45 |
| FAHM | 89.95±1.11 | 88.04±1.45 | 88.30±1.29 | 92.28±0.72 | 92.05±0.35 | 90.32±0.89 | 96.26±1.32 | 94.21±2.27 | 95.21±1.69 |
| FETNet | 90.12±0.52 | 89.34±1.14 | 89.56±1.19 | 89.54±0.36 | 88.02±1.46 | 86.84±1.19 | 96.52±0.99 | 94.84±0.44 | 95.50±0.21 |
| **FSM (Ours)** | **93.84±0.89** | **93.08±0.70** | **92.68±0.26** | **93.15±0.52** | **93.51±0.68** | **91.22±0.85** | **98.43±0.48** | **97.15±0.55** | **97.88±0.62** |

---

## 📄 Citation

If you use this code, please cite:

```bibtex
@article{rachamalla2026fsm,
  author    = {Rachamalla, Tanishq and Das, Aryan and Roy, Swalpa Kumar and Plaza, Antonio},
  title     = {Learnable Fuzzy Spectral Mamba for Uncertainty Aware Hyperspectral Image Classification},
  journal   = {IEEE Geoscience and Remote Sensing Letters},
  volume    = {23},
  pages     = {5503905},
  year      = {2026},
  doi       = {10.1109/LGRS.2026.3687386}
}
```

---

## 📬 Contact

- **Tanishq Rachamalla** — [tanishqrachamalla12@gmail.com](mailto:tanishqrachamalla12@gmail.com)
- **Aryan Das** — [aryan.das2021@vitbhopal.ac.in](mailto:aryan.das2021@vitbhopal.ac.in)
- **Swalpa Kumar Roy** (Corresponding) — [swalpa@tezu.ernet.in](mailto:swalpa@tezu.ernet.in)
- **Antonio Plaza** (Corresponding) — [aplaza@unex.es](mailto:aplaza@unex.es)
