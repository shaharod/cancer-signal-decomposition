
---

# Project: Cancer Signal Decomposition
**Institution:** The Hebrew University of Jerusalem, Nir Friedman Lab

This project utilizes unsupervised deep learning (Autoencoders) and dimensionality reduction (PCA) to isolate cancer-specific biological signals from high-dimensional cfChIP-seq genomic datasets.

## Directory Structure

The project is organized into modular components for model definitions, data processing, and analysis.

```text
cancer-signal-decomposition/
├── core/
│   ├── models/                 # Autoencoder architectures (healthy & disease)
│   ├── engine.py               # Training / evaluation logic
│   └── trainer_utils.py        # Training loop utilities
├── data/                       # Raw and processed cfChIP-seq data
├── latent/
│   ├── latent_run.py           # Run latent space extraction
│   └── latent_utils.py         # Latent-space analysis helpers
├── outputs/
│   ├── real_experiments/       # Results on real cfChIP-seq data
│   │   ├── plots/              # Figures and visualizations
│   │   └── trained_models/     # Saved trained model weights
│   ├── synthetic_experiments/  # Results on synthetic mixed data
│   │   ├── plots/              # Figures and visualizations
│   │   └── trained_models/     # Saved trained model weights
│   ├── disease_mix/            # Disease-mix model outputs
│   └── healthy/                # Healthy model outputs
├── sanity_checks/
│   ├── check_theta.py          # Validate θ (disease fraction) behavior
│   ├── count_data.py           # Dataset statistics and counts
│   └── syn_data_create.py      # Synthetic data generation
├── utils/
│   ├── analysis_utils.py       # Shared analysis helpers
│   ├── data_utils.py           # Data loading and preprocessing
│   ├── pca_utils.py            # PCA fitting and transformations
│   └── plots_utils.py          # Plotting utilities
├── train_disease_mix.py        # Train disease autoencoder with mixing logic
├── analyzer.py                 # General analysis entry point
├── healthy_analysis.py         # Healthy model analysis
├── disease_mix_analysis.py     # Disease-mix analysis on real data
└── disease_mix_syn_analysis.py # Disease-mix analysis on synthetic data

```

---

## Technical Features

* **Hardware Acceleration**: Native support for **Apple Silicon GPU (MPS)** and NVIDIA GPU (CUDA) via PyTorch.
* **Hybrid Deconvolution**: Separates mixed genomic signals into known "Healthy" components (PCA-based) and "Disease" signals (AE-based).
* **Sensitivity Analysis**: Validates the model's Limit of Detection (LOD) by correlating outputs against biological signature libraries across varying disease fractions ().

---

## Core Workflow

### 1. Phase 1: Healthy Tournament

Execute `python3 train_healthy.py` to train a collection of models on healthy genomic data. This builds a reference library of healthy signals across various encoding sizes (16, 32, 64, 128).

### 2. Phase 2: Disease Mix Tournament

Execute `python3 train_disease_mix.py`. This script freezes a healthy base (PCA or AE) and trains a disease-specific Autoencoder to extract cancer signals from mixed samples.

### 3. Phase 3: Biological Validation

Run `python3 disease_mix_analysis.py` to compare isolated signals () against the **Signature Library** using Pearson correlation.

---

## Configuration

All critical settings are located in `config.py`:

* **`DEVICE`**: Set to `'mps'` for Mac GPU training.
* **`ENCODING_SIZES`**: Controls the bottleneck dimension (16, 32, 64, 128).
* **`SCALING_OPTIONS`**: Toggle between `scaled` and `unscaled` data pipelines.
