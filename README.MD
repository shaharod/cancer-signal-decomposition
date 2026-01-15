
---

# Project: Cancer Signal Decomposition
**Institution:** The Hebrew University of Jerusalem, Nir Friedman Lab

This project utilizes unsupervised deep learning (Autoencoders) and dimensionality reduction (PCA) to isolate cancer-specific biological signals from high-dimensional cfChIP-seq genomic datasets.

## Directory Structure

The project is organized into modular components for model definitions, data processing, and analysis.

```text
.
├── core/                       # Core model and training logic
│   ├── models/                 # Neural network architectures
│   │   ├── ae_architectures.py # Basic and Layered AE definitions
│   │   ├── components.py       # Reusable torch modules
│   │   ├── mix_model.py        # Hybrid MixModel implementation
│   │   └── model_factory.py    # Factory for instantiating models
│   ├── engine.py               # GPU-accelerated Trainer class
│   └── trainer_utils.py        # Experiment IO and checkpoint helpers
├── data/                       # Dataset storage
│   ├── real/                   # Raw clinical cfChIP-seq matrices
│   └── synthetic/              # Generated synthetic mix data
├── utils/                      # Helper modules
│   ├── data_utils.py           # Scaling, tensor prep, and inverse scaling
│   ├── pca_utils.py            # sklearn PCA wrappers and IO
│   ├── plots_utils.py          # Tournament and sensitivity visualization
│   └── analysis_utils.py       # Biological validation helpers
├── experiments/                # (Auto-generated) Results and weights
├── train_healthy.py            # Entry point for Phase 1 (Healthy library)
├── train_disease_mix.py        # Entry point for Phase 2 (Disease tournament)
├── disease_mix_analysis.py     # Biological validation for real data
├── disease_mix_syn_analysis.py # Validation for synthetic benchmarks
├── healthy_analysis.py         # Latent space and PCA analysis
├── config.py                   # Global hyperparameters and DEVICE (MPS/CUDA)
├── .gitignore                  # Exclusion rules for data and models
└── README.MD                   # Project documentation

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
