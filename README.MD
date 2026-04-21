
---

# Project: Cancer Signal Decomposition
**Institution:** The Hebrew University of Jerusalem, Nir Friedman Lab

This project utilizes unsupervised deep learning (Autoencoders) and dimensionality reduction (PCA) to isolate cancer-specific biological signals from high-dimensional cfChIP-seq genomic datasets.

## Directory Structure

The project is organized into modular components for model definitions, data processing, and analysis.

cancer-signal-decomposition/
├── core/
│   ├── models/                 # Neural network architectures and wrappers
│   │   ├── ae_architectures.py # Raw Autoencoder models (Basic_AE, Layered_AE)
│   │   ├── components.py       # Universal adapters (PCAComponent, AEComponent)
│   │   ├── mix_model.py        # UniversalMixModel and deconvolution logic
│   │   └── model_factory.py    # Orchestrator for building/wrapping models
│   ├── engine.py               # Trainer class (Adam opt, unscaled MSE, early stopping)
│   └── trainer_utils.py        # ExperimentIO for saving/loading model states
├── data/                       # Root data directory
│   ├── real/                   # Real cfChIP-seq datasets
│   │   ├── real_[variant]/     # Processed run folders for specific real variants (e.g., 'basic')
│   │   └── *.csv               # Raw real data files located in the main directory
│   └── synthetic/              # Synthetically generated mixed datasets
│       └── synthetic_[type]/   # Complexity levels (e.g., 'synthetic_complex')
│           └── [variant_name]/ # Variant-specific folders (e.g., 'dif_dp', 'theta_0.1')
├── outputs/
│   ├── disease_mix_all/        # Phase 2 outputs (Mix model weights, histories, meta)
│   ├── healthy/                # Phase 1 outputs (Baseline weights, histories, meta)
│   ├── real_experiments/       # Final inference results on real cfChIP-seq data
│   │   └── plots/              # Real data visualizations (Scatter, Boxplots)
│   └── synthetic_experiments/  # Final inference results on synthetic mixed data
│       └── plots/              # Synthetic data visualizations
├── utils/
│   ├── analysis_utils.py       # Helpers for loading saved training data and model states
│   ├── data_utils.py           # Data loading, formatting, and scaling pipelines
│   ├── inference_utils.py      # Latent extraction (Z -> 2D UMAP/PCA/t-SNE)
│   ├── metrics_utils.py        # Statistical evaluations (MSE, p-value calculations)
│   ├── pca_utils.py            # Standalone Scikit-learn PCA fitting
│   ├── plots_biology_utils.py  # Biological reconstructions and domain-specific graphs
│   └── plots_utils.py          # Pure Matplotlib logic (Scatter grids, annotations)
├── latent/                     # Latent space extraction and mapping
│   ├── latent_run.py           # Phase 3 Execution: Generate latent visual grids
│   └── latent_utils.py         # Helper functions for UMAP/PCA and latent math
├── config.py                   # Centralized string-based configuration and dynamic pathing
├── train_all_healthy.py        # Phase 1 Execution: Train statistical and neural baselines
├── train_disease_mix.py        # Phase 2 Execution: Cross-architecture disease tournament
├── analyzer.py                 # Phase 3 Execution: Evaluate MSE and training curves
└── analyze_reconstruction.py   # Phase 3 Execution: Scatter recon plots and disease drivers

Configuration & Data Variants
All critical settings and dataset routing are managed centrally in config.py. There's no need to manually change file paths to run different experiments, simply need to update the variables at the top of the config file.

Data Source & Variant Routing:
The pipeline dynamically builds paths based on the exact data variant we want to test.

DATA_SOURCE: Choose 'real' or 'synthetic'.

SYNTHETIC_TYPE: If using synthetic data, choose the complexity (e.g., 'complex').

VARIANT: Select the specific experimental condition you are testing.

Real Options: 'basic', 'gene_reduc'.

Synthetic Options: 'dif_dp', 'dif_hp', 'theta_0.1', 'theta_0.01', 'theta_0.001', 'theta_0.005', 'theta_lim_0.7', 'theta_no_lim'

Hardware & Hyperparameters:

DEVICE: Set to 'mps' for Mac GPU training, 'cuda' for NVIDIA, or 'cpu'.

THETA_EXP_MODE: Controls tumor fraction behavior ('true' , 'fixed', 'random' - 'true' is also for uniform on synthetic data)

ENCODING_SIZES: Automatically scales network depth based on whether you are running simple or complex data.
Currently running with 2, 4, 8, 16 - our smaller encoding size set.
HIDDEN LAYERS: Size of hidden layers is 512, 64 to match smaller encoding size.

---

## Core Workflow
The pipeline is completely modular. You can run the entire process end-to-end, or run individual phases independently as long as the previous phase's outputs exist in the /outputs folder.

### 0. Create Synthetic Data:
Run `python3 data/*` file of choice to create data. Ensure you have under data/real the original files to extract data from.

### 1. Phase 1: Healthy Tournament

Run `python3 train_healthy.py` 
What it does: 
-Trains PCA and Autoencoder baselines on purely healthy cfChIP-seq profiles. 
-Saves weights and history.json to the /outputs/variant healthy directory, as well as train/eval values.
This builds a reference library of healthy signals across various encoding sizes, as chosen in config.py file.


Phase 2: Disease Deconvolution (train_disease_mix.py)


### 2. Phase 2: Disease Mix Tournament
Run python3 `train_disease_mix.py`.
What it does:
-Loads the frozen healthy baselines and pairs them with untrained disease Autoencoders.
-Trains the disease branch on mixed data to isolate the cancer signal.

### 3. Phase 3: Biological Validation

Because the pipeline outputs several different types of data (reconstructions, training histories, and latent spaces), there are three distinct execution scripts we can run to analyze the results:

python3 `analyzer.py` 
Purpose: Evaluates standard model performance. 
It loads the history.json files and best_meta files to plot train/test learning curves.

python3 `analyze_reconstruction.py` 
Purpose: Focuses on biological outputs. It generates reconstruction scatter plots (Input vs. Reconstruction and Pure Disease Input vs. Disease Model Reconstruction), calculates disease-specific MSE, and isolates individual "disease driver" genes.

python3 `latent/latent_run.py` 
Purpose: Extracts the latent representations Z. It uses the helpers in latent/latent_utils.py to compress the latent space into 2D coordinates (PCA/UMAP) and generates visual comparison grids.

---

## Configuration

All critical settings are located in `config.py`:

* **`DEVICE`**: Set to `'mps'` for Mac GPU training.
* **`ENCODING_SIZES`**: Controls the bottleneck dimension (16, 32, 64, 128).
* **`SCALING_OPTIONS`**: List to decide which scaling option to run, and if to run `scaled` and `unscaled` data pipelines.
