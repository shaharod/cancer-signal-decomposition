import io
import traceback
import joblib
import os
import config as cfg
import utils.analysis_utils as au
import utils.data_utils as du

import utils.plots_utils as pu
import torch
import numpy as np
from scipy import stats
from core.models.model_factory import ModelFactory

import pandas as pd

SCALED = True
UNSCALED = False

def load_reconstruction_data(phase):
    """
    Loads the validation data (Mixed Input and Clean Ground Truth).
    Matches your requested structure using config paths.
    """
    if phase == "healthy":
        mix_file = cfg.HEALTHY_GENES_PATH  # Input is pure healthy data
        truth_file = cfg.HEALTHY_GENES_PATH
    else:
        mix_file = cfg.DISEASE_GENES_PATH  # Input is mixed data
        truth_file = cfg.DATA_SUB / 'pure_disease_truth.csv' # Truth is pure disease
    print(f"truth_file: {truth_file}")
    print(f"mix file: {mix_file}")
    # 2. Validation
    if not mix_file.exists() or not truth_file.exists():
        print(f"⚠️ Warning: Reconstruction data not found:\n {mix_file}\n {truth_file}")
        return None, None
        
    # 3. Load & Transpose (Genes should be columns for the model)
    # Using 'T' because typically gene files are (Genes x Samples), but models expect (Samples x Genes)
    df_mixed = pd.read_csv(mix_file, index_col=0).T
    df_pure  = pd.read_csv(truth_file, index_col=0).T
    
    return df_mixed, df_pure



def calculate_theta_invariant_correlation(truth_df, recon_df, theta_values):
    """
    Calculates the partial correlation between Truth and Recon, 
    removing the variance explained by the mixing proportion (theta).
    """
    # Flatten everything to 1D arrays for a global check
    y_true = truth_df.values.flatten()
    y_pred = recon_df.values.flatten()
    
    # Repeat theta for every gene in every sample to match lengths
    num_genes = truth_df.shape[1]
    theta_repeated = np.repeat(theta_values, num_genes)

    # 1. Standard Correlation (Pearson)
    r_standard, _ = stats.pearsonr(y_true, y_pred)

    # 2. Partial Correlation Logic:
    # We want the correlation between (y_true | theta) and (y_pred | theta)
    # Step A: Residuals of truth regressed on theta
    res_true = y_true - stats.linregress(theta_repeated, y_true).slope * theta_repeated
    # Step B: Residuals of pred regressed on theta
    res_pred = y_pred - stats.linregress(theta_repeated, y_pred).slope * theta_repeated
    
    # Step C: Correlation of the residuals
    r_partial, _ = stats.pearsonr(res_true, res_pred)

    print(f"Standard R: {r_standard:.4f}")
    print(f"Partial R (Controlled for Theta): {r_partial:.4f}")
    
    return r_standard, r_partial

def get_best_model_tag(phase, scale_bool, encoding_size):
    """
    Scans the trained_models folder to find the model with the 
    lowest MSE for a specific encoding size.
    """
    tag = "scaled" if scale_bool else "unscaled"
    root = cfg.get_path(phase, folder_type=cfg.MODELS_SUBFOLDER) / tag
    
    best_mse = float('inf')
    best_tag = None
    
    # Iterate through all model folders (mix_H-ae_D-ae, etc.)
    for model_folder in root.iterdir():
        if not model_folder.is_dir(): continue
        
        meta_path = model_folder / f"enc_{encoding_size}" / "best_meta.json"
        results_path = model_folder / f"enc_{encoding_size}" / "results.json"
        
        # Check AE meta or PCA results
        for p in [meta_path, results_path]:
            if p.exists():
                data = io.load_results(p.parent, p.name)
                # Standardize keys: 'best_val' for AE, 'val_mse' for PCA
                mse = data.get('best_val', data.get('val_mse', float('inf')))
                
                if mse < best_mse:
                    best_mse = mse
                    best_tag = model_folder.name
                    
    return best_tag

def analyze_decoder_gene_signatures(model, gene_names, top_n=10):
    """
    Extracts weights from the decoder to identify which genes 
    are most influenced by each latent dimension.
    """
    # Navigate to the disease decoder weights
    # Assuming UniversalMixModel -> disease_model -> decoder -> last layer
    decoder_weights = model.disease_model.decoder[-1].weight.data.cpu().numpy()
    
    # Shape: (input_dim, latent_dim) - if it's (latent, input), transpose it
    if decoder_weights.shape[0] != len(gene_names):
        decoder_weights = decoder_weights.T

    signatures = {}
    for i in range(decoder_weights.shape[1]):
        latent_vector = decoder_weights[:, i]
        
        # Get indices of the largest positive weights
        top_indices = np.argsort(latent_vector)[-top_n:][::-1]
        top_genes = [(gene_names[idx], latent_vector[idx]) for idx in top_indices]
        
        signatures[f"Latent_{i}"] = top_genes
        
    return signatures

def analyze_reconstruction_grid(labels_dict, phase, scale_bool, save_path):
    """
    One function to rule them all. 
    If phase='disease', it handles 'mix' parsing. 
    If phase='healthy', it handles standalone parsing.
    """

    # 1. Load Data & Tensors
    input_df, truth_df = load_reconstruction_data(phase)
    if input_df is None: return
    
    tag = "scaled" if scale_bool else "unscaled"
    input_size = input_df.shape[1]
    input_tensor = torch.tensor(input_df.values).float().to("cpu")
    # log_truth = np.log1p(truth_df.values).flatten()

    # 2. Determine if we are looping through Tournament Bases (Disease) or just Labels (Healthy)
    # This detects if labels_dict is {Base: {Models}} or just {Models}
    if phase == 'disease':
        iterator = labels_dict.items()
    else:
        # Wrap healthy labels in a dummy base so the loop structure is the same
        iterator = [("Standalone", labels_dict)]

    for base_name, models in iterator:
        n_rows = len(cfg.ENCODING_SIZES)
        n_cols = len(models)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)
        fig.suptitle(f"Expression Distributions: Healthy Base = {base_name.upper()}\nPhase: {phase.capitalize()} | Data: {tag.capitalize()}", 
        fontsize=20, fontweight='bold', y=0.98)
                     
        for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
            for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                ax = axes[row_idx, col_idx]
                is_pca = None
                is_mix=True if "mix" in folder_tag else False
                try:
                    # --- UNIFIED LOADING LOGIC ---
                    if is_mix:
                        # Disease Mix Logic
                        parts = folder_tag.split('_H-')
                        h_and_d = parts[1].split('_D-')
                        h_type, d_type = h_and_d[0], h_and_d[1]
                        if d_type == 'pca': is_pca = d_type
            
                        h_model = ModelFactory.create_model(h_type, input_size, enc, cfg.H1, cfg.H2)
                        d_model = ModelFactory.create_model(d_type, input_size, enc, cfg.H1, cfg.H2)
                        model = ModelFactory.create_mix_model(h_model, d_model)
                    else:
                        if folder_tag.lower() == "pca":
                            is_pca = "pca"
                        # Healthy / Standalone Logic
                        model = ModelFactory.create_model(folder_tag, input_size, enc, cfg.H1, cfg.H2)

                    # --- LOAD WEIGHTS & INFER ---
                    ext = "model.joblib" if is_pca else "model.pt"
                    model_path = cfg.get_path(phase, tag, folder_tag, enc, cfg.MODELS_SUBFOLDER) / ext
                    if not model_path.exists():
                        ax.text(0.5, 0.5, "Model Not Found", ha='center'); continue
                    if is_pca:
                        pca_sk = joblib.load(model_path)
                        
                        if is_mix:
                            model.disease.mean.data = torch.tensor(pca_sk.mean_, dtype=torch.float32)
                            model.disease.components.data = torch.tensor(pca_sk.components_, dtype=torch.float32)
                        else:    
                            # Manually inject the weights into the PyTorch buffers
                            model.mean.data = torch.tensor(pca_sk.mean_, dtype=torch.float32)
                            model.components.data = torch.tensor(pca_sk.components_, dtype=torch.float32)
                        
                    else:

                        checkpoint = torch.load(model_path, map_location="cpu")
                        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
                        model.load_state_dict(state_dict)
                    model.eval()

                    with torch.no_grad():
                        output = model(input_tensor)
                        reconstructed = output[0] if isinstance(output, (tuple, list)) else output
                    
                    # log_recon = np.log1p(reconstructed.numpy()).flatten()
                    recon_raw = reconstructed.numpy().flatten()
                    truth_raw = input_df.values.flatten()
                    corr = np.corrcoef(truth_raw, recon_raw)[0, 1]
                    # --- PLOTTING ---
                    # We create a list of data arrays to plot side-by-side
                    data_to_plot = [truth_raw, recon_raw]
                    
                    # Using patch_artist to color the boxes for distinction
                    bp = ax.boxplot(data_to_plot, tick_labels=['Truth', 'Pred'], patch_artist=True, 
                                    showfliers=False, widths=0.6)
                    # Final Formatting
                    # Fix the Y-axis so we see the 0.5 mixing level (50) clearly
                    ax.set_ylim(-5, 110) 
                    ax.axhline(50, color='gray', linestyle='--', alpha=0.3, lw=1, label='Mix Baseline')
                  
                    # # 3. Clean up formatting
                    for patch, color in zip(bp['boxes'], ['#bdc3c7', '#1abc9c']):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.5) # Lower alpha so points show through
                    # Formatting
                    if row_idx == 0: ax.set_title(f"{model_label}", fontsize=14, fontweight='bold')
                    if col_idx == 0: ax.set_ylabel(f"Enc: {enc}\nExpression Level", fontsize=12, fontweight='bold')
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    ax.text(0.5, 0.95, f"R: {corr:.3f}", transform=ax.transAxes, 
                            ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    # ax.set_ylim(0, 100)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error Loading Model, {str(e)}", ha='center', color='red')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        folder = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER) / f"Tournament_H-{base_name}"
        os.makedirs(folder, exist_ok=True)
        full_path = os.path.join(folder, f"{save_path}_{tag}")
        plt.savefig(full_path, dpi=150)
        plt.close()


def analyze_reconstruction_combined(labels_dict, phase, scale_bool, save_path):
    # 1. Load Data
    input_df, truth_df = load_reconstruction_data(phase)
    if input_df is None: return
    
    input_raw = input_df.values.flatten()  # The reference (contains the 50s)
    tag = "scaled" if scale_bool else "unscaled"
    input_size = input_df.shape[1]
    input_tensor = torch.tensor(input_df.values).float().to("cpu")

    # 2. Handle Phase Logic
    iterator = labels_dict.items() if phase == 'disease' else [("Standalone", labels_dict)]

    for base_name, models in iterator:
        n_rows = len(cfg.ENCODING_SIZES)
        n_cols = len(models) + 1 # +1 for the Input reference
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 6 * n_rows), squeeze=False)
        fig.suptitle(f"Tournament Distribution Comparison: Healthy Base = {base_name.upper()}\nPhase: {phase.capitalize()} | Data: {tag.capitalize()}", 
                     fontsize=20, fontweight='bold', y=0.98)
                             
        for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
            # --- COLUMN 0: THE INPUT REFERENCE (Plotted once per row) ---
            ax_input = axes[row_idx, 0]
            sns.violinplot(y=input_raw, ax=ax_input, color='#ecf0f1', inner='quartile', alpha=0.4)
            sns.stripplot(y=input_raw, ax=ax_input, color="#7a96a4", size=2, alpha=0.1, jitter=True)
            
            ax_input.set_ylim(-5, 115)
            ax_input.axhline(50, color='red', linestyle='--', alpha=0.3)
            ax_input.set_ylabel(f"Enc: {enc}\nExpression", fontsize=12, fontweight='bold')
            if row_idx == 0: 
                ax_input.set_title("Original\n(Mixed Input)", fontweight='bold', color='darkblue')

            # --- COLUMNS 1 TO N: THE MODELS ---
            for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                ax = axes[row_idx, col_idx + 1] # Offset by 1
                is_pca = None
                is_mix = True if "mix" in folder_tag else False
                
                try:
                    # --- MODEL LOADING ---
                    if is_mix:
                        parts = folder_tag.split('_H-')
                        h_and_d = parts[1].split('_D-')
                        h_type, d_type = h_and_d[0], h_and_d[1]
                        if d_type == 'pca': is_pca = d_type
                        
                        h_model = ModelFactory.create_model(h_type, input_size, enc, cfg.H1, cfg.H2)
                        d_model = ModelFactory.create_model(d_type, input_size, enc, cfg.H1, cfg.H2)
                        model = ModelFactory.create_mix_model(h_model, d_model)
                    else:
                        if "pca" in folder_tag.lower(): is_pca = "pca"
                        model = ModelFactory.create_model(folder_tag, input_size, enc, cfg.H1, cfg.H2)

                    # --- WEIGHT LOADING ---
                    ext = "model.joblib" if is_pca else "model.pt"
                    model_path = cfg.get_path(phase, tag, folder_tag, enc, cfg.MODELS_SUBFOLDER) / ext
                    
                    if not model_path.exists():
                        ax.text(0.5, 0.5, "Model Not Found", ha='center'); continue

                    if is_pca:
                        import joblib
                        pca_sk = joblib.load(model_path)
                        target = model.disease if is_mix else model
                        target.mean.data = torch.tensor(pca_sk.mean_, dtype=torch.float32)
                        target.components.data = torch.tensor(pca_sk.components_, dtype=torch.float32)
                    else:
                        checkpoint = torch.load(model_path, map_location="cpu")
                        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
                        model.load_state_dict(state_dict)

                    # --- INFERENCE ---
                    model.eval()
                    with torch.no_grad():
                        output = model(input_tensor)
                        reconstructed = output[0] if isinstance(output, (tuple, list)) else output
                    
                    recon_raw = reconstructed.numpy().flatten()
                    corr = np.corrcoef(input_raw, recon_raw)[0, 1]

                    # --- PLOTTING ---
                    sns.violinplot(y=recon_raw, ax=ax, color='#1abc9c', inner='quartile', alpha=0.4)
                    sns.stripplot(y=recon_raw, ax=ax, color="#057861", size=2, alpha=0.1, jitter=True)
                    
                    # ax.set_ylim(-5, 115)
                    ax.axhline(50, color='gray', linestyle='--', alpha=0.3)
                    
                    if row_idx == 0: 
                        ax.set_title(f"{model_label}", fontsize=14, fontweight='bold')
                    
                    ax.text(0.5, 0.95, f"R: {corr:.3f}", transform=ax.transAxes, ha='center', 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                except Exception as e:
                    ax.text(0.5, 0.5, f"Error:\n{str(e)[:30]}...", ha='center', color='red')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        folder = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER) / f"Tournament_H-{base_name}"
        os.makedirs(folder, exist_ok=True)
        plt.savefig(folder / f"{save_path}_{tag}.png", dpi=150)
        plt.close()


def load_revived_model(h_type, d_type=None, enc=8, scale_bool=False):
    """
    Rebuilds a standalone Healthy model OR a Disease Mix model.
    Handles both PyTorch (.pt) and PCA (.joblib) formats.
    """
    tag = "scaled" if scale_bool else "unscaled"
    input_size = 1000  # For your synthetic data

    # Helper to load a single component (Healthy or Disease)
    def prepare_component(m_type, phase, folder_tag):
        path = cfg.get_path(phase, tag, folder_tag, enc, cfg.MODELS_SUBFOLDER)
        is_pca = m_type.lower() == 'pca'
        ext = "model.joblib" if is_pca else "model.pt"
        
        if not (path / ext).exists():
            return None

        if is_pca:
            return joblib.load(path / ext)
        else:
            obj = ModelFactory.create_model(m_type, input_size, enc, cfg.H1, cfg.H2)
            checkpoint = torch.load(path / ext, map_location="cpu")
            if isinstance(checkpoint, dict):
                # Try different common keys used in your project
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'best_state' in checkpoint:
                    state_dict = checkpoint['best_state']
                else:
                    # If it's a dict but neither key exists, 
                    # it's likely the state_dict itself
                    state_dict = checkpoint
            else:
                # If it's not a dict, it's the raw state_dict
                state_dict = checkpoint
                obj.load_state_dict(state_dict)
            obj.eval()
            return obj

    # Logic for Mix vs Standalone
    if d_type:
        # Load Healthy base and Disease component for a Mix
        h_obj = prepare_component(h_type, "healthy", h_type)
        mix_tag = f"mix_H-{h_type}_D-{d_type}"
        d_obj = prepare_component(d_type, "disease", mix_tag)
        
        if h_obj and d_obj:
            return ModelFactory.create_mix_model(h_obj, d_obj)
    else:
        # Load Standalone Healthy
        return prepare_component(h_type, "healthy", h_type)
    
    return None

import seaborn as sns
import matplotlib.pyplot as plt

def plot_consolidated_heatmaps(base_name, models_dict, scale_bool=False):
    """
    Generates a single grid of heatmaps: 
    Rows = Encoding Sizes (8, 16)
    Cols = Models (Basic, Layered, etc.)
    """
    enc_sizes = cfg.ENCODING_SIZES # e.g., [8, 16]
    n_rows = len(enc_sizes)
    n_cols = len(models_dict)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    fig.suptitle(f"Gene Module Discovery: Healthy Base = {base_name.upper()}", fontsize=22, y=1.02)

    for r_idx, enc in enumerate(enc_sizes):
        for c_idx, (model_label, folder_tag) in enumerate(models_dict.items()):
            ax = axes[r_idx, c_idx]
            d_arch = folder_tag.split("_D-")[-1]
            
            # 1. Load the model using your revived loader
            model = load_revived_model(base_name.lower().replace("-", "_"), d_arch, enc, scale_bool)
            
            if model is None or d_arch == 'pca':
                ax.text(0.5, 0.5, "PCA/Missing", ha='center')
                continue

            try:
                # 2. Extract weights from the disease component
                # UniversalMixModel -> AEComponent -> AE -> decoder
                weights = model.disease.ae.decoder[-1].weight.data.cpu().numpy()
                if weights.shape[0] != 1000: 
                    weights = weights.T

                # 3. Plot Heatmap
                sns.heatmap(weights, ax=ax, cmap="RdBu_r", center=0, cbar=(c_idx == n_cols-1))
                ax.axhline(500, color='black', linestyle='--', linewidth=1.5)
                
                if r_idx == 0: ax.set_title(f"Disease Model: {model_label}", fontsize=14, fontweight='bold')
                if c_idx == 0: ax.set_ylabel(f"Enc {enc}\nGenes (0-999)", fontsize=12, fontweight='bold')
                
            except Exception as e:
                ax.text(0.5, 0.5, "Weight Error", ha='center', color='red')

    plt.tight_layout()
    save_path = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER) / f"Consolidated_Heatmap_{base_name}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Consolidated heatmap saved to: {save_path}")
    plt.close()

def run_comprehensive_analysis():
    gene_names = [f"Gene_{i}" for i in range(1000)]
    theta_values = pd.read_csv(cfg.THETA_PATH).values.flatten()
    input_df, truth_df = load_reconstruction_data()

    # Define the "Tournament" combinations we want to deep-dive into
    architectures = [('pca', 'pca'), ('ae_layered', 'ae_layered'), ('ae_basic', 'ae_basic')]
    
    for h_arch, d_arch in architectures:
        for enc in cfg.ENCODING_SIZES:
            # 1. Revive the model
            model = load_revived_model(h_arch, d_arch, enc)
            if model is None: continue

            # 2. Extract Latent/Weight Insights (Interpretability)
            if hasattr(model, 'disease_model'): # If it's a Mix with an AE disease component
                # Signatures
                sigs = analyze_decoder_gene_signatures(model, gene_names)

            # 3. Perform Statistical Validation
            with torch.no_grad():
                input_tensor = torch.tensor(input_df.values).float()
                recon = model(input_tensor)
                recon_df = pd.DataFrame(recon.numpy(), columns=truth_df.columns)
                
            # Partial Correlation (Theta-Invariant)
            r_std, r_inv = calculate_theta_invariant_correlation(truth_df, recon_df, theta_values)

def save_performance_comparison(results_list):
    """
    Creates and saves a bar plot comparing Partial R for all models.
    """
    # Convert the list of results to a DataFrame
    df = pd.DataFrame(results_list)
    
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Create the bar plot
    ax = sns.barplot(data=df, x="h_arch", y="partial_r", hue="d_arch", palette="viridis")
    
    # Add labels and title
    plt.title("Tournament Performance: Partial Correlation (Controlled for Theta)", fontsize=15)
    plt.ylabel("Partial Correlation (R)", fontsize=12)
    plt.xlabel("Healthy Base Architecture", fontsize=12)
    plt.ylim(-0.1, 1.1) # Set limits to see the 0 and 1 clearly
    
    # Add the R values on top of the bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

    # Save it in the disease plots folder
    save_path = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER) / "tournament_summary_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Performance summary saved to: {save_path}")
    plt.close()


def analyze_modular_reconstruction(labels_dict, phase, scale_bool, save_path):
    """
    Analyzes reconstruction by grouping genes by their Pure Target values (0 vs 100).
    Works for both Step 1 (Healthy only) and Step 2 (Mix of Healthy and Disease).
    """
    # 1. Load the specific data used for this inference run
    input_df, truth_df = load_reconstruction_data(phase)
    if input_df is None: return
    
    num_samples = len(input_df)
    tag = "scaled" if scale_bool else "unscaled"
    input_size = input_df.shape[1]
    
    # 2. Load Theta values to build the [Genes | Theta] tensor
    theta_df = pd.read_csv(cfg.THETA_PATH, index_col=0)
    if phase == 'healthy':
        thetas = np.zeros((num_samples, 1)) 
    else:
        # Step 2 uses the mix percentages or fixed 0.5
        thetas = (np.full((num_samples, 1), 0.5) 
                  if cfg.FIXED_THETA_EXP 
                  else theta_df.values[:num_samples])
    
    input_tensor = torch.cat([torch.tensor(input_df.values).float(), 
                              torch.tensor(thetas).float()], dim=1).to("cpu")

    # 3. Load the "Pure" templates (What the branches SHOULD ideally reconstruct)
    # These contain the 0 and 100 signals
    pure_h_truth = pd.read_csv(cfg.SYN_MIX_HEALTHY_PART, index_col=0).T
    pure_d_truth = pd.read_csv(cfg.SYN_MIX_DISEASE_PART, index_col=0).T

    iterator = labels_dict.items() if phase == 'disease_mix' else [("Standalone", labels_dict)]

    for base_name, models in iterator:
        n_rows = len(cfg.ENCODING_SIZES)
        n_cols = len(models)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False)
        fig.suptitle(f"Reconstruction Fidelity ({phase.upper()})\n"
                     f"Comparing Predicted Genes against Ground Truth Targets (0 or 100)", 
                     fontsize=20, fontweight='bold', y=0.98)

        for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
            for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                ax = axes[row_idx, col_idx]
                is_mix = "mix" in folder_tag
                
                try:
                    # --- MODEL & WEIGHT LOADING ---
                    if is_mix:
                        parts = folder_tag.split('_H-')
                        h_and_d = parts[1].split('_D-')
                        h_type, d_type = h_and_d[0], h_and_d[1]
                        h_model = ModelFactory.create_model(h_type, input_size, enc, cfg.H1, cfg.H2)
                        d_model = ModelFactory.create_model(d_type, input_size, enc, cfg.H1, cfg.H2)
                        model = ModelFactory.create_mix_model(h_model, d_model)
                    else:
                        model = ModelFactory.create_model(folder_tag, input_size, enc, cfg.H1, cfg.H2)

                    is_pca = "pca" in folder_tag.lower()
                    ext = "model.joblib" if is_pca else "model.pt"
                    model_path = cfg.get_path(phase, tag, folder_tag, enc, cfg.MODELS_SUBFOLDER) / ext
                    
                    if not model_path.exists():
                        ax.text(0.5, 0.5, "Model Not Found", ha='center'); continue

                    if is_pca:
                        pca_sk = joblib.load(model_path)
                        target = model.disease if is_mix else model
                        target.mean.data = torch.tensor(pca_sk.mean_, dtype=torch.float32)
                        target.components.data = torch.tensor(pca_sk.components_, dtype=torch.float32)
                    else:
                        checkpoint = torch.load(model_path, map_location="cpu")
                        if isinstance(checkpoint, dict):
                            state_dict = checkpoint.get('model_state_dict', 
                                        checkpoint.get('best_state', 
                                        checkpoint))
                        else:
                            state_dict = checkpoint
                        # state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint
                        model.load_state_dict(state_dict)

                    # --- INFERENCE ---
                    model.eval()
                    with torch.no_grad():
                        if is_mix:
                            outputs = model(input_tensor)
                            # unpacked as (mix_recon, disease_recon, healthy_recon, latent)
                            _, recon_d, recon_h, _ = outputs
                        else:
                            # Slice to remove the theta column (last column) for standalone models
                            current_input = input_tensor[:, :input_size] 
                            outputs = model(current_input)
                            
                            # Standalone models return (recon, latent) or just recon
                            recon_h = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                            recon_d = torch.zeros_like(recon_h) # Step 1 has no disease branch
                    # --- DYNAMIC TARGET GROUPING ---
                    # We always evaluate the Disease Branch's ability to reconstruct the PURE disease signal.
                    # We slice the template truth to match the current sample count.
                    flat_recon = recon_d.numpy().flatten()
                    flat_truth = pure_d_truth.iloc[:num_samples].values.flatten()
                    
                    # Create labels based on what the signal value IS in the truth
                    labels = np.where(flat_truth > 50, "Target: 100", "Target: 0")

                    plot_df = pd.DataFrame({
                        'Expression': np.concatenate([flat_truth, flat_recon]),
                        'Source': (['True Pure'] * len(flat_truth) + ['Branch Recon'] * len(flat_recon)),
                        'Group': np.concatenate([labels, labels])
                    })

                    # --- PLOTTING (Per your Sketch) ---
                    sns.boxplot(data=plot_df, x='Group', y='Expression', hue='Source', ax=ax, 
                                palette=['#bdc3c7', '#3498db'], showfliers=False, order=['Target: 0', 'Target: 100'])
                    
                    sns.stripplot(data=plot_df, x='Group', y='Expression', hue='Source', ax=ax, 
                                  palette=['black', 'red'], size=1, alpha=0.1, jitter=True, dodge=True)
                    
                    ax.axhline(0, color='red', linestyle='--', alpha=0.3)
                    ax.axhline(100, color='green', linestyle='--', alpha=0.3)
                    ax.set_ylim(-20, 150)
                    
                    # Labels
                    if row_idx == 0: ax.set_title(f"{model_label}", fontsize=14, fontweight='bold')
                    if col_idx == 0: ax.set_ylabel(f"Enc: {enc}\nExpression", fontsize=12, fontweight='bold')

                except Exception as e:
                    print(f"Error plotting {model_label} at Enc {enc}")
                    traceback.print_exc()
                    ax.text(0.5, 0.5, f"Error: {type(e).__name__}", ha='center', color='red')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_folder = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER) / f"Modular_H-{base_name}"
        os.makedirs(out_folder, exist_ok=True)
        plt.savefig(out_folder / f"{save_path}_{tag}.png", dpi=150)
        plt.close()

def analyze_mix_reconstruction(labels_dict, scale_bool, save_path, mode):
    """
    Analyzes Step 2 (Mix Model) disease branch. 
    Showcases how the branch recovers the Pure Disease signal from a Mixed Input.
    """
    # 1. Load Data (Phase is 'disease_mix')
    input_df, truth_df = load_reconstruction_data('disease')
    if input_df is None: return
    
    num_samples = len(input_df)
    tag = "scaled" if scale_bool else "unscaled"
    input_size = input_df.shape[1] 
    # 1. LOAD AND CONCATENATE DATA (Mirroring your training logic)
    # Healthy samples (Theta = 0)
    df_h = du.prepare_and_align_data(cfg.HEALTHY_GENES_PATH, theta_path=None)
    # Disease samples (Theta > 0)
    df_d = du.prepare_and_align_data(cfg.DISEASE_GENES_PATH, theta_path=cfg.THETA_PATH, mode=mode)
    
    # Combined DF
    df_combined = pd.concat([df_h, df_d]).sample(frac=1, random_state=42)
    
    # 2. GET TEST SPLIT (Using the tournament/mix split path)
    # This ensures we are evaluating on the samples the model DID NOT see during training
    tournament_split_path = cfg.get_split_path("disease", tag)
    train_df, test_df = du.get_split_data(df_combined, split_path=tournament_split_path)
    
    # 3. PREPARE TENSORS
    # We use your get_ready_tensors_df to handle scaling and theta attachment
    train_tensor, test_tensor, _ = du.get_ready_tensors_df(train_df, test_df, scale_bool) # Using test as both for helper
    
    num_samples = len(test_df)
    input_size = test_df.shape[1] - 1 # Subtract 1 for the theta column
    
    # 4. LOAD PURE TRUTH TEMPLATES (For the modular boxes)
    # We need to align these with the specific samples in our test_df
    pure_h_truth_full = pd.read_csv(cfg.SYN_MIX_HEALTHY_PART, index_col=0).T
    pure_d_truth_full = pd.read_csv(cfg.SYN_MIX_DISEASE_PART, index_col=0).T
    test_truth = pure_d_truth_full.reindex(test_df.index).fillna(0.0)
    # Load Theta values for inference
    test_thetas = torch.tensor(test_df[['theta_value']].values).float().to("cpu")

    # Re-verify the input tensor has exactly what the model saw during training
    test_genes = torch.tensor(test_df.drop(columns=['theta_value']).values).float().to("cpu")
    test_tensor = torch.cat([test_genes, test_thetas], dim=1)

    # Phase 2 iterates through the different Healthy Bases
    for base_name, models in labels_dict.items():
        n_rows = len(cfg.ENCODING_SIZES)
        n_cols = len(models)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False)
        fig.suptitle(f"Disease Signal Recovery (Phase: DISEASE MIX | Base: {base_name.upper()})\n"
                     f"Comparing Mixed Input vs. Disease Branch Pure Reconstruction", 
                     fontsize=20, fontweight='bold', y=0.98)

        for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
            for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                ax = axes[row_idx, col_idx]
                
                try:
                    # --- LOAD MIX MODEL ---
                    parts = folder_tag.split('_H-')
                    h_type, d_type = parts[1].split('_D-')[0], parts[1].split('_D-')[1]
                    
                    h_model = ModelFactory.create_model(h_type, input_size, enc, cfg.H1, cfg.H2)
                    d_model = ModelFactory.create_model(d_type, input_size, enc, cfg.H1, cfg.H2)
                    model = ModelFactory.create_mix_model(h_model, d_model)

                    model_path = cfg.get_path('disease', tag, folder_tag, enc, cfg.MODELS_SUBFOLDER) / "model.pt"
                    if not model_path.exists(): continue

                    checkpoint = torch.load(model_path, map_location="cpu")
                    state_dict = checkpoint.get('model_state_dict', checkpoint.get('best_state', checkpoint))
                    model.load_state_dict(state_dict)

                    # --- INFERENCE ---
                    model.eval()
                    with torch.no_grad():
                        # UniversalMixModel uses the 1001-column tensor
                        _, recon_d, _, _ = model(test_tensor)

                    # --- MODULE LOGIC (0-499 vs 500-999) ---
                    flat_input = test_genes.cpu().numpy().flatten()      # The 50% / Mixed signal
                    flat_recon = recon_d.cpu().numpy().flatten()      # The Model's attempt at 100%
                    flat_truth = test_truth.values.flatten() # The 100% Goal

                    # Label groups based on the Disease profile
                    labels = np.where(flat_truth > 50, "Disease Module (500-999)", "Healthy Module (0-499)")

                    plot_df = pd.DataFrame({
                        'Expression': np.concatenate([flat_input, flat_recon]),
                        'Source': (['Mixed Input'] * len(flat_input) + ['Branch Recon (Pure)'] * len(flat_recon)),
                        'Module': np.concatenate([labels, labels])
                    })

                    # --- PLOTTING ---
                    sns.boxplot(data=plot_df, x='Module', y='Expression', hue='Source', ax=ax, 
                                palette=['#f39c12', '#e74c3c'], showfliers=False)
                    
                    ax.axhline(0, color='blue', linestyle='--', alpha=0.3, label="Target 0")
                    ax.axhline(100, color='green', linestyle='--', alpha=0.3, label="Target 100")
                    ax.set_ylim(-10, 150)

                    if row_idx == 0: ax.set_title(model_label, fontweight='bold')
                    if col_idx == 0: ax.set_ylabel(f"Enc: {enc}\nExpression")

                except Exception as e:
                    traceback.print_exc()
                    ax.text(0.5, 0.5, "Inference Error", ha='center', color='red')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER) / f"Tournament_H-{base_name}"
        os.makedirs(out_folder, exist_ok=True)
        plt.savefig(out_folder / f"{save_path}_{tag}.png", dpi=150)
        plt.close()

def fix_df_data(scale_bool, mode):
    tag = "scaled" if scale_bool else "unscaled"
    df_d = du.prepare_and_align_data(cfg.DISEASE_GENES_PATH, theta_path=cfg.THETA_PATH, mode=mode)
    df_h = du.prepare_and_align_data(cfg.HEALTHY_GENES_PATH, theta_path=None)
    df_combined = pd.concat([df_h, df_d]).sample(frac=1, random_state=42)
    tournament_split_path = cfg.get_split_path_new("disease", tag, True, mode)
    train_df, test_df = du.get_split_data(df_combined, split_path=tournament_split_path)
    disease_only_split_path = cfg.get_split_path_new("disease", tag, False, mode)
    disease_df_train, disease_df_test = du.get_split_data(df_d, split_path=disease_only_split_path)
    return train_df, test_df, disease_df_train, disease_df_test


def analyze_d_portion_recon_new(labels_dict, scale_bool, save_path, mode):
    """
    Can deal with a mixed dataset, will divide to 4 plots where we have the 2 plots of 500 genes and then 
    plot separatley for the healthy samples in the mix, and for the disease samples in the mix
    """
    mix_disease, true_disease  = load_reconstruction_data('disease') 
    _, true_healthy = load_reconstruction_data('healthy')
    if mix_disease is None: return
    tag = "scaled" if scale_bool else "unscaled"
    input_size = mix_disease.shape[1]
    print(f"input size {input_size}") 
    gene_size = input_size
    print(f"gene size {gene_size}") 
    train_df, test_df, _, _ = fix_df_data(scale_bool=scale_bool, mode=mode) ## The test has the mix samples!! of healthy and disease
    print (f'test size is {test_df.shape}')
    test_no_theta_t = torch.Tensor(test_df.drop(columns=['theta_value']).values).float()
    test_theta_t = torch.Tensor(test_df[['theta_value']].values).float()
    test_w_theta_t = torch.Tensor(test_df.values).float()
    for base_name, models in labels_dict.items():
        n_rows = len(cfg.ENCODING_SIZES)
        n_cols = len(models)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False)
        fig.suptitle(f"Signal Reconstruction (Phase: DISEASE MIX | Base: {base_name.upper()})\n"
                     f"Branch Reconstruction Comparison to Ground Truth  (theta: {mode})", 
                     fontsize=20, fontweight='bold', y=0.98)

        for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
            for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                ax = axes[row_idx, col_idx]
                try:
                    parts = folder_tag.split('_H-')
                    h_and_d = parts[1].split('_D-')
                    h_type, d_type = h_and_d[0], h_and_d[1]
                    h_model = ModelFactory.create_model(h_type, gene_size, enc, cfg.H1, cfg.H2)
                    d_model = ModelFactory.create_model(d_type, gene_size, enc, cfg.H1, cfg.H2)
                    model = ModelFactory.create_mix_model(h_model, d_model)
                    is_pca = "pca" in d_type.lower()
                    ext = "model.joblib" if is_pca else "model.pt"
                    model_path = cfg.get_path_new('disease', tag, folder_tag, enc, cfg.MODELS_SUBFOLDER, all_or_no=True) / ext
                    if not model_path.exists():
                        ax.text(0.5, 0.5, "Model Not Found", ha='center'); continue
                    
                    if is_pca:
                        pca_sk = joblib.load(model_path)
                        model.disease.mean.data = torch.tensor(pca_sk.mean_, dtype=torch.float32)
                        model.disease.components.data = torch.tensor(pca_sk.components_, dtype=torch.float32)
                    else:
                        checkpoint = torch.load(model_path, map_location="cpu")
                        if isinstance(checkpoint, dict):
                            state_dict = checkpoint.get('model_state_dict', 
                                        checkpoint.get('best_state', 
                                        checkpoint))
                        else:
                            state_dict = checkpoint
                        model.load_state_dict(state_dict)
                    # Check which branches actually have weights
                    print("--- [Weight Check] ---")
                    has_healthy_weights = any(p.sum() != 0 for p in model.healthy.parameters())
                    has_disease_weights = any(p.sum() != 0 for p in model.disease.parameters())

                    print(f"Healthy Branch has non-zero weights: {has_healthy_weights}")
                    print(f"Disease Branch has non-zero weights: {has_disease_weights}")

                    # List the first few keys to ensure they match the 'healthy.' and 'disease.' prefix
                    print(f"First 3 state_dict keys: {list(model.state_dict().keys())[:3]}")
                    model.eval()
                    print(f"test_w_theta_t size is {test_w_theta_t} and size is {test_w_theta_t.shape}")
                    with torch.no_grad():
                        model_outputs = model(test_w_theta_t)
                        is_mix = "mix" in folder_tag
                        if not is_mix: raise ValueError("why is it not a mix model")
                        recon_mix, recon_d, recon_h, _ = model_outputs
                    test_truth_disease = true_disease.reindex(test_df.index).values
                    test_truth_healthy = true_healthy.reindex(test_df.index).values
                    num_samples = test_no_theta_t.shape[0]
                    sample_is_disease = (test_df['theta_value'] > 0).values # Boolean array [num_samples]                   

                    benchmark_truth = np.zeros_like(test_truth_healthy)
                    for i in range(num_samples):
                        if sample_is_disease[i]:
                            benchmark_truth[i] = test_truth_disease[i] # Target: Pure Cancer
                        else:
                            benchmark_truth[i] = test_truth_healthy[i] # Target: Pure Healthy

                    flat_benchmark_truth = benchmark_truth.flatten()
                    flat_d_recon = recon_d.detach().cpu().numpy().flatten()
                    flat_h_recon = recon_h.detach().cpu().numpy().flatten()
 
                    sample_labels = np.where(sample_is_disease, "Disease Sample", "Healthy Sample")
                    gene_template = np.array(['Healthy Genes (0-499)'] * 500 + ['Disease Genes (500-999)'] * 500)
                    flat_gene_labels = np.tile(gene_template, num_samples)
                    flat_sample_labels = np.repeat(sample_labels, 1000)

                    # Combine into a single categorical label: "Disease Sample | Disease Genes", etc.
                    flat_combined_labels = [f"{s}\n{g}" for s, g in zip(flat_sample_labels, flat_gene_labels)]

                    # 5. Build the Plotting DataFrame
                    plot_df = pd.DataFrame({
                        'Expression': np.concatenate([flat_benchmark_truth, flat_h_recon, flat_d_recon]),
                        'Source': (['Benchmark Truth'] * len(flat_benchmark_truth) + 
                                ['Healthy Branch (Frozen)'] * len(flat_h_recon) + 
                                ['Disease Branch (Trainable)'] * len(flat_d_recon)),
                        'Module Group': np.tile(flat_combined_labels, 3)
                    })
                    # plot_df = pd.DataFrame({
                    #     'Expression': np.concatenate([flat_benchmark_truth, flat_disease_recon]),
                    #     'Source': (['Benchmark Truth'] * len(flat_benchmark_truth) + ['Disease Branch Recon'] * len(flat_disease_recon)),
                    #     'Module Group': np.concatenate([flat_combined_labels, flat_combined_labels])
                    # })
                    plot_order = [
                        "Healthy Sample\nHealthy Genes (0-499)", 
                        "Healthy Sample\nDisease Genes (500-999)",
                        "Disease Sample\nHealthy Genes (0-499)", 
                        "Disease Sample\nDisease Genes (500-999)"
                    ]
                    # 3. Plot with specific colors for clarity
                    sns.boxplot(
                        data=plot_df, 
                        x='Module Group', 
                        y='Expression', 
                        hue='Source', 
                        ax=ax, 
                        palette=['#95a5a6', '#2ecc71', '#3498db'], # Gray, Green, Blue
                        showfliers=False, 
                        order=plot_order
                    )

                    # 4. Clean up the labels (Small and Centered)
                    ax.tick_params(axis='x', labelsize=7)
                    plt.setp(ax.get_xticklabels(), rotation=15, ha="center", rotation_mode="anchor")
                    ax.legend(fontsize='x-small', title_fontsize='8', loc='upper right')
                    # plt.setp(ax.get_xticklabels(), rotation=15, ha="center")
                    ax.axhline(0, color="#f97a7a", linestyle='--', alpha=0.3, label="Target 0")
                    ax.axhline(100, color="#85f492", linestyle='--', alpha=0.3, label="Target 100")
                    ax.set_ylim(-10, 150)

                    if row_idx == 0: ax.set_title(model_label, fontweight='bold')
                    if col_idx == 0: ax.set_ylabel(f"Enc: {enc}\nExpression")

                except Exception as e:
                    traceback.print_exc()
                    ax.text(0.5, 0.5, "Inference Error", ha='center', color='red')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_folder = cfg.get_path_new("disease", folder_type=cfg.PLOTS_SUBFOLDER, all_or_no=True) / f"Tournament_H-{base_name}"
        os.makedirs(out_folder, exist_ok=True)
        plt.savefig(out_folder / f"{save_path}_{tag}.png", dpi=150)
        plt.close()


def analyze_disease_mix_reconstruction(labels_dict, scale_bool, save_path, mode):
    """
    Analyzes Step 2 (Mix Model) disease branch. 
    Showcases how the branch recovers the Pure Disease signal from a Mixed Input.
    """
    # 1. Load Data (Phase is 'disease_mix')
    input_df, truth_df = load_reconstruction_data('disease')
    if input_df is None: return
    
    num_samples = len(input_df)
    tag = "scaled" if scale_bool else "unscaled"
    input_size = input_df.shape[1] 
    train_all_df, test_all_df, train_d_df_test_d_df = fix_df_data(scale_bool, mode)
    input_genes_matrix = test_all_df.drop(columns=['theta_value']).values
    # 3. PREPARE TENSORS
    # We use your get_ready_tensors_df to handle scaling and theta attachment
    train_tensor, test_tensor, _ = du.get_ready_tensors_df(train_all_df, test_all_df, scale_bool) # Using test as both for helper
    
    num_samples = len(test_all_df)
    input_size = test_all_df.shape[1] - 1 # Subtract 1 for the theta column
    
    # 4. LOAD PURE TRUTH TEMPLATES (For the modular boxes)
    # We need to align these with the specific samples in our test_all_df
    pure_h_truth_full = pd.read_csv(cfg.SYN_MIX_HEALTHY_PART, index_col=0).T
    pure_d_truth_full = pd.read_csv(cfg.SYN_MIX_DISEASE_PART, index_col=0).T
    test_truth = pure_d_truth_full.reindex(test_all_df.index).fillna(0.0)
    # Load Theta values for inference
    test_thetas = torch.tensor(test_all_df[['theta_value']].values).float().to("cpu")

    # Re-verify the input tensor has exactly what the model saw during training
    test_genes = torch.tensor(test_all_df.drop(columns=['theta_value']).values).float().to("cpu")
    test_tensor = torch.cat([test_genes, test_thetas], dim=1)

    # Phase 2 iterates through the different Healthy Bases
    for base_name, models in labels_dict.items():
        n_rows = len(cfg.ENCODING_SIZES)
        n_cols = len(models)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False)
        fig.suptitle(f"Disease Signal Recovery (Phase: DISEASE MIX | Base: {base_name.upper()})\n"
                     f"Comparing Mixed Input vs. Disease Branch Pure Reconstruction", 
                     fontsize=20, fontweight='bold', y=0.98)

        for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
            for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                ax = axes[row_idx, col_idx]
                is_mix = "mix" in folder_tag
                try:
                    # --- LOAD MIX MODEL ---
                    if is_mix:
                        parts = folder_tag.split('_H-')
                        h_and_d = parts[1].split('_D-')
                        h_type, d_type = h_and_d[0], h_and_d[1]
                        h_model = ModelFactory.create_model(h_type, input_size, enc, cfg.H1, cfg.H2)
                        d_model = ModelFactory.create_model(d_type, input_size, enc, cfg.H1, cfg.H2)
                        model = ModelFactory.create_mix_model(h_model, d_model)
                    else:
                        model = ModelFactory.create_model(folder_tag, input_size, enc, cfg.H1, cfg.H2)

                    is_pca = "pca" in d_type.lower()
                    ext = "model.joblib" if is_pca else "model.pt"
                    model_path = cfg.get_path('disease', tag, folder_tag, enc, cfg.MODELS_SUBFOLDER) / ext
                    if not model_path.exists():
                        ax.text(0.5, 0.5, "Model Not Found", ha='center'); continue

                    if is_pca:
                        pca_sk = joblib.load(model_path)
                        target = model.disease if is_mix else model
                        target.mean.data = torch.tensor(pca_sk.mean_, dtype=torch.float32)
                        target.components.data = torch.tensor(pca_sk.components_, dtype=torch.float32)
                    else:
                        checkpoint = torch.load(model_path, map_location="cpu")
                        if isinstance(checkpoint, dict):
                            state_dict = checkpoint.get('model_state_dict', 
                                        checkpoint.get('best_state', 
                                        checkpoint))
                        else:
                            state_dict = checkpoint
                        # state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint
                        model.load_state_dict(state_dict)

                    # --- INFERENCE ---
                    model.eval()
                    with torch.no_grad():
                        if is_mix:
                            outputs = model(test_tensor)
                            # unpacked as (mix_recon, disease_recon, healthy_recon, latent)
                            recon_mix, recon_d, recon_h, _ = outputs
                        else:
                            raise ValueError("why am I here???????? ")
                            
                            # Standalone models return (recon, latent) or just recon
                            recon_h = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                            recon_d = torch.zeros_like(recon_h) # Step 1 has no disease branch
                    # --- DYNAMIC TARGET GROUPING ---

                    # parts = folder_tag.split('_H-')
                    # h_type, d_type = parts[1].split('_D-')[0], parts[1].split('_D-')[1]
                    
                    # h_model = ModelFactory.create_model(h_type, input_size, enc, cfg.H1, cfg.H2)
                    # d_model = ModelFactory.create_model(d_type, input_size, enc, cfg.H1, cfg.H2)
                    # model = ModelFactory.create_mix_model(h_model, d_model)

                    # model_path = cfg.get_path('disease', tag, folder_tag, enc, cfg.MODELS_SUBFOLDER) / "model.pt"
                    # if not model_path.exists(): continue

                    # checkpoint = torch.load(model_path, map_location="cpu")
                    # state_dict = checkpoint.get('model_state_dict', checkpoint.get('best_state', checkpoint))
                    # model.load_state_dict(state_dict)

                    # # --- INFERENCE ---
                    # model.eval()
                    # with torch.no_grad():
                    #     # UniversalMixModel uses the 1001-column tensor
                    #     recon_mix, recon_d, _, _ = model(test_tensor)

                    # --- MODULE LOGIC (0-499 vs 500-999) ---
                    flat_input = input_genes_matrix.flatten()
                    flat_recon = recon_mix.cpu().numpy().flatten()      # The Model's attempt at 100%

                    # 1. Get dimensions
                    num_samples = input_genes_matrix.shape[0]
                    num_genes = input_genes_matrix.shape[1] # Should be 1000

                    # 2. Create the per-sample template: 500 healthy labels, 500 disease labels
                    module_template = np.array(["Healthy Module (0-499)"] * 500 + ["Disease Module (500-999)"] * 500)

                    # 3. Repeat this template for every sample in the test set
                    flat_labels = np.tile(module_template, num_samples)

                    # 4. Build the Plotting DataFrame
                    plot_df = pd.DataFrame({
                        'Expression': np.concatenate([flat_input, flat_recon]),
                        'Source': (['Input'] * len(flat_input) + ['Reconstructed'] * len(flat_recon)),
                        'Module': np.concatenate([flat_labels, flat_labels])
                    })
                    # labels = np.where(flat_input > 75, " (Input ~100)", 
                    #          np.where(flat_input > 25, " (Input ~50)", "Low Signal (Input ~0)"))
                    
                    # plot_df = pd.DataFrame({
                    #     'Expression': np.concatenate([flat_input, flat_recon]),
                    #     'Source': (['Input'] * len(flat_input) + ['Recon'] * len(flat_recon)),
                    #     'Module': np.concatenate([labels, labels])
                    # })

                    # --- PLOTTING ---
                    sns.boxplot(data=plot_df, x='Module', y='Expression', hue='Source', ax=ax, 
                                palette=['#95a5a6', '#3498db'], showfliers=False)
                    
                    ax.axhline(0, color='blue', linestyle='--', alpha=0.3, label="Target 0")
                    ax.axhline(100, color='green', linestyle='--', alpha=0.3, label="Target 100")
                    ax.set_ylim(-10, 150)

                    if row_idx == 0: ax.set_title(model_label, fontweight='bold')
                    if col_idx == 0: ax.set_ylabel(f"Enc: {enc}\nExpression")

                except Exception as e:
                    traceback.print_exc()
                    ax.text(0.5, 0.5, "Inference Error", ha='center', color='red')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER) / f"Tournament_H-{base_name}"
        os.makedirs(out_folder, exist_ok=True)
        plt.savefig(out_folder / f"{save_path}_{tag}.png", dpi=150)
        plt.close()


##### graph for when we learned only with disease samples #####
def analyze_disease_portion_reconstruction(labels_dict, scale_bool, save_path, mode):
    mix_disease, true_disease  = load_reconstruction_data('disease') 
    theta = pd.read_csv(cfg.THETA_PATH, index_col=0)
    if mode == 'fixed': 
        theta = mix_disease['theta_value'] = 0.5
    elif mode == 'true':
        mix_disease['theta_value'] = theta.iloc[:, 0]
    else:
        raise ValueError("what theta do i even have??")
    
    if mix_disease is None: return
    tag = "scaled" if scale_bool else "unscaled"
    input_size = mix_disease.shape[1]
    print(f"input size {input_size}") 
    gene_size = input_size-1
    print(f"gene size {gene_size}") 

    tournament_split_path = cfg.get_split_path("disease", tag)
    train_df, test_df = du.get_split_data(mix_disease, split_path=tournament_split_path)
    train_all_try, test_all_try, train_disease_try, test_disease_try = fix_df_data(scale_bool=scale_bool, mode=mode)
    if train_df.equals(train_disease_try) and test_df.equals(test_disease_try):
        print ("THE TRAIN/TEST DFS FOR DISEASE ONLY ARE EQUAL LIKE THIS")
    else:
        raise ValueError("TRAIN/TEST WERE NOT EQUAL")
    print (f'test size is {test_df.shape}')
    test_no_theta_t = torch.Tensor(test_df.drop(columns=['theta_value']).values).float()
    test_theta_t = torch.Tensor(test_df[['theta_value']].values).float()
    test_w_theta_t = torch.Tensor(test_df.values).float()
    test_size = test_df.shape[1]-1
    print(f"test size after taking off theta is: {test_size}")
    for base_name, models in labels_dict.items():
        n_rows = len(cfg.ENCODING_SIZES)
        n_cols = len(models)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False)
        fig.suptitle(f"Disease Signal Reconstruction (Phase: DISEASE MIX | Base: {base_name.upper()})\n"
                     f"Comparing Pure Disease Input vs. Disease Branch Reconstruction (theta: {mode})", 
                     fontsize=20, fontweight='bold', y=0.98)

        for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
            for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                ax = axes[row_idx, col_idx]
                try:
                    parts = folder_tag.split('_H-')
                    h_and_d = parts[1].split('_D-')
                    h_type, d_type = h_and_d[0], h_and_d[1]
                    h_model = ModelFactory.create_model(h_type, gene_size, enc, cfg.H1, cfg.H2)
                    d_model = ModelFactory.create_model(d_type, gene_size, enc, cfg.H1, cfg.H2)
                    model = ModelFactory.create_mix_model(h_model, d_model)
                    is_pca = "pca" in d_type.lower()
                    ext = "model.joblib" if is_pca else "model.pt"
                    model_path = cfg.get_path('disease', tag, folder_tag, enc, cfg.MODELS_SUBFOLDER) / ext
                    if not model_path.exists():
                        ax.text(0.5, 0.5, "Model Not Found", ha='center'); continue
                    
                    if is_pca:
                        pca_sk = joblib.load(model_path)
                        model.disease.mean.data = torch.tensor(pca_sk.mean_, dtype=torch.float32)
                        model.disease.components.data = torch.tensor(pca_sk.components_, dtype=torch.float32)
                    else:
                        checkpoint = torch.load(model_path, map_location="cpu")
                        if isinstance(checkpoint, dict):
                            state_dict = checkpoint.get('model_state_dict', 
                                        checkpoint.get('best_state', 
                                        checkpoint))
                        else:
                            state_dict = checkpoint
                        model.load_state_dict(state_dict)
                    model.eval()
                    print(f"test_w_theta_t size is {test_w_theta_t} and size is {test_w_theta_t.shape}")
                    with torch.no_grad():
                        model_outputs = model(test_w_theta_t)
                        is_mix = "mix" in folder_tag
                        if not is_mix: raise ValueError("why is it not a mix model")
                        recon_mix, recon_d, recon_h, _ = model_outputs
                    
                    test_truth_disease = true_disease.reindex(test_df.index)
                    
                    flat_disease_input = test_truth_disease.values.flatten()
                    flat_disease_recon = recon_d.numpy().flatten()
                    print(f"Input flattened length: {len(flat_disease_input)}")
                    print(f"Recon flattened length: {len(flat_disease_recon)}")

                    template = np.array(['Healthy Genes (0-499)'] * 500 + ['Disease Genes (500-999)'] * 500)
                    num_samples = test_no_theta_t.shape[0]

                    flat_labels = np.tile(template, num_samples)
                    if len(flat_labels) != len(flat_disease_input) or len(flat_labels) != len(flat_disease_recon):
                        raise ValueError("not the same length!!")
                    plot_df = pd.DataFrame({
                        'Expression': np.concatenate([flat_disease_input, flat_disease_recon]),
                        'Source': (['Input'] * len(flat_disease_input) + ['Reconstructed'] * len(flat_disease_recon)),
                        'Module': np.concatenate([flat_labels, flat_labels])
                    })
                    
                    sns.boxplot(data=plot_df, x='Module', y='Expression', hue='Source', ax=ax, 
                                palette=['#95a5a6', '#3498db'], showfliers=False)
                    
                    ax.axhline(0, color="#f97a7a", linestyle='--', alpha=0.3, label="Target 0")
                    ax.axhline(100, color="#85f492", linestyle='--', alpha=0.3, label="Target 100")
                    ax.set_ylim(-10, 150)

                    if row_idx == 0: ax.set_title(model_label, fontweight='bold')
                    if col_idx == 0: ax.set_ylabel(f"Enc: {enc}\nExpression")

                except Exception as e:
                    traceback.print_exc()
                    ax.text(0.5, 0.5, "Inference Error", ha='center', color='red')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER) / f"Tournament_H-{base_name}"
        os.makedirs(out_folder, exist_ok=True)
        plt.savefig(out_folder / f"{save_path}_{tag}.png", dpi=150)
        plt.close()




def interpret_disease_mix(phase='disease', mode="true"):

    gene_names = [f"Gene_{i}" for i in range(1000)]
    theta_df = pd.read_csv(cfg.THETA_PATH)
    # theta_values = pd.to_numeric(theta_df.iloc[:, -1], errors='coerce').dropna().values
    # input_df, truth_df = load_reconstruction_data(phase)
    # all possible combinations of healthy baselines with disease
    disease_mix_labels = {
        'PCA':
        {   "pca": "mix_H-pca_D-pca",
            "ae_basic": "mix_H-pca_D-ae_basic",
            "ae_layered": "mix_H-pca_D-ae_layered"
            
        }
        # ,
        # 'AE-Basic':
        # {
        #     "ae_basic": "mix_H-ae_basic_D-ae_basic",
        #     "ae_layered": "mix_H-ae_basic_D-ae_layered",
        #     "pca": "mix_H-ae_basic_D-pca"
        # },
        # 'AE-Layered':
        # {
        #     "ae_basic": "mix_H-ae_layered_D-ae_basic",
        #     "ae_layered": "mix_H-ae_layered_D-ae_layered",
        #     "pca": "mix_H-ae_layered_D-pca"
        # }
    }

    # for baseline, labels in disease_mix_labels.items():

        ##unscaled data reconstructions
    # analyze_reconstruction_grid(disease_mix_labels, phase='disease', 
    #                                 scale_bool=False, save_path="reconstructed_grid_boxplot", 
    #                                 )
    # analyze_reconstruction_combined(disease_mix_labels, phase='disease', 
    #                                 scale_bool=False, save_path="new_reconstructed_grid", 
    #                                 )
    # analyze_modular_reconstruction(disease_mix_labels, phase='disease', 
    #                                 scale_bool=False, save_path="modular_recon", 
    #                                 )

    # analyze_disease_mix_reconstruction(disease_mix_labels, False, save_path="Mix_recon", mode=mode)
    analyze_d_portion_recon_new(disease_mix_labels, False, save_path="Mix_disease_recon", mode=mode)
    # print("################### DISEASE PORTION RECON FUNCTION ###################")
    # analyze_disease_portion_reconstruction(disease_mix_labels, False, save_path="Disease_reconstruction", mode=mode)

    all_results = []
    # for base_label, models in disease_mix_labels.items():
    #     plot_consolidated_heatmaps(base_label, models)
    #     h_arch = arch_map[base_label]
        
    #     for model_label, folder_tag in models.items():
    #         d_arch = folder_tag.split("_D-")[-1]
            
    #         for enc in cfg.ENCODING_SIZES:
    #             model = load_revived_model(h_arch, d_arch, enc)
    #             if model is None: continue


    #             # 2. Statistical Proof: Theta-Invariant Correlation
    #             with torch.no_grad():
    #                 input_tensor = torch.tensor(input_df.values).float()
    #                 output = model(input_tensor)
    #                 recon = output[0] if isinstance(output, tuple) else output
    #                 recon_df = pd.DataFrame(recon.numpy(), columns=truth_df.columns)
    #             # print(theta_values)
    #             r_std, r_partial = calculate_theta_invariant_correlation(truth_df, recon_df, theta_values)
    #             print(f"Validated {folder_tag} Enc {enc}: Partial R = {r_partial:.4f}")   
    #             all_results.append({
    #             'h_arch': base_label,
    #             'd_arch': folder_tag.split("_D-")[-1],
    #             'partial_r': r_partial,
    #             'enc': enc
    #             })
    # save_performance_comparison(all_results)


def analyze_healthy_reconstruction(labels_dict, scale_bool, save_path):
    """
    Analyzes Step 1 (Healthy Model) reconstruction by splitting genes into 
    Active (0-499) and Inactive (500-999) healthy modules.
    """
    # 1. Load Data (Phase is strictly 'healthy')
    input_df, truth_df = load_reconstruction_data('healthy')
    if input_df is None: return
    
    num_samples = len(input_df)
    tag = "scaled" if scale_bool else "unscaled"
    input_size = input_df.shape[1] # Should be 1000

    # Load Pure Healthy Truth (Active/Inactive template)
    pure_h_truth = pd.read_csv(cfg.SYN_MIX_HEALTHY_PART, index_col=0).T
    
    # Iterate through model architectures (PCA, AE, etc.)
    for base_name, models in [("Standalone", labels_dict)]:
        n_rows = len(cfg.ENCODING_SIZES)
        n_cols = len(models)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False)
        fig.suptitle(f"Healthy Baseline Fidelity (Phase: HEALTHY)\n"
                     f"Comparing Active (0-499) vs Inactive (500-999) Modules", 
                     fontsize=20, fontweight='bold', y=0.98)

        for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
            for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                ax = axes[row_idx, col_idx]
                
                try:
                    # --- LOAD HEALTHY MODEL ---
                    model = ModelFactory.create_model(folder_tag, input_size, enc, cfg.H1, cfg.H2)
                    
                    is_pca = "pca" in folder_tag.lower()
                    ext = "model.joblib" if is_pca else "model.pt"
                    model_path = cfg.get_path('healthy', tag, folder_tag, enc, cfg.MODELS_SUBFOLDER) / ext
                    
                    if not model_path.exists(): continue

                    # Flexible loading for PCA or AE
                    if is_pca:
                        pca_sk = joblib.load(model_path)
                        model.mean.data = torch.tensor(pca_sk.mean_, dtype=torch.float32)
                        model.components.data = torch.tensor(pca_sk.components_, dtype=torch.float32)
                    else:
                        checkpoint = torch.load(model_path, map_location="cpu")
                        state_dict = checkpoint.get('model_state_dict', 
                                     checkpoint.get('best_state', checkpoint))
                        model.load_state_dict(state_dict)

                    # --- INFERENCE ---
                    model.eval()
                    with torch.no_grad():
                        # Healthy models only expect 1000 genes
                        current_input = torch.tensor(input_df.values).float()
                        recon_h = model(current_input)
                        if isinstance(recon_h, (tuple, list)): recon_h = recon_h[0]

                    # --- MODULE LOGIC (0-499 vs 500-999) ---
                    # We compare actual Input vs Reconstruction for each module
                    flat_input = current_input.numpy().flatten()
                    flat_recon = recon_h.numpy().flatten()
                    flat_truth = pure_h_truth.iloc[:num_samples].values.flatten()

                    # Define groups based on the Healthy template values
                    labels = np.where(flat_truth > 50, "Healthy Active (0-499)", "Healthy Inactive (500-999)")

                    plot_df = pd.DataFrame({
                        'Expression': np.concatenate([flat_input, flat_recon]),
                        'Source': (['Original Input'] * len(flat_input) + ['Model Recon'] * len(flat_recon)),
                        'Module': np.concatenate([labels, labels])
                    })

                    # --- PLOTTING ---
                    sns.boxplot(data=plot_df, x='Module', y='Expression', hue='Source', ax=ax, 
                                palette=['#95a5a6', '#3498db'], showfliers=False)
                    
                    # Reference lines for 0 and 100
                    ax.axhline(0, color='red', linestyle='--', alpha=0.3)
                    ax.axhline(100, color='green', linestyle='--', alpha=0.3)
                    ax.set_ylim(-10, 130)

                    if row_idx == 0: ax.set_title(model_label, fontweight='bold')
                    if col_idx == 0: ax.set_ylabel(f"Enc: {enc}\nExpression")

                except Exception as e:
                    traceback.print_exc()
                    ax.text(0.5, 0.5, "Inference Error", ha='center', color='red')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = cfg.get_path('healthy', folder_type=cfg.PLOTS_SUBFOLDER)
        os.makedirs(out_path, exist_ok=True)
        plt.savefig(out_path / f"Healthy_Recon_Modular_{tag}.png", dpi=150)
        plt.close()

def interpret_healthy_model(phase='healthy'):

    # setting labels
    model_labels = {
        'Basic-AE': 'ae_basic',
        'Layered-AE': 'ae_layered',
        'PCA': 'pca'
        }
 
    # analyze_reconstruction_grid(model_labels, phase='healthy', 
    #                             scale_bool=False, 
    #                             save_path="reconstructed_grid_boxplot")
    # analyze_reconstruction_combined(model_labels, phase='healthy', 
    #                             scale_bool=False, 
    #                             save_path="new_reconstructed_grid_boxplot")
    # analyze_modular_reconstruction(model_labels, phase='healthy', 
    #                             scale_bool=False, 
    #                             save_path="modular_recon")
    analyze_healthy_reconstruction(model_labels, False, save_path="healthy_Recon")



if __name__ == '__main__':

    # TODO: fix logic, maybe from command lines arguments or something
    # # print(f'model type is: 'synthetic' if cfg.SYNTHETIC_DATA else 'synthetic'}\n\n')
    # print("########### RUNNING HEALTHY MODEL ############")
    # interpret_healthy_model()
    interpret_disease_mix(mode="fixed")
    # cfg.FIXED_THETA_EXP = True
    # print("########### RUNNING MIX MODEL FIXED 0.5 THETA ############")
    # interpret_disease_mix()
    # cfg.FIXED_THETA_EXP = False
    # print("########### RUNNING MIX MODEL UNIFORM THETA ############")
    # interpret_disease_mix()


