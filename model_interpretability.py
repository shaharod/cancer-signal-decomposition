import io
import joblib
import os
import config as cfg
import utils.analysis_utils as au

import utils.plots_utils as pu
import torch
import numpy as np
from scipy import stats

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
                    plot_df = pd.DataFrame({
                        'Value': np.concatenate([truth_raw, recon_raw]),
                        'Type': ['Truth'] * len(truth_raw) + ['Pred'] * len(recon_raw)
                    })
                    sns.violinplot(data=plot_df, x='Type', y='Value', ax=ax, hue='Type',
                    palette=['#bdc3c7', '#1abc9c'], inner=None, alpha=0.3, legend=False)

                    # 3. Overlay the Stripplot (Raw point variability)
                    sns.stripplot(data=plot_df, x='Type', y='Value', ax=ax, hue='Type',
                                palette=['#7a96a4', '#1abc9c'], size=2, alpha=0.15, jitter=True, legend=False)

                    # 4. Final Formatting
                    # Fix the Y-axis so we see the 0.5 mixing level (50) clearly
                    ax.set_ylim(-5, 110) 
                    ax.axhline(50, color='gray', linestyle='--', alpha=0.3, lw=1, label='Mix Baseline')

                    # Color the boxes specifically
                    for patch, color in zip(bp['boxes'], ['#7f8c8d', '#057861']):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.6)
                    
                    # # Colors: Ground Truth (Gray), Prediction (Teal)
                    # colors = ['#bdc3c7', '#1abc9c']
                    # for patch, color in zip(bp['boxes'], colors):
                    #     patch.set_facecolor(color)
                    #     # We create a temporary DataFrame to make Seaborn happy
                    

                    # sns.stripplot(data=plot_df, x='Type', y='Value', ax=ax, hue='Type',
                    #             palette=['yellow', 'red'], # Darker versions of your box colors
                    #             size=2, alpha=0.3, jitter=True, legend=True) #legend=False

                    # # 3. Clean up formatting
                    # for patch, color in zip(bp['boxes'], ['#bdc3c7', '#1abc9c']):
                    #     patch.set_facecolor(color)
                    #     patch.set_alpha(0.5) # Lower alpha so points show through
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

import joblib
import torch
from core.models.model_factory import ModelFactory

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
import seaborn as sns
import matplotlib.pyplot as plt

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

def interpret_disease_mix(phase='disease'):

    gene_names = [f"Gene_{i}" for i in range(1000)]
    theta_df = pd.read_csv(cfg.THETA_PATH)
    # theta_values = pd.to_numeric(theta_df.iloc[:, -1], errors='coerce').dropna().values
    # input_df, truth_df = load_reconstruction_data(phase)
    # all possible combinations of healthy baselines with disease
    disease_mix_labels = {
        'PCA':
        {
            "Basic-AE": "mix_H-pca_D-ae_basic",
            "Layered-AE": "mix_H-pca_D-ae_layered",
            "PCA": "mix_H-pca_D-pca"
        },
        'AE-Basic':
        {
            "Basic-AE": "mix_H-ae_basic_D-ae_basic",
            "Layered-AE": "mix_H-ae_basic_D-ae_layered",
            "PCA": "mix_H-ae_basic_D-pca"
        },
        'AE-Layered':
        {
            "Basic-AE": "mix_H-ae_layered_D-ae_basic",
            "Layered-AE": "mix_H-ae_layered_D-ae_layered",
            "PCA": "mix_H-ae_layered_D-pca"
        }
    }

    # for baseline, labels in disease_mix_labels.items():

        ##unscaled data reconstructions
    analyze_reconstruction_grid(disease_mix_labels, phase='disease', 
                                    scale_bool=False, save_path="reconstructed_grid_boxplot", 
                                    )
    analyze_reconstruction_combined(disease_mix_labels, phase='disease', 
                                    scale_bool=False, save_path="new_reconstructed_grid", 
                                    )
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




def interpret_healthy_model(phase='healthy'):

    # setting labels
    model_labels = {
        'Basic-AE': 'ae_basic',
        'Layered-AE': 'ae_layered',
        'PCA': 'pca'
        }
 
    analyze_reconstruction_grid(model_labels, phase='healthy', 
                                scale_bool=False, 
                                save_path="reconstructed_grid_boxplot")
    analyze_reconstruction_combined(model_labels, phase='healthy', 
                                scale_bool=False, 
                                save_path="new_reconstructed_grid_boxplot")




if __name__ == '__main__':

    # TODO: fix logic, maybe from command lines arguments or something
    # print(f'model type is: 'synthetic' if cfg.SYNTHETIC_DATA else 'synthetic'}\n\n')
    # interpret_healthy_model()
    interpret_disease_mix()

    # if cfg.SYNTHETIC_DATA:    
    #     analyze_reconstruction()



