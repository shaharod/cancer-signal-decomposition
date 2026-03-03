import io
import traceback
import joblib
import os

from matplotlib.lines import Line2D
import config as cfg
import utils.analysis_utils as au
import utils.data_utils as du
import utils.model_utils as mu
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
from core.models.model_factory import ModelFactory

import pandas as pd

SCALED = True
UNSCALED = False

def load_reconstruction_data(phase, mode):
    """
    Loads the validation data (Mixed Input and Clean Ground Truth).
    Matches your requested structure using config paths.
    """
    if phase == "healthy":
        mix_file = cfg.HEALTHY_GENES_PATH  # Input is pure healthy data
        truth_file = cfg.HEALTHY_GENES_PATH
    else:
        mix_file =cfg.get_disease_gene_path(mode)  # Input is mixed data
        truth_file = cfg.DATA_SUB / 'pure_disease_truth.csv' # Truth is pure disease
    print(f"truth_file: {truth_file}")
    print(f"mix file: {mix_file}")
    # 2. Validation
    if not mix_file.exists():
        print(f"⚠️ Warning: Reconstruction data not found:\n {mix_file}")
        return None, None
    if not truth_file.exists():
        print(f"⚠️ Warning: truth file data not found:\n {truth_file}")
        return None, None
        
    # Load & Transpose (Genes should be columns for the model)
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

    # Standard Correlation (Pearson)
    r_standard, _ = stats.pearsonr(y_true, y_pred)

    # Partial Correlation Logic:
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

def get_best_model_tag(phase, scale_bool, encoding_size, is_mixed):
    """
    Scans the trained_models folder to find the model with the 
    lowest MSE for a specific encoding size.
    """
    tag = "scaled" if scale_bool else "unscaled"
    root = cfg.get_path(phase, folder_type=cfg.MODELS_SUBFOLDER, is_mixed=is_mixed) / tag
    
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

def load_revived_model(h_type, d_type=None, enc=8, scale_bool=False, is_mixed=False):
    """
    Rebuilds a standalone Healthy model OR a Disease Mix model.
    Handles both PyTorch (.pt) and PCA (.joblib) formats.
    """
    tag = "scaled" if scale_bool else "unscaled"
    input_size = 1000  # For your synthetic data

    # Helper to load a single component (Healthy or Disease)
    def prepare_component(m_type, phase, folder_tag):
        path = cfg.get_path(phase, tag, folder_tag, enc, cfg.MODELS_SUBFOLDER, is_mixed)
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

def plot_consolidated_heatmaps(base_name, models_dict, scale_bool=False, is_mixed=False):
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
    save_path = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=is_mixed
    ) / f"Consolidated_Heatmap_{base_name}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Consolidated heatmap saved to: {save_path}")
    plt.close()


## this function right now works with interpreting the models that were trained with all samples - healthy and disease
def analyze_d_portion_recon_new(labels_dict, scale_bool, save_path, mode):
    """
    Can deal with a mixed dataset, will divide to 4 plots where we have the 2 plots of 500 genes and then 
    plot separatley for the healthy samples in the mix, and for the disease samples in the mix
    """
    mix_disease, true_disease  = load_reconstruction_data('disease', mode) 
    _, true_healthy = load_reconstruction_data('healthy', mode)
    if mix_disease is None: return
    tag = "scaled" if scale_bool else "unscaled"

    input_size = mix_disease.shape[1]
    print(f"input size {input_size}") 
    gene_size = input_size
    print(f"gene size {gene_size}") 
    train_df, test_df = du.fix_df_data(scale_bool=scale_bool, mode=mode, is_mixed=True) ## The test has the mix samples!! of healthy and disease
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
        
        fig_scatter, axes_scatter = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)
        fig_scatter.suptitle("Input vs. Total Output", fontsize=20)

        for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
            for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                ax = axes[row_idx, col_idx]
                try:
                    recon_mix, recon_d, recon_h, _ = mu.create_load_mix_model(folder_tag=folder_tag, test_set=test_w_theta_t, 
                                                                       gene_size=gene_size, enc=enc, scale_tag=tag)
                    if recon_mix is None:
                        continue    
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
                    flat_recon_mix = recon_mix.detach().cpu().numpy().flatten()
                    
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
                    
                    plot_order = [
                        "Healthy Sample\nHealthy Genes (0-499)", 
                        "Healthy Sample\nDisease Genes (500-999)",
                        "Disease Sample\nHealthy Genes (0-499)", 
                        "Disease Sample\nDisease Genes (500-999)"
                    ]
                    selection_map = {
                        "Healthy Sample\nHealthy Genes (0-499)": ['Benchmark Truth', 'Healthy Branch (Frozen)'],
                        "Healthy Sample\nDisease Genes (500-999)": ['Benchmark Truth', 'Healthy Branch (Frozen)'],
                        "Disease Sample\nHealthy Genes (0-499)": ['Benchmark Truth', 'Disease Branch (Trainable)'],
                        "Disease Sample\nDisease Genes (500-999)": ['Benchmark Truth', 'Disease Branch (Trainable)']
                    }

                    # Apply the filter: Keep only the rows where the Source matches the Module Group choice
                    plot_df = plot_df[plot_df.apply(
                        lambda row: row['Source'] in selection_map.get(row['Module Group'], []), axis=1
                    )]
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
                    ax_scatter = axes_scatter[row_idx, col_idx]
                    
                    x_vals = test_w_theta_t[:, :1000].flatten().numpy() # Input
                    y_vals = recon_mix.flatten().numpy()                # Total Output
                    
                    # Use hexbin for speed/density with 1000s of points
                    # ax_scatter.hexbin(x_vals, y_vals, gridsize=30, cmap='Blues', mincnt=1)
                    color_dict = {"Healthy Sample": "#2ecc71", "Disease Sample": "#e74c3c"}
                    sns.scatterplot(x=x_vals, y=y_vals, s=1, alpha=0.4, hue=flat_sample_labels,
                                    ax=ax_scatter,
                                    hue_order=["Disease Sample", "Healthy Sample"],
                                       palette=color_dict, 
                                       edgecolors='none')
                    # Add Identity Line
                    max_val = max(x_vals.max(), y_vals.max())
                    # ax_scatter.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
                    # Check the labels themselves
                    unique_labels, counts = np.unique(flat_sample_labels, return_counts=True)
                    print(f"--- [PLOT AUDIT] ---")
                    for label, count in zip(unique_labels, counts):
                        print(f"Label: {label} | Count: {count}")

                    ax_scatter.plot([0, max_val], [0, max_val], color='#e74c3c', linestyle='--', linewidth=1, label='Identity')
                    r2 = r2_score(x_vals, y_vals)
                    pearson_r, _ = stats.pearsonr(x_vals, y_vals)

                    # Display them on the plot
                    text_str = f'$R^2 = {r2:.3f}$\nPearson $r = {pearson_r:.3f}$'
                    ax_scatter.text(0.05, 0.95, text_str, transform=ax_scatter.transAxes, 
                                    fontsize=10, verticalalignment='top', 
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

                    ax_scatter.set_title(f"{model_label} (Enc {enc})")

                    if col_idx == 0: ax_scatter.set_ylabel("Total Recon (H+D)")
                    if row_idx == n_rows - 1: ax_scatter.set_xlabel("Original Input")
                except Exception as e:
                    traceback.print_exc()
                    ax.text(0.5, 0.5, "Inference Error", ha='center', color='red')
        
        plt.figure(fig.number)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=True) / f"Tournament_H-{base_name}"
        os.makedirs(out_folder, exist_ok=True)
        plt.savefig(out_folder / f"{save_path}_{tag}_branches.png", dpi=150)
        plt.close()

        plt.figure(fig_scatter.number) # Focus on the scatter figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        fig_scatter.savefig(out_folder / f"{save_path}_{tag}_total.png", dpi=150)
        plt.close(fig_scatter)

##### graph for when we learned only with disease samples #####
def analyze_disease_portion_reconstruction(labels_dict, scale_bool, save_path, mode):
    mix_disease, true_disease  = load_reconstruction_data('disease', mode) 
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
    if gene_size != 1000:
        raise ValueError("in disease portion, something is wrongggg in gene size")
    
    tournament_split_path = cfg.get_split_path("disease", tag, False)
    train_df, test_df = du.get_split_data(mix_disease, split_path=tournament_split_path)
    train_disease_try, test_disease_try = du.fix_df_data(scale_bool=scale_bool, mode=mode, is_mixed=False)
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
        fig_scatter, axes_scatter = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)
        fig_scatter.suptitle(f"Disease Samples Training only: Input vs Recon\nBase: {base_name.upper()} | Theta: {mode}", 
                            fontsize=20, fontweight='bold', y=0.98)
        for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
            for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                ax = axes[row_idx, col_idx]
                ax_scatter = axes_scatter[row_idx, col_idx]
                try:
                    recon_mix, recon_d, recon_h, _ = mu.create_load_mix_model(folder_tag=folder_tag, test_set=test_w_theta_t, 
                                                                       gene_size=gene_size, enc=enc, scale_tag=tag)
                     
                    test_truth_disease = true_disease.reindex(test_df.index)
                    
                    flat_disease_input = test_truth_disease.values.flatten()
                    flat_disease_recon = recon_d.numpy().flatten()
                    x_vals = test_no_theta_t.flatten().numpy()
                    y_vals = recon_mix.flatten().numpy()
                    ax_scatter.scatter(x_vals, y_vals, s=1, alpha=0.1, color="#d20d0d", edgecolor='none')
                
                    # Identity Line (Target)
                    max_val = max(x_vals.max(), y_vals.max())
                    ax_scatter.plot([0, max_val], [0, max_val], color='#e74c3c', linestyle='--', linewidth=1, label='Identity')
                    r2 = r2_score(x_vals, y_vals)
                    pearson_r, _ = stats.pearsonr(x_vals, y_vals)

                    text_str = f'$R^2 = {r2:.3f}$\n$r = {pearson_r:.3f}$'
                    ax_scatter.text(0.05, 0.95, text_str, transform=ax_scatter.transAxes, 
                                    fontsize=12, verticalalignment='top', 
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax_scatter.set_title(f"{model_label} (Enc {enc})")
                    if col_idx == 0: ax_scatter.set_ylabel("Total Recon")
                    if row_idx == n_rows - 1: ax_scatter.set_xlabel("Total Input")
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

        plt.figure(fig.number)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=False) / f"Tournament_H-{base_name}"
        os.makedirs(out_folder, exist_ok=True)

        plt.savefig(out_folder / f"{save_path}_{tag}.png", dpi=150)
        plt.close()
        plt.figure(fig_scatter.number)
        fig_scatter.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_scatter.savefig(out_folder / f"{save_path}_{tag}_mix_vs_input.png")
        plt.close(fig_scatter)

def analyze_disease_portion_reconstruction_s(labels_dict, scale_bool, save_path, mode):
    """
    Plots the continuous True Pure Disease Profile against the Autoencoder's 
    Disease Branch Reconstruction (recon_d) using a scatter plot.
    """
    mix_disease, true_disease  = load_reconstruction_data('disease', mode) 
    theta = pd.read_csv(cfg.THETA_PATH, index_col=0)
    
    if mode == 'fixed': 
        theta = mix_disease['theta_value'] = 0.5
    elif mode == 'true':
        mix_disease['theta_value'] = theta.iloc[:, 0]
    else:
        raise ValueError("what theta do i even have??")
    
    if mix_disease is None: return
    
    tag = "scaled" if scale_bool else "unscaled"
    
    # 1. Prepare Splits
    tournament_split_path = cfg.get_split_path("disease", tag, False)
    train_q_type, test_w_type = du.get_split_data(mix_disease, split_path=tournament_split_path)
    train_df, test_df = du.fix_df_data(scale_bool=scale_bool, mode=mode, is_mixed=False)
    ## fix_df returns with theta only, load recon returns with theta and with type
    input_size = train_df.shape[1]
    gene_size = input_size - 1
    print(f"Input size: {input_size} | Gene size: {gene_size}") 
    
    print(f'Test size is {test_df.shape}')
    test_w_theta_t = torch.Tensor(test_df.values).float()

    # 2. Plotting Loop
    for base_name, models in labels_dict.items():
        n_rows = len(cfg.ENCODING_SIZES)
        n_cols = len(models)
        
        # Setup a single figure for the Scatter Plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)
        fig.suptitle(f"Disease Signal Isolation (Phase: DISEASE MIX | Base: {base_name.upper()})\n"
                     f"True Pure Disease vs. Disease Branch Output (theta: {mode})", 
                     fontsize=18, fontweight='bold', y=0.98)
                     
        for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
            for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                ax = axes[row_idx, col_idx]
                try:
                    recon_mix, recon_d, recon_h, _ = mu.create_load_mix_model(folder_tag=folder_tag, test_set=test_w_theta_t, 
                                                                       gene_size=gene_size, enc=enc, scale_tag=tag)
                    
                    # Extract Data for Plotting
                    test_truth_disease = true_disease.reindex(test_df.index)
                    
                    x_vals = test_truth_disease.values.flatten()  # Ground Truth Pure Disease
                    y_vals = recon_d.numpy().flatten()            # Disease Branch Output
                    
                    # Scatter Plot
                    # ax.scatter(x_vals, y_vals, s=1, alpha=0.3, color="#c8250c", edgecolor='none')

                    color_map = {"Disease A (CRC)": "#d43220", "Disease B (SCLC)": "#870fb6"}
                    disease_map = {1: "Disease A (CRC)", 2: "Disease B (SCLC)"}
                    test_truth_disease = true_disease.reindex(test_df.index)
                    flat_input = test_truth_disease.values.flatten()
                    flat_recon = recon_d.numpy().flatten()
                    if 'disease_type' in test_w_type.columns:
                        sample_labels = test_w_type['disease_type'].map(disease_map).fillna("Unknown")
                    else:
                        sample_labels = pd.Series(["Disease"] * len(test_df))
                        color_map = {"Disease": "#8e44ad"} # Fallback color
                        
                    # Multiply the patient labels so they match the flattened genes!
                    flat_labels = np.repeat(sample_labels.values, gene_size)

                    sns.scatterplot(
                        x=flat_input, 
                        y=flat_recon, 
                        hue=flat_labels, 
                        palette=color_map, 
                        s=1, 
                        alpha=0.3, 
                        ax=ax, 
                        edgecolor='none'
                    )
                    # Identity Line
                    max_val = max(np.nanmax(x_vals), np.nanmax(y_vals))
                    min_val = min(np.nanmin(x_vals), np.nanmin(y_vals))
                    ax.plot([min_val, max_val], [min_val, max_val], color="#9a1b0c", linestyle='--', linewidth=1, label='Identity')
                    
                    # Calculate & Display Metrics
                    r2 = r2_score(x_vals, y_vals)
                    pearson_r, _ = stats.pearsonr(x_vals, y_vals)
                    text_str = f'$R^2 = {r2:.3f}$\n$r = {pearson_r:.3f}$'
                    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
                            fontsize=12, verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Formatting
                    ax.set_title(f"{model_label} (Enc {enc})")
                    ax.set_xlabel("True Pure Disease (Ground Truth)")
                    ax.set_ylabel("Disease Branch Recon")
                    if col_idx == 0: ax.set_ylabel("Disease Branch Output")
                    if row_idx == n_rows - 1: ax.set_xlabel("True Pure Disease Input")
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    ax.text(0.5, 0.5, "Inference Error", ha='center', color='red')

        # 3. Save Figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=False) / f"Tournament_H-{base_name}"
        os.makedirs(out_folder, exist_ok=True)

        # Save single scatter compilation
        plt.savefig(out_folder / f"{save_path}_{tag}_disease_scatter.png", dpi=150)
        plt.close(fig)

#def plot_total_reconstruction_scatter(labels_dict, scale_bool, save_path, mode, is_mixed, disease_map=None, color_map=None):
#     """
#     Universal Scatter Plot: Input vs Total Output.
#     Automatically maps numeric disease types to readable legend labels.
#     """
#     tag = "scaled" if scale_bool else "unscaled"
    
#     # --- 💡 THE DICTIONARY ---
#     # Default mapping if don't pass one in the function call
#     if disease_map is None:
#         disease_map = {
#             0: "Healthy", 
#             1: "Disease 1", 
#             2: "Disease 2"
#         }
#     if color_map is None:
#         color_map = {
#             "Healthy": "#2ecc71",   # Emerald Green
#             "Disease 1": "#e74c3c", # Red
#             "Disease 2": "#9b59b6"  # Purple
#         }
#     # 1. Load the mixed dataset
#     train_df, test_df = du.fix_df_data(scale_bool=scale_bool, mode=mode, is_mixed=is_mixed)
    
#     # 2. Safely separate metadata from tensors
#     metadata_cols = ['theta_value']
#     if 'disease_type' in test_df.columns:
#         metadata_cols.append('disease_type')
        
#     test_genes = test_df.drop(columns=metadata_cols, errors='ignore')
#     gene_size = test_genes.shape[1]
    
#     # Create strict [Genes + Theta] tensor for the model
#     test_no_theta_t = torch.tensor(test_genes.values, dtype=torch.float32)
#     test_theta_t = torch.tensor(test_df[['theta_value']].values, dtype=torch.float32)
#     test_w_theta_t = torch.cat([test_no_theta_t, test_theta_t], dim=1)

#     #  MAP THE NUMBERS TO NAMES FOR THE LEGEND
#     if 'disease_type' in test_df.columns:
#         # This instantly translates [0, 1, 2] -> ["Healthy", "Disease 1", "Disease 2"]
#         # If a number isn't in the dict (e.g., 3), it safely leaves it as "3"
#         mapped_series = test_df['disease_type'].map(disease_map).fillna(test_df['disease_type'].astype(str))
#         sample_labels = mapped_series.values
#     else:
#         # Fallback if the column doesn't exist
#         sample_labels = np.where(test_df['theta_value'] > 0, disease_map.get(1, "Disease"), disease_map.get(0, "Healthy"))

#     # Repeat labels so every single gene point knows what patient it came from
#     flat_sample_labels = np.repeat(sample_labels, gene_size)

#     # 4. Loop through models and plot (The rest of the logic remains exactly the same!)
#     for base_name, models in labels_dict.items():
#         n_rows = len(cfg.ENCODING_SIZES)
#         n_cols = len(models)
        
#         fig_scatter, axes_scatter = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)
#         fig_scatter.suptitle(f"Total Signal Reconstruction (Phase: DISEASE MIX | Base: {base_name.upper()})\n"
#                              f"Input vs. Total Output (theta: {mode})", 
#                              fontsize=20, fontweight='bold', y=0.98)

#         for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
#             for col_idx, (model_label, folder_tag) in enumerate(models.items()):
#                 ax_scatter = axes_scatter[row_idx, col_idx]
#                 try:
#                     # Run Inference
#                     recon_mix, _, _, _ = mu.create_load_mix_model(
#                         folder_tag=folder_tag, test_set=test_w_theta_t, 
#                         gene_size=gene_size, enc=enc, scale_tag=tag
#                     )
                    
#                     if recon_mix is None:
#                         print(f"Where the hell is my reconstruced data????? with folder tag:\n {folder_tag}")
#                         continue    
                    
#                     x_vals = test_no_theta_t.flatten().numpy()
#                     y_vals = recon_mix.detach().cpu().numpy().flatten()
                    
#                     # Scatter Plot (Seaborn will automatically use your mapped dictionary strings!)
#                     sns.scatterplot(
#                         x=x_vals, y=y_vals, s=1, alpha=0.4, 
#                         hue=flat_sample_labels,
#                         ax=ax_scatter,
#                         edgecolors='none',
#                     palette=color_map

#                     )

#                     # Identity Line
#                     max_val = max(x_vals.max(), y_vals.max())
#                     min_val = min(x_vals.min(), y_vals.min())
#                     ax_scatter.plot([min_val, max_val], [min_val, max_val], color='#e74c3c', linestyle='--', linewidth=1, label='Identity')
                    
#                     # Metrics
#                     r2 = r2_score(x_vals, y_vals)
#                     pearson_r, _ = stats.pearsonr(x_vals, y_vals)
#                     text_str = f'$R^2 = {r2:.3f}$\nPearson $r = {pearson_r:.3f}$'
#                     ax_scatter.text(0.05, 0.95, text_str, transform=ax_scatter.transAxes, 
#                                     fontsize=10, verticalalignment='top', 
#                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

#                     # Formatting
#                     ax_scatter.set_title(f"{model_label} (Enc {enc})")
#                     if col_idx == 0: ax_scatter.set_ylabel("Total Recon (H+D)")
#                     if row_idx == n_rows - 1: ax_scatter.set_xlabel("Original Input")
#                     ax_scatter.legend(fontsize='x-small', title_fontsize='8', loc='lower right')

#                 except Exception as e:
#                     import traceback
#                     traceback.print_exc()
#                     ax_scatter.text(0.5, 0.5, "Inference Error", ha='center', color='red')
        
#         # Save Figure
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=is_mixed) / f"Tournament_H-{base_name}"
#         os.makedirs(out_folder, exist_ok=True)
        
#         fig_scatter.savefig(out_folder / f"{save_path}_{tag}_scatter.png", dpi=150)
#         plt.close(fig_scatter)

def _plot_simple_boxplot(ax, flat_input, flat_recon, num_samples, model_label, enc):
    """Handles the blocky synthetic data boxplots."""
    template = np.array(['Healthy Genes (0-499)'] * 500 + ['Disease Genes (500-999)'] * 500)
    flat_labels = np.tile(template, num_samples)
    
    plot_df = pd.DataFrame({
        'Expression': np.concatenate([flat_input, flat_recon]),
        'Source': (['True Input'] * len(flat_input) + ['Reconstruction'] * len(flat_recon)),
        'Module': np.concatenate([flat_labels, flat_labels])
    })
    
    sns.boxplot(data=plot_df, x='Module', y='Expression', hue='Source', ax=ax, 
                palette=['#95a5a6', '#3498db'], showfliers=False)
    
    ax.axhline(0, color="#f97a7a", linestyle='--', alpha=0.3)
    ax.axhline(100, color="#85f492", linestyle='--', alpha=0.3)
    ax.set_ylim(-10, 150)
    
    ax.set_xlabel("Gene Module")
    ax.set_ylabel(f"Enc: {enc}\nExpression Level")
    ax.set_title(model_label, fontweight='bold')

def _plot_complex_scatter(ax, flat_input, flat_recon, flat_labels, color_map, model_label, enc):
    """Handles the continuous real biological data scatter plots, colored by disease type."""
    sns.scatterplot(
        x=flat_input, 
        y=flat_recon, 
        hue=flat_labels, 
        palette=color_map, 
        s=1, 
        alpha=0.3, 
        ax=ax, 
        edgecolor='none',
        legend=False  # Crucial: Stops Matplotlib from freezing on loc="best"
    )
    
    # Identity Line
    max_val = max(np.nanmax(flat_input), np.nanmax(flat_recon))
    min_val = min(np.nanmin(flat_input), np.nanmin(flat_recon))
    ax.plot([min_val, max_val], [min_val, max_val], color="#9a1b0c", linestyle='--', linewidth=1)
    
    # Metrics
    r2 = r2_score(flat_input, flat_recon)
    pearson_r, _ = stats.pearsonr(flat_input, flat_recon)
    text_str = f'$R^2 = {r2:.3f}$\n$r = {pearson_r:.3f}$'
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    ax.set_xlabel("True Pure Disease (Ground Truth)")
    ax.set_ylabel(f"Enc: {enc}\nDisease Branch Recon")
    ax.set_title(model_label, fontweight='bold')

def _plot_simple_total_scatter(ax, flat_input, flat_recon, model_label, enc):
    """Handles the basic single-color scatter for simple synthetic data."""
    ax.scatter(flat_input, flat_recon, s=1, alpha=0.1, color="#d20d0d", edgecolor='none')
    
    max_val = max(np.nanmax(flat_input), np.nanmax(flat_recon))
    min_val = min(np.nanmin(flat_input), np.nanmin(flat_recon))
    ax.plot([min_val, max_val], [min_val, max_val], color='#e74c3c', linestyle='--', linewidth=1)
    
    r2 = r2_score(flat_input, flat_recon)
    pearson_r, _ = stats.pearsonr(flat_input, flat_recon)
    text_str = f'$R^2 = {r2:.3f}$\n$r = {pearson_r:.3f}$'
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    ax.set_title(f"{model_label} (Enc {enc})")
    ax.set_xlabel("Total Mixed Input")
    ax.set_ylabel("Total Mixed Recon")

def _plot_complex_total_scatter(ax, flat_input, flat_recon, flat_labels, color_map, model_label, enc):
    """Handles the 3-color scatter for real biological data."""
    sns.scatterplot(
        x=flat_input, 
        y=flat_recon, 
        hue=flat_labels, 
        palette=color_map, 
        s=1, 
        alpha=0.3, 
        ax=ax, 
        edgecolor='none',
        legend=False # Crucial for speed!
    )
    
    max_val = max(np.nanmax(flat_input), np.nanmax(flat_recon))
    min_val = min(np.nanmin(flat_input), np.nanmin(flat_recon))
    ax.plot([min_val, max_val], [min_val, max_val], color="#34495e", linestyle='--', linewidth=1)
    
    r2 = r2_score(flat_input, flat_recon)
    pearson_r, _ = stats.pearsonr(flat_input, flat_recon)
    text_str = f'$R^2 = {r2:.3f}$\n$r = {pearson_r:.3f}$'
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    ax.set_title(f"{model_label} (Enc {enc})")
    ax.set_xlabel("Total Mixed Input (Ground Truth)")
    ax.set_ylabel("Total Mixed Recon (Model Output)")

def analyze_total_reconstruction(labels_dict, scale_bool, save_path, mode, is_simple=False, is_mixed=False):
    """
    Evaluates the Total Mix Reconstruction (recon_mix).
    Colors the complex data by 3 classes: Healthy, Disease A, and Disease B.
    """
    # Load your mixed data
    mix_disease, _ = load_reconstruction_data('disease', mode) 
    theta = pd.read_csv(cfg.THETA_PATH, index_col=0)
    
    if mode == 'fixed': 
        mix_disease['theta_value'] = 0.5
    elif mode == 'true':
        mix_disease['theta_value'] = theta.iloc[:, 0]
    else:
        raise ValueError("Unknown theta mode!")
    
    if mix_disease is None: return
    
    tag = "scaled" if scale_bool else "unscaled"
    
    # 1. Prepare Splits
    tournament_split_path = cfg.get_split_path("disease", tag, False)
    train_w_type, test_w_type = du.get_split_data(mix_disease, split_path=tournament_split_path)
    train_df, test_df = du.fix_df_data(scale_bool=scale_bool, mode=mode, is_mixed=False)
    
    input_size = test_df.shape[1]
    gene_size = input_size - 1
    print(f"Total Recon -> Input size: {input_size} | Gene size: {gene_size}") 
    
    # test_df has theta dropped/isolated, so test_no_theta_t is purely genes
    test_no_theta_t = torch.Tensor(test_df.drop(columns=['theta_value'], errors='ignore').values).float()
    test_w_theta_t = torch.Tensor(test_df.values).float()

    # 2. Plotting Loop
    for base_name, models in labels_dict.items():
        n_rows = len(cfg.ENCODING_SIZES)
        n_cols = len(models)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)
        fig.suptitle(f"Total Mix Reconstruction (Phase: DISEASE MIX | Base: {base_name.upper()})\n"
                     f"Total Input vs. Total Recon (theta: {mode})", 
                     fontsize=18, fontweight='bold', y=0.98)
                     
        for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
            for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                ax = axes[row_idx, col_idx]
                try:
                    # Run Inference
                    recon_mix, recon_d, recon_h, _ = mu.create_load_mix_model(
                        folder_tag=folder_tag, test_set=test_w_theta_t, 
                        gene_size=gene_size, enc=enc, scale_tag=tag
                    )
                    
                    if recon_mix is None:
                        continue
                        
                    # Flatten the data for scatter
                    flat_input = test_no_theta_t.flatten().numpy()
                    flat_recon = recon_mix.detach().cpu().numpy().flatten()
                    
                    # 🔀 Route to correct plot
                    if is_simple:
                        _plot_simple_total_scatter(ax, flat_input, flat_recon, model_label, enc)
                    else:
                        # --- 3-CLASS COLOR MAPPING ---
                        disease_map = {0: "Healthy", 1: "Disease A (CRC)", 2: "Disease B (SCLC)"}
                        color_map = {
                            "Healthy": "#2ecc71",         # Green
                            "Disease A (CRC)": "#d43220", # Red
                            "Disease B (SCLC)": "#870fb6", # Purple
                            "Disease": "#d43220"      # Fallback Red
                        }
                        
                        if 'disease_type' in test_w_type.columns:
                            sample_labels = test_w_type['disease_type'].map(disease_map).fillna("Unknown")
                        else:
                            sample_labels = np.where(test_df['theta_value'] == 0, "Healthy", "Disease")
                            sample_labels = pd.Series(sample_labels)
                            
                        # Multiply labels by 20,000 genes
                        flat_labels = np.repeat(sample_labels.values, gene_size)
                        
                        _plot_complex_total_scatter(ax, flat_input, flat_recon, flat_labels, color_map, model_label, enc)
                        
                    # Clean up grid inner labels
                    if col_idx > 0: ax.set_ylabel("")
                    if row_idx < n_rows - 1: ax.set_xlabel("")
                    
                except Exception as e:
                    traceback.print_exc()
                    ax.text(0.5, 0.5, "Inference Error", ha='center', color='red')

        # 3. Add the Universal Master Legend for the 3 Classes
        if not is_simple:
            unique_classes = pd.unique(sample_labels) 
            
            legend_elements = []
            # Dynamically build a dot for each class present
            for cls in unique_classes:
                color = color_map.get(cls, "#7f8c8d")
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', label=cls, markerfacecolor=color, markersize=8)
                )
            
            # Add the identity line
            legend_elements.append(Line2D([0], [0], color='#34495e', linestyle='--', linewidth=1, label='Perfect Reconstruction'))

            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)

        # 4. Save Figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=False) / f"Tournament_H-{base_name}"
        os.makedirs(out_folder, exist_ok=True)

        data_tag = "simple" if is_simple else "complex"
        plt.savefig(out_folder / f"{save_path}_{tag}_{data_tag}_total_recon_scatter.png", dpi=150)
        plt.close(fig)

def analyze_disease_portion_reconstruction_scatter(labels_dict, scale_bool, save_path, mode, is_simple=False):
    """
    Main execution block for evaluating Disease Branch Reconstruction.
    Dynamically switches between Boxplots and Scatter plots based on is_simple.
    """
    mix_disease, true_disease  = load_reconstruction_data('disease', mode) 
    theta = pd.read_csv(cfg.THETA_PATH, index_col=0)
    
    if mode == 'fixed': 
        theta = mix_disease['theta_value'] = 0.5
    elif mode == 'true':
        mix_disease['theta_value'] = theta.iloc[:, 0]
    else:
        raise ValueError("what theta do i even have??")
    
    if mix_disease is None: return
    
    tag = "scaled" if scale_bool else "unscaled"
    
    # 1. Prepare Splits
    tournament_split_path = cfg.get_split_path("disease", tag, False)
    train_w_type, test_w_type = du.get_split_data(mix_disease, split_path=tournament_split_path)
    train_df, test_df = du.fix_df_data(scale_bool=scale_bool, mode=mode, is_mixed=False)
    
    input_size = train_df.shape[1]
    gene_size = input_size - 1
    
    if is_simple and gene_size != 1000:
        raise ValueError("Simple synthetic data expects exactly 1000 genes.")
        
    print(f"Input size: {input_size} | Gene size: {gene_size}") 
    print(f'Test size is {test_df.shape}')
    
    test_w_theta_t = torch.Tensor(test_df.values).float()

    # 2. Plotting Loop
    for base_name, models in labels_dict.items():
        n_rows = len(cfg.ENCODING_SIZES)
        n_cols = len(models)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)
        plot_type = "Boxplot" if is_simple else "Scatter"
        fig.suptitle(f"Disease Signal Isolation ({plot_type} | Base: {base_name.upper()})\n"
                     f"True Pure Disease vs. Disease Branch Output (theta: {mode})", 
                     fontsize=18, fontweight='bold', y=0.98)
                     
        for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
            for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                ax = axes[row_idx, col_idx]
                try:
                    # Run Inference using your updated utility function
                    recon_mix, recon_d, recon_h, _ = mu.create_load_mix_model(
                        folder_tag=folder_tag, test_set=test_w_theta_t, 
                        gene_size=gene_size, enc=enc, scale_tag=tag
                    )
                    
                    if recon_d is None:
                        continue
                        
                    # Extract Data
                    test_truth_disease = true_disease.reindex(test_df.index)
                    flat_input = test_truth_disease.values.flatten()
                    flat_recon = recon_d.detach().cpu().numpy().flatten()
                    
                    # 🔀 Route to the correct plot type
                    if is_simple:
                        _plot_simple_boxplot(ax, flat_input, flat_recon, test_w_theta_t.shape[0], model_label, enc)
                    else:
                        color_map = {"Disease A (CRC)": "#d43220", "Disease B (SCLC)": "#870fb6"}
                        disease_map = {1: "Disease A (CRC)", 2: "Disease B (SCLC)"}
                        
                        if 'disease_type' in test_w_type.columns:
                            sample_labels = test_w_type['disease_type'].map(disease_map).fillna("Unknown")
                        else:
                            sample_labels = pd.Series(["Disease"] * len(test_df))
                            color_map = {"Disease": "#8e44ad"} 
                            
                        flat_labels = np.repeat(sample_labels.values, gene_size)
                        _plot_complex_scatter(ax, flat_input, flat_recon, flat_labels, color_map, model_label, enc)
                        
                    # Clean up grid inner labels
                    if col_idx > 0: ax.set_ylabel("")
                    if row_idx < n_rows - 1: ax.set_xlabel("")
                    
                except Exception as e:
                    traceback.print_exc()
                    ax.text(0.5, 0.5, "Inference Error", ha='center', color='red')

        # 3. Add the Universal Master Legend (Only for Complex Data)
        if not is_simple:
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Disease A (CRC)', markerfacecolor='#d43220', markersize=8),
                Line2D([0], [0], marker='o', color='w', label='Disease B (SCLC)', markerfacecolor='#870fb6', markersize=8),
                Line2D([0], [0], color="#2D2A2A", linestyle='--', linewidth=1, label='Perfect Reconstruction')
            ]
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)

        # 4. Save Figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=False) / f"Tournament_H-{base_name}"
        os.makedirs(out_folder, exist_ok=True)

        data_tag = "simple" if is_simple else "complex"
        plt.savefig(out_folder / f"{save_path}_{tag}_{data_tag}_disease_recon.png", dpi=150)
        plt.close(fig)
        
def interpret_disease_mix(phase='disease', mode="true"):

    gene_names = [f"Gene_{i}" for i in range(1000)]
    theta_df = pd.read_csv(cfg.THETA_PATH)
    # theta_values = pd.to_numeric(theta_df.iloc[:, -1], errors='coerce').dropna().values
    # input_df, truth_df = load_reconstruction_data(phase)
    # all possible combinations of healthy baselines with disease
    labels_dict = {
        'PCA':
        {   "pca": "mix_H-pca_D-pca",
            "ae_basic": "mix_H-pca_D-ae_basic",
            "ae_layered": "mix_H-pca_D-ae_layered"
            
        }
    }


    ######### NOTE: these functions dont work with the more complex data ###############
    ##unscaled data reconstructions
    # analyze_d_portion_recon_new(labels_dict=labels_dict, scale_bool=False, save_path="analyze_recon_allSamples_dif", mode=mode)
    # # print("################### DISEASE PORTION RECON FUNCTION ###################")
    # analyze_disease_portion_reconstruction(labels_dict=labels_dict, scale_bool=False, save_path="analyze_recon_dSamplesOnly", mode=mode)

    ######################################################################################
    analyze_disease_portion_reconstruction_scatter(labels_dict=labels_dict, scale_bool=UNSCALED, save_path="dOnly_recon_vs_truth", mode=mode, is_simple=cfg.SYN_SIMPLE)

    analyze_total_reconstruction(
    labels_dict=labels_dict, 
    scale_bool=UNSCALED,                     
    save_path="total_recon",   
    mode=mode,                          
    is_simple=cfg.SYN_SIMPLE,
    is_mixed=False           
)
    analyze_total_reconstruction(labels_dict, UNSCALED, 
                                 "total_recon",
                                 mode, cfg.SYN_SIMPLE, True)
    
#     plot_total_reconstruction_scatter(
#     labels_dict=labels_dict, 
#     scale_bool=False, 
#     save_path="allSamples", 
#     mode=mode, 
#     is_mixed=True,
#     disease_map=None  # <-- Pass it right here!
# )    
#     plot_total_reconstruction_scatter(
#     labels_dict=labels_dict, 
#     scale_bool=False, 
#     save_path="dOnly", 
#     mode=mode, 
#     is_mixed=False,
#     disease_map=None  # <-- Pass it right here!
# )

    # all_results = []
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


def analyze_healthy_reconstruction(labels_dict, scale_bool, save_path, mode="true"):
    """
    Analyzes Step 1 (Healthy Model) reconstruction by splitting genes into 
    Active (0-499) and Inactive (500-999) healthy modules.
    """
    # 1. Load Data (Phase is strictly 'healthy')
    input_df, truth_df = load_reconstruction_data('healthy', mode)
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
                    model_path = cfg.get_path('healthy', tag, folder_tag, enc, cfg.MODELS_SUBFOLDER, False) / ext
                    
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
        out_path = cfg.get_path('healthy', folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=False)
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
 
    analyze_healthy_reconstruction(model_labels, False, save_path="healthy_Recon")



if __name__ == '__main__':

    # TODO: fix logic, maybe from command lines arguments or something
    # # print(f'model type is: 'synthetic' if cfg.SYNTHETIC_DATA else 'synthetic'}\n\n')
    # print("########### RUNNING HEALTHY MODEL ############")
    # interpret_healthy_model()
    cfg.FIXED_THETA_EXP = True
    cfg.DISEASE_GENES_PATH = cfg.DATA_SUB / "disease_data_theta05.csv"
    print("########### RUNNING MIX MODEL FIXED 0.5 THETA ############")
    interpret_disease_mix(mode="fixed")
    cfg.FIXED_THETA_EXP = False
    cfg.DISEASE_GENES_PATH = cfg.DATA_SUB / "disease_data_uniform_theta.csv"
    print("########### RUNNING MIX MODEL UNIFORM THETA ############")
    interpret_disease_mix(mode="true")


