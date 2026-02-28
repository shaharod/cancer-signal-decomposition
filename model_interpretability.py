import io
import traceback
import joblib
import os
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
                    parts = folder_tag.split('_H-')
                    h_and_d = parts[1].split('_D-')
                    h_type, d_type = h_and_d[0], h_and_d[1]
                    h_model = ModelFactory.create_model(h_type, gene_size, enc, cfg.H1, cfg.H2)
                    d_model = ModelFactory.create_model(d_type, gene_size, enc, cfg.H1, cfg.H2)
                    model = ModelFactory.create_mix_model(h_model, d_model)
                    is_pca = "pca" in d_type.lower()
                    ext = "model.joblib" if is_pca else "model.pt"
                    model_path = cfg.get_path('disease', tag, folder_tag, enc, cfg.MODELS_SUBFOLDER, is_mixed=False) / ext
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
    }


    ##unscaled data reconstructions

    analyze_d_portion_recon_new(disease_mix_labels, scale_bool=False, save_path="analyze_recon_allSamples_dif", mode=mode)
    # print("################### DISEASE PORTION RECON FUNCTION ###################")
    analyze_disease_portion_reconstruction(disease_mix_labels, scale_bool=False, save_path="analyze_recon_dSamplesOnly", mode=mode)

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


