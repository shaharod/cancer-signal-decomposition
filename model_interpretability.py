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

def load_reconstruction_data():
    """
    Loads the validation data (Mixed Input and Clean Ground Truth).
    Matches your requested structure using config paths.
    """
    # 1. Define Paths
    mix_file = cfg.DISEASE_GENES_PATH
    # Assuming 'pure_disease_truth.csv' exists in your data folder
    truth_file = cfg.DATA_SUB / 'pure_disease_truth.csv' 

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

# import seaborn as sns
# import matplotlib.pyplot as plt

# def plot_decoder_weights_heatmap(model, gene_names, base_name, enc):
#     """
#     Creates a heatmap of the decoder weights to visualize gene-latent connections.
#     """
#     # Extract weights from the final layer of the disease decoder
#     weights = model.disease_model.decoder[-1].weight.data.cpu().numpy()
    
#     # Handle Transpose if necessary (Output Genes x Latent Neurons)
#     if weights.shape[0] != len(gene_names):
#         weights = weights.T
        
#     # Select top 50 most variable genes in the weights for better visibility
#     weight_variance = np.var(weights, axis=1)
#     top_genes_idx = np.argsort(weight_variance)[-50:]
#     filtered_weights = weights[top_genes_idx, :]
#     filtered_names = [gene_names[i] for i in top_genes_idx]

#     plt.figure(figsize=(12, 10))
#     sns.heatmap(filtered_weights, xticklabels=[f"L{i}" for i in range(enc)],
#                 yticklabels=filtered_names, cmap="RdBu_r", center=0)
    
#     plt.title(f"Decoder Weight Connections: {base_name} (Enc {enc})")
    
#     # Save into the specific model's plot folder
#     save_dir = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER) / base_name
#     plt.savefig(save_dir / f"weights_heatmap_enc{enc}.png")
#     plt.close()

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
    from core.models.model_factory import ModelFactory
    from sklearn.decomposition import PCA
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. Load Data & Tensors
    input_df, truth_df = load_reconstruction_data()
    if input_df is None: return
    
    tag = "scaled" if scale_bool else "unscaled"
    input_size = input_df.shape[1]
    input_tensor = torch.tensor(input_df.values).float().to("cpu")
    log_truth = np.log1p(truth_df.values).flatten()

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
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
        fig.suptitle(f"Tournament Results: Healthy Base = {base_name.upper()}\nPhase: {phase.capitalize()} | Data: {tag.capitalize()}", 
                     fontsize=20, fontweight='bold', y=0.98)
                     
        for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
            for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                ax = axes[row_idx, col_idx]
                
                try:
                    # --- UNIFIED LOADING LOGIC ---
                    if "mix" in folder_tag:
                        # Disease Mix Logic
                        parts = folder_tag.split('_H-')
                        h_and_d = parts[1].split('_D-')
                        h_type, d_type = h_and_d[0], h_and_d[1]

                        def prepare_obj(m_type):
                            if m_type.lower() == 'pca':
                                obj = PCA(n_components=enc)
                                obj.mean_, obj.n_components_ = np.zeros(input_size), enc
                                obj.components_ = np.zeros((enc, input_size))
                                return obj
                            return ModelFactory.create_model(m_type, input_size, enc)

                        model = ModelFactory.create_mix_model(prepare_obj(h_type), prepare_obj(d_type))
                    else:
                        # Healthy / Standalone Logic
                        model = ModelFactory.create_model(folder_tag, input_size, enc)

                    # --- LOAD WEIGHTS & INFER ---
                    is_pca = "pca" in folder_tag.lower()
                    ext = "model.joblib" if is_pca else "model.pt"
                    model_path = cfg.get_path(phase, tag, folder_tag, enc, cfg.MODELS_SUBFOLDER) / ext

                    if not model_path.exists():
                        ax.text(0.5, 0.5, f"Missing:\n{folder_tag}", ha='center')
                        continue

                    if is_pca:
                        # PCA doesn't need state_dict loading, the Factory handles the object
                        # But we need to ensure the d_type is loaded if it's a standalone PCA
                        checkpoint = joblib.load(model_path)
                        # If the factory returns a fresh object, you might need to assign 
                        # the loaded components to it here.
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
                    truth_raw = truth_df.values.flatten()
                    # corr = np.corrcoef(log_truth, log_recon)[0, 1]
                    corr = np.corrcoef(truth_raw, recon_raw)[0, 1]
                    # --- PLOTTING ---
                    # ax.hexbin(log_truth, log_recon, gridsize=70, cmap='YlGnBu', mincnt=1, bins='log')
                    # ax.plot([log_truth.min(), log_truth.max()], [log_truth.min(), log_truth.max()], 'r--', lw=1)
                    ax.hexbin(truth_raw, recon_raw, gridsize=70, cmap='YlGnBu', mincnt=1)

                    # Identity line based on raw min/max
                    ax.plot([truth_raw.min(), truth_raw.max()], [truth_raw.min(), truth_raw.max()], 'r--', lw=1)
                    if row_idx == 0: ax.set_title(f"{model_label}", fontsize=12, fontweight='bold')
                    if col_idx == 0: ax.set_ylabel(f"Enc: {enc}", fontsize=10, fontweight='bold')
                    ax.text(0.05, 0.95, f"R: {corr:.3f}", transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                except Exception as e:
                    ax.text(0.5, 0.5, f"Error Loading Model", ha='center', color='red')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        folder = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER) / f"Tournament_H-{base_name}"
        os.makedirs(folder, exist_ok=True)
        full_path = os.path.join(folder, f"{save_path}_{tag}")
        plt.savefig(full_path, dpi=150)
        plt.close()


def interpret_disease_mix(phase='disease'):

    # all possible combinations of healthy baselines with disease
    disease_mix_labels = {
        'PCA':
        {
            "basic": "mix_H-pca_D-ae_basic",
            "layered": "mix_H-pca_D-ae_layered",
            "pca-based": "mix_H-pca_D-pca"
        },
        'AE-Basic':
        {
            "basic": "mix_H-ae_basic_D-ae_basic",
            "layered": "mix_H-ae_basic_D-ae_layered",
            "pca-based": "mix_H-ae_basic_D-pca"
        },
        'AE-Layered':
        {
            "basic": "mix_H-ae_layered_D-ae_basic",
            "layered": "mix_H-ae_layered_D-ae_layered",
            "pca-based": "mix_H-ae_layered_D-pca"
        }
    }


    for baseline, labels in disease_mix_labels.items():

        ##unscaled data reconstructions
        analyze_reconstruction_grid(disease_mix_labels, phase='disease', 
                                    scale_bool=False, save_path="reconstructed_grid", 
                                    )
        
        







def interpret_healthy_model(phase='healthy'):

    # setting labels
    model_labels = {
        'Basic-AE': 'ae_basic',
        'Layered-AE': 'ae_layered',
        'PCA': 'pca'
        }
 
    analyze_reconstruction_grid(model_labels, phase='healthy', 
                                scale_bool=False, 
                                save_path="reconstructed_grid")




if __name__ == '__main__':

    # TODO: fix logic, maybe from command lines arguments or something
    print(f'model type is: {'synthetic' if cfg.SYNTHETIC_DATA else 'synthetic'}\n\n')
    interpret_healthy_model()
    interpret_disease_mix()

    # if cfg.SYNTHETIC_DATA:    
    #     analyze_reconstruction()


