import os
import joblib
import torch
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
import utils.model_utils as mu

import config as cfg
from core.models.model_factory import ModelFactory


# -------------------------------------------------------------------
# 1. LATENT EXTRACTION (DATA -> Z)
# -------------------------------------------------------------------

def get_standalone_latents(model_type, input_size, enc_sizes, scale_bool, test_set, phase):
    latents = {}
    for enc in enc_sizes:
        _, z = mu.create_load_standalone_model(phase=phase, m_type=model_type, enc=enc, scale_bool=scale_bool, input_size=input_size, test_t=test_set)
        latents[f"{model_type}_enc:{enc}"] = z
    return latents

def get_mix_latents(mix_type, input_size, enc_sizes, scale_tag, is_mixed, test_t):
    latents = {}
    for enc in enc_sizes:
        _, _, _, z = mu.create_load_mix_model(folder_tag=mix_type, test_set=test_t, gene_size=input_size, enc=enc, scale_tag=scale_tag)
        latents[f"{mix_type}_enc:{enc}"] = z
        ## NOTE: the latent in mix models is the latent of disease part
    return latents


def choose_sig_list(phase):
    if phase == "synthetic":
        return cfg.SYN_SIG_LIST
    if phase == "real":
        return cfg.REAL_SYN_LIST
    raise ValueError("Add here other signature lists i might want and have better phase names") #TODO in future, do as error states

# -------------------------------------------------------------------
# 2. COORDINATE GENERATION & SAVING (Z -> 2D)
# -------------------------------------------------------------------

def generate_coords(Z, method="umap", **kwargs):
    """
    Unified interface for dimensionality reduction.
    """
    if method == "pca":
        return PCA(n_components=2).fit_transform(Z)
    elif method == "umap":
        reducer = umap.UMAP(
            n_neighbors=kwargs.get('n_neighbors', 15),
            min_dist=kwargs.get('min_dist', 0.1),
            random_state=42
        )
        return reducer.fit_transform(Z)
    elif method == "tsne":
        return TSNE(
            n_components=2, 
            perplexity=kwargs.get('perplexity', 30),
            init="pca", 
            random_state=42
        ).fit_transform(Z)
    raise ValueError(f"Unknown method: {method}")


def save_latent_batch(latents_dict, phase, scaled, color_df, methods=["pca", "umap"], is_mixed=False):
    """
    More general batch processor that iterates through ANY provided columns in color_df.
   
    """
    scale_str = "scaled" if scaled else "unscaled"
    
    for name, Z in latents_dict.items():
        # name: "mix_H-pca_D-ae_layered_enc16"
        model_tag = name.split("_enc")[0]
        enc_size = name.split("_enc")[-1]
        base_name = model_tag.split("H-")[1].split("_D-")[0] if "H-" in model_tag else "standalone"
        model_root = cfg.get_path(phase, scale_str, model_tag, enc_size, folder_type=cfg.MODELS_SUBFOLDER, is_mixed=is_mixed)
        coord_dir = model_root / "latent_space"
        coord_dir.mkdir(parents=True, exist_ok=True)

        ## path for visual summaries - inside plots/tournament-H..
        summary_root = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=is_mixed) / f"Tournament_H-{base_name}" / "latent_analysis"
        summary_root.mkdir(parents=True, exist_ok=True)

        for m in methods:
            coords = generate_coords(Z, method=m)
            np.save(coord_dir / f"{m}_coords.npy", coords)
            
            # General Loop: Color by every column provided in the dataframe
            for col in color_df.columns:
                title = f"{model_tag} ({enc_size}) {m.upper()} | {col}"
                # Save one copy in the model folder (archive) and one in the tournament folder (review)
                save_path = coord_dir / f"{m}_{col}.png"
                plot_latent_space(coords, color_df[col].values, col, title, save_path)

# -------------------------------------------------------------------
# 3. PLOTTING SCATTER PLOT
# -------------------------------------------------------------------

def plot_latent_space(coords, color_values, label_name, title, save_path, cmap="magma"):
    """
    Standardized scatter plot for any coordinate/color combination.
    Can be used to view correlation of theta to latent vectors too
    """
    plt.figure(figsize=(10, 8))
    is_categorical = pd.api.types.is_categorical_dtype(color_values) or isinstance(color_values[0], str)
    
    # Using 'magma' or 'viridis' for continuous values like Theta
    sc = plt.scatter(
        coords[:, 0], coords[:, 1], 
        c=color_values if not is_categorical else None,
        cmap=cmap if not is_categorical else None,
        s=15, alpha=0.7, edgecolors='none'
    )
    if not is_categorical:
        plt.colorbar(sc, label=label_name)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------------
# 4. GLOBAL COMPARISON GRIDS
# -------------------------------------------------------------------

def plot_general_comparison_grid(phase, scaled, color_values, label_name, 
                                 row_keys, col_keys, method="umap", save_subdir="latent_grids"):
    """
    Creates a flexible grid where:
    Rows = Encoding Sizes
    Cols = Model Architectures
    """
    scale_str = "scaled" if scaled else "unscaled"
    fig, axes = plt.subplots(
        nrows=len(row_keys),
        ncols=len(col_keys),
        figsize=(5 * len(col_keys), 4 * len(row_keys)),
        constrained_layout=True
    )

    # Handle 1D axes arrays
    if len(row_keys) == 1: axes = np.expand_dims(axes, axis=0)
    if len(col_keys) == 1: axes = np.expand_dims(axes, axis=-1)

    vmin, vmax = color_values.min(), color_values.max()

    for i, row_val in enumerate(row_keys):
        for j, col_val in enumerate(col_keys):
            ax = axes[i, j]
            
            # General path logic using the row/col identifiers
            path = cfg.get_path(phase, scale_str, col_val, row_val) / "latent_space" / f"{method}_coords.npy"
            
            if path.exists():
                coords = np.load(path)
                sc = ax.scatter(coords[:, 0], coords[:, 1], c=color_values, 
                                cmap="magma", vmin=vmin, vmax=vmax, s=8, alpha=0.6)
                ax.set_title(f"{col_val} | {row_val}", fontsize=10)
            else:
                ax.axis("off")
            
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"{method.upper()} Grid | Target: {label_name}", fontsize=16)
    
    # Save to a dynamic path
    summary_dir = cfg.BASE_EXP_DIR / phase / "plots" / save_subdir / scale_str
    summary_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(summary_dir / f"grid_{method}_{label_name}.png", dpi=200)
    plt.close()

def plot_comprehensive_comparison_grid(phase, scaled, sig_name, sig_values, perplexities, split_info=None):
    """
    Creates a grid: 
    Rows = Model Families (AEs, PCA) 
    Cols = [PCA Projection, t-SNE Perp A, t-SNE Perp B, ...]
    """
    scale_str = "scaled" if scaled else "unscaled"
    model_families = cfg.MODEL_TYPES + ["pca"]
    encoding_sizes = cfg.ENCODING_SIZES
    
    # Define columns: 1 for PCA baseline + 1 for each perplexity
    methods = ["pca"] + [f"tsne_perp_{p}" for p in perplexities]
    
    # Create the figure: Total Rows = (Models * Encodings), Total Cols = Methods
    total_rows = len(model_families) * len(encoding_sizes)
    fig, axes = plt.subplots(
        nrows=total_rows,
        ncols=len(methods),
        figsize=(4 * len(methods), 3.5 * total_rows),
        constrained_layout=True
    )

    vmin, vmax = sig_values.min(), sig_values.max()
    n_tr = split_info["n_train"] if split_info else len(sig_values)

    row_idx = 0
    for model_tag in model_families:
        for enc in encoding_sizes:
            for col_idx, method in enumerate(methods):
                ax = axes[row_idx, col_idx]
                
                # Load saved coordinates
                coord_file = f"{method.lower()}_coords.npy"
                path = cfg.get_path(phase, scale_str, model_tag, enc) / "latent_space" / coord_file
                
                if not path.exists():
                    ax.axis("off")
                    continue

                coords = np.load(path)
                
                # Plotting with Shapes
                # 1. Train/Val (Circles)
                ax.scatter(coords[:n_tr, 0], coords[:n_tr, 1], c=sig_values[:n_tr], 
                           cmap="viridis", vmin=vmin, vmax=vmax, s=10, alpha=0.5, marker='o')
                
                # 2. Test (Crosses) - Plotted on top for visibility
                if split_info:
                    ax.scatter(coords[n_tr:, 0], coords[n_tr:, 1], c=sig_values[n_tr:], 
                               cmap="viridis", vmin=vmin, vmax=vmax, s=25, alpha=0.9, marker='x')

                # Labels and Styling
                if row_idx == 0:
                    ax.set_title(f"Method: {method.upper()}", fontsize=14, fontweight='bold')
                
                if col_idx == 0:
                    ax.set_ylabel(f"{model_tag}\nEnc: {enc}", fontsize=12, fontweight='bold')
                
                ax.set_xticks([]); ax.set_yticks([])

            row_idx += 1

    fig.suptitle(f"Latent Space Cross-Reference | Phase: {phase} | Sig: {sig_name}\nCircle=Train, X=Test", 
                 fontsize=18, y=1.02)

    # Add a global colorbar
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap="viridis")
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label(f"Signature Level: {sig_name}", fontsize=12)

    # Save to a dedicated summary folder
    summary_dir = cfg.HEALTHY_OUT_DIR / "plots" / "comprehensive_grids" / scale_str
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = summary_dir / f"comprehensive_{sig_name}.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Comprehensive Grid Saved] {out_path}")


def plot_model_family_grids(phase, scaled, sig_name, sig_values, perplexities, split_info=None):
    """
    Creates one grid per Model Family (e.g., one for 'ae_layered', one for 'vae').
    Rows = Encoding Sizes (16, 32, 64)
    Cols = [PCA, t-SNE P5, t-SNE P30, ...]
    """
    scale_str = "scaled" if scaled else "unscaled"
    model_families = cfg.MODEL_TYPES + ["pca"]
    encoding_sizes = cfg.ENCODING_SIZES
    methods = ["pca"] + [f"tsne_perp_{p}" for p in perplexities]
    
    vmin, vmax = sig_values.min(), sig_values.max()
    n_tr = split_info["n_train"] if split_info else len(sig_values)

    for model_tag in model_families:
        fig, axes = plt.subplots(
            nrows=len(encoding_sizes),
            ncols=len(methods),
            figsize=(4 * len(methods), 4 * len(encoding_sizes)),
            constrained_layout=True
        )

        # Handle 1D axes case if only one encoding size exists
        if len(encoding_sizes) == 1: axes = np.expand_dims(axes, axis=0)

        for i, enc in enumerate(encoding_sizes):
            for j, method in enumerate(methods):
                ax = axes[i, j]
                path = cfg.get_path(phase, scale_str, model_tag, enc) / "latent_space" / f"{method.lower()}_coords.npy"
                
                if not path.exists():
                    ax.axis("off")
                    continue

                coords = np.load(path)
                
                # Plot Train (Circle) and Test (X)
                ax.scatter(coords[:n_tr, 0], coords[:n_tr, 1], c=sig_values[:n_tr], 
                           cmap="viridis", vmin=vmin, vmax=vmax, s=12, alpha=0.5, marker='o')
                if split_info:
                    ax.scatter(coords[n_tr:, 0], coords[n_tr:, 1], c=sig_values[n_tr:], 
                               cmap="viridis", vmin=vmin, vmax=vmax, s=30, alpha=0.9, marker='x')

                # Title only on the top row
                if i == 0: ax.set_title(method.upper(), fontsize=12)
                # Label only on the first column
                if j == 0: ax.set_ylabel(f"Dim: {enc}", fontsize=12, fontweight='bold')
                
                ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(f"Model: {model_tag.upper()} | Phase: {phase} | Sig: {sig_name}", fontsize=16)
        
        # Save to a subfolder named after the model
        summary_dir = cfg.HEALTHY_OUT_DIR / "plots" / "model_specific_grids" / scale_str / model_tag
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(summary_dir / f"{sig_name}_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Finished generating model-specific grids for {sig_name}")