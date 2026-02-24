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


def save_latent_visuals(latents_dict, phase, scaled, sig_df, methods=["pca", "umap"]):
    """
    Iterates through latents and generates the requested 2D projections.
    """
    scale_str = "scaled" if scaled else "unscaled"
    
    for name, Z in latents_dict.items():
        # name format: "mix_H-pca_D-ae_basic_enc8"
        model_tag = name.split("_enc")[0]
        enc_size = name.split("_enc")[-1]

        latent_dir = cfg.get_path(phase, scale_str, model_tag, enc_size) / "latent_space"
        latent_dir.mkdir(parents=True, exist_ok=True)

        for m in methods:
            coords = generate_coords(Z, method=m)
            np.save(latent_dir / f"{m}_coords.npy", coords)
            
            # Color by Theta if available in sig_df
            for col in sig_df.columns:
                plot_latent_comparison(
                    coords, sig_df[col].values, col,
                    title=f"{model_tag} ({enc_size}) {m.upper()}\nColored by: {col}",
                    save_path=latent_dir / f"{m}_{col}.png"
                )

# -------------------------------------------------------------------
# 3. PLOTTING SCATTER PLOT
# -------------------------------------------------------------------

def plot_and_save_single(coords, sig_values, sig_name, method, folder, model_tag, enc):
    """Helper to plot a single scatter map."""
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=sig_values, cmap='viridis', s=15, alpha=0.7)
    
    plt.colorbar(sc, label=f"Count: {sig_name}")
    plt.title(f"{model_tag} ({enc}) {method} Projection\nSignature: {sig_name}")
    plt.xlabel(f"{method} 1")
    plt.ylabel(f"{method} 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(folder / f"{method.lower()}_{sig_name}.png", dpi=150)
    plt.close()

def plot_latent_comparison(coords, color_values, label_name, title, save_path):
    """
    Standardized scatter plot for any coordinate/color combination.
    """
    plt.figure(figsize=(10, 8))
    # Using 'magma' or 'viridis' for continuous values like Theta
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=color_values, 
                      cmap='magma', s=15, alpha=0.7)
    plt.colorbar(sc, label=label_name)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
# -------------------------------------------------------------------
# 4. GLOBAL COMPARISON GRIDS
# -------------------------------------------------------------------

def plot_global_comparison_grid(phase, scaled, sig_name, sig_values, method="pca"):
    """Aggregates saved .npy files from all models into one large comparison grid."""
    scale_str = "scaled" if scaled else "unscaled"
    model_families = cfg.MODEL_TYPES + ["pca"]
    encoding_sizes = cfg.ENCODING_SIZES

    fig, axes = plt.subplots(
        nrows=len(encoding_sizes),
        ncols=len(model_families),
        figsize=(5 * len(model_families) + 1, 4 * len(encoding_sizes)),
        gridspec_kw={"right": 0.85}
    )

    # Force 2D axes for indexing
    if len(encoding_sizes) == 1: axes = np.expand_dims(axes, axis=0)
    if len(model_families) == 1: axes = np.expand_dims(axes, axis=-1)

    vmin, vmax = sig_values.min(), sig_values.max()
    coord_file = f"{method.lower()}_coords.npy"

    for i, enc in enumerate(encoding_sizes):
        for j, model_tag in enumerate(model_families):
            ax = axes[i, j]
            
            # Build path to previously saved coordinates
            path = cfg.get_path(phase, scale_str, model_tag, enc) / "latent_space" / coord_file
            
            if not path.exists():
                ax.axis("off")
                continue

            coords = np.load(path)
            ax.scatter(coords[:, 0], coords[:, 1], c=sig_values, cmap="viridis", 
                       vmin=vmin, vmax=vmax, s=8, alpha=0.6)
            
            ax.set_title(f"{model_tag} | Enc: {enc}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Global {method.upper()} Comparison | Phase: {phase} | Sig: {sig_name}", fontsize=16)
    
    # Global colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap="viridis")
    fig.colorbar(sm, cax=cbar_ax).set_label(f"Signature: {sig_name}")

    # Central Summary Folder
    summary_dir = cfg.HEALTHY_OUT_DIR / "plots" / "latent_grids" / scale_str
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = summary_dir / f"grid_{method}_{sig_name}.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Grid Saved] {out_path}")


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