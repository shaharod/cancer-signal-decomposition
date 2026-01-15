import os
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path

import config as cfg
from core.models.model_factory import ModelFactory


# FIXME: i have no idea what is going on here, change or fix or whatever

# -------------------------------------------------------------------
# 1. LOADERS (MODEL & WEIGHTS)
# -------------------------------------------------------------------

def load_ae_models(phase, model_tag, scaled, encoding_sizes, input_size, device=cfg.DEVICE):
    """Global Loader for AE models across phases."""

    scale_str = "scaled" if scaled else "unscaled"
    models_dict = {}

    for enc in encoding_sizes:
        enc_dir = cfg.get_path(phase, scale_str, model_tag, enc)
        model_path = enc_dir / "model.pt"

        if not model_path.exists():
            print(f"[Warning] Model weights not found: {model_path}")
            continue

        model = ModelFactory.create_model(model_tag, input_size, enc).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        models_dict[enc] = model
        print(f"[Loaded] {model_tag} ({enc}-dim) Phase: {phase}")

    return models_dict


def load_pca_models(phase, scaled, encoding_sizes):
    """Loads saved sklearn PCA models."""
    scale_str = "scaled" if scaled else "unscaled"
    pca_dict = {}

    for enc in encoding_sizes:
        enc_dir = cfg.get_path(phase, scale_str, "pca", enc)
        pca_path = enc_dir / "model.joblib"

        if not pca_path.exists():
            continue

        pca = joblib.load(pca_path)
        pca_dict[enc] = pca
    return pca_dict

# -------------------------------------------------------------------
# 2. LATENT EXTRACTION (DATA -> Z)
# -------------------------------------------------------------------

def get_ae_latents(phase, model_tag, scaled, encoding_sizes, X_tensor, device="cpu"):
    """Extracts latent Z from AE models."""
    input_size = X_tensor.shape[1]
    models_dict = load_ae_models(phase, model_tag, scaled, encoding_sizes, input_size, device)
    
    latents = {}
    for enc_dim, model in models_dict.items():
        with torch.no_grad():
            # Standardized access via ModelFactory encoder
            Z = model.encoder(X_tensor.to(device)).cpu().numpy()
        latents[f"{model_tag}_{enc_dim}"] = Z
    return latents

def get_pca_latents(phase, scaled, encoding_sizes, X_np):
    """Extracts latent Z from PCA objects."""
    pca_models = load_pca_models(phase, scaled, encoding_sizes)
    latents = {}
    for enc, pca in pca_models.items():
        latents[f"pca_{enc}"] = pca.transform(X_np)
    return latents

# -------------------------------------------------------------------
# 3. COORDINATE GENERATION & SAVING (Z -> 2D)
# -------------------------------------------------------------------

def save_latent_visuals(latents_dict, phase, scaled, sig_df, perplexities, split_info=None):
    """Saves .npy coordinates and individual .png plots to model folders."""
    scale_str = "scaled" if scaled else "unscaled"
    
    for name, Z in latents_dict.items():
        # Parse name e.g., "ae_layered_32"
        parts = name.split("_")
        model_tag = "_".join(parts[:-1])
        enc_size = parts[-1]

        # Resolve folder
        model_dir = cfg.get_path(phase, scale_str, model_tag, enc_size)
        latent_dir = model_dir / "latent_space"
        latent_dir.mkdir(parents=True, exist_ok=True)

        # 1. Compute and Save PCA 2D
        pca_2d = PCA(n_components=2).fit_transform(Z)
        np.save(latent_dir / "pca_coords.npy", pca_2d)

        # 2. Compute and Save t-SNE 2D
        for perp in perplexities:
            tsne_2d = TSNE(
                n_components=2, 
                perplexity=perp, 
                init="pca", 
                learning_rate='auto',
                random_state=42
            ).fit_transform(Z)

            np.save(latent_dir / f"tsne_perp_{perp}_coords.npy", tsne_2d)

            for sig in cfg.SIG_LIST:
                if sig in sig_df.columns:
                    sig_values = sig_df[sig].values
                    plot_and_save_single(tsne_2d, sig_values, sig, f"t-SNE (Perp {perp})", latent_dir, model_tag, enc_size)

        # 3. Generate individual signature plots
        for sig in cfg.SIG_LIST:
            if sig in sig_df.columns:
                sig_values = sig_df[sig].values
                plot_and_save_single(pca_2d, sig_values, sig, "PCA", latent_dir, model_tag, enc_size)

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