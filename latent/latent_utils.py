import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import config as cfg

# -------------------------------------------------------------------
# LATENT SPACE PLOTTING (Coordinates -> Images)
# -------------------------------------------------------------------

def plot_latent_space(coords, color_values, label_name, title, save_path, cmap="magma"):
    """Standardized scatter plot for any coordinate/color combination."""
    plt.figure(figsize=(10, 8))
    is_categorical = pd.api.types.is_categorical_dtype(color_values) or isinstance(color_values[0], str)
    
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
    
    # Ensure parent directories exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_general_comparison_grid(phase, scaled, color_values, label_name, 
                                 row_keys, col_keys, method="umap", save_subdir="latent_grids", is_mixed=False):
    """Creates a flexible grid: Rows = Encoding Sizes, Cols = Model Architectures."""
    tournament_name_dict = {"pca": "PCA", "ae_basic": "Basic_AE", "ae_layered": "Layered_AE"}
    scale_str = "scaled" if scaled else "unscaled"
    fig, axes = plt.subplots(
        nrows=len(row_keys), ncols=len(col_keys),
        figsize=(5 * len(col_keys), 4 * len(row_keys)), constrained_layout=True
    )

    if len(row_keys) == 1: axes = np.expand_dims(axes, axis=0)
    if len(col_keys) == 1: axes = np.expand_dims(axes, axis=-1)
    
    first_col = col_keys[0]
    base_name = first_col.split("H-")[1].split("_D-")[0] if "H-" in first_col else "standalone"
    tournament_folder = f"Tournament_H-{tournament_name_dict.get(base_name, base_name)}"
    
    is_categorical = (label_name == "disease_type")
    vmin, vmax = color_values.min(), color_values.max()
    
    if is_categorical:
        num_classes = int(vmax - vmin + 1)
        color_dict = {0: "#2ecc71", 1: "#d43220", 2: "#870fb6"}
        color_list = [color_dict.get(int(val), "black") for val in range(int(vmin), int(vmax) + 1)]
        cmap = mcolors.ListedColormap(color_list)
        plot_vmin, plot_vmax = vmin - 0.5, vmax + 0.5
    else:
        cmap = plt.get_cmap("magma")
        plot_vmin, plot_vmax = vmin, vmax

    for i, row_val in enumerate(row_keys):
        for j, col_val in enumerate(col_keys):
            ax = axes[i, j]
            path = cfg.get_path(phase, scale_str, col_val, row_val, folder_type=cfg.MODELS_SUBFOLDER, is_mixed=is_mixed) / f"{method}_coords.npy"
            
            if path.exists():
                coords = np.load(path)
                sc = ax.scatter(coords[:, 0], coords[:, 1], c=color_values, 
                                cmap=cmap, vmin=plot_vmin, vmax=plot_vmax, s=12, alpha=0.8)
                ax.set_title(f"{col_val} | {row_val}", fontsize=10)
            else:
                ax.axis("off")
            
            ax.set_xticks([]); ax.set_yticks([])

    mix_str = "MIXED (H+D)" if is_mixed else "DISEASE ONLY"
    fig.suptitle(f"{method.upper()} Grid | Target: {label_name} | {mix_str}", fontsize=16)
    
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=plot_vmin, vmax=plot_vmax), cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    
    if is_categorical:
        ticks = np.arange(vmin, vmax + 1)
        cbar.set_ticks(ticks)
        class_dict = {0: "Healthy", 1: "Disease A (CRC)", 2: "Disease B (SCLC)"}
        cbar.set_ticklabels([class_dict.get(int(t), f"Class {int(t)}") for t in ticks])
        cbar.set_label("Ground Truth Classification", fontsize=12, fontweight='bold')
    else:
        cbar.set_label(label_name, fontsize=12, fontweight='bold')

    plot_root = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=is_mixed)
    summary_dir = plot_root / tournament_folder / save_subdir / scale_str
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(summary_dir / f"grid_{method}_{label_name}_{scale_str}.png", dpi=200)
    plt.close()

def plot_combined_comparison_grid(phase, scaled, theta_values, disease_values, 
                                  row_keys, col_keys, method="umap", save_subdir="latent_grids", is_mixed=False):
    """Creates a master grid mapping Theta to Color (Magma) and Disease Type to Marker Shape."""
    tournament_name_dict = {"pca": "PCA", "ae_basic": "Basic_AE", "ae_layered": "Layered_AE"}
    scale_str = "scaled" if scaled else "unscaled"

    fig, axes = plt.subplots(nrows=len(row_keys), ncols=len(col_keys),
                             figsize=(5 * len(col_keys), 4 * len(row_keys)), constrained_layout=True)

    if len(row_keys) == 1: axes = np.expand_dims(axes, axis=0)
    if len(col_keys) == 1: axes = np.expand_dims(axes, axis=-1)

    first_col = col_keys[0]
    base_name = first_col.split("H-")[1].split("_D-")[0] if "H-" in first_col else "standalone"
    tournament_folder = f"Tournament_H-{tournament_name_dict.get(base_name, base_name)}"

    theta_array = np.array(theta_values, dtype=float)
    vmin, vmax = np.nanmin(theta_array), np.nanmax(theta_array)

    marker_dict = {0: 'o', 1: '^', 2: 's'}
    label_dict = {0: "Healthy", 1: "Disease A (CRC)", 2: "Disease B (SCLC)"}
    sc = None 

    for i, row_val in enumerate(row_keys):
        for j, col_val in enumerate(col_keys):
            ax = axes[i, j]
            path = cfg.get_path(phase, scale_str, col_val, row_val, folder_type=cfg.MODELS_SUBFOLDER, is_mixed=is_mixed) / f"{method}_coords.npy"
            
            if path.exists():
                coords = np.load(path)
                unique_diseases = np.unique(disease_values)
                for d_type in unique_diseases:
                    mask = (disease_values == d_type)
                    sc_layer = ax.scatter(
                        coords[mask, 0], coords[mask, 1], 
                        c=theta_array[mask], cmap="magma", vmin=vmin, vmax=vmax, 
                        marker=marker_dict.get(int(d_type), 'x'),
                        s=15, alpha=0.8, edgecolors='none'
                    )
                    if sc_layer is not None: sc = sc_layer
                ax.set_title(f"{col_val} | {row_val}", fontsize=10)
            else:
                ax.axis("off")
            ax.set_xticks([]); ax.set_yticks([])

    mix_str = "MIXED (H+D)" if is_mixed else "DISEASE ONLY"
    fig.suptitle(f"{method.upper()} Combined Grid | Color: Theta | Shape: Disease Type | {mix_str}", fontsize=16)
    
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label("Theta Value (Mixing Proportion)", fontsize=11, fontweight='bold')
    
    legend_elements = [
        Line2D([0], [0], marker=marker_dict[d_key], color='w', label=label_dict[d_key],
               markerfacecolor='gray', markersize=8)
        for d_key in np.unique(disease_values) if int(d_key) in marker_dict
    ]
    
    fig.legend(handles=legend_elements, loc='center right', title="Disease Type", 
               bbox_to_anchor=(1.12, 0.5), fontsize=10, title_fontsize=11)

    plot_root = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=is_mixed)
    summary_dir = plot_root / tournament_folder / save_subdir / scale_str
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = summary_dir / f"grid_{method}_combined_{scale_str}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()