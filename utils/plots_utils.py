import matplotlib.pyplot as plt
import utils.analysis_utils as au
import config as cfg
import numpy as np
import os

DATA_TYPE = 'Synthetic' if cfg.SYNTHETIC_DATA else 'True'
ENC_SIZES = cfg.ENCODING_SIZES


def plot_test_mse_bars(data_s, data_u, fig_title, save_path):
    """
    Bar plots of Test MSE
    """

    num_rows = len(ENC_SIZES)
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows), squeeze=False)
    fig.suptitle(fig_title, fontsize=16, fontweight='bold')

    col_titles = ["Pipeline: Trained on Scaled Data", "Pipeline: Trained on Raw Data"]
    datasets = [data_s, data_u]

    colors = ['#8dbade', '#558e3e', '#e1968b'] # Matching colors

    for row_idx, enc in enumerate(ENC_SIZES):
        for col_idx, data in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            model_names = list(data.keys())

            mse_vals = [data[m][au.TEST_MSE_IDX].get(enc, [0])[0] for m in model_names]

            bars = ax.bar(model_names, mse_vals, color=colors[:len(model_names)], 
                          edgecolor='black', alpha=0.9, width=0.6)
            
            ax.set_title(f"Encoding {enc} | {col_titles[col_idx]}", fontsize=12)
            ax.set_ylabel("MSE (Original Units)")
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            plt.setp(ax.get_xticklabels(), fontweight='bold', fontsize=9)

            for b in bars:
                height = b.get_height()
                ax.text(b.get_x() + b.get_width()/2, height + (max(mse_vals) * 0.01),
                        f'{height:.4g}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            ax.set_ylim(0, max(mse_vals) * 1.15)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path / f'{fig_title}.png')
    plt.close()


def plot_learning_curves(data_s, data_u, fig_title_prefix, save_path, zoom_params=None):
    """
    Generates one figure per Model Type.
    Rows: Encoding Sizes
    Cols: Scaled vs Unscaled Data
    """
    
    # 1. Identify unique models (excluding PCA)
    all_models = sorted(list(set(list(data_s.keys()) + list(data_u.keys()))))
    models_to_plot = [m for m in all_models if 'pca' not in m.lower()]

    datasets = [("Scaled Data", data_s), ("Unscaled Data", data_u)]

    for model_name in models_to_plot:
        
        num_rows = len(ENC_SIZES)
        # Increased height slightly to accommodate headers
        fig, axes = plt.subplots(num_rows, 2, figsize=(13, 4.5 * num_rows), squeeze=False)
        title = ' '.join(fig_title_prefix.split('_'))
        fig.suptitle(f"{title}: {model_name}", fontsize=18, fontweight='bold', y=0.96)

        for row_idx, enc in enumerate(ENC_SIZES):
            for col_idx, (col_name, data) in enumerate(datasets):
                ax = axes[row_idx, col_idx]

                # --- Titles & Headers ---
                # 1. Standard Subplot Title (Encoding Size)
                ax.set_title(f"Encoding Size: {enc}", fontsize=11, fontweight='bold')

                # 2. Column Header: Manually placed significantly HIGHER (Only on top row)
                if row_idx == 0:
                    ax.text(0.5, 1.12, col_name, 
                            transform=ax.transAxes, 
                            ha='center', va='bottom', 
                            fontsize=14, fontweight='bold') 

                # --- Data Extraction ---
                if model_name not in data:
                    ax.text(0.5, 0.5, "Model not in dataset", ha='center', transform=ax.transAxes)
                    continue

                model_tuple = data[model_name]
                train_h = model_tuple[au.TRAIN_LOSS_IDX].get(enc, [])
                eval_h  = model_tuple[au.EVAL_LOSS_IDX].get(enc, [])

                # --- Plotting ---
                if len(train_h) > 0:
                    epochs = np.arange(1, len(train_h) + 1)
                    # Train: Blue Line
                    ax.plot(epochs, train_h, label="Train", color='tab:blue', linewidth=1.5)

                    if len(eval_h) > 0:
                        step = len(train_h) // len(eval_h)
                        e_epochs = np.arange(step, len(train_h) + 1, step)
                        # Eval: Orange Dashed Line with Markers
                        ax.plot(e_epochs, eval_h, label="Eval", color='tab:orange', 
                                linestyle='--', marker='.', markersize=6, alpha=0.8)

                # --- Zoom & formatting ---
                if zoom_params:
                    if 'last_n_epochs' in zoom_params and len(train_h) > 0:
                        max_e = len(train_h)
                        ax.set_xlim(max(1, max_e - zoom_params['last_n_epochs']), max_e)
                    
                    if 'ylim_top' in zoom_params and zoom_params['ylim_top'] is not None:
                        ax.set_ylim(0, zoom_params['ylim_top'])

                ax.set_ylabel("Loss (MSE)")
                ax.set_xlabel("Epochs")
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(fontsize='small', loc='upper right')

        # Adjusted rect to leave more room at the top for the manual headers
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) 
        
        file_name = f"{fig_title_prefix}_{model_name}.png"
        plt.savefig(save_path / file_name)
        plt.close()
        print(f"Saved plot: {file_name}")


def plot_reconstruction_scatter(original, reconstructed, title, save_path, log_scale=False):
    """Plot 4: Individual Gene Reconstruction with Pearson Correlation."""
    plt.figure(figsize=(8, 8))
    x, y = np.array(original).flatten(), np.array(reconstructed).flatten()
    
    # Calculate Correlation Coefficient
    corr = np.corrcoef(x, y)[0, 1]
    
    plt.scatter(x, y, alpha=0.4, s=10, color='#3498db', edgecolors='none', label='Genes')
    
    # Identity Line
    ma = max(x.max(), y.max())
    plt.plot([0, ma], [0, ma], color='red', linestyle='--', linewidth=1.5, label='Identity (y=x)')

    plt.title(f"{title}\nPearson Correlation: {corr:.4f}", fontsize=14, fontweight='bold')
    plt.xlabel("Ground Truth Expression", fontweight='bold')
    plt.ylabel("Model Reconstruction", fontweight='bold')
    
    if log_scale:
        plt.xscale('log'); plt.yscale('log')

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path) # Analyzer provides the full filename here
    plt.close()


def plot_mse_vs_encoding(data_s, data_u, fig_title, save_path):
    """
    Test MSE loss change over encoding sizes
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(fig_title, fontsize=16, fontweight='bold')
    col_titles = ["Pipeline: Scaled Data", "Pipeline: Raw Data"]
    datasets = [data_s, data_u]
    markers = ['o', 's', 'D', '^', 'v']

    for col_idx, data in enumerate(datasets):
        ax = axes[col_idx]
        for m_idx, (model_name, model_tuple) in enumerate(data.items()):
            mse_dict = model_tuple[2]
            y_vals = [float(np.array(mse_dict.get(enc, 0)).flatten()[0]) for enc in ENC_SIZES]
            ax.plot(ENC_SIZES, y_vals, marker=markers[m_idx % len(markers)], label=model_name, linewidth=2)
            
            # Annotate exact values
            for x_val, y_val in zip(ENC_SIZES, y_vals):
                ax.annotate(f'{y_val:.4g}', (x_val, y_val), textcoords="offset points", 
                            xytext=(0,10), ha='center', fontsize=8, fontweight='bold')

        ax.set_title(col_titles[col_idx]); ax.set_xticks(ENC_SIZES)
        ax.set_ylabel("Final Test MSE"); ax.grid(True, alpha=0.4)
        ax.set_xlabel("Encoding Size"); ax.grid(True, alpha=0.4)
        ax.legend()
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.2) # Headroom

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path / f'{fig_title}.png')
    plt.close()


def plot_training_vs_pca(data_s, data_u, fig_title_prefix, save_path):
    """
    Plots Training Loss vs PCA Baseline.
    One figure per Model Type.
    Rows: Encoding Sizes
    Cols: Scaled vs Unscaled Data
    """
    
    # 1. Separate PCA models from AE models
    all_keys = sorted(list(set(list(data_s.keys()) + list(data_u.keys()))))
    pca_keys = [k for k in all_keys if 'pca' in k.lower()]
    ae_models = [k for k in all_keys if 'pca' not in k.lower()]

    datasets = [("Scaled Data", data_s), ("Unscaled Data", data_u)]

    for model_name in ae_models:
        
        num_rows = len(ENC_SIZES)
        fig, axes = plt.subplots(num_rows, 2, figsize=(12, 3.5 * num_rows), squeeze=False)
        fig.suptitle(f"{fig_title_prefix}: {model_name} vs PCA", fontsize=18, fontweight='bold', y=0.98)

        for row_idx, enc in enumerate(ENC_SIZES):
            for col_idx, (col_name, data) in enumerate(datasets):
                ax = axes[row_idx, col_idx]

                # --- Headers & Titles ---
                ax.set_title(f"Encoding: {enc}", fontsize=11, fontweight='bold')
                
                if row_idx == 0:
                    ax.text(0.5, 1.08, col_name, 
                            transform=ax.transAxes, 
                            ha='center', va='bottom', 
                            fontsize=14, fontweight='bold') 

                # --- Get PCA Baseline for this specific data/encoding ---
                pca_val = None
                # Find the corresponding PCA model in this dataset
                current_pca_key = next((k for k in data.keys() if 'pca' in k.lower()), None)
                
                if current_pca_key:
                    # Usually PCA is constant, so we take the first value or min value of its "history"
                    pca_hist = data[current_pca_key][au.TRAIN_LOSS_IDX].get(enc, [])
                    if pca_hist:
                        pca_val = pca_hist[-1] # Taking the final converged value

                # --- Get AE Model Data ---
                if model_name in data:
                    train_h = data[model_name][au.TRAIN_LOSS_IDX].get(enc, [])
                    
                    if len(train_h) > 0:
                        epochs = np.arange(1, len(train_h) + 1)
                        # Main Model Line
                        ax.plot(epochs, train_h, label=f"{model_name} (Train)", 
                                color='tab:blue', linewidth=2)
                        
                        # PCA Baseline Line
                        if pca_val is not None:
                            ax.axhline(y=pca_val, color='tab:green', linestyle='--', 
                                       linewidth=2, label=f"PCA Baseline ({pca_val:.2f})")

                # --- Formatting ---
                ax.set_ylabel("Training Loss (MSE)")
                ax.set_xlabel("Epochs")
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(fontsize='small', loc='upper right')

        # Tight layout logic matches your preference
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
        file_name = f"{fig_title_prefix}_{model_name}_vs_PCA.png"
        plt.savefig(save_path / file_name)
        plt.close()
        print(f"Saved plot: {file_name}")




                    #### Sarina Plots Fixing Logic - Try ####

## Sarina plot addition ##

# -- CURVE PLOTS -- #
def plot_train_eval_curves(data_s, data_u, save_name, folder_path, 
                                 include_pca=False, zoom_params=None):
    """
    Plots Train vs Eval for all models.
    zoom_params: dict with {'last_n_epochs': int, 'ylim_top': float} or None
    """
    color_map = {
        'basic': '#1f77b4',    # Blue
        'layered': '#2ca02c',  # Green
        'pca': '#EC7063',
        'default': '#7f7f7f'   # Gray fallback
    }
    # Get encoding sizes from the first available model
    first_model = next(iter(data_s.values()))
    encoding_sizes = list(first_model[0].keys())
    n_enc = len(encoding_sizes)
    
    # Identify AE models vs PCA baseline
    pca_keys = ['pca-based', 'PCA', 'pca']
    models_to_plot = [m for m in data_s.keys() if m not in pca_keys]
    
    for model_key in models_to_plot:
        fig, axes = plt.subplots(n_enc, 2, figsize=(14, 4 * n_enc), squeeze=False)
        title_suffix = " (Zoomed)" if zoom_params else ""
        fig.suptitle(f"Training History: {model_key}{title_suffix}", fontsize=16, y=1.02)

        for i, enc in enumerate(encoding_sizes):
            for j, (data_dict, title) in enumerate([(data_s, "Scaled"), (data_u, "Unscaled")]):
                ax = axes[i, j]
                
                # Unpack Model Data
                train_curve = data_dict[model_key][0].get(enc, [])
                eval_curve = data_dict[model_key][1].get(enc, [])
                
                if not train_curve: continue
                
                # Setup X-axis
                epochs = np.arange(1, len(train_curve) + 1)
                
                # Plotting AE
                current_color = color_map.get(model_key, color_map['default'])                  
                ax.plot(epochs, train_curve, label=f"Train", color=current_color, linestyle='-', lw=1.5)
                ax.plot(epochs, eval_curve, label=f"Eval", color=current_color, linestyle='--', lw=1.2)

                # Optional PCA Lines
                if include_pca:
                    # Find which PCA key exists in this dataset
                    current_pca_key = next((k for k in pca_keys if k in data_dict), None)
                    if current_pca_key:
                        p_train = data_dict[current_pca_key][0].get(enc, [None])[0]
                        p_eval = data_dict[current_pca_key][1].get(enc, [None])[0]
                        pca_color = color_map['pca']
                        if p_train is not None:
                            ax.axhline(y=p_train, color=pca_color, linestyle='-', alpha=0.6, label="PCA Train")
                        if p_eval is not None:
                            ax.axhline(y=p_eval, color=pca_color, linestyle='--', alpha=0.6, label="PCA Eval")

                # Zoom Logic (Only if zoom_params is not None)
                if zoom_params:
                    last_n = zoom_params.get('last_n_epochs', 0)
                    y_max = zoom_params.get('ylim_top')
                    
                    if last_n > 0:
                        start_x = max(0, len(train_curve) - last_n)
                        ax.set_xlim(start_x, len(train_curve))
                    
                    if y_max is not None:
                        ax.set_ylim(0, y_max)

                # Formatting
                ax.set_title(f"{title} | Enc: {enc}")
                ax.set_ylabel("MSE")
                ax.grid(True, which='both', linestyle=':', alpha=0.5)
                ax.legend(loc='upper right', fontsize='x-small')

        plt.tight_layout()
        
        # Save file
        os.makedirs(folder_path, exist_ok=True)
        filename = f"{save_name}_{model_key}.png"
        plt.savefig(os.path.join(folder_path, filename), bbox_inches='tight', dpi=150)
        plt.close()

def compare_models_side_by_side_grid(losses_ae_basic_s, losses_ae_layered_s, losses_pca_s, 
                                     losses_ae_basic_u, losses_ae_layered_u, losses_pca_u,
                                     encoding_sizes, 
                                     save_path, folder_path, runtag, ylim_top, zoom_x=50,
                                     name1="Basic AE", name2="Layered AE"):
    """
    Plots a 2x2 grid: comparing loss curves for basic and layered compared to the pca
    Rows: Scaled Data, Unscaled Data
    Cols: Basic AE vs PCA, Layered AE vs PCA
    Each plot holds all enc size curves
    """
    if not encoding_sizes:
        raise ValueError("encoding_sizes is empty.")

    # 1. Setup the Grid (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey='row', sharex='col')
    
    # Data mapping for the loop
    # Row 0: Scaled, Row 1: Unscaled
    row_data = [
        {"ae_dicts": [losses_ae_basic_s, losses_ae_layered_s], "pca_dict": losses_pca_s, "label": "Scaled"},
        {"ae_dicts": [losses_ae_basic_u, losses_ae_layered_u], "pca_dict": losses_pca_u, "label": "Unscaled"}
    ]
    model_names = [name1, name2]

    # Calculate global max epochs for x-axis scaling
    all_dicts = [losses_ae_basic_s, losses_ae_layered_s, losses_ae_basic_u, losses_ae_layered_u]
    max_epochs_global = 1
    for d in all_dicts:
        for enc in encoding_sizes:
            curve = d.get(enc, [])
            if curve:
                max_epochs_global = max(max_epochs_global, len(curve))

    # 2. Iterate through Rows (Scaling) and Cols (Architecture)
    for row_idx, r_info in enumerate(row_data):
        for col_idx, (ae_dict, m_name) in enumerate(zip(r_info["ae_dicts"], model_names)):
            ax = axes[row_idx, col_idx]
            pca_dict = r_info["pca_dict"]
            
            for enc in encoding_sizes:
                ae_curve = ae_dict.get(enc, [])
                pca_val = pca_dict.get(enc, None)
                
                # Plot AE Curve (Solid)
                if ae_curve:
                    epochs = np.arange(1, len(ae_curve) + 1)
                    line, = ax.plot(epochs, ae_curve, linewidth=1.8, label=f"Enc {enc}")
                    line_color = line.get_color()

                    # Plot PCA Line (Dashed) - Match color to corresponding AE encoding
                    if pca_val is not None:
                        ax.hlines(pca_val, xmin=1, xmax=len(ae_curve), linestyles="--", 
                                  linewidth=1.4, color=line_color, alpha=0.7)

            # --- Styling ---
            if row_idx == 0:
                ax.set_title(f"{m_name} ({r_info['label']})", fontsize=14, fontweight='bold')
            else:
                ax.set_title(f"{m_name} ({r_info['label']})", fontsize=14)
                
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle="--", alpha=0.5)
            
            # Zoom Logic
            start_epoch = max(1, max_epochs_global - zoom_x + 1)
            ax.set_xlim(start_epoch, max_epochs_global)
            ax.set_ylim(bottom=0, top=ylim_top)
            
            if col_idx == 0:
                ax.set_ylabel(f"{r_info['label']} MSE Loss", fontweight='bold')

    # 3. Handle Legend
    # Since colors are consistent by encoding size, we only need one legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Add dummy entries for the line styles
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='gray', lw=1.8, linestyle='-'),
                    Line2D([0], [0], color='gray', lw=1.4, linestyle='--')]
    
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 0.95), title="Encoding Sizes")
    fig.legend(custom_lines, ["AE Model", "PCA Baseline"], loc="upper left", bbox_to_anchor=(1.0, 0.75), title="Line Style")

    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    
    # Save logic
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    full_path = os.path.join(folder_path, f"{save_path}_{runtag}.png")
    plt.savefig(full_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_test_mse_comparison_lines(data_s, data_u, encoding_sizes, title, save_path, folder_path):
    """
    Plots Test MSE vs Encoding Size using the standard data_s/data_u dictionary format.
    
    :param data_s: Dict of {model_label: (train, eval, mse_dict)} for Scaled data
    :param data_u: Dict of {model_label: (train, eval, mse_dict)} for Unscaled data
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    
    # We map specific labels to specific styles to keep the plot clean
    style_map = {
        'basic': {'color': '#1f77b4', 'marker': 'o', 'label': 'Basic AE'},
        'layered': {'color': '#2ca02c', 'marker': 's', 'label': 'Layered AE'},
        'pca-based': {'color': '#EC7063', 'marker': '^', 'label': 'PCA Baseline'},
        'PCA': {'color': '#EC7063', 'marker': '^', 'label': 'PCA Baseline'}
    }

    pipelines = [
        (ax1, data_s, "Pipeline: Trained on Scaled Data"),
        (ax2, data_u, "Pipeline: Trained on Raw Data")
    ]

    for ax, data_type_dict, col_title in pipelines:
        for model_key, results_tuple in data_type_dict.items():
            # Index 2 is the mse_dict: {enc_size: [mse_val]}
            mse_master_dict = results_tuple[au.TEST_MSE_IDX]
            
            y_values = []
            valid_encodings = []

            for enc in encoding_sizes:
                if enc in mse_master_dict:
                    val = mse_master_dict[enc]
                    # Extract the float from the list/array
                    try:
                        clean_val = float(np.array(val).flatten()[0])
                        y_values.append(clean_val)
                        valid_encodings.append(enc)
                    except (IndexError, TypeError):
                        continue
            
            # Get style or use defaults
            style = style_map.get(model_key, {'color': 'gray', 'marker': 'x', 'label': model_key})
            
            # Plot the line
            ax.plot(valid_encodings, y_values, 
                    label=style['label'], 
                    color=style['color'], 
                    marker=style['marker'], 
                    linewidth=2, markersize=8)

            # Add Data Labels
            for x_val, y_val in zip(valid_encodings, y_values):
                ax.annotate(f'{y_val:.4g}', (x_val, y_val), textcoords="offset points", 
                            xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

        # Formatting
        ax.set_title(col_title, fontsize=14, pad=15)
        ax.set_xlabel("Encoding Size (Latent Dimension)", fontsize=12)
        ax.set_ylabel("Test MSE (Original Units)", fontsize=12)
        ax.set_xticks(encoding_sizes)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

        # Add headroom so labels don't get cut off at the top
        curr_ylim = ax.get_ylim()
        ax.set_ylim(curr_ylim[0], curr_ylim[1] * 1.25)

    fig.suptitle(title, fontsize=18, y=1.05)
    plt.tight_layout()
    
    os.makedirs(folder_path, exist_ok=True)
    output_path = os.path.join(folder_path, save_path)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

def plot_comprehensive_comparison_bars(data_s, data_u, encoding_sizes, title, save_path, folder_path):
    """
    Generates a vertical grid of bar charts comparing model performance (MSE) 
    across different encoding sizes and preprocessing pipelines.

    The function creates an (N x 2) grid of subplots where:
    - Each row corresponds to a specific encoding size (latent dimension).
    - The left column displays results for the 'Scaled' data pipeline.
    - The right column displays results for the 'Raw' (Unscaled) data pipeline.
    - Each bar within a subplot represents a specific model architecture 
      (e.g., Basic AE, Layered AE, PCA).

    Parameters:
    -----------
    data_s : dict
        A dictionary containing results for the scaled pipeline. 
        Expected format: {model_label: (train_history, eval_history, mse_dict)}
        where mse_dict is {encoding_size: [mse_value]}.
    data_u : dict
        A dictionary containing results for the raw/unscaled pipeline.
        Same format as data_s.
    encoding_sizes : list of int
        The list of latent dimensions (e.g., [2, 4, 8, 16, 32]) to be plotted as rows.
    title : str
        The main title for the entire figure.
    save_path : str
        The filename for the output image (e.g., 'comparison_grid.png').
    folder_path : str or Path
        The directory where the resulting plot will be saved.
    """
    n_enc = len(encoding_sizes)
    fig, axes = plt.subplots(n_enc, 2, figsize=(14, 4 * n_enc), squeeze=False)
    
    # Standard project colors
    style_map = {
        'basic': {'color': '#1f77b4', 'label': 'Basic AE'},
        'layered': {'color': '#2ca02c', 'label': 'Layered AE'},
        'pca-based': {'color': '#EC7063', 'label': 'PCA'},
        'PCA': {'color': '#EC7063', 'label': 'PCA'}
    }

    model_keys = list(data_s.keys())
    x = np.arange(len(model_keys))
    bar_labels = [style_map.get(k, {'label': k})['label'] for k in model_keys]
    colors = [style_map.get(k, {'color': 'gray'})['color'] for k in model_keys]

    for i, enc in enumerate(encoding_sizes):
        for j, (data_dict, col_title) in enumerate([(data_s, "Scaled Data"), (data_u, "Raw Data")]):
            ax = axes[i, j]
            
            # Extract MSE values from the dictionary structure
            vals = []
            for k in model_keys:
                mse_dict = data_dict[k][2] # Index 2 is TEST_MSE_IDX
                val = mse_dict.get(enc, [0])
                # Flatten to get the raw float
                vals.append(float(np.array(val).flatten()[0]))
            
            bars = ax.bar(x, vals, color=colors, edgecolor='black', alpha=0.8, width=0.6)
            
            ax.set_xticks(x)
            ax.set_xticklabels(bar_labels, fontweight='bold')
            ax.set_title(f"Encoding {enc} | {col_title}", fontsize=13, pad=15)
            ax.set_ylabel("MSE (Original Units)")
            ax.grid(axis='y', linestyle='--', alpha=0.3)

            # Value Labels
            for b in bars:
                height = b.get_height()
                ax.text(b.get_x() + b.get_width()/2, height, f'{height:.4g}', 
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            if len(vals) > 0:
                ax.set_ylim(0, max(vals) * 1.2)

    fig.suptitle(title, fontsize=18, y=1.02)
    plt.tight_layout()
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, save_path), bbox_inches="tight", dpi=150)
    plt.close()
    
def plot_io_scatter(original, reconstructed, title, save_path, log_scale=False):
    """
    Optimized Reconstruction Scatter. 
    Uses hexbin for high-density genomic data to avoid memory crashes.
    """
    # 1. Ensure data is on CPU and flattened
    x = np.array(original).flatten()
    y = np.array(reconstructed).flatten()
    
    # 2. Handle Log Scale Safely (log1p)
    if log_scale:
        x = np.log1p(x)
        y = np.log1p(y)
        label_suffix = " (Log1p)"
    else:
        label_suffix = ""

    # 3. Calculate Correlation
    corr = np.corrcoef(x, y)[0, 1]

    plt.figure(figsize=(9, 8))
    
    # Use hexbin for high density - it's much faster than scatter for millions of points
    hb = plt.hexbin(x, y, gridsize=100, cmap='Blues', mincnt=1, bins='log')
    cb = plt.colorbar(hb, label='log10(count)')

    # Identity Line
    max_val = max(x.max(), y.max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Identity (y=x)')

    plt.title(f"{title}\nPearson Correlation: {corr:.4f}", fontsize=14, fontweight='bold')
    plt.xlabel(f"Ground Truth Expression{label_suffix}", fontweight='bold')
    plt.ylabel(f"Model Reconstruction{label_suffix}", fontweight='bold')
    
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    