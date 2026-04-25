import traceback
import torch
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
import utils.analysis_utils as au
import config as cfg
import numpy as np
import utils.data_utils as du

DATA_TYPE = 'Synthetic' if cfg.SYNTHETIC_DATA else 'True'
ENC_SIZES = cfg.ENCODING_SIZES

DISEASE_MAP = {0: "Healthy", 1: "Disease A (CRC)", 2: "Disease B (SCLC)"}
COLOR_MAP = {
    "Healthy": "#2ecc71",         # Green
    "Disease A (CRC)": "#d43220", # Red
    "Disease B (SCLC)": "#870fb6",# Purple
    "Disease": "#d43220"          # Fallback Red
}

style_map = {
        'basic': {'color': '#1f77b4', 'marker': 'o', 'label': 'Basic AE'},
        'Basic-AE': {'color': '#1f77b4', 'marker': 'o', 'label': 'Basic AE'},
        'layered': {'color': '#2ca02c', 'marker': 's', 'label': 'Layered AE'},
        'Layered-AE': {'color': '#2ca02c', 'marker': 's', 'label': 'Layered AE'},
        'pca-based': {'color': '#EC7063', 'marker': '^', 'label': 'PCA Baseline'},
        'PCA': {'color': '#EC7063', 'marker': '^', 'label': 'PCA Baseline'}
    }
# -- CURVE PLOTS -- #

def plot_train_eval_curves(data_s, data_u, save_name, folder_path, 
                                 include_pca=False, zoom_params=None):
    """
    Plots Train vs Eval for all models.
    zoom_params: dict with {'last_n_epochs': int, 'ylim_top': float} or None
    """
    # Get encoding sizes from the first available model
    first_model = next(iter(data_u.values()))
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
                
                if not train_curve: continue  # noqa: E701
                
                # Setup X-axis
                epochs = np.arange(1, len(train_curve) + 1)
                
                # Plotting AE
                style = style_map.get(model_key, {'color': 'gray', 'marker': 'x', 'label': model_key})

                # current_model = style_map.get(model_key, color_map['default'])                
                ax.plot(epochs, train_curve, label="Train", color=style['color'], linestyle='-', lw=1.5)
                if len(eval_curve) > 0:
                    # Calculate how often we validated (e.g., 300 / 60 = 5)
                    epoch_jump = len(train_curve) // len(eval_curve)
                    
                    # Create an x-axis that jumps by that amount: [5, 10, 15... 300]
                    eval_epochs = np.arange(epoch_jump, len(train_curve) + 1, epoch_jump)
                    
                    # Ensure they match exactly in case of rounding
                    eval_epochs = eval_epochs[:len(eval_curve)] 
                    
                    ax.plot(eval_epochs, eval_curve, label="Eval", color=style['color'], linestyle='--', lw=1.2)
                # Optional PCA Lines
                if include_pca:
                    # Find which PCA key exists in this dataset
                    current_pca_key = next((k for k in pca_keys if k in data_dict), None)
                    if current_pca_key:
                        p_train = data_dict[current_pca_key][0].get(enc, [None])[0]
                        p_eval = data_dict[current_pca_key][1].get(enc, [None])[0]
                        pca_style = style_map.get('PCA',{'color': 'gray', 'marker': 'x', 'label': model_key})
                        if p_train is not None:
                            ax.axhline(y=p_train, color=pca_style['color'], linestyle='-', alpha=0.6, label="PCA Train")
                        if p_eval is not None:
                            ax.axhline(y=p_eval, color=pca_style['color'], linestyle='--', alpha=0.6, label="PCA Eval")
                
                # Zoom Logic (Only if zoom_params is not None)
                if zoom_params:
                    last_n = zoom_params.get('last_n_epochs', 0)
                    y_max = zoom_params.get('ylim_top')
                    
                    if last_n > 0:
                        start_x = max(0, len(train_curve) - last_n)
                        ax.set_xlim(start_x, len(train_curve))

                        if y_max is None:
                            zoomed_train = train_curve[start_x:]
                            
                            # 2. Grab only the eval data in the zoomed window
                            # (Convert start_x to eval_epochs index by dividing by the jump)
                            eval_start_idx = start_x // epoch_jump if 'epoch_jump' in locals() else 0
                            zoomed_eval = eval_curve[eval_start_idx:] if len(eval_curve) > 0 else [0]
                            
                            # 3. Find the highest value in that specific window and add 10% padding
                            local_max = max(max(zoomed_train), max(zoomed_eval))
                            
                            # Ignore massive outlier spikes in PCA baseline if it's way off the chart
                            y_max = local_max * 1.1
                    
                    if y_max is not None:
                        ax.set_ylim(0, y_max)

                # Formatting
                ax.set_title(f"{title} | Enc: {enc}")
                ax.set_ylabel("MSE")
                ax.set_xlabel("Epochs")
                ax.grid(True, which='both', linestyle=':', alpha=0.5)
                ax.legend(loc='upper right', fontsize='x-small')

        plt.tight_layout()
        zoom_str = "_zoomed" if zoom_params else ""
        # Save file
        folder_path.mkdir(parents=True, exist_ok=True)
        filename = f"{save_name}_{model_key}{zoom_str}.png"
        
        # Because folder_path is a Path object, we can just use the / operator!
        plt.savefig(folder_path / filename, bbox_inches='tight', dpi=150)
        plt.close()

def plot_test_mse_comparison_lines(data_s, data_u, encoding_sizes, title, save_path, folder_path):
    """
    Plots Test MSE vs Encoding Size using the standard data_s/data_u dictionary format.
    
    :param data_s: Dict of {model_label: (train, eval, mse_dict)} for Scaled data
    :param data_u: Dict of {model_label: (train, eval, mse_dict)} for Unscaled data
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    
    pipelines = [
        (ax1, data_s, "Pipeline: Trained on Scaled Data"),
        (ax2, data_u, "Pipeline: Trained on Raw Data")
    ]

    for ax, data_type_dict, col_title in pipelines:
        for model_key, results_tuple in data_type_dict.items():
            print(f"model_key is {model_key}")
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
            if style['color'] == 'gray':
                print(f'wtf whats the model key -> {model_key}')
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
    
    folder_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(folder_path / save_path, bbox_inches="tight", dpi=150)
    plt.close()

def plot_comprehensive_comparison_bars(data_s, data_u, encoding_sizes, title, save_path, folder_path):
    n_enc = len(encoding_sizes)
    fig, axes = plt.subplots(n_enc, 2, figsize=(14, 4 * n_enc), squeeze=False)
    
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
                mse_dict = data_dict[k][au.TEST_MSE_IDX]
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
    folder_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(folder_path / save_path, bbox_inches="tight", dpi=150)
    plt.close()


# ---------- Used in model interpretability, refactoring ------ #
def plot_disease_mse_lines(master_results, encoding_sizes, save_path, mode, is_mixed=False):
    """
    Plots the pre-calculated Test MSE vs. Encoding Size lines.
    """
    n_subplots = len(master_results)
    if n_subplots == 0:
        print("No MSE data to plot.")
        return
        
    fig, axes = plt.subplots(1, n_subplots, figsize=(7.5 * n_subplots, 6), sharey=False)
    if n_subplots == 1: axes = [axes]
    
    # Standard styles (matching your other line plots)
    style_map = {
        'basic': {'color': '#1f77b4', 'marker': 'o'},
        'layered': {'color': '#2ca02c', 'marker': 's'},
        'pca-based': {'color': '#EC7063', 'marker': '^'},
        'PCA': {'color': '#EC7063', 'marker': '^'}
    }
    fallback_colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for ax, (base_name, models_dict) in zip(axes, master_results.items()):
        color_idx = 0
        
        for model_label, enc_dict in models_dict.items():
            if not enc_dict: continue
            
            valid_encodings = sorted(list(enc_dict.keys()))
            y_values = [enc_dict[enc] for enc in valid_encodings]
            
            match_key = next((k for k in style_map.keys() if k.lower() in model_label.lower()), None)
            style = style_map[match_key] if match_key else {'color': fallback_colors[color_idx % len(fallback_colors)], 'marker': 'd'}
            if not match_key: color_idx += 1
            
            ax.plot(valid_encodings, y_values, label=model_label, color=style['color'], 
                    marker=style['marker'], linewidth=2, markersize=8)

            for x_val, y_val in zip(valid_encodings, y_values):
                ax.annotate(f'{y_val:.4g}', (x_val, y_val), textcoords="offset points", 
                            xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

        ax.set_title(f"Pipeline: {base_name.upper()}", fontsize=14, pad=15)
        ax.set_xlabel("Encoding Size (Latent Dimension)", fontsize=12)
        ax.set_ylabel("Test MSE (Original Units)", fontsize=12)
        ax.set_xticks(encoding_sizes)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
        curr_ylim = ax.get_ylim()
        ax.set_ylim(curr_ylim[0], curr_ylim[1] * 1.15)

    fig.suptitle(f"Disease Branch Reconstruction: Test MSE vs Encoding Size (Theta: {mode})", fontsize=18, y=1.05)
    plt.tight_layout()
    
    # 100% Pathlib Saving
    out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=is_mixed) / "MSE_Lines"
    out_folder.mkdir(parents=True, exist_ok=True)
    
    output_path = out_folder / f"{save_path}_disease_mse_lines.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved Test MSE Line Plot to {output_path}")

def plot_disease_mse_by_theta_lines(master_results, bin_counts, encoding_sizes, save_path, is_mixed=False):
    """
    Plots the pre-calculated binned Test MSE vs. Encoding Size lines.
    """
    n_subplots = len(master_results)
    if n_subplots == 0:
        print("No binned MSE data to plot.")
        return
        
    fig, axes = plt.subplots(1, n_subplots, figsize=(9 * n_subplots, 7), sharey=False)
    if n_subplots == 1: axes = [axes]
    
    color_map = {'basic': '#1f77b4', 'layered': '#2ca02c', 'pca': '#EC7063'}
    style_map = {'Low (<0.33)': 'dotted', 'Med (0.33-0.66)': 'dashed', 'High (>0.66)': 'solid'}
    
    for ax, (base_name, models_dict) in zip(axes, master_results.items()):
        for model_label, bin_dict in models_dict.items():
            m_color = next((v for k, v in color_map.items() if k in model_label.lower()), 'gray')
            
            for bin_name, enc_dict in bin_dict.items():
                if not enc_dict: continue
                
                valid_encodings = sorted(list(enc_dict.keys()))
                y_values = [enc_dict[enc] for enc in valid_encodings]
                
                l_style = style_map.get(bin_name, 'solid')
                n_samples = bin_counts[bin_name]
                label_text = f"{model_label} [{bin_name} | N={n_samples}]"
                
                ax.plot(valid_encodings, y_values, label=label_text, color=m_color, 
                        linestyle=l_style, marker='o', linewidth=2, markersize=6)

                # Data Labels
                for x_val, y_val in zip(valid_encodings, y_values):
                    ax.annotate(f'{y_val:.0f}', (x_val, y_val), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=8)

        ax.set_title(f"Pipeline: {base_name.upper()}", fontsize=14, pad=15)
        ax.set_xlabel("Encoding Size (Latent Dimension)", fontsize=12)
        ax.set_ylabel("Test MSE (Original Units)", fontsize=12)
        ax.set_xticks(encoding_sizes)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        curr_ylim = ax.get_ylim()
        ax.set_ylim(curr_ylim[0], curr_ylim[1] * 1.15)

    fig.suptitle("Disease MSE by Theta Bins vs Encoding Size", fontsize=18, y=1.05)
    plt.tight_layout()
    
    # 100% Pathlib Saving
    out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=is_mixed) / "MSE_Lines"
    out_folder.mkdir(parents=True, exist_ok=True)
    
    output_path = out_folder / f"{save_path}_disease_mse_by_theta.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved Theta Binned Test MSE Line Plot to {output_path}")

def _plot_simple_boxplot(ax, test_truth_disease, test_truth_healthy, recon_h, recon_d, thetas, model_label, enc):
    """Handles the detailed 4-category synthetic data boxplots for mixed datasets."""
    num_samples = len(thetas)
    sample_is_disease = (thetas > 0).values
    
    # 1. Build the dynamic benchmark truth
    benchmark_truth = np.zeros_like(test_truth_healthy)
    for i in range(num_samples):
        if sample_is_disease[i]:
            benchmark_truth[i] = test_truth_disease[i] # Target: Pure Cancer
        else:
            benchmark_truth[i] = test_truth_healthy[i] # Target: Pure Healthy
            
    # 2. Flatten everything
    flat_benchmark_truth = benchmark_truth.flatten()
    flat_h_recon = recon_h.detach().cpu().numpy().flatten()
    flat_d_recon = recon_d.detach().cpu().numpy().flatten()
    
    # 3. Build the 4-category labels
    sample_labels = np.where(sample_is_disease, "Disease Sample", "Healthy Sample")
    gene_template = np.array(['Healthy Genes (0-499)'] * 500 + ['Disease Genes (500-999)'] * 500)
    
    flat_gene_labels = np.tile(gene_template, num_samples)
    flat_sample_labels = np.repeat(sample_labels, 1000)
    flat_combined_labels = [f"{s}\n{g}" for s, g in zip(flat_sample_labels, flat_gene_labels)]
    
    # 4. Build the DataFrame
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

    # Filter out the combinations we don't want to plot
    plot_df = plot_df[plot_df.apply(lambda row: row['Source'] in selection_map.get(row['Module Group'], []), axis=1)]
    
    # 5. Plot
    sns.boxplot(
        data=plot_df, x='Module Group', y='Expression', hue='Source', ax=ax, 
        palette=['#95a5a6', '#2ecc71', '#3498db'], showfliers=False, order=plot_order
    )
    
    # 6. Formatting
    ax.tick_params(axis='x', labelsize=7)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="center", rotation_mode="anchor")
    ax.legend(fontsize='x-small', title_fontsize='8', loc='upper right')
    
    ax.axhline(0, color="#f97a7a", linestyle='--', alpha=0.3)
    ax.axhline(100, color="#85f492", linestyle='--', alpha=0.3)
    ax.set_ylim(-10, 150)
    
    ax.set_title(model_label, fontweight='bold')
    ax.set_ylabel(f"Enc: {enc}\nExpression")
    
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

def _plot_evaluation_scatter(ax, flat_input, flat_recon, flat_labels, color_map, title, xlabel, ylabel, line_color="#34495e"):
    """A universal scatter plot generator for model evaluation."""
    sns.scatterplot(
        x=flat_input, y=flat_recon, hue=flat_labels, 
        palette=color_map, s=1, alpha=0.3, ax=ax, 
        edgecolor='none', legend=False 
    )
    
    # Identity Line
    max_val = max(np.nanmax(flat_input), np.nanmax(flat_recon))
    min_val = min(np.nanmin(flat_input), np.nanmin(flat_recon))
    ax.plot([min_val, max_val], [min_val, max_val], color=line_color, linestyle='--', linewidth=1)
    
    # Metrics
    r2 = r2_score(flat_input, flat_recon)
    pearson_r, _ = stats.pearsonr(flat_input, flat_recon)
    text_str = f'$R^2 = {r2:.3f}$\n$r = {pearson_r:.3f}$'
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    # Set the upper limits of the X and Y axes to 1500
    ax.set_xlim(right=1500)
    ax.set_ylim(top=1500)

      
def _annotate_outliers(ax, flat_input, flat_recon, gene_names, sample_names, gene_size, threshold):
    """Annotates points where the residual error exceeds the threshold with Sample & Gene."""
    # Calculate absolute error
    residuals = np.abs(flat_input - flat_recon)
    
    # Find flat indices of all outliers
    outlier_indices = np.where(residuals > threshold)[0]

    for idx in outlier_indices:
        # ---------------------------------------------------------
        # 🛠️ THE FIX: Translate flat 1D index back to 2D (Row, Col)
        # ---------------------------------------------------------
        sample_idx = idx // gene_size
        gene_idx = idx % gene_size
        
        # Look up the actual names
        sample = sample_names[sample_idx]
        gene = gene_names[gene_idx]
        
        x_val = flat_input[idx]
        y_val = flat_recon[idx]
        
        # Create a combined label
        label_text = f"{sample}\n({gene})"
        
        # Annotate the plot
        ax.annotate(
            label_text, 
            (x_val, y_val), 
            textcoords="offset points", 
            xytext=(5, 5), 
            ha='left', 
            fontsize=7,
            color="#c0392b", # A nice subtle red for outliers
            arrowprops=dict(arrowstyle="->", color='gray', lw=0.5, alpha=0.7)
        )

def print_top_outliers(truth_2d, recon_2d, sample_names, gene_names, threshold=1500, top_n=20):
    """Instantly finds and prints the biggest reconstruction errors."""
    # 1. Calculate the error matrix directly in 2D
    residuals = np.abs(truth_2d - recon_2d)
    
    # 2. Get the 2D coordinates of the outliers
    sample_indices, gene_indices = np.where(residuals > threshold)
    
    # 3. Build a list of dictionaries
    outliers = []
    for s_idx, g_idx in zip(sample_indices, gene_indices):
        outliers.append({
            "Sample": sample_names[s_idx],
            "Gene": gene_names[g_idx],
            "True_Val": round(truth_2d[s_idx, g_idx], 2),
            "Recon_Val": round(recon_2d[s_idx, g_idx], 2),
            "Error": round(residuals[s_idx, g_idx], 2)
        })
        
    # 4. Convert to a DataFrame and sort by the biggest errors
    df_outliers = pd.DataFrame(outliers)
    
    if df_outliers.empty:
        print(f"\n✅ No outliers found above threshold {threshold}")
        return df_outliers
        
    df_outliers = df_outliers.sort_values(by="Error", ascending=False)
    
    # 5. Print a beautiful table to the terminal
    print(f"\n🚨 Found {len(df_outliers)} Outliers (Showing Top {top_n}) 🚨")
    print(df_outliers.head(top_n).to_string(index=False))
    print("-" * 50)
    
    return df_outliers

def plot_reconstruction_grid(labels_dict, inference_cache, test_df_full, test_n_theta, 
                             true_disease_input, gene_size, scaler, scale_bool, 
                             save_path, mode, is_simple=False, is_mixed=False, target_type='total'):
    """
    Master function to plot either Total Mix Reconstruction OR Disease Signal Isolation.
    target_type: 'total' or 'disease'
    """
    tag = "scaled" if scale_bool else "unscaled"
    
    for base_name, models in labels_dict.items():
        n_rows = len(cfg.ENCODING_SIZES)
        n_cols = len(models)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)
        
        # Dynamic Titles
        if target_type == 'total':
            fig.suptitle(f"Total Mix Reconstruction (Phase: DISEASE MIX | Base: {base_name.upper()})\nTotal Input vs. Total Recon (theta: {mode})", fontsize=18, fontweight='bold', y=0.98)
        else:
            plot_type = "Boxplot" if is_simple else "Scatter"
            fig.suptitle(f"Disease Signal Isolation ({plot_type} | Base: {base_name.upper()})\nTrue Pure Disease vs. Disease Branch Output (theta: {mode})", fontsize=18, fontweight='bold', y=0.98)
                     
        for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
            for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                ax = axes[row_idx, col_idx]
                try:
                    # 1. Cache Lookup
                    model_outputs = inference_cache[base_name][enc].get(model_label)
                    
                    target_key = 'mix' if target_type == 'total' else 'disease'
                    if model_outputs is None or model_outputs[target_key] is None:
                        ax.text(0.5, 0.5, "Model / Output Not Found", ha='center', color='red')
                        continue
                        
                    recon_tensor = model_outputs[target_key]
                    
                    # 2. Extract Ground Truth based on target type
                    if target_type == 'total':
                        # truth_array = test_n_theta.detach().cpu().numpy()
                        mask = np.ones(len(test_df_full), dtype=bool)
                    else:
                        mask = (test_df_full['theta_value'] > 0).values

                        # test_truth_d = true_disease_input.reindex(test_df_full.index)
                        # test_truth_h = true_healthy.reindex(test_df_full.index)
                        # benchmark_truth = test_truth_d.fillna(test_truth_h)
                        # truth_array = benchmark_truth.values
                    mask_tensor = torch.tensor(mask)
                    current_df = test_df_full.iloc[mask]
                    recon_sliced = recon_tensor[mask_tensor]
                    if target_type == 'total':
                        truth_array = test_n_theta[mask_tensor].detach().cpu().numpy()
                    else:
                        # Reindex against ONLY the disease samples
                        test_truth_d = true_disease_input.reindex(current_df.index)
                        truth_array = test_truth_d.values
                    
                    truth_array = truth_array[:, :gene_size]

                    # Handle Scaling
                    if scale_bool and scaler is not None:
                        recon_final = du.inverse_scale(scaler, recon_sliced).detach().cpu().numpy()
                    else:
                        recon_final = recon_sliced.detach().cpu().numpy()

                    gene_names = current_df.drop(columns=['theta_value', 'disease_type'], errors='ignore').columns
                    sample_names = current_df.index
                    print(f"\n[Checking {model_label} - Enc {enc}]")
                    print_top_outliers(truth_array, recon_final, sample_names, gene_names, threshold=1500, top_n=30)

                    flat_input = truth_array.flatten()
                    flat_recon = recon_final.flatten()
                    
                    # 4. Route to the correct helper plot
                    if target_type == 'disease' and is_simple:
                        _plot_simple_boxplot(
                            ax=ax, 
                            test_truth_disease=test_truth_d, 
                            test_truth_healthy=None,
                            recon_h=model_outputs['healthy'], 
                            recon_d=recon_tensor, 
                            thetas=test_df_full['theta_value'], 
                            model_label=model_label, 
                            enc=enc
                        )
                    else:
                        # Prepare Labels and Colors
                        if 'disease_type' in current_df.columns:
                            sample_labels = current_df['disease_type'].map(DISEASE_MAP).fillna("Unknown")
                        else:
                            sample_labels = np.where(current_df['theta_value'] == 0, "Healthy", "Disease")

                        sample_labels = pd.Series(sample_labels)  
                        flat_labels = np.repeat(sample_labels.values, gene_size)
                        
                        if is_simple and target_type == 'total':
                            _plot_simple_total_scatter(ax, flat_input, flat_recon, model_label, enc)
                        else:
                            # Use the unified evaluation scatter function you created!
                            title = f"{model_label} (Enc {enc})"
                            xlabel = "Total Mixed Input" if target_type == 'total' else "True Pure Disease"
                            ylabel = "Total Mixed Recon" if target_type == 'total' else "Disease Branch Recon"
                            line_color = "#34495e" if target_type == 'total' else "#9a1b0c"
                            
                            _plot_evaluation_scatter(ax, flat_input, flat_recon, flat_labels, COLOR_MAP, title, xlabel, ylabel, line_color)
                        
                        # 5. Annotations
                        
                        dynamic_thresh = 1500
                        _annotate_outliers(ax, flat_input, flat_recon, gene_names, sample_names, gene_size, dynamic_thresh)

                    # Clean up grid inner labels
                    if col_idx > 0: ax.set_ylabel("")
                    if row_idx < n_rows - 1: ax.set_xlabel("")
                    
                except Exception as e:
                    traceback.print_exc()
                    ax.text(0.5, 0.5, "Plotting Error", ha='center', color='red')

        # Add the Universal Master Legend
        if not is_simple or (not is_simple and target_type == 'disease'):
            unique_classes = pd.unique(sample_labels) if 'sample_labels' in locals() else []
            legend_elements = []
            for cls in unique_classes:
                color = COLOR_MAP.get(cls, "#7f8c8d")
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=cls, markerfacecolor=color, markersize=8))
            
            line_lbl = 'Perfect Reconstruction'
            legend_elements.append(Line2D([0], [0], color='#34495e' if target_type=='total' else '#2D2A2A', linestyle='--', linewidth=1, label=line_lbl))
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)

        # Save Figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=is_mixed) / f"Tournament_H-{base_name}"
        out_folder.mkdir(parents=True, exist_ok=True)

        data_tag = "_simple" if is_simple else "_complex"
        if cfg.SYNTHETIC_DATA != "synthetic": data_tag = ""
        plt.savefig(out_folder / f"{save_path}_{tag}{data_tag}.png", dpi=150)
        plt.close(fig)


def analyze_disease_portion_reconstruction_scatter(labels_dict, inference_cache, test_df_full, true_disease_input, scaler, scale_bool, gene_size, save_path, mode, is_simple=False, is_mixed=False):
    """
    Evaluates Disease Branch Reconstruction using pre-computed inference cache.
    Dynamically switches between Boxplots and Scatter plots based on is_simple.
    """
    tag = "scaled" if scale_bool else "unscaled"
    # input_size = test_df_full.shape[1]
    # gene_size = input_size - 1
    _, true_healthy = du.load_reconstruction_data('healthy', mode)
    
    # Plotting Loop
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
                    model_outputs = inference_cache[base_name][enc].get(model_label)
                    
                    if model_outputs is None or model_outputs['disease'] is None:
                        ax.text(0.5, 0.5, "Model / Output Not Found", ha='center', color='red')
                        continue
                        
                    recon_d = model_outputs['disease']
                        
                    # Extract Data
                    test_truth_d = true_disease_input.reindex(test_df_full.index)
                    test_truth_h = true_healthy.reindex(test_df_full.index)
                    benchmark_truth = test_truth_d.fillna(test_truth_h)
                    if scale_bool and scaler is not None:
                        recon_final = du.inverse_scale(scaler, recon_d).detach().cpu().numpy()
                        # input_final = scaler.inverse_transform(benchmark_truth.values)   
                    else:
                        recon_final = recon_d.detach().cpu().numpy()
                        # input_final = benchmark_truth.values
                    flat_input = benchmark_truth.values.flatten()
                    flat_recon = recon_final.flatten()
                    
                    # Route to the correct plot type
                    if is_simple:
                        _plot_simple_boxplot(
                            ax=ax, 
                            test_truth_disease=test_truth_d, 
                            test_truth_healthy=true_healthy.reindex(test_df_full.index).values, 
                            recon_h=model_outputs['healthy'], 
                            recon_d=recon_d, 
                            thetas=test_df_full['theta_value'], 
                            model_label=model_label, 
                            enc=enc
                        )
                    else:
                        disease_map = {0: "Healthy", 1: "Disease A (CRC)", 2: "Disease B (SCLC)"}
                        color_map = {
                            "Healthy": "#2ecc71",         # Green
                            "Disease A (CRC)": "#d43220", # Red
                            "Disease B (SCLC)": "#870fb6",# Purple
                            "Disease": "#d43220"          # Fallback Red
                        }
                        
                        # color_map = {"Disease A (CRC)": "#d43220", "Disease B (SCLC)": "#870fb6"}
                        # disease_map = {1: "Disease A (CRC)", 2: "Disease B (SCLC)"}
                        
                        if 'disease_type' in test_df_full.columns:
                            sample_labels = test_df_full['disease_type'].map(disease_map).fillna("Unknown")
                        else:
                            sample_labels = pd.Series(["Disease"] * len(test_df_full))
                            color_map = {"Disease": "#8e44ad"} 
                            
                        flat_labels = np.repeat(sample_labels.values, gene_size)
                        # _plot_complex_scatter(ax, flat_input, flat_recon, flat_labels, color_map, model_label, enc)
                        
                        threshold = 1500
                        mask_2d = (recon_final > threshold)  | (benchmark_truth.values > threshold )
                        gene_names = benchmark_truth.columns
                        
                        genes_over_thresh = gene_names[mask_2d.any(axis=0)]
                        
                        if len(genes_over_thresh) > 0:
                            print(f"\n[Model: {model_label} | Enc: {enc}]")
                            print(f"Genes reconstructed > {threshold}: {genes_over_thresh.tolist()}. Num is {len(genes_over_thresh)}")
                        
                        # 2. GRAPHING: Annotate the points on the scatter plot
                        # Find the 1D indices where the flattened reconstruction is over the threshold
                        over_thresh_indices = np.where((flat_recon > threshold) | (flat_input > threshold ))[0]
                        
                        # Limit annotations to prevent text-overlap on extremely dense graphs
                        max_annotations = 30
                        
                        for i, flat_idx in enumerate(over_thresh_indices):
                            if i >= max_annotations:
                                print(f"  ... and {len(over_thresh_indices) - max_annotations} more points (annotations truncated for readability).")
                                break
                                
                            x_val = flat_input[flat_idx]
                            y_val = flat_recon[flat_idx]
                            
                            # Map the 1D index back to the gene column index using modulo
                            gene_col_idx = flat_idx % gene_size
                            gene_name = gene_names[gene_col_idx]
                            
                            # Draw the text next to the point
                            ax.annotate(gene_name, 
                                        (x_val, y_val), 
                                        xytext=(5, 5), # Offset the text slightly
                                        textcoords='offset points',
                                        fontsize=8, 
                                        color='black',
                                        alpha=0.7)
                    # Clean up grid inner labels
                    if col_idx > 0: ax.set_ylabel("")
                    if row_idx < n_rows - 1: ax.set_xlabel("")
                    
                except Exception as e:
                    traceback.print_exc()
                    ax.text(0.5, 0.5, "Plotting Error", ha='center', color='red')

        # Add the Universal Master Legend
        if not is_simple:
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Disease A (CRC)', markerfacecolor='#d43220', markersize=8),
                Line2D([0], [0], marker='o', color='w', label='Disease B (SCLC)', markerfacecolor='#870fb6', markersize=8),
                Line2D([0], [0], color="#2D2A2A", linestyle='--', linewidth=1, label='Perfect Reconstruction')
            ]
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)

        # Save Figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=is_mixed) / f"Tournament_H-{base_name}"
        # os.makedirs(out_folder, exist_ok=True)

        data_tag = "simple" if is_simple else "complex"
        plt.savefig(out_folder / f"{save_path}_{tag}_{data_tag}.png", dpi=150)
        plt.close(fig)
  