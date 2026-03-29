from adjustText import adjust_text
import matplotlib.patheffects as PathEffects
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.colors as mcolors
import io
import traceback
import joblib
import os

from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
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

SCALED, MIXED = True, True
UNSCALED, NOT_MIXED = False, False


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

def generate_inference_cache(labels_dict, test_w_theta_t, gene_size, tag):
    """
    Loops through all models and encodings, runs the forward pass ONCE, 
    and caches the output tensors in memory.
    """
    cache = {}
    for base_name, models in labels_dict.items():
        cache[base_name] = {}
        for enc in cfg.ENCODING_SIZES:
            cache[base_name][enc] = {}
            for model_label, folder_tag in models.items():
                try:
                    
                    # Run the heavy PyTorch inference ONCE
                    recon_mix, recon_d, recon_h, _ = mu.create_load_mix_model(
                        folder_tag=folder_tag, test_set=test_w_theta_t, 
                        gene_size=gene_size, enc=enc, scale_tag=tag
                    )
                    
                    # Store the results
                    cache[base_name][enc][model_label] = {
                        'mix': recon_mix,
                        'disease': recon_d,
                        'healthy': recon_h
                    }
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    cache[base_name][enc][model_label] = None
                    
    return cache

def analyze_total_reconstruction(labels_dict, inference_cache, test_df_full, test_n_theta, gene_size, scaler, scale_bool, save_path, mode, is_simple=False, is_mixed=False):
    """
    Evaluates the Total Mix Reconstruction (recon_mix) using pre-computed inference cache.
    Colors the complex data dynamically based on available classes.
    test_df - has genes with theta column
    test_w_type - has gene with theta and disease type
    """
    tag = "scaled" if scale_bool else "unscaled"
    # input_size = test_df.shape[1/\]

    # gene_size = input_size - 1
    
    # # test_df has theta isolated, so test_no_theta_t is purely genes
    # test_no_theta_t = torch.Tensor(test_df.drop(columns=['theta_value', 'disease_type'], errors='ignore').values).float()

    # Plotting Loop
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
                    # 🚀 INSTANT CACHE LOOKUP (No PyTorch inference here!)
                    model_outputs = inference_cache[base_name][enc].get(model_label)
                    
                    if model_outputs is None or model_outputs['mix'] is None:
                        ax.text(0.5, 0.5, "Model / Output Not Found", ha='center', color='red')
                        continue
                        
                    recon_mix_t = model_outputs['mix']
                    
                    if scale_bool and scaler is not None:
                        recon_mix = du.inverse_scale(scaler, recon_mix_t).detach().cpu().numpy()
                        #inverse scale the input too
                        # input_final = du.inverse_scale(scaler, test_n_theta).detach().cpu().numpy()
                    else:
                        recon_mix = recon_mix_t.detach().cpu().numpy()
                        # input_final = test_n_theta.numpy()
                    # Flatten the data for scatter
                    flat_input = test_n_theta.detach().cpu().numpy().flatten()
                    flat_recon = recon_mix.flatten()
                    
                    # Route to correct plot
                    if is_simple:
                        _plot_simple_total_scatter(ax, flat_input, flat_recon, model_label, enc)
                    else:
                        disease_map = {0: "Healthy", 1: "Disease A (CRC)", 2: "Disease B (SCLC)"}
                        color_map = {
                            "Healthy": "#2ecc71",         # Green
                            "Disease A (CRC)": "#d43220", # Red
                            "Disease B (SCLC)": "#870fb6",# Purple
                            "Disease": "#d43220"          # Fallback Red
                        }
                        
                        if 'disease_type' in test_df_full.columns:
                            sample_labels = test_df_full['disease_type'].map(disease_map).fillna("Unknown")
                        else:
                            sample_labels = np.where(test_df_full['theta_value'] == 0, "Healthy", "Disease")
                            sample_labels = pd.Series(sample_labels)
                            
                        flat_labels = np.repeat(sample_labels.values, gene_size)
                        _plot_complex_total_scatter(ax, flat_input, flat_recon, flat_labels, color_map, model_label, enc)
                        
                    # Clean up grid inner labels
                    if col_idx > 0: ax.set_ylabel("")
                    if row_idx < n_rows - 1: ax.set_xlabel("")
                    
                except Exception as e:
                    traceback.print_exc()
                    ax.text(0.5, 0.5, "Plotting Error", ha='center', color='red')

        # Add the Universal Master Legend for the 3 Classes
        if not is_simple:
            unique_classes = pd.unique(sample_labels) 
            legend_elements = []
            for cls in unique_classes:
                color = color_map.get(cls, "#7f8c8d")
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', label=cls, markerfacecolor=color, markersize=8)
                )
            legend_elements.append(Line2D([0], [0], color='#34495e', linestyle='--', linewidth=1, label='Perfect Reconstruction'))
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)

        # Save Figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=is_mixed) / f"Tournament_H-{base_name}"
        os.makedirs(out_folder, exist_ok=True)

        data_tag = "simple" if is_simple else "complex"
        plt.savefig(out_folder / f"{save_path}_{tag}_{data_tag}.png", dpi=150)
        plt.close(fig)

def analyze_disease_portion_reconstruction_scatter(labels_dict, inference_cache, test_df_full, true_disease_input, scaler, scale_bool, gene_size, save_path, mode, is_simple=False, is_mixed=False):
    """
    Evaluates Disease Branch Reconstruction using pre-computed inference cache.
    Dynamically switches between Boxplots and Scatter plots based on is_simple.
    """
    tag = "scaled" if scale_bool else "unscaled"
    # input_size = test_df_full.shape[1]
    # gene_size = input_size - 1
    _, true_healthy = load_reconstruction_data('healthy', mode)
    
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
                        _plot_complex_scatter(ax, flat_input, flat_recon, flat_labels, color_map, model_label, enc)
                        
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
        os.makedirs(out_folder, exist_ok=True)

        data_tag = "simple" if is_simple else "complex"
        plt.savefig(out_folder / f"{save_path}_{tag}_{data_tag}.png", dpi=150)
        plt.close(fig)
    
def prepare_scatter_data(inference_cache, base_name, enc, model_label, 
                                 test_df_full, disease_target, 
                                 scale_bool, scaler):
    """
     Boilerplate to extract and average data for scatter plots.
     Compares TRUE Healthy baseline vs TARGET Disease cohort residuals.
     Returns strictly positive unscaled (inverse_scaled) counts.
    """
    try:
        model_outputs = inference_cache[base_name][enc].get(model_label)
        if model_outputs is None:
            return None
            
        # 1. Pull Tensors
        recon_mix_tensor = model_outputs['mix'] # Total reconstruction (H+D)
        recon_d_tensor = model_outputs['disease'] # Isolated Disease Signal (Z_d)
        recon_h_tensor = model_outputs['healthy'] # Healthy Branch output (Z_h)

        # 2. Inverse Scaling (Mandatory for these biological counts plots)
        if scale_bool and scaler is not None:
            recon_mix_np = du.inverse_scale(scaler, recon_mix_tensor).detach().cpu().numpy()
            recon_d_np = du.inverse_scale(scaler, recon_d_tensor).detach().cpu().numpy()
            recon_h_np = du.inverse_scale(scaler, recon_h_tensor).detach().cpu().numpy()
        else:
            recon_mix_np = recon_mix_tensor.detach().cpu().numpy()
            recon_d_np = recon_d_tensor.detach().cpu().numpy()
            recon_h_np = recon_h_tensor.detach().cpu().numpy()
            
        # 3. Create Masks
        is_target_disease = (test_df_full['disease_type'] == disease_target).values
        is_true_healthy = (test_df_full['disease_type'] == 0).values # The strict baseline

        if not np.any(is_target_disease) or not np.any(is_true_healthy):
            print(f"⚠️ Missing cohorts for scatter prep (Target:{disease_target})")
            return None

        
        # Note: avg_d_disease can be negative (subtraction), we don't clip it here.
        
        return recon_h_np, is_true_healthy, recon_mix_np,is_target_disease,recon_d_np 

    except Exception as e:
        traceback.print_exc()
        return None
    

def plot_absolute_expression_scatter_template(ax, h_baseline_avg, mix_disease_avg, 
                                             title,
                                             show_identity=True, 
                                             fold_change_lines=None, 
                                             highlight_top_n=10, gene_names=None):
    """
    Absolute Total Expression Comparison (Log-Log)
    
    line_options parameter example: [2.0, 3.0] -> draws 2x and 3x fold change lines
    """
    
    # Set logarithmic scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot the dense gene cloud
    ax.scatter(h_baseline_avg, mix_disease_avg, alpha=0.3, s=8, color='#7f8c8d', edgecolors='none')
    
    # 1. OPTION: Drawing reference lines
    # Calculate plot limits based on data range (min to max)
    lim_min = min(h_baseline_avg.min(), mix_disease_avg.min())
    lim_max = max(h_baseline_avg.max(), mix_disease_avg.max())
    line_range = np.linspace(lim_min, lim_max, 100)
    
    if show_identity:
        ax.plot(line_range, line_range, color='black', linewidth=1.2, label='Identity (Y=X)')

    # if fold_change_lines is not None:
    #     # Example fold_change_lines = [2.0, 3.0]
    #     # Iterate through both up and down lines
    #     for fc in fold_change_lines:
    #         # Upregulation lines (Red dashed)
    #         ax.plot(line_range, line_range * fc, color='#d62728', linestyle='--', linewidth=0.8, alpha=0.7)
    #         # Downregulation lines (Blue dashed)
    #         # ax.plot(line_range, line_range / fc, color='#1f77b4', linestyle='--', linewidth=0.8, alpha=0.7)

    # 2. OPTION: Highlight and Label Top Outliers
    if highlight_top_n > 0 and gene_names is not None:
        # Calculate ratio, handling potential zeros in denominator just in case
        epsilon = 1.0
        ratios = mix_disease_avg / (h_baseline_avg + epsilon)
        
        # Get absolute deviation from 1.0 ratio to find both up and down outliers
        deviation_from_identity = np.abs(ratios - 1.0)
        top_indices = np.argsort(deviation_from_identity)[-highlight_top_n:]
        
        # Scatter the outliers in a prominent color
        ax.scatter(h_baseline_avg[top_indices], mix_disease_avg[top_indices], s=15, color='#e67e22', edgecolors='black', linewidth=0.5)
        
        # Add labels
        for idx in top_indices:
            # Shift label position slightly
            ax.text(h_baseline_avg[idx] * 1.1, mix_disease_avg[idx] * 1.05, gene_names[idx], fontsize=7)

    # Aesthetics
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(r"Avg Healthy Baseline ($Z_{h\_healthy}$)", fontsize=10)
    ax.set_ylabel(r"Avg Total Mix Recon ($Z_{h}+Z_{d}$ Target)", fontsize=10)
    
    # Use generic formatted labels (10^1, 10^2, etc)
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    
    ax.grid(True, which="both", ls="-", alpha=0.2)
    # plt.tight_layout() # Move tight_layout to the call function

def plot_absolute_expression_scatter_with_significance(ax, 
                                                       healthy_cohort_matrix, 
                                                       disease_cohort_matrix, 
                                                       title, gene_names=None,
                                                       alpha_threshold=0.0000005,
                                                       highlight_top_n=5):
    """
    Plots Absolute Expression and colors points by q-value significance.
    Requires the full 2D matrices (Patients x Genes), not just the 1D averages!
    """
    # 1. Calculate Averages for the X and Y axes
    h_avg = healthy_cohort_matrix.mean(axis=0)
    d_avg = disease_cohort_matrix.mean(axis=0)
    
    # 2. Calculate P-Values (Welch's t-test for unequal variances)
    # Compares the distribution of the healthy cohort vs disease cohort per gene
    t_stats, p_values = ttest_ind(healthy_cohort_matrix, disease_cohort_matrix, 
                                  axis=0, equal_var=False)
    
    # Clean up any NaNs (genes with 0 variance)
    p_values = np.nan_to_num(p_values, nan=1.0)
    
    # 3. Calculate Q-Values (FDR Correction)
    _, q_values, _, _ = multipletests(p_values, method='fdr_bh')
    
    # 4. Map to -log10 for the color scale
    # Add a tiny epsilon to prevent log(0) for perfectly separated genes
    neg_log10_q = -np.log10(q_values + 1e-300)
    
    # 5. Create the Masks
    epsilon = 1.0
    fold_change = (d_avg + epsilon) / (h_avg + epsilon)
    
    # Set a minimum biological change (e.g., 1.2 means a 20% increase or decrease)
    fc_threshold =2 
    
    is_stat_sig = q_values <= alpha_threshold
    is_bio_sig = (fold_change >= fc_threshold) | (fold_change <= (1.0 / fc_threshold))
    # is_bio_sig_up = fold_change >= fc_threshold
    # The gene MUST be both statistically different AND physically different
    is_significant = is_stat_sig & is_bio_sig
    up_regulated_sig_mask = is_significant & (fold_change >= fc_threshold)
    # True if significant, False if not
    # is_significant = q_values <= alpha_threshold
    
    # # Set logarithmic scales for the axes
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    
    lim_min = min(h_avg.min(), d_avg.min())
    lim_max = max(h_avg.max(), d_avg.max())
    line_range = np.linspace(lim_min, lim_max, 100)
    
    ax.plot(line_range, line_range * fc_threshold, color="#b84646", linestyle='--', linewidth=0.8, alpha=0.7, )
    ax.plot(line_range, line_range / fc_threshold, color="#b84646", linestyle='--', linewidth=0.8, alpha=0.7)

    epsilon = 1.0
    # ratios = d_avg / (h_avg + epsilon)
    up_indices = np.where(up_regulated_sig_mask)[0]
    if len(up_indices) > 0:
    # 3. Rank them by their fold change value
    # We sort the fold_change values of ONLY the significant up-regulated genes
        top_up_indices = up_indices[np.argsort(fold_change[up_indices])[-highlight_top_n:]]
    else:
        top_up_indices = []
    # Get absolute deviation from 1.0 ratio to find both up and down outliers
    # deviation_from_identity = np.abs(fold_change - 1.0)
    # top_indices = np.argsort(deviation_from_identity)[-highlight_top_n:]
    
    # Scatter the outliers in a prominent color
    ax.scatter(h_avg[top_up_indices], d_avg[top_up_indices], s=15, color='none', edgecolors='black', linewidth=0.5)
    
    # Add labels
    texts = []
    for idx in top_up_indices:
        # We place the text exactly ON the dot. adjust_text will move it later.
        txt = ax.text(h_avg[idx], d_avg[idx], gene_names[idx], 
                        fontsize=8, fontweight='bold', color='black', zorder=10)
        
        # THE CONTRAST FIX: Add a thick white outline to the black text
        txt.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='white')])
        
        texts.append(txt)
    if texts:
        adjust_text(texts, ax=ax, 
                    arrowprops=dict(arrowstyle="-", color='black', lw=0.8, alpha=0.7),
                    expand_points=(1.5, 1.5),  # Force it to push further from dots
                    force_text=(0.5, 0.5))
    # 6. Plotting Part A: The Background (Non-Significant Genes)
    
    # These get plotted first so they sit in the background as light grey
    ax.scatter(h_avg[~is_significant], d_avg[~is_significant], 
               color='#d3d3d3', alpha=0.5, s=8, edgecolors='none', label='Not Significant')
    
    # 7. Plotting Part B: The Significant Genes (Color Mapped)
    # Create a custom colormap from Dark Grey to Red
    cmap = mcolors.LinearSegmentedColormap.from_list("sig_cmap", ["#696969", "#ff0000"])
    min_color_val = -np.log10(alpha_threshold)
    max_color_val = 10.0
    # Scatter the significant points, using 'c' to map the neg_log10_q values to the cmap
    sc = ax.scatter(h_avg[is_significant], d_avg[is_significant], 
                    c=neg_log10_q[is_significant], cmap=cmap, 
                    vmin=min_color_val, vmax=max_color_val,
                    alpha=0.8, s=15, edgecolors='black', linewidth=0.2)
    
    # 8. Add the Colorbar to the axis
    cbar = ax.figure.colorbar(sc, ax=ax)
    def q_val_formatter(x, pos):
        return f"$10^{{-{x:g}}}$"
        
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(q_val_formatter))
    cbar.set_label(r'$-\log_{10}(q\text{-value})$', fontsize=9)
    
    # Draw the Identity line
    ax.axline((1, 1), (10,10), color='black', linewidth=1.2, linestyle='--', label='Identity')
    ax.set_xlim([lim_min, lim_max])
    ax.set_ylim([lim_min, lim_max])
    # Aesthetics
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(r"Avg Healthy Baseline ($Z_{h\_healthy}$)", fontsize=10)
    ax.set_ylabel(r"Avg Total Mix Recon ($Z_{h}+Z_{d}$ Target)", fontsize=10)
    ax.grid(True, which="both", ls="-", alpha=0.2)

def plot_residual_magnitude_scatter_template(ax, h_baseline_avg, d_disease_avg, 
                                             title,
                                             show_zero_line=True, 
                                             abs_threshold_lines=None, 
                                             radiating_ratio_lines=None,
                                             highlight_top_n=10, gene_names=None):
    """
    Residual Magnitude vs Healthy Baseline (Linear-X, Linear-Y)
    
    abs_threshold_lines example: [500, 1000] -> draws Y=±500, Y=±1000
    radiating_ratio_lines example: [0.5, 1.0] -> draws Y = ±0.5*X, Y = ±1.0*X
    """
    
    # X scale is log (baseline expression), Y scale is linear (Zd can be negative)
    # ax.set_xscale('log')
    
    # Plot the gene cloud
    ax.scatter(h_baseline_avg, d_disease_avg, alpha=0.3, s=8, color='#7f8c8d', edgecolors='none')
    ax.axline((0, 0), slope=1, color='black', linestyle='--', label='y = x')
    # 1. OPTION: Zero baseline
    if show_zero_line:
        ax.axhline(0, color='black', linewidth=1.2, label='Zero Workload (Zd=0)')

    # # 2. OPTION: Absolute thresholds (Good for Elephants)
    # if abs_threshold_lines is not None:
    #     for thresh in abs_threshold_lines:
    #         ax.axhline(thresh, color='#d62728', linestyle=':', linewidth=0.8, alpha=0.6)
    #         # ax.axhline(-thresh, color='#1f77b4', linestyle=':', linewidth=0.8, alpha=0.6)

    # 3. OPTION: Radiating Ratio lines (Mandatory for Mice deconvolution check)
    # if radiating_ratio_lines is not None:
    #     # Generate sorting array for smooth lines on log x-axis
    #     sort_idx = np.argsort(h_baseline_avg)
    #     h_sorted = h_baseline_avg[sort_idx]
        
    #     for ratio in radiating_ratio_lines:
    #         # Positive compensation (Add cancer signal)
    #         ax.plot(h_sorted, h_sorted * ratio, color='#d62728', linestyle='--', linewidth=0.8, alpha=0.6)
    #         # Negative compensation (Subtract healthy signal)
    #         ax.plot(h_sorted, h_sorted * -ratio, color='#1f77b4', linestyle='--', linewidth=0.8, alpha=0.6)
            
    # Highlight Outliers 
    if highlight_top_n > 0 and gene_names is not None:
        epsilon = 1.0
        # Ratio of disease added workload compared to healthy baseline
        workload_ratios = np.abs(d_disease_avg) / (h_baseline_avg + epsilon)
        
        # Sort by relative workload
        top_indices = np.argsort(workload_ratios)[-highlight_top_n:]
        
        # Highlight points
        ax.scatter(h_baseline_avg[top_indices], d_disease_avg[top_indices], s=15, color='#e67e22', edgecolors='black', linewidth=0.5)
        
        # Add labels
        for idx in top_indices:
            # Need to shift label based on Zd sign
            y_shift = d_disease_avg[idx] + (np.sign(d_disease_avg[idx]) * (d_disease_avg.max() * 0.05))
            ax.text(h_baseline_avg[idx] * 1.1, y_shift, gene_names[idx], fontsize=7)

    # Aesthetics
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(r"Avg Healthy Baseline ($Z_{h\_healthy}$)", fontsize=10)
    ax.set_ylabel(r"Avg Disease Signal Recon ($Z_{d}$ Target)", fontsize=10)
    
    # Format X axis
    # ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    # ax.grid(True, which="both", ls="-", alpha=0.2)

def avg_disease_exp(recon_h_np, is_true_healthy, recon_mix_np, is_target_disease, recon_d_np):
    avg_h_baseline = recon_h_np[is_true_healthy].mean(axis=0)
        
        # Y-Axis TOTAL (Idea 1): Average Total (Mix) reconstruction from DISEASE patients
    avg_mix_disease = recon_mix_np[is_target_disease].mean(axis=0)
    
    # Y-Axis RESIDUAL (Idea 2): Average Disease signal from DISEASE patients
    avg_d_disease = recon_d_np[is_target_disease].mean(axis=0)
    
    # Clip to strictly positive + epsilon to prevent log(0) errors later
    epsilon = 1.0
    avg_h_baseline = np.clip(avg_h_baseline, epsilon, None)
    avg_mix_disease = np.clip(avg_mix_disease, epsilon, None)

    return avg_h_baseline, avg_mix_disease, avg_d_disease

def analyze_disease_drivers_grid(labels_dict, inference_cache, test_df_full, test_genes_df, 
                                 scale_bool, scaler, save_path, mode, top_n=10, is_mixed=False):
    """
    Evaluates Top Disease Drivers (Relative Compensation) in a grid layout.
    Loops through available disease types and creates a separate grid figure for each.
    """
    tag = "scaled" if scale_bool else "unscaled"
    gene_names = test_genes_df.columns.tolist()
    
    # Identify unique disease types in the dataset, excluding 0 (Healthy)
    unique_diseases = [d for d in test_df_full['disease_type'].unique() if d != 0]
    disease_map = {1: "Disease A (CRC)", 2: "Disease B (SCLC)"}
    
    # Outer Loop: Generate a separate figure for each disease type
    for disease_target in unique_diseases:
        disease_name = disease_map.get(disease_target, f"Disease {disease_target}")
        
        # Inner Loop: Generate grids per Base Architecture
        for base_name, models in labels_dict.items():
            n_rows = len(cfg.ENCODING_SIZES) 
            n_cols = len(models)
            
            fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
            fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
            
            fig1.suptitle(f"Total Expression Deconvolution: {disease_name}\n(Base: {base_name.upper()} | Mode: {mode})", fontsize=16, fontweight='bold', y=0.98)
            fig2.suptitle(f"Disease Branch Workload Diagnostics: {disease_name}\n(Base: {base_name.upper()} | Mode: {mode})", fontsize=16, fontweight='bold', y=0.98)

            for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
                for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                    ax1 = axes1[row_idx, col_idx]
                    ax2 = axes2[row_idx, col_idx]
                    try:
                        prepare_result = prepare_scatter_data(
                            inference_cache=inference_cache, base_name=base_name, enc=enc, 
                            model_label=model_label, test_df_full=test_df_full, 
                            disease_target=disease_target, scale_bool=scale_bool, scaler=scaler
                        )
                        if prepare_result is None:
                            ax1.text(0.5, 0.5, "Model / Output Not Found", ha='center', color='red')
                            ax2.text(0.5, 0.5, "Model / Output Not Found", ha='center', color='red')
                            continue
                        
                        recon_h_np, is_true_healthy, recon_mix_np, is_target_disease, recon_d_np = prepare_result
                        averaged_result =avg_disease_exp(recon_h_np, is_true_healthy, recon_mix_np, is_target_disease, recon_d_np)
                        h_avg, mix_avg, d_avg = averaged_result

                        #figure1 - absolute total of disease samples vs healthy samples
                        
                        # plot_absolute_expression_scatter_template(
                        #     ax=ax1, h_baseline_avg=h_avg, mix_disease_avg=mix_avg,
                        #     title=f"Total Expression Deconvolution (Model:{model_label} CRC)",
                        #     show_identity=True, 
                        #     fold_change_lines=[2.0, 5.0, 10.0], # CUSTOMIZE LINES HERE
                        #     highlight_top_n=top_n, gene_names=gene_names
                        #     )
                        plot_absolute_expression_scatter_with_significance(
                            ax=ax1, healthy_cohort_matrix=recon_h_np[is_true_healthy],
                            disease_cohort_matrix=recon_mix_np[is_target_disease],
                            title=f"Total Expression Deconvolution (Model:{model_label})",
                            gene_names=gene_names 
                        )
                        continue
                        plot_residual_magnitude_scatter_template(
                           ax=ax2, h_baseline_avg=h_avg, d_disease_avg=d_avg,
                           title=f"Disease Branch VS Healthy Expression | Mode: {mode}",
                           show_zero_line=True, abs_threshold_lines=[500, 1000],
                           radiating_ratio_lines=[0.5, 1.0, 2.0],
                           highlight_top_n=top_n, gene_names=gene_names
                       )
                        
                    except Exception as e:
                        traceback.print_exc()
                        fig1.text(0.5, 0.5, "Plotting Error", ha='center', color='red')
                        fig2.text(0.5, 0.5, "Plotting Error", ha='center', color='red')

            # 8. Save Figure
            fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Use your established dynamic pathing
            out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=is_mixed) / f"Tournament_H-{base_name}"
            os.makedirs(out_folder, exist_ok=True)
            
            # Format: save_path_scaled_Disease1.png
            filename1 = f"Absolute_Scatter_{tag}_Disease{disease_target}_{mode}.png"
            fig1.savefig(out_folder / filename1, dpi=150)
            plt.close(fig1)

            filename2 = f"Residual_Scatter_{tag}_Disease{disease_target}_{mode}.png"
            fig2.savefig(out_folder / filename2, dpi=150)
            plt.close(fig2)

            print(f"Saved: {filename1} and {filename2}")


def run_comprehensive_reconstruction_analysis(labels_dict, scale_bool, save_path, mode, is_mixed=False, is_simple=False):
    """
    The Master Pipeline. Loads data, runs all models once, and generates all plots.
    """
    print(f"\n🚀 Starting Evaluation Pipeline (Mode: {mode.upper()} | Simple: {is_simple})")
    
    # ==========================================
    # 1. LOAD AND PREP DATA (Happens exactly once)
    # ==========================================
    tag = "scaled" if scale_bool else "unscaled"

    _, true_disease = load_reconstruction_data('disease', mode) 
   
    train_t, test_t, scaler, info = du.load_and_prep_tensors(
    phase="disease", mode=mode, scale_bool=scale_bool, is_mixed=is_mixed
    )
    test_df_full = info['test_df_full'].fillna(value=0.0)      # Contains [Genes | Theta | Type]

    gene_size = test_t.shape[1] - 1
    # test_w_theta_t = torch.Tensor(test_t.values).float()
    
    print(f"✅ Data Loaded. Genes: {gene_size}, Test Samples: {test_t.shape[0]}")

    # ==========================================
    # 2. GENERATE INFERENCE CACHE 
    # ==========================================
    print("🧠 Running Model Inference Cache...")
    inference_cache = generate_inference_cache(labels_dict, test_t, gene_size, tag)
    metadata_cols = ['theta_value', 'disease_type']
    test_genes_df = test_df_full.drop(columns=metadata_cols, errors='ignore')
    
    actual_gene_size = test_genes_df.shape[1]
    test_no_theta_t = torch.Tensor(test_genes_df.values).float()
    if gene_size != gene_size:
        raise ValueError(f"why arent they the same size?: {gene_size} vs {actual_gene_size}")
    # ==========================================
    # 3. GENERATE VISUALIZATIONS 
    # ==========================================

    if is_mixed:
        print("🎨 Drawing Disease Drivers...")
        analyze_disease_drivers_grid(
            labels_dict=labels_dict,
            inference_cache=inference_cache,
            test_df_full=test_df_full,
            test_genes_df=test_genes_df,
            scale_bool=scale_bool,
            scaler=scaler,               # Passed from du.load_and_prep_tensors
            save_path=save_path,     # Prefix for the saved file
            mode=mode,
            top_n=10,  # Shows top 10 up and top 10 down per subplot
            is_mixed=is_mixed                    
        )
    return
    print("🎨 Drawing Total Mix Scatter Plots...")
    analyze_total_reconstruction(
        labels_dict=labels_dict, 
        inference_cache=inference_cache, 
        test_df_full=test_df_full, 
        test_n_theta=test_no_theta_t,
        gene_size=actual_gene_size, 
        scaler=scaler,
        scale_bool=scale_bool, 
        save_path=save_path+"_total", 
        mode=mode, 
        is_simple=is_simple, 
        is_mixed=is_mixed
    )
    

    print("🎨 Drawing Disease Branch Scatter Plots...")
    analyze_disease_portion_reconstruction_scatter(
        labels_dict=labels_dict, 
        inference_cache=inference_cache, 
        test_df_full=test_df_full, 
        true_disease_input=true_disease, 
        scaler=scaler,
        scale_bool=scale_bool, 
        gene_size=actual_gene_size,
        save_path=save_path+"_disease", 
        mode=mode, 
        is_simple=is_simple
        ,
        is_mixed=is_mixed
    )
    
    print("✅ All analyses and visualizations complete!\n")

def interpret_disease_mix(phase='disease', mode="true"):

    labels_dict = {
        'PCA':
        {   "pca": "mix_H-pca_D-pca",
            "ae_basic": "mix_H-pca_D-ae_basic",
            "ae_layered": "mix_H-pca_D-ae_layered"
            
        }
    }
    for scale in cfg.SCALING_OPTIONS:
        scaling = "scaled" if scale else "unscaled"
        print(f"####### RUNNING WITH {scaling.upper()} DATA")
        print("################# running with mix") 
        run_comprehensive_reconstruction_analysis(labels_dict=labels_dict, 
                                                scale_bool=scale, save_path="analyze_recon_mixed", 
                                                mode=mode, is_mixed=MIXED)
        return
        # print("############ running with no mix") 
        run_comprehensive_reconstruction_analysis(labels_dict=labels_dict, 
                                                scale_bool=scale, save_path=f"analyze_recon_dOnly", 
                                            mode=mode, is_mixed=NOT_MIXED)



    print("end of run")
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


