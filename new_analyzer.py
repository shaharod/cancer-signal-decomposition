## currently has graph of MSE 

import utils.analysis_utils as au
import utils.data_utils as du
import utils.model_utils as mu
import config as cfg
import os
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

SCALED = True
UNSCALED = False
variants = {
        'Theta Limit (0.7)': 'theta_lim_0.7',
        'Diff DP': 'dif_dp',
        'Diff HP': 'dif_hp'
    }
variants_t= {
        # 'Theta Limit (0.7)': 'theta_lim_0.7',
        'Theta Noise 0.001': 'theta_0.001',
        'Theta Noise 0.005': 'theta_0.005',
        'Theta Noise 0.01': 'theta_0.01',
        'Theta Noise 0.1': 'theta_0.1',
    }

variants_lim = {
        'No Theta Limit' : 'no_theta_lim',
        'Theta Limit (0.7)': 'theta_lim_0.7'
    }
def collect_phase_data(phase, model_labels, is_mixed):
    """
    getting trained models history and data from models
    
    :param phase: 'heathy', 'disease' or whatever option
    :param model_labels: possible models to retreive data from
    """

    # data dictionaries to collect
    data_s = {}
    data_u = {}

    for label, model_tag in model_labels.items():
        data_s[label] = au.load_data_for_analysis(SCALED, model_tag, phase, is_mixed)
        data_u[label] = au.load_data_for_analysis(UNSCALED, model_tag, phase, is_mixed)

    return data_s, data_u

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

def calculate_disease_branch_mse(labels_dict, inference_cache, test_df_full, true_disease_input, scaler, scale_bool):
    """
    Extracts the disease branch predictions, isolates true disease samples, 
    and calculates MSE against the pure disease target for a single variant.
    """
    variant_results = {}
    variant_results_bins = {}
    # Isolate ONLY the sample IDs that are true disease samples
    disease_mask = test_df_full.index.isin(true_disease_input.index)
    test_df_disease = test_df_full[disease_mask]

    # 2. Get the valid indices to align the truth matrix
    # valid_disease_indices = test_df_full.index[disease_mask]
    benchmark_truth = true_disease_input.reindex(test_df_disease.index).values
    thetas = test_df_disease['theta_value'].values
    
    theta_bins = {
        'Low (<0.33)': (thetas >= 0.0) & (thetas < 0.33),
        'Med (0.33-0.66)': (thetas >= 0.33) & (thetas < 0.66),
        'High (>0.66)': (thetas >= 0.66) & (thetas <= 1.0)
    }
    bin_counts = {bin_name: np.sum(mask) for bin_name, mask in theta_bins.items()}
    for base_name, models in labels_dict.items():
        variant_results[base_name] = {}
        variant_results_bins[base_name] = {}
        
        for enc in cfg.ENCODING_SIZES:
            for model_label, folder_tag in models.items():
                if model_label not in variant_results[base_name]:
                    variant_results[base_name][model_label] = {}
                    variant_results_bins[base_name][model_label] = {bin_name: {} for bin_name in theta_bins.keys()}
                    
                try:
                    model_outputs = inference_cache[base_name][enc].get(model_label)
                    if model_outputs is None or model_outputs['disease'] is None:
                        continue
                        
                    recon_d = model_outputs['disease']
                    
                    # Convert to numpy and inverse scale if necessary
                    if scale_bool and scaler is not None:
                        print("calling inverse scale")
                        recon_final = du.inverse_scale(scaler, recon_d).detach().cpu().numpy()
                    else:
                        recon_final = recon_d.detach().cpu().numpy()
                        
                    # Slice out ONLY the true disease samples
                    recon_disease_only = recon_final[disease_mask]
                    
                    for bin_name, bin_mask in theta_bins.items():
                        # Skip if a bin has no samples (can happen in 'fixed 0.5' mode)
                        if not np.any(bin_mask):
                            continue
                            
                        truth_binned = benchmark_truth[bin_mask]
                        recon_binned = recon_disease_only[bin_mask]
                        
                        bin_mse = np.mean((truth_binned - recon_binned) ** 2)
                        variant_results_bins[base_name][model_label][bin_name][enc] = bin_mse
                        
                    # Calculate Global MSE against the pure disease truth
                    global_mse = np.mean((benchmark_truth - recon_disease_only) ** 2)
                    variant_results[base_name][model_label][enc] = global_mse
                    
                except Exception as e:
                    print(f"Error calculating Disease MSE for {base_name}-{model_label}-{enc}: {e}")
                    continue
                    
    return variant_results, variant_results_bins, bin_counts

def plot_disease_variant_multi_model_grid(master_variant_data, enc_sizes, save_dir, baseline_name):
    """
    Plots Disease Branch Test MSE vs Data Variant.
    Rows = Encoding Sizes
    Columns = Scaled / Unscaled
    Lines = Different Models (Basic AE, Layered AE, PCA)
    X-axis = Variants
    """
    variants = list(master_variant_data.keys())
    n_rows = len(enc_sizes)
    
    # Enforce exactly two columns: Scaled and Unscaled
    pipelines = ['scaled', 'unscaled']
    n_cols = len(pipelines)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle(f"Disease Branch Reconstruction Across Data Variants\n(Baseline: {baseline_name})", 
                 fontsize=18, fontweight='bold', y=1.02)
    
    # Line styles for your models
    style_map = {
        'basic': {'color': '#1f77b4', 'marker': 'o'},
        'layered': {'color': '#2ca02c', 'marker': 's'},
        'pca': {'color': '#EC7063', 'marker': '^'}
    }
    fallback_colors = ['#9467bd', '#8c564b', '#e377c2']

    for row_idx, enc in enumerate(enc_sizes):
        for col_idx, base_name in enumerate(pipelines):
            ax = axes[row_idx, col_idx]
            
            # Find all models that exist for this scaled/unscaled pipeline across all variants
            all_models = set()
            for var_dict in master_variant_data.values():
                if base_name in var_dict:
                    all_models.update(var_dict[base_name].keys())
            
            # If a pipeline has absolutely no data, label it and skip gracefully
            if not all_models:
                if row_idx == 0:
                    ax.set_title(f"Pipeline: {base_name.upper()} (No Data)", fontsize=14, pad=15)
                ax.set_xticks(range(len(variants)))
                ax.set_xticklabels(variants, rotation=15)
                continue

            color_idx = 0
            for model_label in all_models:
                y_values = []
                valid_x = []
                
                # Fetch the exact MSE for this variant, pipeline, model, and encoding size
                for v_idx, var_name in enumerate(variants):
                    try:
                        mse_val = master_variant_data[var_name][base_name][model_label][enc]
                        y_values.append(mse_val)
                        valid_x.append(v_idx)
                    except KeyError:
                        continue 
                
                if y_values:
                    match_key = next((k for k in style_map.keys() if k.lower() in model_label.lower()), None)
                    if match_key:
                        style = style_map[match_key]
                    else:
                        style = {'color': fallback_colors[color_idx % len(fallback_colors)], 'marker': 'd'}
                        color_idx += 1

                    # Draw the line
                    ax.plot(valid_x, y_values, label=model_label, 
                            color=style['color'], marker=style['marker'], 
                            linewidth=2.5, markersize=8)
                    
                    # Annotate the exact MSE values (staggered vertically)
                    for i, (x_val, y_val) in enumerate(zip(valid_x, y_values)):
                        y_offset = 10 + ((i % 2) * 12) 
                        ax.annotate(f'{y_val:.4g}', (x_val, y_val), textcoords="offset points", 
                                    xytext=(0, y_offset), ha='center', fontsize=9, 
                                    fontweight='bold', color=style['color'])

            # --- Formatting ---
            if row_idx == 0:
                ax.set_title(f"Pipeline: {base_name.upper()}", fontsize=14, pad=15)
                
            if col_idx == 0:
                ax.set_ylabel(f"Disease MSE\n(Enc: {enc})", fontsize=13, fontweight='bold')
            else:
                ax.set_ylabel("Disease MSE", fontsize=12)
            
            ax.set_xticks(range(len(variants)))
            if row_idx == n_rows - 1:
                ax.set_xticklabels(variants, rotation=15, fontsize=11)
            else:
                ax.set_xticklabels([])
                
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='best', fontsize=10)
            
            # Headroom for annotations
            curr_ylim = ax.get_ylim()
            y_range = curr_ylim[1] - curr_ylim[0] if curr_ylim[1] != curr_ylim[0] else 1
            ax.set_ylim(curr_ylim[0] - (y_range * 0.05), curr_ylim[1] + (y_range * 0.35))

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    file_path = save_dir / f"disease_branch_variant_comparison.png"
    plt.savefig(file_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✅ Saved Disease Variant Grid Plot to: {file_path}")


def plot_variant_mse_lines(master_data, enc_sizes, model_key, save_dir):
    """
    Plots Test MSE vs Data Variant using lines representing different Encoding Sizes.
    Matches the styling of plot_test_mse_comparison_lines.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    variants = list(master_data.keys())
    x_indices = np.arange(len(variants))
    
    pipelines = [
        (ax1, 'scaled', "Pipeline: Trained on Scaled Data"),
        (ax2, 'unscaled', "Pipeline: Trained on Raw Data")
    ]
    
    # Styles for different encoding sizes (since X-axis is now Variants)
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#2ca02c', '#EC7063', '#9467bd', '#8c564b']
    
    for ax, p_key, col_title in pipelines:
        for idx, enc in enumerate(enc_sizes):
            y_values = []
            valid_x = []
            
            for v_idx, var_name in enumerate(variants):
                data_dict = master_data[var_name][p_key]
                if data_dict and model_key in data_dict:
                    mse_dict = data_dict[model_key][au.TEST_MSE_IDX]
                    if enc in mse_dict:
                        val = mse_dict[enc]
                        try:
                            clean_val = float(np.array(val).flatten()[0])
                            y_values.append(clean_val)
                            valid_x.append(v_idx)
                        except (IndexError, TypeError):
                            continue
            
            if y_values:
                # Plot the line for this encoding size
                ax.plot(valid_x, y_values, label=f"Enc Size: {enc}", 
                        color=colors[idx % len(colors)], marker=markers[idx % len(markers)], 
                        linewidth=2, markersize=8)
                
                # Add Data Labels
                for x_val, y_val in zip(valid_x, y_values):
                    ax.annotate(f'{y_val:.4g}', (x_val, y_val), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

        # Formatting
        ax.set_title(col_title, fontsize=14, pad=15)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(variants, rotation=15, fontsize=11)
        ax.set_ylabel("Test MSE (Original Units)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

        # Add headroom
        curr_ylim = ax.get_ylim()
        ax.set_ylim(curr_ylim[0], curr_ylim[1] * 1.25)

    fig.suptitle(f"Data Variability Impact on Test MSE ({model_key})", fontsize=18, y=1.05)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir / f"variant_mse_lines_{model_key}.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✅ Saved MSE Line Plot to: {save_dir / f'variant_mse_lines_{model_key}.png'}")


def plot_variant_curves(master_data, target_enc, model_key, save_dir, phase):
    """
    Plots Train vs Eval curves for all variants on a specific model and encoding size.
    Rows = Variants, Columns = Scaled/Unscaled.
    """
    variants = list(master_data.keys())
    n_vars = len(variants)
    
    fig, axes = plt.subplots(n_vars, 2, figsize=(14, 4 * n_vars), squeeze=False)
    fig.suptitle(f"Training History Across Data Variants\n({model_key} | Phase: {phase} | Enc: {target_enc})", fontsize=16, y=1.02)
    
    for i, var_name in enumerate(variants):
        for j, (p_key, col_title) in enumerate([('scaled', "Scaled"), ('unscaled', "Unscaled")]):
            ax = axes[i, j]
            data_dict = master_data[var_name][p_key]
            
            if not data_dict or model_key not in data_dict:
                if not data_dict:
                    word = "data_dict"
                if model_key not in data_dict:
                    word = " hold model key"
                print(f"no {word} for: Training History Across Data Variants\n({model_key} | Enc: {target_enc})")
                ax.set_title(f"Variant: {var_name} ({col_title}) - NO DATA")
                continue
                
            train_curve = data_dict[model_key][au.TRAIN_LOSS_IDX].get(target_enc, [])
            eval_curve = data_dict[model_key][au.EVAL_LOSS_IDX].get(target_enc, [])
            
            if not train_curve:
                ax.set_title(f"Variant: {var_name} ({col_title}) - NO CURVE")
                continue
            
            # Plot Train
            epochs = np.arange(1, len(train_curve) + 1)
            ax.plot(epochs, train_curve, label="Train", color='#1f77b4', linestyle='-', lw=1.5)
            
            # Plot Eval (with epoch jump logic from your code)
            if len(eval_curve) > 0:
                epoch_jump = len(train_curve) // len(eval_curve)
                eval_epochs = np.arange(epoch_jump, len(train_curve) + 1, epoch_jump)[:len(eval_curve)]
                ax.plot(eval_epochs, eval_curve, label="Eval", color='#ff7f0e', linestyle='--', lw=1.2)
                
            # Formatting
            ax.set_title(f"Variant: {var_name} | {col_title}")
            ax.set_ylabel("MSE")
            ax.grid(True, which='both', linestyle=':', alpha=0.5)
            ax.legend(loc='upper right', fontsize='x-small')
            
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"variant_{phase}_curves_{model_key}_enc{target_enc}.png"
    plt.savefig(save_dir / file_name, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✅ Saved Curve Plot to: {save_dir / file_name}")

def plot_variant_multi_model_grid(master_data, enc_sizes, model_labels, save_dir, baseline_name, phase):
    """
    Plots Test MSE vs Data Variant.
    Rows = Encoding Sizes, Columns = Scaled / Unscaled.
    Lines = Different Models (e.g., Basic AE vs Layered AE).
    """
    variants = list(master_data.keys())
    n_rows = len(enc_sizes)
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows), squeeze=False)
    if phase == 'disease':
        fig.suptitle(f"Model Comparison Across Data Variants\n(Healthy Model Baseline: {baseline_name})", fontsize=18, fontweight='bold', y=1.02)
    elif phase == 'healthy':
        fig.suptitle(f"Model Comparison Across Data Variants\n(Healthy Model Evaluation)", fontsize=18, fontweight='bold', y=1.02)
    pipelines = [
        (0, 'scaled', "Pipeline: Scaled Data"),
        (1, 'unscaled', "Pipeline: Raw Data")
    ]
    
    # Colors/Markers mapped to MODELS now
    colors = ['#1f77b4', '#2ca02c', '#EC7063', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v']

    for row_idx, enc in enumerate(enc_sizes):
        for col_idx, p_key, col_title in pipelines:
            ax = axes[row_idx, col_idx]
            
            # Draw a line for EACH model
            for m_idx, (model_display_name, model_tag) in enumerate(model_labels.items()):
                y_values = []
                valid_x = []
                
                for v_idx, var_name in enumerate(variants):
                    data_dict = master_data[var_name][p_key]
                    # collect_phase_data uses model_display_name as the dictionary key
                    if data_dict and model_display_name in data_dict:
                        mse_dict = data_dict[model_display_name][au.TEST_MSE_IDX]
                        if enc in mse_dict:
                            val = mse_dict[enc]
                            try:
                                clean_val = float(np.array(val).flatten()[0])
                                y_values.append(clean_val)
                                valid_x.append(v_idx)
                            except (IndexError, TypeError):
                                continue
                
                if y_values:
                    # Plot Model Line
                    model_color = colors[m_idx % len(colors)]
                    ax.plot(valid_x, y_values, label=model_display_name, 
                            color=model_color, marker=markers[m_idx % len(markers)], 
                            linewidth=2.5, markersize=8)
                    
                    # Add exact values, slightly staggered vertically to prevent overlap
                    for x_val, y_val in zip(valid_x, y_values):
                        y_offset = 10 + (m_idx * 12) # Stagger text for overlapping points
                        ax.annotate(f'{y_val:.4g}', (x_val, y_val), textcoords="offset points", 
                                    xytext=(0, y_offset), ha='center', fontsize=9, 
                                    fontweight='bold', color=model_color)

            # Formatting
            if row_idx == 0:
                ax.set_title(col_title, fontsize=14, pad=15)
            if col_idx == 0:
                ax.set_ylabel(f"Test MSE\n(Enc: {enc})", fontsize=13, fontweight='bold')
            
            ax.set_xticks(range(len(variants)))
            if row_idx == n_rows - 1:
                ax.set_xticklabels(variants, rotation=15, fontsize=11)
            else:
                ax.set_xticklabels([])
                
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Show legend with model names
            if len(model_labels) > 0:
                ax.legend(loc='best', fontsize=10)
            
            # Headroom for the staggered annotations
            curr_ylim = ax.get_ylim()
            y_range = curr_ylim[1] - curr_ylim[0] if curr_ylim[1] != curr_ylim[0] else 1
            ax.set_ylim(curr_ylim[0] - (y_range * 0.05), curr_ylim[1] + (y_range * 0.40))

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    file_path = save_dir / f"variant_phase_{phase}_multi_model_grid.png"
    plt.savefig(file_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✅ Saved Multi-Model Grid Plot to: {file_path}")

import copy

def aggregate_and_plot_variants(baseline='PCA', target_models=['basic','layered'], phase='disease', 
                                is_mixed=True, variants={
                                                        # 'Simple': '',
                                                        'Theta Limit (0.7)': 'theta_lim_0.7',
                                                        'Diff DP': 'dif_dp',
                                                        'Diff HP': 'dif_hp'
                                                    }, variant_char=''):
    """
    Loops through variant folders, extracts all history/metrics, and passes 
    the master dictionary to the line and curve plotting functions.
    """
    
    # model_tag = f"mix_H-{baseline.lower()}_D-ae_{target_model}"
    if phase == 'disease':
        model_labels = {
            f"Basic AE": f"mix_H-{baseline.lower()}_D-ae_basic",
            f"Layered AE": f"mix_H-{baseline.lower()}_D-ae_layered",
            f"PCA": f"mix_H-{baseline.lower()}_D-pca"
        }
    elif phase == 'healthy':
        model_labels = {
            f"Basic AE": "ae_basic",
            f"Layered AE": "ae_layered",
            f"PCA":"pca"
        }
    enc_sizes = cfg.ENCODING_SIZES

    # Backup original config paths
    original_base = cfg.BASE_EXP_DIR
    original_healthy = cfg.HEALTHY_OUT_DIR
    original_disease = cfg.DISEASE_OUT_DIR

    # This will hold ALL data for ALL variants
    master_data = {}

    for variant_name, suffix in variants.items():
        # Override Config Paths for this variant
        cfg.BASE_EXP_DIR = cfg.PROJECT_ROOT / 'outputs' / f"synthetic_experiments_{suffix}"
        cfg.HEALTHY_OUT_DIR = cfg.BASE_EXP_DIR / 'healthy'
        cfg.DISEASE_OUT_DIR = cfg.BASE_EXP_DIR / 'disease_mix'

        print(f"\n--- Loading Variant Data: {variant_name} ---")
        try:
            data_s, data_u = collect_phase_data(phase, model_labels, is_mixed)
            # Store everything inside the master dict
            master_data[variant_name] = {'scaled': data_s, 'unscaled': data_u}
        except Exception as e:
            print(f"Could not load pre-saved metrics for {variant_name}: {e}")
            master_data[variant_name] = {'scaled': {}, 'unscaled': {}}

    # Restore Config Paths immediately after loop
    cfg.BASE_EXP_DIR = original_base
    cfg.HEALTHY_OUT_DIR = original_healthy
    cfg.DISEASE_OUT_DIR = original_disease

    # --- Generate the Plots ---
    save_dir = cfg.PROJECT_ROOT / 'outputs' / 'plots' / f'cross_variant_comparisons{variant_char}'
    
    # 1. The Line Plot (MSE vs Variants)
    plot_variant_multi_model_grid(
        master_data=master_data, 
        enc_sizes=enc_sizes, 
        model_labels=model_labels, 
        save_dir=save_dir,
        baseline_name=baseline,
        phase=phase
    )
    
    # 2. The Curve Plots (Train/Eval History)
    # Usually you don't want 40 curve plots, so we pick the top 1 or 2 encoding sizes to visualize.
    # We will just plot the highest and lowest encoding size from your list as an example:
    target_encodings_to_plot = [enc_sizes[0], enc_sizes[-1]] 
    target_model_dict = {
        "pca": "PCA",
        "basic": 'Basic AE',
        "layered": "Layered AE"
    }
    print("######### TARGET MODEL IS #######")
    for target_model in target_models:
        print(target_model)
        for enc in target_encodings_to_plot:
            plot_variant_curves(master_data, target_enc=enc, model_key=f"{target_model_dict[target_model]}", save_dir=save_dir, phase=phase)

def plot_per_var_d_mse_bins(results, bin_counts, is_mixed, save_path):
    n_subplots = len(results)
    fig, axes = plt.subplots(1, n_subplots, figsize=(9 * n_subplots, 7), sharey=False)
    if n_subplots == 1: axes = [axes]
    
    # Base colors for models
    color_map = {'basic': '#1f77b4', 'layered': '#2ca02c', 'pca': '#EC7063'}
    # Line styles for theta bins
    style_map = {'Low (<0.33)': 'dotted', 'Med (0.33-0.66)': 'dashed', 'High (>0.66)': 'solid'}
    
    for ax, (base_name, models_dict) in zip(axes, results.items()):
        
        for model_label, bin_dict in models_dict.items():
            # Get base color for the model
            m_color = next((v for k, v in color_map.items() if k in model_label.lower()), 'gray')
            
            for bin_name, enc_dict in bin_dict.items():
                if not enc_dict: continue
                
                valid_encodings = sorted(list(enc_dict.keys()))
                y_values = [enc_dict[enc] for enc in valid_encodings]
                
                l_style = style_map.get(bin_name, 'solid')
                n_samples = bin_counts[bin_name]
                label_text = f"{model_label} [{bin_name} | N={n_samples}]"
                # Plot the binned line
                ax.plot(valid_encodings, y_values, 
                        label=label_text, 
                        color=m_color, 
                        linestyle=l_style,
                        marker='o', 
                        linewidth=2, markersize=6)

                # Add Data Labels (optional, might get crowded with 9 lines, you can comment this loop out if it's too messy)
                for x_val, y_val in zip(valid_encodings, y_values):
                    ax.annotate(f'{y_val:.0f}', (x_val, y_val), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=8)

        # Formatting Subplot
        ax.set_title(f"Pipeline: {base_name.upper()}", fontsize=14, pad=15)
        ax.set_xlabel("Encoding Size (Latent Dimension)", fontsize=12)
        ax.set_ylabel("Test MSE (Original Units)", fontsize=12)
        ax.set_xticks(cfg.ENCODING_SIZES)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Move legend outside to prevent overlapping lines
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        curr_ylim = ax.get_ylim()
        ax.set_ylim(curr_ylim[0], curr_ylim[1] * 1.15)

    fig.suptitle(f"Disease MSE by Theta Bins vs Encoding Size", fontsize=18, y=1.05)
    plt.tight_layout()
    # plt.show()
    out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=is_mixed) / "MSE_Lines"
    os.makedirs(out_folder, exist_ok=True)
    output_path = out_folder / f"{save_path}_disease_mse_by_theta.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved Theta Binned Test MSE Line Plot to {output_path}")

def plot_per_var_d_mse(results, is_mixed, save_path, mode):
    n_subplots = len(results)
    fig, axes = plt.subplots(1, n_subplots, figsize=(7.5 * n_subplots, 6), sharey=False)
    
    # Ensure axes is iterable even if there's only 1 subplot
    if n_subplots == 1: axes = [axes]
    
    # Style map to keep lines consistent across subplots
    style_map = {
        'basic': {'color': '#1f77b4', 'marker': 'o'},
        'layered': {'color': '#2ca02c', 'marker': 's'},
        'pca-based': {'color': '#EC7063', 'marker': '^'},
        'PCA': {'color': '#EC7063', 'marker': '^'}
    }
    # Fallback colors if model_label isn't in style_map
    fallback_colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for ax, (base_name, models_dict) in zip(axes, results.items()):
        color_idx = 0
        
        for model_label, enc_dict in models_dict.items():
            # Skip if no valid data was calculated for this model
            if not enc_dict: continue
            
            # Sort encodings just in case
            valid_encodings = sorted(list(enc_dict.keys()))
            y_values = [enc_dict[enc] for enc in valid_encodings]
            
            # Get style
            match_key = next((k for k in style_map.keys() if k.lower() in model_label.lower()), None)
            if match_key:
                style = style_map[match_key]
            else:
                style = {'color': fallback_colors[color_idx % len(fallback_colors)], 'marker': 'd'}
                color_idx += 1
            
            # Plot the line
            ax.plot(valid_encodings, y_values, 
                    label=model_label, 
                    color=style['color'], 
                    marker=style['marker'], 
                    linewidth=2, markersize=8)

            # Add Data Labels
            for x_val, y_val in zip(valid_encodings, y_values):
                ax.annotate(f'{y_val:.4g}', (x_val, y_val), textcoords="offset points", 
                            xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

        # Formatting Subplot
        ax.set_title(f"Pipeline: {base_name.upper()}", fontsize=14, pad=15)
        ax.set_xlabel("Encoding Size (Latent Dimension)", fontsize=12)
        ax.set_ylabel("Test MSE (Original Units)", fontsize=12)
        ax.set_xticks(cfg.ENCODING_SIZES)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

        # Add headroom so labels don't get cut off at the top
        curr_ylim = ax.get_ylim()
        ax.set_ylim(curr_ylim[0], curr_ylim[1] * 1.15)

    # Master Figure Formatting
    fig.suptitle(f"Disease Branch Reconstruction: Test MSE vs Encoding Size (Theta: {mode})", fontsize=18, y=1.05)
    plt.tight_layout()
    
    # Saving
    out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=is_mixed) / "MSE_Lines"
    os.makedirs(out_folder, exist_ok=True)
    
    output_path = out_folder / f"{save_path}_disease_mse_lines.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved Test MSE Line Plot to {output_path}")

def aggregate_and_plot_disease_branch_variants(variants_dict, labels_dict, baseline='PCA', 
                                               mode='true', is_mixed=True, variant_char=''):
    """
    Loops through variant folders, dynamically iterates through both scaled/unscaled pipelines, 
    loads the specific data, generates the inference cache, calculates the isolated 
    disease-branch MSE, and plots the cross-variant grid.
    """
    original_base = cfg.BASE_EXP_DIR
    master_variant_data = {}
    
    for variant_name, suffix in variants_dict.items():
        # 1. Override Config Paths for this variant
        cfg.BASE_EXP_DIR = cfg.PROJECT_ROOT / 'outputs' / f"synthetic_experiments_{suffix}"
        cfg.change_data_dir(suffix)
        cfg.DATA_SUB = cfg.DATA_PATH/ "synthetic_complex" /f"{suffix}"
        print(f"\n--- Processing Disease Branch MSE for Variant: {variant_name} ---")
        
        # Initialize the dictionary for this variant
        master_variant_data[variant_name] = {}
        
        try:
            # 2. Iterate over the pipelines defined in your labels_dict (e.g., 'scaled', 'unscaled')
            for pipeline_name, pipeline_models in labels_dict.items():
                
                # Dynamically set the boolean and tag based on the dictionary key
                scale_bool = (pipeline_name.lower() == 'scaled')
                scale = 'scaled' if scale_bool else 'unscaled'
                tag = pipeline_name.lower()
                
                print(f"  -> Running Pipeline: {pipeline_name.upper()} (scale_bool={scale_bool})")
                
                # Load Core Data specific to THIS pipeline's scaling needs
                _, true_disease = du.load_reconstruction_data('disease', mode) 
                
                train_t, test_t, scaler, info = du.load_and_prep_tensors(
                    phase="disease", mode=mode, scale_bool=scale_bool, is_mixed=is_mixed
                )
                
                test_df_full = info['test_df_full'].fillna(value=0.0)      
                gene_size = test_t.shape[1] - 1
                
                # Isolate the dictionary for JUST this pipeline so the cache generator 
                # doesn't try to run unscaled models on scaled data, etc.
                single_pipeline_dict = {pipeline_name: pipeline_models}

                # 3. Generate PyTorch Cache 
                cache = generate_inference_cache(single_pipeline_dict, test_t, gene_size, tag)
                print(f"########3 CALCLUATING DISEASE MSE FOR {'UNSCALED' if not scale_bool else 'SCALED'}")
                print(f"pipeline is for {single_pipeline_dict.keys()}")
                # 4. Calculate Isolated Disease MSE
                var_mse_dict, var_mse_dict_bins, bin_counts = calculate_disease_branch_mse(
                    labels_dict=single_pipeline_dict, 
                    inference_cache=cache, 
                    test_df_full=test_df_full, 
                    true_disease_input=true_disease, 
                    scaler=scaler, 
                    scale_bool=scale_bool
                )
                
                # Merge this pipeline's results into the master variant dictionary
                master_variant_data[variant_name].update(var_mse_dict)
                
                plot_per_var_d_mse(
                    results=var_mse_dict,
                    is_mixed=is_mixed,
                    save_path=f"analyze_recon_mixed_{scale}",
                    mode=mode
                )
                
                plot_per_var_d_mse_bins(
                    results=var_mse_dict_bins,
                    bin_counts=bin_counts,
                    is_mixed=is_mixed,
                    save_path=f"analyze_recon_mixed_{scale}"
                )
        except Exception as e:
            print(f"  Failed to process {variant_name}: {e}")
            import traceback
            traceback.print_exc()
            
    # Restore config globally after loop finishes
    cfg.BASE_EXP_DIR = original_base
    
    # 5. Send master dictionary to Plotter
    print("\n🎨 Drawing cross-variant disease branch MSE grid...")
    save_dir = cfg.PROJECT_ROOT / 'outputs' / 'plots' / f'cross_variant_comparisons{variant_char}'
    
    plot_disease_variant_multi_model_grid(
        master_variant_data=master_variant_data, 
        enc_sizes=cfg.ENCODING_SIZES, 
        save_dir=save_dir, 
        baseline_name=baseline
    )



def run_aggregation_disease_mse():
    
    
    baseline = 'PCA'
    
    # Define BOTH pipelines here. The function will loop over these keys
    # and adjust the scaling logic automatically.
    labels_dict = {
        'scaled': {
            f"Basic AE": f"mix_H-{baseline.lower()}_D-ae_basic",
            f"Layered AE": f"mix_H-{baseline.lower()}_D-ae_layered",
            f"PCA": f"mix_H-{baseline.lower()}_D-pca"
        },
        'unscaled': {
            f"Basic AE": f"mix_H-{baseline.lower()}_D-ae_basic",
            f"Layered AE": f"mix_H-{baseline.lower()}_D-ae_layered",
            f"PCA": f"mix_H-{baseline.lower()}_D-pca"
        }
    }

    aggregate_and_plot_disease_branch_variants(
        variants_dict=variants_lim,
        labels_dict=labels_dict,
        baseline='PCA',
        mode='true',           # Pass your mode
        is_mixed=True,
        variant_char='_lim'
    )
    print("##############3 IM HERE ##############")
    aggregate_and_plot_disease_branch_variants(
        variants_dict=variants,
        labels_dict=labels_dict,
        baseline='PCA',
        mode='true',           # Pass your mode
        is_mixed=True
    )
    aggregate_and_plot_disease_branch_variants(
        variants_dict=variants_t,
        labels_dict=labels_dict,
        baseline='PCA',
        mode='true',           # Pass your mode
        is_mixed=True,
        variant_char='_t'
    )



if __name__ == '__main__':
    print("Starting Analysis Pipeline...")

    run_aggregation_disease_mse()
    raise
    # variants = {
    #     # 'Simple': '',
    #     'Theta Limit (0.7)': 'theta_lim_0.7',
    #     'Diff DP': 'dif_dp',
    #     'Diff HP': 'dif_hp'
    # }

    # variants_t = {
    #     # 'Simple': '',
    #     # 'Theta Limit (0.7)': 'theta_lim_0.7',
    #     'Theta Noise 0.001': 'theta_0.001',
    #     'Theta Noise 0.005': 'theta_0.005',
    #     'Theta Noise 0.01': 'theta_0.01',
    #     'Theta Noise 0.1' : 'theta_0.1'
    # }

    # variants_lim = {
    #     'No Theta Limit' : 'no_theta_lim',
    #     'Theta Limit (0.7)': 'theta_lim_0.7'
    # }
    
    # ---------------------------------------------------------
    # THE NEW CROSS-VARIANT COMPARISON PIPELINE
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(">>> STARTING CROSS-VARIANT MSE COMPARISON")
    print("="*50)

    for phase in ['healthy', 'disease']: 
        #comparing limit of theta vs no limit 
        aggregate_and_plot_variants(
            baseline='PCA', 
            target_models=['basic', 'layered'], 
            phase=phase, 
            is_mixed=True,
            variants=variants_lim,
            variant_char='_lim'
        )
        
        #comapring profile variants  
        aggregate_and_plot_variants(
            baseline='PCA', 
            target_models=['basic', 'layered'], 
            phase=phase, 
            is_mixed=True,
            variants=variants,
            variant_char=''
        )
        
        #comapring theta variants
        aggregate_and_plot_variants(
            baseline='PCA', 
            target_models=['basic', 'layered'], 
            phase=phase, 
            is_mixed=True,
            variants=variants_t,
            variant_char='_t'
        )


    
    # aggregate_and_plot_t_variants(
    #     baseline='PCA', 
    #     target_models=['basic','layered'],
    #     phase='disease', 
    #     is_mixed=True
    # )

    # Example 2: Compare variants for PCA + Basic AE model
    # aggregate_and_plot_t_variants(
    #     baseline='PCA', 
    #     target_model=['basic', 'layered'], 
    #     phase='disease', 
    #     is_mixed=True
    # )
    
    print("\n✅ All variant comparisons complete! Check the outputs/plots/cross_variant_comparisons folder.")