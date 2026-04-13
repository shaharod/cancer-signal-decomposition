## currently has graph of MSE 

import utils.analysis_utils as au
import config as cfg
import os
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

SCALED = True
UNSCALED = False

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


def plot_variant_curves(master_data, target_enc, model_key, save_dir):
    """
    Plots Train vs Eval curves for all variants on a specific model and encoding size.
    Rows = Variants, Columns = Scaled/Unscaled.
    """
    variants = list(master_data.keys())
    n_vars = len(variants)
    
    fig, axes = plt.subplots(n_vars, 2, figsize=(14, 4 * n_vars), squeeze=False)
    fig.suptitle(f"Training History Across Data Variants\n({model_key} | Enc: {target_enc})", fontsize=16, y=1.02)
    
    for i, var_name in enumerate(variants):
        for j, (p_key, col_title) in enumerate([('scaled', "Scaled"), ('unscaled', "Unscaled")]):
            ax = axes[i, j]
            data_dict = master_data[var_name][p_key]
            
            if not data_dict or model_key not in data_dict:
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
    file_name = f"variant_curves_{model_key}_enc{target_enc}.png"
    plt.savefig(save_dir / file_name, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✅ Saved Curve Plot to: {save_dir / file_name}")

def plot_variant_multi_model_grid(master_data, enc_sizes, model_labels, save_dir, baseline_name):
    """
    Plots Test MSE vs Data Variant.
    Rows = Encoding Sizes, Columns = Scaled / Unscaled.
    Lines = Different Models (e.g., Basic AE vs Layered AE).
    """
    variants = list(master_data.keys())
    n_rows = len(enc_sizes)
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows), squeeze=False)
    fig.suptitle(f"Model Comparison Across Data Variants\n(Healthy Baseline: {baseline_name})", fontsize=18, fontweight='bold', y=1.02)
    
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
    file_path = save_dir / f"variant_multi_model_grid_H-{baseline_name}.png"
    plt.savefig(file_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✅ Saved Multi-Model Grid Plot to: {file_path}")

import copy

def aggregate_and_plot_variants(baseline='PCA', target_model='layered', phase='disease', is_mixed=True):
    """
    Loops through variant folders, extracts all history/metrics, and passes 
    the master dictionary to the line and curve plotting functions.
    """
    variants = {
        # 'Simple': '',
        '0.1t': '_0.1t',
        'Diff DP': '_dif_dp',
        'Diff HP': '_dif_hp'
    }
    
    # model_tag = f"mix_H-{baseline.lower()}_D-ae_{target_model}"
    model_labels = {
        f"Basic AE": f"mix_H-{baseline.lower()}_D-ae_basic",
        f"Layered AE": f"mix_H-{baseline.lower()}_D-ae_layered",
        f"PCA": f"mix_H-{baseline.lower()}_D-pca"
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
        cfg.BASE_EXP_DIR = cfg.PROJECT_ROOT / 'outputs' / f"synthetic_experiments{suffix}"
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
    save_dir = cfg.PROJECT_ROOT / 'outputs' / 'plots' / 'cross_variant_comparisons'
    
    # 1. The Line Plot (MSE vs Variants)
    plot_variant_multi_model_grid(
        master_data=master_data, 
        enc_sizes=enc_sizes, 
        model_labels=model_labels, 
        save_dir=save_dir,
        baseline_name=baseline
    )
    
    # 2. The Curve Plots (Train/Eval History)
    # Usually you don't want 40 curve plots, so we pick the top 1 or 2 encoding sizes to visualize.
    # We will just plot the highest and lowest encoding size from your list as an example:
    target_encodings_to_plot = [enc_sizes[0], enc_sizes[-1]] 
    
    for enc in target_encodings_to_plot:
        plot_variant_curves(master_data, target_enc=enc, model_key=f"{baseline}-{target_model}", save_dir=save_dir)


if __name__ == '__main__':
    print("Starting Analysis Pipeline...")

    # ---------------------------------------------------------
    # YOUR EXISTING PIPELINE (Commented out for now to save time)
    # ---------------------------------------------------------
    # analyze_healthy_model()
    # 
    # for mode in ["true"]: 
    #     print(f"\n" + "="*40)
    #     print(f">>> STARTING SYNTHETIC EXPERIMENT: {mode.upper()}")
    #     print("="*40)
    #     
    #     if mode == "true":
    #         cfg.RANDOM_THETA_EXP = False
    #         cfg.FIXED_THETA_EXP = False
    #     elif mode == "fixed":
    #         cfg.RANDOM_THETA_EXP = False
    #         cfg.FIXED_THETA_EXP = True
    #
    #     analyze_disease_mix(is_mixed=True) 
    #     analyze_disease_mix(is_mixed=False) 


    # ---------------------------------------------------------
    # THE NEW CROSS-VARIANT COMPARISON PIPELINE
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(">>> STARTING CROSS-VARIANT MSE COMPARISON")
    print("="*50)

    # Example 1: Compare variants for the PCA + Layered AE model
    # is_mixed=True looks in the 'disease_mix_all' subfolders
    aggregate_and_plot_variants(
        baseline='PCA', 
        target_model='layered', 
        phase='disease', 
        is_mixed=True
    )

    # Example 2: Compare variants for PCA + Basic AE model
    aggregate_and_plot_variants(
        baseline='PCA', 
        target_model='basic', 
        phase='disease', 
        is_mixed=True
    )
    
    print("\n✅ All variant comparisons complete! Check the outputs/plots/cross_variant_comparisons folder.")