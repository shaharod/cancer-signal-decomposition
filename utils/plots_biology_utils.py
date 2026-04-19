import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.patheffects as PathEffects
from adjustText import adjust_text

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


def plot_significance_scatter(ax, h_avg, d_avg, is_significant, up_regulated_mask, neg_log10_q, 
                              fold_change, title, gene_names, alpha_threshold=0.0000005, fc_threshold=2.0, highlight_top_n=5):
    """Plots the biological scatter with q-value colormapping and labeled outliers."""
    
    lim_min = min(h_avg.min(), d_avg.min())
    lim_max = max(h_avg.max(), d_avg.max())
    line_range = np.linspace(lim_min, lim_max, 100)
    
    # Fold Change Lines
    ax.plot(line_range, line_range * fc_threshold, color="#b84646", linestyle='--', linewidth=0.8, alpha=0.7)
    ax.plot(line_range, line_range / fc_threshold, color="#b84646", linestyle='--', linewidth=0.8, alpha=0.7)

    # 1. Background (Non-Significant)
    ax.scatter(h_avg[~is_significant], d_avg[~is_significant], 
               color='#d3d3d3', alpha=0.5, s=8, edgecolors='none', label='Not Significant')
    
    # 2. Significant Genes (Color Mapped)
    cmap = mcolors.LinearSegmentedColormap.from_list("sig_cmap", ["#696969", "#ff0000"])
    min_color_val = -np.log10(alpha_threshold)
    max_color_val = 10.0
    
    sc = ax.scatter(h_avg[is_significant], d_avg[is_significant], 
                    c=neg_log10_q[is_significant], cmap=cmap, 
                    vmin=min_color_val, vmax=max_color_val,
                    alpha=0.8, s=15, edgecolors='black', linewidth=0.2)
    
    # Colorbar
    cbar = ax.figure.colorbar(sc, ax=ax)
    def q_val_formatter(x, pos): return f"$10^{{-{x:g}}}$"
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(q_val_formatter))
    cbar.set_label(r'$-\log_{10}(q\text{-value})$', fontsize=9)

    # 3. Label Top Outliers
    up_indices = np.where(up_regulated_mask)[0]
    if len(up_indices) > 0:
        top_up_indices = up_indices[np.argsort(fold_change[up_indices])[-highlight_top_n:]]
        
        ax.scatter(h_avg[top_up_indices], d_avg[top_up_indices], s=15, color='none', edgecolors='black', linewidth=0.5)
        
        texts = []
        for idx in top_up_indices:
            txt = ax.text(h_avg[idx], d_avg[idx], gene_names[idx], 
                          fontsize=8, fontweight='bold', color='black', zorder=10)
            txt.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='white')])
            texts.append(txt)
            
        if texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color='black', lw=0.8, alpha=0.7),
                        expand_points=(1.5, 1.5), force_text=(0.5, 0.5))

    # Aesthetics
    ax.axline((1, 1), (10,10), color='black', linewidth=1.2, linestyle='--', label='Identity')
    ax.set_xlim([lim_min, lim_max])
    ax.set_ylim([lim_min, lim_max])
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
