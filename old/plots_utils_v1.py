import matplotlib.pyplot as plt
import numpy as np
import os
import math


# --- CURVE PLOTS ---
def compare_scaling_impact(losses_from_scaled_pipe, losses_from_unscaled_pipe, losses_pca_unscaled, 
                           encoding_sizes, save_path, folder_path, runtag, ylim_top, 
                           model_name="AE", zoom_x=50):
    """
    Compares the performance of a model trained on Scaled data vs Unscaled data.
    Both are plotted in the ORIGINAL units (Unscaled MSE) to see if scaling helped.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Left: Model trained on Scaled data, then projected back
    # Right: Model trained on Unscaled data directly
    axes = [ax1, ax2]
    data_sources = [losses_from_scaled_pipe, losses_from_unscaled_pipe]
    titles = [f"{model_name} (Trained on Scaled)", f"{model_name} (Trained on Raw)"]

    for ax, data_dict, title in zip(axes, data_sources, titles):
        for i, enc in enumerate(encoding_sizes):
            # Extract the 'unscaled' curve from the respective pipeline
            curve = data_dict.get(enc, [])
            pca_val = losses_pca_unscaled.get(enc, None)
            
            if curve:
                epochs = np.arange(1, len(curve) + 1)
                line, = ax.plot(epochs, curve, linewidth=1.8, label=f"Enc={enc}")
                
                if pca_val is not None:
                    ax.hlines(pca_val, xmin=1, xmax=len(curve), linestyles="--", 
                              linewidth=1.6, color=line.get_color(), alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Focus on final convergence in original units
        ax.set_ylim(0, ylim_top)
        max_epochs = max([len(c) for c in data_dict.values()] or [1])
        ax.set_xlim(max(1, max_epochs - zoom_x), max_epochs)

    ax1.set_ylabel("MSE (Original Units)")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f"scaling_logic_comp_{runtag}.png"), bbox_inches="tight")
    plt.close()


def compare_models_side_by_side(losses_ae_basic, losses_ae_layered, losses_pca, encoding_sizes, 
                                save_path, folder_path, runtag, ylim_top, zoom_x=50,
                                name1="Basic AE", name2="Layered AE"):
    """
    Plots two AE architectures side-by-side.
    - Each plot compares the AE curves to PCA baselines.
    - Uses the zoomed styling (last N epochs) and grid style from your previous code.
    """
    if not encoding_sizes:
        raise ValueError("encoding_sizes is empty.")

    # Calculate global max epochs for x-axis scaling
    max_epochs_global = 1
    for d in [losses_ae_basic, losses_ae_layered]:
        for enc in encoding_sizes:
            curve = d.get(enc, [])
            if curve:
                max_epochs_global = max(max_epochs_global, len(curve))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    axes = [ax1, ax2]
    model_dicts = [losses_ae_basic, losses_ae_layered]
    model_names = [name1, name2]

    for ax, ae_dict, m_name in zip(axes, model_dicts, model_names):
        for enc in encoding_sizes:
            ae_curve = ae_dict.get(enc, [])
            pca_val = losses_pca.get(enc, None)
            
            # 1. Plot AE Curve (Solid)
            if ae_curve:
                epochs = np.arange(1, len(ae_curve) + 1)
                line, = ax.plot(epochs, ae_curve, linewidth=1.8, label=f"{m_name} (enc={enc})")
                line_color = line.get_color() # Match PCA color to AE color

                # 2. Plot PCA Line (Dashed)
                if pca_val is not None:
                    xmax = len(ae_curve)
                    ax.hlines(pca_val, xmin=1, xmax=xmax, linestyles="--", 
                              linewidth=1.6, color=line_color, label=f"PCA (enc={enc})")

        # Styling from your preferred plot
        ax.set_title(f"{m_name} Performance")
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Zoom Logic
        start_epoch = max(1, max_epochs_global - zoom_x + 1)
        ax.set_xlim(start_epoch, max_epochs_global)
        ax.set_ylim(bottom=0, top=ylim_top)

    ax1.set_ylabel("MSE Loss")

    # Legend Merging (from your old logic)
    merged = {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            merged[label] = handle

    fig.legend(merged.values(), merged.keys(), loc="upper left", bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    full_path = os.path.join(folder_path, f"{save_path}_{runtag}.png")
    plt.savefig(full_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def plot_grid_train_vs_eval_scaled_unscaled(train_s, eval_s, train_u, eval_u, encoding_sizes, epoch_jump, ylim_top, save_path, folder_path, model_name="AE"):
    """Generates a grid of Train vs Eval curves for both Scaled and Unscaled data."""
    n = len(encoding_sizes)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), squeeze=False)

    for i, enc in enumerate(encoding_sizes):
        for j, (tr, ev, title) in enumerate([(train_s, eval_s, "Scaled"), (train_u, eval_u, "Unscaled")]):
            ax = axes[i, j]
            t_curve = tr.get(enc, [])
            e_curve = ev.get(enc, [])
            
            if t_curve:
                ax.plot(np.arange(1, len(t_curve)+1), t_curve, label="Train")
            if e_curve:
                ax.plot(np.arange(epoch_jump, epoch_jump * len(e_curve) + 1, epoch_jump), e_curve, '--', label="Eval")
            
            ax.set_title(f"{model_name} (enc={enc}) | {title}")
            ax.set_ylim(0, ylim_top)
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, save_path))
    plt.close()


# -- LINE PLOT FOR MSE -- #
def plot_test_mse_comparison_lines(
    m1_s, m2_s, pca_s,  # Scaled results (dicts)
    m1_u, m2_u, pca_u,  # Unscaled results (dicts)
    encoding_sizes, title, save_path, folder_path,
    labels=["Basic AE", "Layered AE", "PCA"]
):
    """
    Plots Test MSE as a function of Encoding Size.
    Left Plot: Scaled Pipeline | Right Plot: Raw Pipeline
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    
    colors = ['#5DADE2', 'green', '#EC7063'] # Blue, Green, Red
    markers = ['o', 's', '^'] # Circle, Square, Triangle
    
    pipelines = [
        (ax1, [m1_s, m2_s, pca_s], "Pipeline: Trained on Scaled Data"),
        (ax2, [m1_u, m2_u, pca_u], "Pipeline: Trained on Raw Data")
    ]

    for ax, model_dicts, col_title in pipelines:
        for i, (m_dict, label) in enumerate(zip(model_dicts, labels)):
            # Extract values for each encoding size (ensure they are floats)
            y_values = []
            for enc in encoding_sizes:
                val = m_dict.get(enc, 0)
                # Handle potential list/tensor wrap as seen in previous errors
                y_values.append(float(np.array(val).flatten()[0]))
            
            # Plot the line
            ax.plot(encoding_sizes, y_values, label=label, color=colors[i], 
                    marker=markers[i], linewidth=2, markersize=8)

            # --- Data Labels (Optional: shows exact MSE value next to points) ---
            for x_val, y_val in zip(encoding_sizes, y_values):
                ax.annotate(f'{y_val:.4g}', (x_val, y_val), textcoords="offset points", 
                            xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

        # Formatting
        ax.set_title(col_title, fontsize=14, pad=15)
        ax.set_xlabel("Encoding Size (Latent Dimension)", fontsize=12)
        ax.set_ylabel("Test MSE (Original Units)", fontsize=12)
        ax.set_xticks(encoding_sizes)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

        # Add headroom for labels
        curr_ylim = ax.get_ylim()
        ax.set_ylim(curr_ylim[0], curr_ylim[1] * 1.2)

    fig.suptitle(title, fontsize=18, y=1.05)
    plt.tight_layout()
    
    output_path = os.path.join(folder_path, save_path)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Line comparison plot saved to: {output_path}")
    plt.close()


# --- BAR PLOTS ---
def plot_model_comparison_bars(mse_ae, mse_pca, encoding_sizes, title, save_path, folder_path, labels=["AE", "PCA"]):
    """Compare two models (or model vs PCA) side-by-side per encoding size."""
    n = len(encoding_sizes)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, enc in enumerate(encoding_sizes):

        ax = axes[i]
        vals = [mse_ae.get(enc, 0), mse_pca.get(enc, 0)]

        x = np.arange(len(labels))
        bars = ax.bar(x, vals, color=['skyblue', 'salmon'], edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"Enc Size: {enc}")
        
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'{b.get_height():.4g}', ha='center', va='bottom')

    for j in range(i + 1, len(axes)): axes[j].axis('off')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(folder_path, save_path))
    plt.close()


def plot_comprehensive_comparison_bars(
    m1_s, m2_s, pca_s,  # Scaled Pipeline results (lists)
    m1_u, m2_u, pca_u,  # Unscaled Pipeline results (lists)
    encoding_sizes, title, save_path, folder_path, 
    labels=["Basic AE", "Layered AE", "PCA"]
):
    """
    Rows: Encoding Sizes
    Cols: Scaled Pipeline vs Unscaled Pipeline (both in original units)
    Bars: Shows the specific MSE value on top of each model bar.
    """
    n_enc = len(encoding_sizes)
    # Increased width to 14 to make room for text labels
    fig, axes = plt.subplots(n_enc, 2, figsize=(14, 4 * n_enc), squeeze=False)
    
    colors = ['#5DADE2', 'green', '#EC7063'] # Blue, green, Red
    x = np.arange(len(labels))

    for i, enc in enumerate(encoding_sizes):
        # Column data mapping
        col_data = [
            (m1_s.get(enc, 0), m2_s.get(enc, 0), pca_s.get(enc, 0)), # Scaled side
            (m1_u.get(enc, 0), m2_u.get(enc, 0), pca_u.get(enc, 0))  # Unscaled side
        ]
        col_titles = ["Pipeline: Trained on Scaled Data", "Pipeline: Trained on Raw Data"]

        for j, (vals, col_title) in enumerate(zip(col_data, col_titles)):
            ax = axes[i, j]
            bars = ax.bar(x, vals, color=colors, edgecolor='black', alpha=0.8, width=0.6)
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontweight='bold')
            ax.set_title(f"Encoding {enc} | {col_title}", fontsize=13, pad=15)
            ax.set_ylabel("MSE (Original Units)")
            ax.grid(axis='y', linestyle='--', alpha=0.3)

            # --- VALUE LABELS LOGIC ---
            for b in bars:
                height = b.get_height()
                # Format to 4 significant figures. 
                # Uses a slight offset (va='bottom') to sit just above the bar.
                ax.text(
                    b.get_x() + b.get_width()/2, 
                    height,
                    f'{height:.4g}', 
                    ha='center', 
                    va='bottom', 
                    fontsize=11, 
                    fontweight='bold',
                    color='black'
                )
            
            # Add some headroom on the Y-axis so labels don't get cut off
            ax.set_ylim(0, max(vals) * 1.15)

    fig.suptitle(title, fontsize=18, y=1.02)
    plt.tight_layout()
    
    # Save with high DPI for crisp text
    output_path = os.path.join(folder_path, save_path)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Comprehensive bar plot saved to: {output_path}")
    plt.close()


def plot_bars(train_s, train_u, test_s, test_u, encoding_sizes, title, save_path, folder_path):
    """Visualizes the 4-way comparison: Scaled/Unscaled x Train/Test."""
    n = len(encoding_sizes)
    fig, axes = plt.subplots(math.ceil(n/2), 2, figsize=(12, 5 * math.ceil(n/2)), squeeze=False)
    axes = axes.flatten()
    
    labels = ["Tr Scaled", "Te Scaled", "Tr Raw", "Te Raw"]
    for i, enc in enumerate(encoding_sizes):
        ax = axes[i]
        vals = [train_s.get(enc, 0), test_s.get(enc, 0), train_u.get(enc, 0), test_u.get(enc, 0)]
        ax.bar(labels, vals, color=['blue', 'lightblue', 'green', 'lightgreen'])
        ax.set_title(f"Encoding: {enc}")

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, save_path))
    plt.close()


## scatter plot of recon for samples
def plot_reconstruction_validation(pure_truth, mixed_input, reconstructed_disease, 
                                   sample_idx, folder_path, runtag, 
                                   arch_name="AE", enc=16):
    """
    Validation Scatter: Compares Model Reconstruction vs Ground Truth.
    Dots = Genes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # helper for diagonal line
    def add_identity(ax, a, b):
        ma = max(a.max(), b.max())
        ax.plot([0, ma], [0, ma], linewidth=1.5, c='red', linestyle='--', alpha=0.8, label="Identity")

    # Plot 1: Input vs Truth (The Problem)
    ax1.scatter(pure_truth, mixed_input, s=10, alpha=0.4, c='tab:gray', edgecolors='none')
    add_identity(ax1, pure_truth, mixed_input)
    ax1.set_title(f"Input (Mixed) vs. Ground Truth\nSample {sample_idx}", fontsize=13)
    ax1.set_xlabel("Pure Disease Signal (Ground Truth)")
    ax1.set_ylabel("Mixed Input Signal (Observed)")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Plot 2: Reconstruction vs Truth (The Solution)
    ax2.scatter(pure_truth, reconstructed_disease, s=10, alpha=0.4, c='tab:red', edgecolors='none')
    add_identity(ax2, pure_truth, reconstructed_disease)
    
    # Calculate Correlation for the title
    corr = np.corrcoef(pure_truth, reconstructed_disease)[0, 1]
    ax2.set_title(f"Model Reconstruction vs. Ground Truth\nCorrelation: {corr:.4f}", fontsize=13)
    ax2.set_xlabel("Pure Disease Signal (Ground Truth)")
    ax2.set_ylabel("Extracted Disease Signal (Model Output)")
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(f"Reconstruction Accuracy: {arch_name} (Enc={enc})", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save logic matching your existing style
    save_name = f"reconstruction_val_{arch_name}_enc{enc}_sample{sample_idx}_{runtag}.png"
    full_path = os.path.join(folder_path, save_name)
    plt.savefig(full_path, dpi=150, bbox_inches="tight")
    print(f"Validation scatter saved to: {full_path}")
    plt.close(fig)


def plot_multi_model_reconstruction(pure_truth, mixed_input, recon_dict, 
                                     sample_idx, folder_path, runtag, enc=16):
    """
    Creates a grid comparing different architectures' ability to reconstruct 
    the same sample.
    recon_dict: {'PCA': data, 'Basic AE': data, 'Layered AE': data}
    """
    n_models = len(recon_dict)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 5), sharey=True)
    
    # helper for diagonal line
    def add_identity(ax, a, b):
        ma = max(a.max(), b.max())
        ax.plot([0, ma], [0, ma], linewidth=1.5, c='red', linestyle='--', alpha=0.8)

    # 1. The Input (Reference)
    axes[0].scatter(pure_truth, mixed_input, s=8, alpha=0.3, c='tab:gray')
    add_identity(axes[0], pure_truth, mixed_input)
    axes[0].set_title(f"Mixed Input\n(Sample {sample_idx})", fontsize=12)
    axes[0].set_ylabel("Observed / Extracted Value")

    # 2. The Model Reconstructions
    for i, (name, recon_val) in enumerate(recon_dict.items()):
        ax = axes[i+1]
        ax.scatter(pure_truth, recon_val, s=8, alpha=0.4, c='tab:blue' if 'AE' in name else 'tab:green')
        add_identity(ax, pure_truth, recon_val)
        
        corr = np.corrcoef(pure_truth, recon_val)[0, 1]
        ax.set_title(f"{name}\nCorr: {corr:.4f}", fontsize=12)
        ax.set_xlabel("Ground Truth")

    plt.tight_layout()
    save_path = os.path.join(folder_path, f"multi_model_recon_sample{sample_idx}_{runtag}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Multi-model scatter saved to: {save_path}")
    plt.close()



import matplotlib.pyplot as plt
import numpy as np

def plot_bar_grid(data_s, data_u, encoding_sizes, base_name, model_keys):
    # Set your custom colors here
    colors = {
        "basic": "#4A90E2",     # Blue
        "layered": "#50E3C2",   # Teal
        "pca-based": "#F5A623"  # Orange
    }
    
    fig, axes = plt.subplots(len(encoding_sizes), 2, figsize=(14, 3 * len(encoding_sizes)), sharey='row')
    fig.suptitle(f'MSE Comparison: Healthy Base {base_name}', fontsize=18, fontweight='bold', y=1.02)

    for i, enc in enumerate(encoding_sizes):
        for j, (data_dict, title) in enumerate(zip([data_s, data_u], ["Scaled", "Unscaled"])):
            ax = axes[i, j]
            
            # Prepare data for this specific subplot
            current_labels = []
            values = []
            current_colors = []
            
            for key in model_keys:
                mse_val = data_dict[key][2].get(enc, [0])[0]
                current_labels.append(key)
                values.append(mse_val)
                current_colors.append(colors.get(key, 'gray'))
            
            bars = ax.bar(current_labels, values, color=current_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
            
            # Formatting
            if i == 0: ax.set_title(title, fontsize=14, pad=10)
            if j == 0: ax.set_ylabel(f"Size: {enc}\nMSE", fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            
            # Add text labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_learning_curves(data_s, data_u, encoding_sizes, base_name, model_keys):
    # Color config: Easy to change
    # Note: 'pca-based' gets the same color across all figures
    style_cfg = {
        "basic": {"train": "#1f77b4", "eval": "#aec7e8"},
        "layered": {"train": "#2ca02c", "eval": "#98df8a"},
        "pca-based": {"train": "red", "eval": "darkred"} # Fixed color for PCA
    }

    pca_key = "pca-based"
    other_models = [m for m in model_keys if m != pca_key]

    for model_key in other_models:
        fig, axes = plt.subplots(len(encoding_sizes), 2, figsize=(15, 4 * len(encoding_sizes)))
        fig.suptitle(f'Base: {base_name} | Model: {model_key} vs PCA', fontsize=16, y=1.01)

        for i, enc in enumerate(encoding_sizes):
            for j, (data_dict, title) in enumerate(zip([data_s, data_u], ["Scaled", "Unscaled"])):
                ax = axes[i, j]
                
                # Plot AE Train/Eval
                train_loss = data_dict[model_key][0].get(enc, [])
                eval_loss = data_dict[model_key][1].get(enc, [])
                epochs = range(1, len(train_loss) + 1)
                
                ax.plot(epochs, train_loss, label=f'Train {model_key}', 
                        color=style_cfg[model_key]["train"], lw=2)
                ax.plot(epochs, eval_loss, label=f'Eval {model_key}', 
                        color=style_cfg[model_key]["eval"], lw=1.5, linestyle='-')

                # Plot PCA Baseline (Horizontal lines)
                pca_train = data_dict[pca_key][0].get(enc, [0])[0]
                pca_eval = data_dict[pca_key][1].get(enc, [0])[0]

                ax.axhline(y=pca_train, color=style_cfg[pca_key]["train"], 
                           linestyle='--', label='PCA Train MSE', alpha=0.8)
                ax.axhline(y=pca_eval, color=style_cfg[pca_key]["eval"], 
                           linestyle=':', label='PCA Eval MSE', alpha=0.8)

                # Formatting
                if i == 0: ax.set_title(title, fontsize=14)
                ax.set_xlabel("Epochs")
                ax.set_ylabel(f"Size {enc}")
                ax.legend(loc='upper right', fontsize='small', frameon=True)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# def plot_training_convergence(
#     losses_ae_basic, losses_ae_layered, losses_pca, 
#     encoding_sizes, save_path, folder_path, title="Model Convergence"
# ):
#     """
#     Plots AE training history (lines) against a static PCA baseline (horizontal line).
#     """
#     plt.figure(figsize=(10, 6))
    
#     # 1. Plot Basic AE Line (over epochs)
#     basic_curve = np.array(losses_ae_basic).flatten()
#     epochs = np.arange(1, len(basic_curve) + 1)
#     plt.plot(epochs, basic_curve, label='Basic AE', color='#5DADE2', linewidth=2)
    
#     # 2. Plot Layered AE Line (over epochs)
#     layered_curve = np.array(losses_ae_layered).flatten()
#     plt.plot(epochs, layered_curve, label='Layered AE', color='green', linewidth=2)
    
#     # 3. Plot PCA Baseline (Horizontal Line)
#     # PCA doesn't have epochs, so we draw it across the entire x-range
#     pca_val = float(np.array(losses_pca).flatten()[0])
#     plt.axhline(y=pca_val, color='#EC7063', linestyle='--', linewidth=2, label=f'PCA Baseline ({pca_val:.4g})')

#     # Labeling the specific final values next to the last data points
#     plt.text(epochs[-1], basic_curve[-1], f' {basic_curve[-1]:.4g}', color='#5DADE2', va='center', fontweight='bold')
#     plt.text(epochs[-1], layered_curve[-1], f' {layered_curve[-1]:.4g}', color='green', va='center', fontweight='bold')

#     # Formatting
#     plt.title(f"{title} (Encoding Size: {encoding_size})", fontsize=14)
#     plt.xlabel("Epochs", fontsize=12)
#     plt.ylabel("MSE Loss", fontsize=12)
#     plt.grid(True, linestyle=':', alpha=0.6)
#     plt.legend()
    
#     # Save the plot
#     os.makedirs(folder_path, exist_ok=True)
#     output_path = os.path.join(folder_path, save_path)
#     plt.savefig(output_path, bbox_inches="tight", dpi=150)
#     plt.show()


