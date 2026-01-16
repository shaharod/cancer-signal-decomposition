import matplotlib.pyplot as plt
import utils.analysis_utils as au
import config as cfg
import numpy as np

DATA_TYPE = 'Synthetic' if cfg.SYNTHETIC_DATA else 'True'
ENC_SIZES = cfg.ENCODING_SIZES



def plot_test_mse_bars(data_s, data_u, fig_title, save_path):
    """
    Bar plot of Test MSE.
    Matches the style in image_3dc3e9.png.
    """
    num_rows = len(ENC_SIZES)
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows), squeeze=False)
    fig.suptitle(fig_title, fontsize=16, fontweight='bold')

    col_titles = ["Pipeline: Trained on Scaled Data", "Pipeline: Trained on Raw Data"]
    datasets = [data_s, data_u]
    colors = ['#8dbade', '#558e3e', '#e1968b'] # Matching colors from image_3dc3e9.png

    for row_idx, enc in enumerate(ENC_SIZES):
        for col_idx, data in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            
            # Extract names and values
            model_names = list(data.keys())
            mse_vals = [data[m][2].get(enc, [0])[0] for m in model_names]

            # Create bars
            bars = ax.bar(model_names, mse_vals, color=colors[:len(model_names)], 
                          edgecolor='black', alpha=0.9, width=0.6)
            
            # --- STYLING (Matching image_3dc3e9.png) ---
            ax.set_title(f"Encoding {enc} | {col_titles[col_idx]}", fontsize=12)
            ax.set_ylabel("MSE (Original Units)")
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            plt.setp(ax.get_xticklabels(), fontweight='bold', fontsize=9)

            # Add Value Labels on top of bars
            for b in bars:
                height = b.get_height()
                ax.text(
                    b.get_x() + b.get_width()/2, height + (max(mse_vals) * 0.01),
                    f'{height:.4g}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10
                )
            
            # Add headroom so labels don't hit the top
            ax.set_ylim(0, max(mse_vals) * 1.15)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path / f'{fig_title}.png')
    plt.close()




def plot_mse_vs_encoding(data_s, data_u, fig_title, save_path):
    """
    Plot 3: Line plot showing Test MSE vs. Encoding Size.
    Displays two columns: Scaled vs. Raw data.
    Each line represents a different model variant.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(fig_title, fontsize=16, fontweight='bold')

    enc_sizes = cfg.ENCODING_SIZES
    col_titles = ["Pipeline: Scaled Data", "Pipeline: Raw Data"]
    datasets = [data_s, data_u]
    
    # Standard markers for clear visibility
    markers = ['o', 's', 'D', '^', 'v']

    for col_idx, data in enumerate(datasets):
        ax = axes[col_idx]
        
        for m_idx, (model_name, model_tuple) in enumerate(data.items()):
            # Index 2 is the test_mse dictionary
            mse_dict = model_tuple[2]
            
            # Align MSE values with the global ENCODING_SIZES list
            y_values = []
            for enc in enc_sizes:
                val = mse_dict.get(enc, [0])
                # Support both list-wrapped and direct float values
                y_values.append(val[0] if isinstance(val, list) else val)

            # Plot line for the specific model
            ax.plot(enc_sizes, y_values, marker=markers[m_idx % len(markers)], 
                    label=model_name, linewidth=2, markersize=8, alpha=0.9)

        ax.set_title(col_titles[col_idx], fontsize=13)
        ax.set_xlabel("Encoding Size (Latent Space Dimensions)", fontweight='bold')
        ax.set_ylabel("Final Test MSE", fontweight='bold')
        
        # Ensure all encoding sizes are visible on the X-axis
        ax.set_xticks(enc_sizes)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize='medium', frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path / f'{fig_title}.png')
    plt.close()




def plot_learning_curves(data_s, data_u, fig_title, save_path):
    """
    Plot 1: Train vs Eval Learning Curves.
    Uses unique colors for each model to distinguish Basic AE from Layered AE.
    """
    enc_sizes = cfg.ENCODING_SIZES
    num_rows = len(enc_sizes)
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows), squeeze=False)
    fig.suptitle(fig_title, fontsize=16, fontweight='bold')

    col_titles = ["Pipeline: Scaled Data", "Pipeline: Raw Data"]
    datasets = [data_s, data_u]
    
    # Use a standard color cycle so each model gets a unique color
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for row_idx, enc in enumerate(enc_sizes):
        for col_idx, data in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            
            # To store PCA for drawing horizontal baselines at the end
            pca_baselines = [] 

            # Iterate with index to pick a unique color per model
            for m_idx, (model_name, model_tuple) in enumerate(data.items()):
                train_h = model_tuple[au.TRAIN_LOSS_IDX].get(enc, [])
                eval_h  = model_tuple[au.EVAL_LOSS_IDX].get(enc, [])
                
                m_color = colors[m_idx % len(colors)]

                if 'pca' in model_name.lower() and 'mix' not in model_name.lower():
                    # Save PCA values; we'll draw them as dashed baselines later
                    if train_h: pca_baselines.append((train_h[0], m_color, f"{model_name} Baseline"))
                
                elif len(train_h) > 0:
                    train_epochs = range(1, len(train_h) + 1)
                    # Solid line for Training
                    ax.plot(train_epochs, train_h, label=f"{model_name} Train", 
                            color=m_color, linestyle='-', linewidth=1.5)
                    
                    if len(eval_h) > 0:
                        step = len(train_h) // len(eval_h)
                        eval_epochs = range(step, len(train_h) + 1, step)
                        if len(eval_epochs) != len(eval_h):
                            eval_epochs = range(1, len(eval_h) + 1)
                        
                        # Dashed or Dotted line for Evaluation to distinguish from Train
                        ax.plot(eval_epochs, eval_h, label=f"{model_name} Eval", 
                                color=m_color, linestyle='--', alpha=0.7)

            # Add horizontal lines for PCA
            for val, color, label in pca_baselines:
                ax.axhline(y=val, color=color, linestyle=':', label=label, linewidth=2)

            ax.set_title(f"Encoding {enc} | {col_titles[col_idx]}", fontsize=11)
            ax.set_ylabel("Loss (MSE)")
            ax.set_xlabel("Epochs")
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(fontsize='x-small', loc='upper right', ncol=1) # ncol=2 helps if many labels

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path / f'{fig_title}.png')
    plt.close()



def plot_reconstruction_scatter(original, reconstructed, title, save_path, log_scale=False):
    """
    Plot 4: Individual Gene Reconstruction Scatter Plot.
    Compares original input values (X) vs reconstructed output values (Y).
    Includes a y=x line representing perfect reconstruction.
    """
    plt.figure(figsize=(8, 8))
    
    # Flatten arrays if they come in as (1, N) or (N, 1) tensors
    x = np.array(original).flatten()
    y = np.array(reconstructed).flatten()

    # Scatter plot of individual genes
    plt.scatter(x, y, alpha=0.5, s=10, color='#3498db', label='Genes')

    # Identity Line (y=x)
    limit = max(np.max(x), np.max(y))
    plt.plot([0, limit], [0, limit], color='red', linestyle='--', linewidth=2, label='Perfect Reconstruction (y=x)')

    # Styling
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Original Expression Value", fontweight='bold')
    plt.ylabel("Reconstructed Expression Value", fontweight='bold')
    
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f"{title} (Log Scale)", fontsize=14, fontweight='bold')

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()