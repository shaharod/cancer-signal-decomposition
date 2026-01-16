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
    Test MSE loss change over encding sizes
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