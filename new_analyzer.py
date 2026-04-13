import config as cfg
import utils.data_utils as du
import matplotlib.pyplot as plt
import numpy as np
import model_interpretability_copy as mi

def generate_multi_run_meta_cache(labels_dict, data_variants, theta_mode="true", scale_bool=True):
    """
    Creates a master cache holding inferences from multiple different data runs.
    Structure: meta_cache[variant_name][base_name][enc][model_label] = outputs
    """
    meta_cache = {}
    master_info = {} # Store test_df_full for each variant
    
    for variant in data_variants:
        print(f"\n📥 Loading Data for Variant: {variant.upper()}")
        
        # 1. Dynamically set global config (if your du.load requires it) 
        # Alternatively, pass 'variant' directly into du.load_and_prep_tensors
        if variant == 'simple':
            cfg.SYN_SIMPLE = True
            cfg.SYN_CMPLX = False
        elif variant == 'complex':
            cfg.SYN_SIMPLE = False
            cfg.SYN_CMPLX = True
            
        # 2. Load the specific data for this variant
        train_t, test_t, scaler, info = du.load_and_prep_tensors(
            phase="disease", mode=theta_mode, scale_bool=scale_bool, is_mixed=True
        )
        
        gene_size = test_t.shape[1] - 1
        tag = "scaled" if scale_bool else "unscaled"
        
        # 3. Generate the standard cache for this specific variant
        print(f"🧠 Running Inference for {variant}...")
        run_cache = mi.generate_inference_cache(labels_dict, test_t, gene_size, tag)
        
        # 4. Store in the Meta-Cache
        meta_cache[variant] = run_cache
        master_info[variant] = {
            'test_df_full': info['test_df_full'].fillna(value=0.0),
            'scaler': scaler
        }
        
    return meta_cache, master_info

def plot_architecture_vs_data_complexity(meta_cache_scaled, meta_cache_unscaled, 
                                         master_info_scaled, master_info_unscaled, 
                                         true_disease_inputs_dict, base_name='PCA'):
    """
    Grid: 
      Rows = Encoding Sizes
      Cols = Scaled (Left) vs Unscaled (Right)
    Inside Plot: 
      X-Axis = Model Type (Categorical)
      Lines = Data Variants
    """
    # 1. Setup the Grid
    enc_sizes = cfg.ENCODING_SIZES
    n_rows = len(enc_sizes)
    n_cols = 2 # 0: Scaled, 1: Unscaled
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True)
    fig.suptitle("Architectural Complexity vs. Data Complexity", fontsize=16, fontweight='bold')
    
    # The categorical X-axis values (Assuming these match your labels_dict keys)
    # Order matters here! Easiest to hardest complexity.
    models_x = ['pca', 'ae_basic', 'ae_layered'] 
    
    # Visual Styles for Data Variants
    variant_styles = {
        'simple': {'color': '#2ecc71', 'marker': 'o'},
        'complex': {'color': '#e74c3c', 'marker': 's'},
        'noisy': {'color': '#9b59b6', 'marker': '^'}
    }

    # 2. Iterate through the Grid (Rows = Enc Size)
    for row_idx, enc in enumerate(enc_sizes):
        
        # 3. Iterate through Columns (Scaled vs Unscaled)
        for col_idx, is_scaled in enumerate([True, False]):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]
            
            # Point to the correct cache based on column
            current_cache = meta_cache_scaled if is_scaled else meta_cache_unscaled
            current_info = master_info_scaled if is_scaled else master_info_unscaled
            
            # 4. Iterate through Data Variants (The Lines)
            for variant, cache in current_cache.items():
                
                # Setup ground truth for this variant
                test_df_full = current_info[variant]['test_df_full']
                scaler = current_info[variant]['scaler'] if is_scaled else None
                true_disease_input = true_disease_inputs_dict[variant]
                
                disease_mask = test_df_full.index.isin(true_disease_input.index)
                benchmark_truth = true_disease_input[disease_mask].values
                
                # Calculate MSE for each model on the X-axis
                y_mse_values = []
                valid_models_x = []
                
                for model_label in models_x:
                    try:
                        model_outputs = cache[base_name][enc].get(model_label)
                        if not model_outputs: continue
                        
                        recon_d = model_outputs['disease']
                        
                        # Inverse scale if we are in the Left Column
                        if is_scaled and scaler is not None:
                            recon_final = du.inverse_scale(scaler, recon_d).detach().cpu().numpy()
                        else:
                            recon_final = recon_d.detach().cpu().numpy()
                            
                        recon_disease_only = recon_final[disease_mask]
                        mse = np.mean((benchmark_truth - recon_disease_only) ** 2)
                        
                        y_mse_values.append(mse)
                        valid_models_x.append(model_label)
                        
                    except Exception as e:
                        print(f"Missing data for {variant}-{model_label}-enc{enc}")
                
                # 5. Plot the Line for this Variant
                if y_mse_values:
                    style = variant_styles.get(variant, {'color': 'gray', 'marker': 'd'})
                    # Plotting categorical X values natively maps to indices 0, 1, 2...
                    ax.plot(valid_models_x, y_mse_values, 
                            label=f"Data: {variant.title()}", 
                            color=style['color'], marker=style['marker'], 
                            linewidth=2.5, markersize=8)

            # Subplot Formatting
            title_scale = "Scaled" if is_scaled else "Unscaled"
            ax.set_title(f"Latent Enc: {enc} ({title_scale})", fontweight='bold')
            ax.set_ylabel("Disease MSE")
            ax.grid(True, linestyle='--', alpha=0.4)
            if row_idx == 0 and col_idx == 1:
                ax.legend(loc='upper right')

    # Formatting Bottom Row X-axis
    for ax in axes[-1, :]:
        ax.set_xlabel("Architecture Complexity")
        # Optional: Clean up names for the plot
        ax.set_xticklabels(['PCA Baseline', 'Shallow AE', 'Deep AE'])

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()