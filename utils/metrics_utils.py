import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import utils.data_utils as du

def calculate_disease_mse_from_cache(labels_dict, inference_cache, test_df_full, true_disease_input, scaler, scale_bool, encoding_sizes):
    """
    Isolates pure disease samples from the cache and calculates the Global MSE.
    Returns a nested dictionary: master_results[base_name][model_label][enc_size] = mse_val
    """
    master_results = {}
    disease_test_indices = test_df_full.index.intersection(true_disease_input.index)
    benchmark_truth = true_disease_input.loc[disease_test_indices].values
    disease_row_locs = [test_df_full.index.get_loc(idx) for idx in disease_test_indices]

    for base_name, models in labels_dict.items():
        master_results[base_name] = {}
        for enc in encoding_sizes:
            for model_label, _ in models.items():
                if model_label not in master_results[base_name]:
                    master_results[base_name][model_label] = {}
                    
                try:
                    model_outputs = inference_cache[base_name][enc].get(model_label)
                    if model_outputs is None or model_outputs['disease'] is None:
                        continue
                        
                    recon_d = model_outputs['disease']
                    
                    if scale_bool and scaler is not None:
                        recon_final = du.inverse_scale(scaler, recon_d).detach().cpu().numpy()
                    else:
                        recon_final = recon_d.detach().cpu().numpy()

                    # Calculate Global MSE on JUST the disease samples
                    recon_disease_only = recon_final[disease_row_locs]
                    global_mse = np.mean((benchmark_truth - recon_disease_only) ** 2)
                    
                    master_results[base_name][model_label][enc] = global_mse
                except Exception as e:
                    print(f"Error calculating MSE for {base_name}-{model_label}-{enc}: {e}")
                    continue
                    
    return master_results

def calculate_binned_disease_mse_from_cache(labels_dict, inference_cache, test_df_full, true_disease_input, scaler, scale_bool, encoding_sizes):
    """
    Calculates the Test MSE for Disease Branch Reconstruction, binned by Theta ranges.
    Returns:
        master_results: Nested dict of MSE values [base_name][model_label][bin_name][enc_size]
        bin_counts: Dict of how many samples fell into each theta bin
    """
    # 1. Isolate DISEASE samples and get truth
    disease_mask = test_df_full.index.isin(true_disease_input.index)
    test_df_disease = test_df_full[disease_mask]
    benchmark_truth = true_disease_input.reindex(test_df_disease.index).values
    
    # 2. Create Theta Bins
    thetas = test_df_disease['theta_value'].values
    theta_bins = {
        'Low (<0.33)': (thetas >= 0.0) & (thetas < 0.33),
        'Med (0.33-0.66)': (thetas >= 0.33) & (thetas < 0.66),
        'High (>0.66)': (thetas >= 0.66) & (thetas <= 1.0)
    }
    bin_counts = {bin_name: np.sum(mask) for bin_name, mask in theta_bins.items()}

    # 3. Calculate Binned MSE
    master_results = {}
    for base_name, models in labels_dict.items():
        master_results[base_name] = {}
        for enc in encoding_sizes:
            for model_label, _ in models.items():
                if model_label not in master_results[base_name]:
                    master_results[base_name][model_label] = {bin_name: {} for bin_name in theta_bins.keys()}
                    
                try:
                    model_outputs = inference_cache[base_name][enc].get(model_label)
                    if model_outputs is None or model_outputs['disease'] is None:
                        continue
                        
                    recon_d = model_outputs['disease']
                    
                    if scale_bool and scaler is not None:
                        recon_final = du.inverse_scale(scaler, recon_d).detach().cpu().numpy()
                    else:
                        recon_final = recon_d.detach().cpu().numpy()
                        
                    recon_disease_only = recon_final[disease_mask]
                    
                    # Calculate MSE for EACH theta bin
                    for bin_name, bin_mask in theta_bins.items():
                        if not np.any(bin_mask):
                            continue # Skip empty bins
                            
                        truth_binned = benchmark_truth[bin_mask]
                        recon_binned = recon_disease_only[bin_mask]
                        
                        bin_mse = np.mean((truth_binned - recon_binned) ** 2)
                        master_results[base_name][model_label][bin_name][enc] = bin_mse
                        
                except Exception as e:
                    print(f"Error calculating binned MSE for {base_name}-{model_label}-{enc}: {e}")
                    continue

    return master_results, bin_counts

def calculate_differential_expression(healthy_matrix, disease_matrix, alpha_threshold=0.0000005, fc_threshold=2.0):
    """Calculates q-values and fold changes to find significantly upregulated genes."""
    h_avg = healthy_matrix.mean(axis=0)
    d_avg = disease_matrix.mean(axis=0)
    
    # Calculate P-Values (Welch's t-test)
    _, p_values = ttest_ind(healthy_matrix, disease_matrix, axis=0, equal_var=False)
    p_values = np.nan_to_num(p_values, nan=1.0)
    
    # Calculate Q-Values (FDR Correction)
    _, q_values, _, _ = multipletests(p_values, method='fdr_bh')
    
    # Map to -log10
    neg_log10_q = -np.log10(q_values + 1e-300)
    
    # Fold Change and Significance Masks
    epsilon = 1.0
    fold_change = (d_avg + epsilon) / (h_avg + epsilon)
    
    is_stat_sig = q_values <= alpha_threshold
    is_bio_sig = (fold_change >= fc_threshold) | (fold_change <= (1.0 / fc_threshold))
    
    is_significant = is_stat_sig & is_bio_sig
    up_regulated_mask = is_significant & (fold_change >= fc_threshold)
    
    return h_avg, d_avg, neg_log10_q, is_significant, up_regulated_mask, fold_change

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
