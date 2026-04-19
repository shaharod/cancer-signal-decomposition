import traceback

import numpy as np

import config as cfg
import utils.model_utils as mu
import utils.data_utils as du

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
    