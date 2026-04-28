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
    

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# -------------------------------------------------------------------
# LATENT EXTRACTION & COORDINATE GENERATION
# -------------------------------------------------------------------

def get_standalone_latents(model_type, input_size, enc_sizes, scale_bool, test_set, phase):
    latents = {}
    for enc in enc_sizes:
        _, z = mu.create_load_standalone_model(phase=phase, m_type=model_type, enc=enc, scale_bool=scale_bool, input_size=input_size, test_t=test_set)
        latents[f"{model_type}_enc{enc}"] = z
    return latents

def get_mix_latents(mix_type, input_size, enc_sizes, scale_tag, is_mixed, test_t):
    latents = {}
    for enc in enc_sizes:
        _, _, _, z = mu.create_load_mix_model(folder_tag=mix_type, test_set=test_t, gene_size=input_size, enc=enc, scale_tag=scale_tag)
        latents[f"{mix_type}_enc{enc}"] = z
        ## NOTE: the latent in mix models is the latent of disease part
    return latents

def generate_coords(Z, method="umap", **kwargs):
    """Unified interface for dimensionality reduction."""
    if method == "pca":
        return PCA(n_components=2).fit_transform(Z)
    elif method == "umap":
        reducer = umap.UMAP(
            n_neighbors=kwargs.get('n_neighbors', 15),
            min_dist=kwargs.get('min_dist', 0.1),
            random_state=42
        )
        return reducer.fit_transform(Z)
    elif method == "tsne":
        return TSNE(
            n_components=2, 
            perplexity=kwargs.get('perplexity', 30),
            init="pca", 
            random_state=42
        ).fit_transform(Z)
    raise ValueError(f"Unknown method: {method}")

def save_latent_batch(latents_dict, phase, scaled, methods=["pca", "umap"], is_mixed=False):
    """Processes PyTorch latents into 2D coordinates and caches them."""
    scale_str = "scaled" if scaled else "unscaled"
    
    for name, Z in latents_dict.items():
        model_tag = name.split("_enc")[0]
        enc_size = name.split("_enc")[-1]
        model_root = cfg.get_path(phase, scale_str, model_tag, enc_size, folder_type=cfg.MODELS_SUBFOLDER, is_mixed=is_mixed)

        for m in methods:
            coords = generate_coords(Z, method=m)
            np.save(model_root / f"{m}_coords.npy", coords)