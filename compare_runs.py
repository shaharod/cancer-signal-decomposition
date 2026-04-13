import config as cfg
import utils.data_utils as du
import utils.model_utils as mu
import os
from pathlib import Path

# ---- Project Root ----
PROJECT_ROOT = Path(__file__).resolve().parent
# OUTPUTS_ROOT = PROJECT_ROOT / 'outputs'

def get_comparison_paths(phase, model_type, enc_size, scale_tag='scaled', 
                         data_variants=None, theta_modes=None, base_outputs_dir=None):
    """
    Traverses the outputs directory to find specific model configurations 
    across different synthetic data generations.

    Args:
        model_type (str): e.g., 'ae_layered', 'pca'
        enc_size (int): e.g., 16, 32, 64
        scale_tag (str): 'scaled' or 'unscaled'
        data_variants (list): List of suffix strings
        theta_modes (list): List of theta modes

    Returns:
        list of dicts: Contains the metadata and the exact Path object for each found run.
    """
    
    if base_outputs_dir is None:
        base_outputs_dir = PROJECT_ROOT / "outputs"

    if data_variants is None:
        data_variants = ['0.1t', 'dif_hp', 'dif_dp'] 
    
    if theta_modes is None:
        theta_modes = ['uniform', 'fixed']

    collected_runs = []

    for variant in data_variants:
        # 1. Target the specific experiment root
        exp_dir = "outputs" / f'synthetic_experiments_{variant}'
        
        if not exp_dir.exists():
            print(f"Skipping missing experiment directory: {exp_dir.name}")
            continue
        
        if phase == 'healthy':
            model_dir = exp_dir / 'healthy' / 'trained_models' / scale_tag / model_type / f"enc_{enc_size}"
            
            if model_dir.exists():
                collected_runs.append({
                    'phase': 'healthy',
                    'data_variant': variant,
                    'theta_mode': None, # Healthy runs don't use theta
                    'model_type': model_type,
                    'enc_size': enc_size,
                    'scale_tag': scale_tag,
                    'model_path': model_dir,
                    'plots_path': exp_dir / 'healthy' / 'plots' / scale_tag / model_type / f"enc_{enc_size}"
                })

        elif phase == 'disease':
            disease_root = exp_dir / 'disease_mix_all'
            
            if theta_modes is None:
                theta_modes = ['uniform', 'fixed']

            for theta_mode in theta_modes:
                if theta_mode == 'fixed':
                    theta_dir_name = 'disease_mix_fixed_0.5'
                else:
                    theta_dir_name = f'disease_mix_{theta_mode}_theta'
                    
                model_dir = disease_root / theta_dir_name / 'trained_models' / scale_tag / model_type / f"enc_{enc_size}"
                
                if model_dir.exists():
                    collected_runs.append({
                        'phase': 'disease',
                        'data_variant': variant,
                        'theta_mode': theta_mode,
                        'model_type': model_type,
                        'enc_size': enc_size,
                        'scale_tag': scale_tag,
                        'model_path': model_dir,
                        'plots_path': disease_root / theta_dir_name / 'plots' / scale_tag / model_type / f"enc_{enc_size}"
                    })
    return collected_runs

import torch

def build_dynamic_meta_cache(phase, base_name, model_types, enc_sizes, data_variants, theta_mode, gene_size, scale_bool=True):
    """
    Uses the discovered paths to load models and run inference,
    building the meta-cache required for the cross-variant plotting grids.
    """
    meta_cache = {}
    master_info = {}
    scale_tag = "scaled" if scale_bool else "unscaled"

    for variant in data_variants:
        print(f"\n📥 Loading Data & Models for Variant: {variant.upper()}")
        meta_cache[variant] = {base_name: {enc: {} for enc in enc_sizes}}
        
        # 1. LOAD THE DATA FOR THIS SPECIFIC VARIANT
        # Assuming you update load_and_prep_tensors to accept the 'variant' string
        train_t, test_t, scaler, info = du.load_and_prep_tensors(
            phase=phase, mode=theta_mode, scale_bool=scale_bool, 
            is_mixed=(phase=='disease'), variant=variant 
        )
        master_info[variant] = {'test_df_full': info['test_df_full'], 'scaler': scaler}
        
        # 2. LOOP THROUGH REQUESTED ARCHITECTURES
        for enc in enc_sizes:
            for model_type in model_types:
                
                # Fetch the exact path using our new function
                runs = get_comparison_paths(
                    phase=phase, model_type=model_type, enc_size=enc, 
                    scale_tag=scale_tag, data_variants=[variant], theta_modes=[theta_mode]
                )
                
                if not runs:
                    continue # Skip if this model wasn't trained for this variant
                    
                target_path = runs[0]['model_path']
                
                # 3. RUN INFERENCE 
                try:
                    # Note: You will need to ensure your model_utils (mu) has a function 
                    # that can load directly from a Path object rather than a folder_tag.
                    # e.g., model = torch.load(target_path / 'model.pt')
                    model_outputs = mu.run_inference_from_explicit_path(
                        target_model_dir=target_path, 
                        folder_tag=model_type,  # e.g., 'mix_H-pca_D-ae_layered'
                        test_set=test_t, 
                        gene_size=gene_size, 
                        enc=enc, 
                        scale_tag=scale_tag
                    )

                    recon_mix, recon_d, recon_h, _ = model_outputs
                    
                    # Store in the expected Meta-Cache format
                    meta_cache[variant][base_name][enc][model_type] = {
                        'mix': recon_mix,
                        'disease': recon_d,
                        'healthy': recon_h
                    }
                except Exception as e:
                    print(f"Failed to run inference for {model_type} at {target_path}: {e}")
            
    return meta_cache, master_info
