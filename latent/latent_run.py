import sys
import os
from pathlib import Path


# FIXME: i have no idea what is going on here, change or fix or whatever

# Get the path of the current file's directory
current_file = Path(__file__).resolve()

# Go up enough levels to reach the project root where config.py is
# If config.py is two folders up:
project_root = current_file.parents[1] 

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
import torch
import pandas as pd
import config as cfg
import latent_utils as lu
import utils.data_utils as du

def run_visualization_pipeline(phase, model_tags, data_path):
    """
    Runs the full t-SNE generation and plotting for a specific phase.
    """
    perplexities = [5, 30, 50, 100]
    
    for is_scaled in [True, False]:
        tag = "scaled" if is_scaled else "unscaled"
        print(f"\n>>> Starting Comprehensive Analysis: {phase} ({tag})")        
        # 1. Dynamically resolve the split path for THIS specific experiment/tag
        split_path = cfg.get_split_path(phase, tag, False) #TODO notice the is_mixed is False, if we want latent of all samples when running disease - need to change this
        
        # 2. Load the IDs for THIS phase/tag
        train_ids, test_ids = du.load_split(split_path)
        all_viz_ids = train_ids + test_ids
        
        # 3. Load signatures ONLY for the relevant IDs
        full_df_sig = pd.read_csv(cfg.SIG_PATH, index_col=0).loc[all_viz_ids, cfg.SIG_LIST]

        # 4. Load tensors using the phase-specific data path
        train_t, test_t, _ = du.get_ready_tensors(
            data_path, 
            split_path=split_path,
            use_scaling=is_scaled
        )
        X_combined = torch.cat([train_t, test_t], dim=0)
        X = X_combined[:, :-1]
        
        # shape-based plotting (O vs X)
        split_info = {"n_train": len(train_t),
                      "n_test": len(test_t)}
        latents_all = {}

        # 2. Extract Latents for all specified models
        for model_tag in model_tags:
            if model_tag == "pca":
                pca_lat = lu.get_pca_latents(phase,is_scaled, cfg.ENCODING_SIZES, X.numpy())
                latents_all.update(pca_lat)
            else:
                ae_lat = lu.get_ae_latents(phase, model_tag, is_scaled, cfg.ENCODING_SIZES, X)
                latents_all.update(ae_lat)

        # 3. Generate Coordinates (PCA and Multiple t-SNEs)
        # This saves .npy files for PCA and each perplexity in our list
        lu.save_latent_visuals(
            latents_all, 
            phase, 
            is_scaled, 
            full_df_sig, 
            perplexities=perplexities,
            split_info=split_info
        )

        # 4. Generate Cross-Reference Grids
        # We create a separate comparison grid for every gene signature
        for sig in cfg.SIG_LIST:
            sig_values = full_df_sig[sig].values
            
            lu.plot_comprehensive_comparison_grid(
                phase=phase,
                scaled=is_scaled,
                sig_name=sig,
                sig_values=sig_values,
                perplexities=perplexities,
                split_info=split_info
            )

            # Focused grids by model
            lu.plot_model_family_grids(
                phase=phase,
                scaled=is_scaled,
                sig_name=sig,
                sig_values=sig_values,
                perplexities=perplexities,
                split_info=split_info
            )

if __name__ == '__main__':
    # Run for Healthy
    run_visualization_pipeline("healthy", cfg.MODEL_TYPES + ["pca"], cfg.HEALTHY_GENES_PATH)
     #TODO add here for disease when we reach this stage of latent representation
    
    # EXAMPLE: Run for Disease (Add your specific mixed model tags here)
    # disease_models = ["mix_H-pca_D-pca", "mix_H-ae_layered_D-ae_layered"]
    # run_visualization_pipeline("disease", disease_models, cfg.)