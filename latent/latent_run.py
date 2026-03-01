"""
Latent Analysis Pipeline: Cancer Signal Decomposition
=====================================================
This script extracts disease-specific latents (Zd) from UniversalMixModels
and generates UMAP/PCA visualizations.

Usage Examples:
---------------
1. Run Tournament Analysis for Synthetic Data:
   $ python latent_run.py

2. Customizing for 'Smarter' Data:
   Ensure your input CSV contains 'theta_value' and 'disease_type' columns.
   The script will automatically generate grids for both PCA and UMAP.

Key Logic:
----------
- Extracts Z_d only (ignores the frozen healthy branch).
- Automatically colors by mixing proportion (Theta) and categorical disease type.
- Saves raw coordinates (.npy) in model folders and grids in the plots folder.
"""

import sys
from pathlib import Path
print(sys.executable)
# Get the path of the current file's directory
current_file = Path(__file__).resolve()

# Go up enough levels to reach the project root where config.py is
# If config.py is two folders up:
project_root = current_file.parents[1] 

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
import torch
import config as cfg
import latent_utils as lu
import utils.data_utils as du



def run_comprehensive_latent_analysis(phase, is_mixed, mode):
    """
    General pipeline to extract latents and generate grids for any phase/mode.
    """
    print(f"\n>>> STARTING LATENT ANALYSIS | Phase: {phase} | Mixed: {is_mixed} | Mode: {mode}")
    
    # Tournament for visualization
    disease_models = ["mix_H-pca_D-ae_basic", "mix_H-pca_D-ae_layered", "mix_H-pca_D-pca"]
    healthy_models = ["ae_basic", "ae_layered", "pca"]
    
    model_tags = disease_models if phase == "disease" else healthy_models
    
    for scale in cfg.SCALING_OPTIONS:
        tag = "scaled" if scale else "unscaled"
        
        # Load the specific Test Set and Metadata (Theta, etc.)
        # ensures we color points by the correct sample IDs
        if phase == "disease":
            _, test_df = du.fix_df_data(scale, mode=mode, is_mixed=is_mixed)
        elif phase == "healthy":
            pass
        test_t = torch.Tensor(test_df.values).float()
        
        # Extract Latents
        # use the correct loader based on the phase
        input_size = test_t.shape[1] - 1
        if phase == "disease":
            latents = {}
            for m_tag in model_tags:
                # Extracts Z_d (Disease latent)
                lats = lu.get_mix_latents(m_tag, input_size, cfg.ENCODING_SIZES, tag, is_mixed, test_t)
                latents.update(lats)
        elif phase == "healthy":
            latents = {}
            for m_tag in model_tags:
                lats = lu.get_standalone_latents(m_tag, input_size, cfg.ENCODING_SIZES, scale, test_t, phase)
                latents.update(lats)
        else:
            raise ValueError("Was there a reason for me to be here???")
        
        visualization_targets = ['theta_value']
        if 'disease_type' in test_df.columns:
            visualization_targets.append('disease_type')
        
        color_df = test_df[visualization_targets]
        # Batch Process & Save Visuals (Raw coords + Individual plots)
        # pass the test_df as the 'color_df' to automatically color by Theta and other genes
        lu.save_latent_batch(latents, phase, scale, color_df=color_df, methods=["pca", "umap"], is_mixed=is_mixed)

        # Generate Global Comparison Grids

        # color by Theta to see the signal separation
        for target in visualization_targets:
            lu.plot_general_comparison_grid(
                phase=phase, 
                scaled=scale, 
                color_values=test_df[target].values, 
                label_name=target,
                row_keys=cfg.ENCODING_SIZES, 
                col_keys=model_tags, 
                method="umap",
                is_mixed=is_mixed
            )

            lu.plot_general_comparison_grid(
            phase=phase, 
            scaled=False, 
            color_values=test_df[target].values, 
            label_name=target,
            row_keys=cfg.ENCODING_SIZES, 
            col_keys=model_tags, 
            method="pca",   # <--- Loads pca_coords.npy
            is_mixed=is_mixed
            )

if __name__ == '__main__':
    
    # Analyze Healthy Baselines (Phase 1)
    run_comprehensive_latent_analysis("healthy", is_mixed=False, mode="true")
    
    # Execute the "Tournament" latent review for both Synthetic Modes
    # for mode in ["true", "fixed"]:
    #     # Update config flags to match experiment
    #     if mode == "true":
    #         cfg.RANDOM_THETA_EXP = False
    #         cfg.FIXED_THETA_EXP = False
    #     elif mode == "fixed":
    #         cfg.RANDOM_THETA_EXP = False
    #         cfg.FIXED_THETA_EXP = True
        
    #     # Analyze Disease Models trained with ALL samples (H + D)
    #     run_comprehensive_latent_analysis("disease", is_mixed=True, mode=mode)
        
    #     # Analyze Disease Models trained with Disease Samples Only
    #     run_comprehensive_latent_analysis("disease", is_mixed=False, mode=mode)


