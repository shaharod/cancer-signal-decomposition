import os
import config as cfg
from utils import plots_utils as pu
from utils import analysis_utils as au

def main():
    # --- 1. Load Everything (Basic, Layered, and PCA) ---
    pca_tag = "pca"
    
    # Dictionaries to hold data for multiple models
    data_s = {} # Scaled
    data_u = {} # Unscaled

    print("Loading data for all models...")
    for tag in cfg.MODEL_TYPES + [pca_tag]:
        data_s[tag] = au.load_data_for_analysis(True, tag)
        data_u[tag] = au.load_data_for_analysis(False, tag)

    # Shortcut references for PCA - train and mse, scaled and unscaled
    tr_pca_s, _, mse_pca_s = data_s[pca_tag]
    tr_pca_u, _, mse_pca_u = data_u[pca_tag]

    # --- Setup Plotting Environment ---
    plot_root = cfg.get_path("healthy", folder_type=cfg.PLOTS_SUBFOLDER)
    os.makedirs(plot_root, exist_ok=True)
    plot_folder_str = str(plot_root) + os.sep


    ## plot test mse comparisons: row per enc, col scaled/unscaled, 3 bars, one per model
    pu.plot_comprehensive_comparison_bars(
            m1_s=data_s["ae_basic"][2],   
            m2_s=data_s["ae_layered"][2], 
            pca_s=mse_pca_s,        
            m1_u=data_u["ae_basic"][2], 
            m2_u=data_u["ae_layered"][2], 
            pca_u=mse_pca_u, 
            encoding_sizes=cfg.ENCODING_SIZES,
            title="Performance Tournament: Scaled vs Raw Pipeline (Original Units)",
            save_path="final_architecture_vs_scaling_bars.png",
            folder_path=plot_folder_str,
            labels=["Basic AE", "Layered AE", "PCA"]
        )

    # --- Generate Comparison Graphs ---
    m1_tag, m2_tag = cfg.MODEL_TYPES[0], cfg.MODEL_TYPES[1]
    
    tr_ae_1_s, _, _ = data_s[m1_tag]
    tr_ae_2_s, _, _ = data_s[m2_tag]
    tr_ae_1_u, _, _ = data_u[m1_tag]
    tr_ae_2_u, _, _ = data_u[m2_tag]

    ## comparing losses: ae vs pca - scaled
    pu.compare_models_side_by_side(
        losses_ae_basic=tr_ae_1_s,
        losses_ae_layered=tr_ae_2_s,
        losses_pca=tr_pca_s,
        encoding_sizes=cfg.ENCODING_SIZES,
        save_path="arch_comparison_vs_pca_scaled",
        folder_path=plot_folder_str,
        runtag=f"e{cfg.EPOCHS_NUM}",
        ylim_top=150, # Focus on the convergence area
        zoom_x=50,    # See the last 50 epochs clearly
        name1=m1_tag,
        name2=m2_tag
    )
    ## comparing losses: ae vs pca - unscaled
    pu.compare_models_side_by_side(
        losses_ae_basic=tr_ae_1_u,
        losses_ae_layered=tr_ae_2_u,
        losses_pca=tr_pca_u,
        encoding_sizes=cfg.ENCODING_SIZES,
        save_path="arch_comparison_vs_pca_unscaled",
        folder_path=plot_folder_str,
        runtag=f"e{cfg.EPOCHS_NUM}",
        ylim_top=150, # Focus on the convergence area
        zoom_x=50,    # See the last 50 epochs clearly
        name1=m1_tag,
        name2=m2_tag
    )

    for ae_tag in cfg.MODEL_TYPES:
        tr_ae_s, ev_ae_s, mse_ae_s = data_s[ae_tag]
        tr_ae_u, ev_ae_u, mse_ae_u = data_u[ae_tag]

        pu.compare_scaling_impact(
            losses_from_scaled_pipe=tr_ae_s,
            losses_from_unscaled_pipe=tr_ae_u, 
            losses_pca_unscaled=tr_pca_u,
            encoding_sizes=cfg.ENCODING_SIZES,
            save_path=f"scaling_effect_{ae_tag}",
            folder_path=plot_folder_str,
            runtag=f"e{cfg.EPOCHS_NUM}",
            ylim_top=100, # Adjust based on your raw data MSE range
            model_name=ae_tag
        )

        # Training Dynamics (Grid)
        pu.plot_grid_train_vs_eval_scaled_unscaled(
            tr_ae_s, ev_ae_s, tr_ae_u, ev_ae_u,
            cfg.ENCODING_SIZES, cfg.EPOCH_JUMP, 300,
            f"dynamics_{ae_tag}.png", plot_folder_str,
            model_name=ae_tag
        )

    # # --- 4. Tournament Winner: Basic vs Layered ---
    # pu.plot_model_comparison_bars(
    #     data_s["ae_basic"][2], data_s["ae_layered"][2], cfg.ENCODING_SIZES,
    #     title="Final Healthy Tournament: Basic vs Layered AE",
    #     save_path="tournament_arch_comparison.png",
    #     folder_path=plot_folder_str,
    #     labels=["ae_basic", "ae_layered"]
    # )

if __name__ == "__main__":
    main()