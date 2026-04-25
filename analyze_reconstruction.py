import traceback
import torch
import config as cfg

import utils.analysis_utils as au
import utils.data_utils as du
import utils.inference_utils as iu
import utils.metrics_utils as mu
import utils.plots_biology_utils as pbu
import utils.metrics_utils as mu
import utils.plots_utils as pu

import matplotlib.pyplot as plt
import torch
import numpy as np

SCALED, MIXED = True, True
UNSCALED, NOT_MIXED = False, False


def analyze_disease_reconstruction_mse_lines(labels_dict, inference_cache, test_df_full, true_disease_input, 
                                             scaler, scale_bool, save_path, mode, is_mixed=False):
    """
    Calculates the global Test MSE for Disease Branch Reconstruction from the inference cache
    and plots it as a line graph.
    """
    # 1. Math
    master_results = mu.calculate_disease_mse_from_cache(
        labels_dict, inference_cache, test_df_full, true_disease_input, 
        scaler, scale_bool, cfg.ENCODING_SIZES
    )
    
    # 2. Plot
    pu.plot_disease_mse_lines(master_results, cfg.ENCODING_SIZES, save_path, mode, is_mixed)

def analyze_disease_reconstruction_mse_by_theta(labels_dict, inference_cache, test_df_full, 
                                                true_disease_input, scaler, scale_bool, 
                                                save_path, mode, is_mixed=False):
    """
    Calculates the Test MSE for Disease Branch Reconstruction binned by Theta, 
    and plots the results.
    """
    # 1. Math
    master_results, bin_counts = mu.calculate_binned_disease_mse_from_cache(
        labels_dict, inference_cache, test_df_full, true_disease_input, 
        scaler, scale_bool, cfg.ENCODING_SIZES
    )
    
    # 2. Plot
    pu.plot_disease_mse_by_theta_lines(
        master_results, bin_counts, cfg.ENCODING_SIZES, save_path, is_mixed
    )

def analyze_disease_drivers_grid(labels_dict, inference_cache, test_df_full, test_genes_df, 
                                 scale_bool, scaler, save_path, mode, top_n=10, is_mixed=False):
    """
    Evaluates Top Disease Drivers (Relative Compensation) in a grid layout.
    Loops through available disease types and creates a separate grid figure for each.
    """
    tag = "scaled" if scale_bool else "unscaled"
    gene_names = test_genes_df.columns.tolist()
    
    # Identify unique disease types in the dataset, excluding 0 (Healthy)
    unique_diseases = [d for d in test_df_full['disease_type'].unique() if d != 0]
    disease_map = {1: "Disease A (CRC)", 2: "Disease B (SCLC)"}
    
    # Outer Loop: Generate a separate figure for each disease type
    for disease_target in unique_diseases:
        disease_name = disease_map.get(disease_target, f"Disease {disease_target}")
        
        # Inner Loop: Generate grids per Base Architecture
        for base_name, models in labels_dict.items():
            n_rows = len(cfg.ENCODING_SIZES) 
            n_cols = len(models)
            
            fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
            fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
            
            fig1.suptitle(f"Total Expression Deconvolution: {disease_name}\n(Base: {base_name.upper()} | Mode: {mode})", fontsize=16, fontweight='bold', y=0.98)
            fig2.suptitle(f"Disease Branch Workload Diagnostics: {disease_name}\n(Base: {base_name.upper()} | Mode: {mode})", fontsize=16, fontweight='bold', y=0.98)

            for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
                for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                    ax1 = axes1[row_idx, col_idx]
                    ax2 = axes2[row_idx, col_idx]
                    try:
                        # 1. Fetch Tensors (from Inference Utils)
                        prepare_result = iu.prepare_scatter_data(
                            inference_cache=inference_cache, base_name=base_name, enc=enc, 
                            model_label=model_label, test_df_full=test_df_full, 
                            disease_target=disease_target, scale_bool=scale_bool, scaler=scaler
                        )
                        
                        if prepare_result is None:
                            ax1.text(0.5, 0.5, "Model / Output Not Found", ha='center', color='red')
                            ax2.text(0.5, 0.5, "Model / Output Not Found", ha='center', color='red')
                            continue
                        
                        recon_h_np, is_true_healthy, recon_mix_np, is_target_disease, recon_d_np = prepare_result
                        
                        # 2. Math for Figure 1: Differential Expression (from Metrics Utils)
                        healthy_matrix = recon_h_np[is_true_healthy]
                        disease_matrix = recon_mix_np[is_target_disease]
                        
                        h_avg, d_avg, neg_log10_q, is_sig, up_mask, fc = mu.calculate_differential_expression(
                            healthy_matrix, disease_matrix
                        )

                        # 3. Plot Figure 1 (from Biology Plots Utils)
                        pbu.plot_significance_scatter(
                            ax=ax1, h_avg=h_avg, d_avg=d_avg, is_significant=is_sig, 
                            up_regulated_mask=up_mask, neg_log10_q=neg_log10_q, fold_change=fc,
                            title=f"Total Expression Deconvolution (Model:{model_label})",
                            gene_names=gene_names, highlight_top_n=top_n
                        )

                        # 4. Math for Figure 2: Residual Averages (from Metrics Utils)
                        _, _, avg_d_disease = mu.avg_disease_exp(recon_h_np, is_true_healthy, recon_mix_np, is_target_disease, recon_d_np)
                        
                        # 5. Plot Figure 2 (from Biology Plots Utils)
                        pbu.plot_residual_magnitude_scatter_template(
                            ax=ax2, h_baseline_avg=h_avg, d_disease_avg=avg_d_disease,
                            title=f"Disease Branch VS Healthy Expression | Mode: {mode}",
                            show_zero_line=True, abs_threshold_lines=[500, 1000],
                            radiating_ratio_lines=[0.5, 1.0, 2.0],
                            highlight_top_n=top_n, gene_names=gene_names
                        )
                        
                    except Exception as e:
                        traceback.print_exc()
                        fig1.text(0.5, 0.5, "Plotting Error", ha='center', color='red')
                        fig2.text(0.5, 0.5, "Plotting Error", ha='center', color='red')

            # 6. Save Figures (Pure Pathlib!)
            fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            out_folder = cfg.get_path("disease", folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=is_mixed) / f"Tournament_H-{base_name}"
            out_folder.mkdir(parents=True, exist_ok=True)
            
            filename1 = f"Absolute_Scatter_{tag}_Disease{disease_target}_{mode}.png"
            fig1.savefig(out_folder / filename1, dpi=150)
            plt.close(fig1)

            filename2 = f"Residual_Scatter_{tag}_Disease{disease_target}_{mode}.png"
            fig2.savefig(out_folder / filename2, dpi=150)
            plt.close(fig2)

            print(f"Saved: {filename1} and {filename2}")

def run_comprehensive_reconstruction_analysis(labels_dict, scale_bool, save_path, mode, is_mixed=False, is_simple=False):
    """
    The Master Pipeline. Loads data, runs all models once, and generates all plots.
    """
    print(f"\n🚀 Starting Evaluation Pipeline (Mode: {mode.upper()} | Simple: {is_simple})")
    
    # ==========================================
    # 1. LOAD AND PREP DATA (Happens exactly once)
    # ==========================================
    tag = "scaled" if scale_bool else "unscaled"

    _, true_disease = du.load_reconstruction_data('disease', mode) 
   
    train_t, val_t, test_t, scaler, info = du.load_and_prep_tensors(
    phase="disease", mode=mode, scale_bool=scale_bool, is_mixed=is_mixed
    )
    test_df_full = info['test_df_full'].fillna(value=0.0)      # Contains [Genes | Theta | Type]

    # test_w_theta_t = torch.Tensor(test_t.values).float()
    

    bad_samples = ["SCLC0232-519_NA_H3K4me3-725_01072023-95"]
    keep_mask = ~test_df_full.index.isin(bad_samples)
    # 2. Filter the Pandas DataFrame
    test_df_full = test_df_full[keep_mask]
    true_disease = true_disease.drop(index=bad_samples, errors='ignore')

    test_t = test_t[torch.tensor(keep_mask)]
    gene_size = test_t.shape[1] - 1
    print(f"✅ Data Loaded. Genes: {gene_size}, Test Samples: {test_t.shape[0]}")
    # ==========================================
    # 2. GENERATE INFERENCE CACHE 
    # ==========================================
    print("🧠 Running Model Inference Cache...")
    inference_cache = iu.generate_inference_cache(labels_dict, test_t, gene_size, tag)
    metadata_cols = ['theta_value', 'disease_type']
    test_genes_df = test_df_full.drop(columns=metadata_cols, errors='ignore')
    
    actual_gene_size = test_genes_df.shape[1]
    test_no_theta_t = torch.Tensor(test_genes_df.values).float()
    if gene_size != actual_gene_size:
        raise ValueError(f"why arent they the same size?: {gene_size} vs {actual_gene_size}")
    
    ## removing metadatacols also from true disease?
    true_disease = true_disease.drop(columns=metadata_cols, errors="ignore")



    # ==========================================
    # 3. GENERATE VISUALIZATIONS 
    # ==========================================
    
    # print("🎨 Drawing Disease Branch Scatter Plots...")
    # pu.plot_reconstruction_grid(
    #     labels_dict=labels_dict, inference_cache=inference_cache, 
    #     test_df_full=test_df_full, test_n_theta=test_no_theta_t,
    #     true_disease_input=true_disease, gene_size=actual_gene_size, 
    #     scaler=scaler, scale_bool=scale_bool, save_path=save_path+"_disease", 
    #     mode=mode, is_simple=is_simple, is_mixed=is_mixed, target_type='disease'
    # )
    
    print("🎨 Drawing Total Mix Scatter Plots...")
    pu.plot_reconstruction_grid(
        labels_dict=labels_dict, inference_cache=inference_cache, 
        test_df_full=test_df_full, test_n_theta=test_no_theta_t,
        true_disease_input=true_disease, gene_size=actual_gene_size, 
        scaler=scaler, scale_bool=scale_bool, save_path=save_path+"_total", 
        mode=mode, is_simple=is_simple, is_mixed=is_mixed, target_type='total'
    )
    raise
    pu.plot_reconstruction_grid(
        labels_dict=labels_dict, inference_cache=inference_cache, 
        test_df_full=test_df_full, test_n_theta=test_no_theta_t,
        true_disease_input=true_disease, gene_size=actual_gene_size, 
        scaler=scaler, scale_bool=scale_bool, save_path=save_path+"_total", 
        mode=mode, is_simple=is_simple, is_mixed=is_mixed, target_type='total'
    )
    
    print("🎨 Drawing Disease Recon MSE Lines...")
    analyze_disease_reconstruction_mse_lines( 
        labels_dict=labels_dict, inference_cache=inference_cache, 
        test_df_full=test_df_full, true_disease_input=true_disease, 
        scaler=scaler, scale_bool=scale_bool, save_path=save_path, 
        mode=mode, is_mixed=is_mixed
    )
    
    print("🎨 Drawing Disease Recon MSE by Theta Bins...")
    analyze_disease_reconstruction_mse_by_theta( 
        labels_dict=labels_dict, inference_cache=inference_cache, 
        test_df_full=test_df_full, true_disease_input=true_disease, 
        scaler=scaler, scale_bool=scale_bool, save_path=save_path, 
        mode=mode, is_mixed=is_mixed
    )

    print("🎨 Drawing Disease Drivers (Biological Differentials)...")
    analyze_disease_drivers_grid( 
        labels_dict=labels_dict, inference_cache=inference_cache, 
        test_df_full=test_df_full, test_genes_df=test_genes_df, 
        scale_bool=scale_bool, scaler=scaler, save_path=save_path, 
        mode=mode, top_n=10, is_mixed=is_mixed
    )
    
    print("✅ All analyses and visualizations complete!\n")

def interpret_disease_mix(phase='disease', mode="true"):

    labels_dict = {
        'PCA':
        {   "pca": "mix_H-pca_D-pca",
            "ae_basic": "mix_H-pca_D-ae_basic",
            "ae_layered": "mix_H-pca_D-ae_layered"
            
        }
    }
    for scale in cfg.SCALING_OPTIONS:
        scaling = "scaled" if scale else "unscaled"
        print(f"####### RUNNING WITH {scaling.upper()} DATA")
        print("################# running with mix") 
        run_comprehensive_reconstruction_analysis(labels_dict=labels_dict, scale_bool=scale, 
                                                  save_path="analyze_recon_mixed", mode=mode, 
                                                  is_mixed=MIXED, is_simple=False)


    print("end of run")
if __name__ == '__main__':

    mode = "true" 
    print(f"########### RUNNING MIX MODEL | MODE: {mode.upper()} ############")
    
    cfg.THETA_EXP_MODE = mode
    interpret_disease_mix(mode=mode)


