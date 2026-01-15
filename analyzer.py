import config as cfg
from utils import analysis_utils as au
from utils import plots_utils as pu
import os


def get_model_results(is_scaled, model_tag, phase='healthy'):
    
    data = au.load_data_for_analysis(is_scaled, model_tag, phase)

    # extract specific metrices using the constants from analysis_utils
    train_curves = data[au.TRAIN_LOSS_IDX]
    eval_curvs = data[au.EVAL_LOSS_IDX]

    final_mses = {
        enc: (val[-1] if isinstance(val, list) else val)
        for enc, val in data[au.TEST_MSE_IDX].items()
    }

    return train_curves, eval_curvs, final_mses


def run_analysis1(phase='healthy'):
    
    # getting plots path
    models = cfg.MODEL_TYPES
    plot_dir = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER)

    # getting results for both scaled and unscaled models
    res_s = {m: get_model_results(True, m, phase) for m in models}
    res_u = {m: get_model_results(False, m, phase) for m in models}

    # getting pca results
    _, _, pca_mse_s = get_model_results(True, 'pca', phase)
    _, _, pca_mse_u = get_model_results(False, 'pca', phase)

    pu.plot_comprehensive_comparison_bars(  
        m1_s=res_s["ae_basic"][au.EVAL_LOSS_IDX], 
        m2_s=res_s["ae_layered"][au.EVAL_LOSS_IDX],
        pca_s=pca_mse_s,
        m1_u=res_u["ae_basic"][au.EVAL_LOSS_IDX],
        m2_u=res_u["ae_layered"][au.EVAL_LOSS_IDX],
        pca_u=pca_mse_u,
        encoding_sizes=cfg.ENCODING_SIZES,
        title=f"Tournament: {phase.upper()}",
        save_path=f"{phase}_comparison.png",
        folder_path=str(plot_dir)
    )


def run_analysis(phase):

    pca_tag = 'pca'
    models_tags = cfg.MODEL_TYPES + [pca_tag]

    data_s = {}
    data_u = {}

    print('loading data for all models...')
    for model_tag in models_tags:
        data_s[model_tag] = au.load_data_for_analysis(True, model_tag, phase)
        data_u[model_tag] = au.load_data_for_analysis(False, model_tag, phase)
    

    tr_pca_s, _, mse_pca_s = data_s[pca_tag]    # train_loss, eval_loss = _, test_mse
    tr_pca_u, _, mse_pca_u = data_s[pca_tag]    # train_loss, eval_loss = _, test_mse


    # plotting environment
    plot_root = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER)
    os.makedirs(plot_root, exist_ok=True)
    plot_folder_str = str(plot_root) + os.sep

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



if __name__ == "__main__":
    # Run analysis for healthy first
    print("Running Healthy Phase Analysis...")
    run_analysis(phase='healthy')
    
    # If you have disease data ready, you can uncomment this:
    # print("Running Disease Mix Phase Analysis...")
    # run_analysis(phase='disease_mix_syn')