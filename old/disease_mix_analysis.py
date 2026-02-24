import os
import config as cfg
from utils import plots_utils_v1 as pu
from utils import analysis_utils as au
# from utils.analysis_utils import au.TRAIN_LOSS_IDX, EVAL_LOSS_IDX, TEST_MSE_IDX


def main():
    print(">>> ANALYZING MIXES WITH HEALTHY PCA BASE")
    phase = "disease"

    # 1. Define the specific mixes using PCA as Healthy Base
    labels_l = [
        {
            "basic": "mix_H-pca_D-ae_basic",
            "layered": "mix_H-pca_D-ae_layered",
            "pca-based": "mix_H-pca_D-pca"
        },
        {
            "basic": "mix_H-ae_basic_D-ae_basic",
            "layered": "mix_H-ae_basic_D-ae_layered",
            "pca-based": "mix_H-ae_basic_D-pca"
        },
        {
            "basic": "mix_H-ae_layered_D-ae_basic",
            "layered": "mix_H-ae_layered_D-ae_layered",
            "pca-based": "mix_H-ae_layered_D-pca"
        }
        ]
    
    healthy_base_name_l = ["PCA", "AE-Basic", "AE-Layered"]

    for name, labels in zip(healthy_base_name_l, labels_l):

        data_s = {} # Scaled
        data_u = {} # Unscaled

        # load data
        for key, folder_tag in labels.items():
            # returned: (train_dict, eval_dict, mse_dict)
            data_s[key] = au.load_data_for_analysis(True, folder_tag, phase=phase)
            data_u[key] = au.load_data_for_analysis(False, folder_tag, phase=phase)


        # resolve plot root via config.py
        plot_root = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER)
        os.makedirs(plot_root, exist_ok=True)
        plot_folder_str = str(plot_root) + os.sep

        # getting pca-pca MSE
        mse_pca_mix_s = data_s["pca-based"][au.TEST_MSE_IDX]
        mse_pca_mix_u = data_u["pca-based"][au.TEST_MSE_IDX]


        # plotting bar comparison
        pu.plot_comprehensive_comparison_bars(
            m1_s=data_s["basic"][au.TEST_MSE_IDX],
            m2_s=data_s["layered"][au.TEST_MSE_IDX],
            pca_s=mse_pca_mix_s,
            m1_u=data_u["basic"][au.TEST_MSE_IDX],
            m2_u=data_u["layered"][au.TEST_MSE_IDX],
            pca_u=mse_pca_mix_u,
            encoding_sizes=cfg.ENCODING_SIZES,
            title=f"Disease Tournament: Impact of AE vs PCA (Healthy Base = {name})",
            save_path=f"{name.lower()}_base_tournament_bars.png",
            folder_path=plot_folder_str,
            labels=["Disease Basic AE", "Disease Layered AE", "Disease PCA"]
        )

        pu.plot_test_mse_comparison_lines(
            m1_s=data_s["basic"][au.TEST_MSE_IDX],
            m2_s=data_s["layered"][au.TEST_MSE_IDX],
            pca_s=mse_pca_mix_s,
            m1_u=data_u["basic"][au.TEST_MSE_IDX],
            m2_u=data_u["layered"][au.TEST_MSE_IDX],
            pca_u=mse_pca_mix_u,
            encoding_sizes=cfg.ENCODING_SIZES,
            title=f"Disease Tournament: AE Basic vs AE Layered vs PCA (Healthy Base = {name})",
            save_path=f"{name.lower()}_base_tournament_lines.png",
            folder_path=plot_folder_str,
            labels=["Disease Basic AE", "Disease Layered AE", "Disease PCA"]
        )

        # plotting training dynamics
        pu.compare_models_side_by_side(
            losses_ae_basic=data_s["basic"][au.TRAIN_LOSS_IDX],     # Training curves
            losses_ae_layered=data_s["layered"][au.TRAIN_LOSS_IDX], # Training curves
            losses_pca=data_s["pca-based"][au.TRAIN_LOSS_IDX],      # Final MSE lines, FIXME: WAS EVAL_LOSS_IDX
            encoding_sizes=cfg.ENCODING_SIZES,
            save_path=f"dynamics_on_{name.lower()}_base",
            folder_path=plot_folder_str,
            runtag=f"e{cfg.EPOCHS_NUM}",
            ylim_top=100, 
            zoom_x=100,
            name1=f"D-Basic (H-{name})",
            name2=f"D-Layered (H-{name})"
        )




if __name__ == "__main__":
    for mode in ["true", "fixed"]:
        print(f"\n" + "="*40)
        print(f">>> STARTING PLOT: {mode.upper()}")
        print("="*40)
        
        # Set the flags so get_path and get_ready_tensors behave correctly
        if mode == "true":
            cfg.RANDOM_THETA_EXP = False
            cfg.FIXED_THETA_EXP = False
        # elif mode == "random":
        #     cfg.RANDOM_THETA_EXP = True
        #     cfg.FIXED_THETA_EXP = False
        elif mode == "fixed":
            cfg.RANDOM_THETA_EXP = False
            cfg.FIXED_THETA_EXP = True
    
        main()