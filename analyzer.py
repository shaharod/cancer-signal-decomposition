import os

import joblib
import config as cfg
import utils.analysis_utils as au
from core.models.model_factory import ModelFactory

import utils.plots_utils as pu
import torch
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

SCALED = True
UNSCALED = False

def collect_phase_data(phase, model_labels, is_mixed):
    """
    getting trained models history and data from models
    
    :param phase: 'heathy', 'disease' or whatever option
    :param model_labels: possible models to retreive data from
    """

    # data dictionaries to collect
    data_s = {}
    data_u = {}

    for label, model_tag in model_labels.items():
        data_s[label] = au.load_data_for_analysis(SCALED, model_tag, phase, is_mixed)
        data_u[label] = au.load_data_for_analysis(UNSCALED, model_tag, phase, is_mixed)

    return data_s, data_u

def print_data(data_s, data_u):
    """
    Prints a summary of the loaded data structure for both Scaled and Unscaled sets.
    debug function
    """
    datasets = {"SCALED": data_s, "UNSCALED": data_u}
    
    for label, ds in datasets.items():
        print(f"\n{'='*20} {label} DATA {'='*20}")
        
        for model_name, model_tuple in ds.items():
            # Unpacking based on analysis_utils structure
            train_dict = model_tuple[au.TRAIN_LOSS_IDX]
            eval_dict  = model_tuple[au.EVAL_LOSS_IDX]
            mse_dict   = model_tuple[au.TEST_MSE_IDX]
            
            print(f"\nModel: {model_name}")
            
            # Since all dicts share the same encoding keys
            for enc in train_dict.keys():
                # Get lengths or values for a quick overview
                t_len = len(train_dict[enc])
                e_len = len(eval_dict[enc])
                # Test MSE is usually a single-item list
                mse_val = mse_dict[enc][0] if isinstance(mse_dict[enc], list) else mse_dict[enc]
                
                print(f"  [Enc {enc:3}]: Train Pts: {t_len:4} | Eval Pts: {e_len:4} | Test MSE: {mse_val:.6f}")


def analyze_disease_mix(is_mixed, phase='disease'):

    # all possible combinations of healthy baselines with disease
    disease_mix_labels = {
        'PCA':
        {
            "basic": "mix_H-pca_D-ae_basic",
            "layered": "mix_H-pca_D-ae_layered",
            "pca-based": "mix_H-pca_D-pca"
        }
        # ,
        # 'AE-Basic':
        # {
        #     "basic": "mix_H-ae_basic_D-ae_basic",
        #     "layered": "mix_H-ae_basic_D-ae_layered",
        #     "pca-based": "mix_H-ae_basic_D-pca"
        # },
        # 'AE-Layered':
        # {
        #     "basic": "mix_H-ae_layered_D-ae_basic",
        #     "layered": "mix_H-ae_layered_D-ae_layered",
        #     "pca-based": "mix_H-ae_layered_D-pca"
        # }
    }


    for baseline, labels in disease_mix_labels.items():

        # collect data for the current baseline
        data_s, data_u = collect_phase_data(phase, model_labels=labels, is_mixed=is_mixed)
        if not data_s and not data_u:
            print(f"!!! WARNING: No data found for {baseline} with is_mixed={is_mixed}. Check your folder structure.")
            continue
        # getting mse values for plotting
        for data_type in [data_s, data_u]:
            for model_name in data_type:

                # extracting mse
                mse_dict = data_type[model_name][au.TEST_MSE_IDX]

                # make sure mse is wrapped in a list for plot_utils function needs
                for enc in mse_dict:
                    val = mse_dict[enc]
                    if not isinstance(val, (list, np.ndarray)):
                        mse_dict[enc] = [val]

        # group-level save path per baseline
        group_save_path = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER, is_mixed=is_mixed) / f"Tournament_H-{baseline}"
        group_save_path.mkdir(parents=True, exist_ok=True)

        zoom = {
            'last_n_epochs': 300,
            'ylim_top': None
        }

        # generating plots once per baseline
        print(f"Generating Group Plots for: {baseline}")

        pu.plot_train_eval_curves(data_s, data_u, save_name=f'tournament_H-{baseline}', folder_path=group_save_path, include_pca=False, zoom_params=None)    # <--- No zoom
        
        ## with pca train/test vals too
        pu.plot_train_eval_curves(data_s, data_u, save_name=f'tournament_H-{baseline}', folder_path=group_save_path, include_pca=True, zoom_params=None)    # <--- No zoom

        pu.plot_test_mse_comparison_lines(data_s, data_u, cfg.ENCODING_SIZES, f'MSE Performance: H-{baseline}', f'mse_line_comparison_H-{baseline}.png', group_save_path)

        pu.plot_comprehensive_comparison_bars(
            data_s, data_u,
            encoding_sizes=cfg.ENCODING_SIZES,
            title=f"Disease Tournament: Impact of AE vs PCA (Healthy Base = {baseline})",
            save_path=f"{baseline.lower()}_base_tournament_bars.png",
            folder_path=group_save_path
        )
     




def analyze_healthy_model(phase='healthy'):

    # setting labels
    model_labels = {
        'Basic-AE': 'ae_basic',
        'Layered-AE': 'ae_layered',
        'PCA': 'pca'
        }

    zoom = {
        'last_n_epochs': 100, 
        'ylim_top': 1000     # Adjust this value based on your typical final MSE
    }

    # getting data and path
    data_s, data_u = collect_phase_data(phase, model_labels=model_labels, is_mixed=False)
    save_path = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER)
    

    # plotting
                ## Sarina plot additions ##
    pu.plot_train_eval_curves(data_s, data_u, save_name='healthy_train_history', folder_path=save_path, include_pca=False, zoom_params=None)    # <--- No zoom
    
    ## with pca train/test vals too
    pu.plot_train_eval_curves(data_s, data_u, save_name='healthy_train_history', folder_path=save_path, include_pca=True, zoom_params=None)    # <--- No zoom

    pu.plot_test_mse_comparison_lines(data_s, data_u, cfg.ENCODING_SIZES, 'Healthy Model Performance', 'mse_line_comparison.png', save_path)
    pu.plot_comprehensive_comparison_bars(data_s, data_u, cfg.ENCODING_SIZES, title="Performance Tournament: Scaled vs Raw Pipeline (Original Units)",
                                                          save_path="final_architecture_vs_scaling_bars.png",
                                                          folder_path=save_path)

if __name__ == '__main__':
    for mode in ["true", "fixed"]:
        print(f"\n" + "="*40)
        print(f">>> STARTING SYNTHETIC EXPERIMENT: {mode.upper()}")
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
    # TODO: fix logic, maybe from command lines arguments or something
    # print(f'model type is: {'synthetic' if cfg.SYNTHETIC_DATA else 'synthetic'}\n\n')
        analyze_disease_mix(is_mixed=True) #both healthy and disease samples in training
        analyze_disease_mix(is_mixed=False) #disease samples only
    analyze_healthy_model()

    # if cfg.SYNTHETIC_DATA:    
    #     analyze_reconstruction()


