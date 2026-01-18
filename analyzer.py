import os
import config as cfg
import utils.analysis_utils as au

import utils.plots_utils as pu
import torch
import numpy as np

import pandas as pd

SCALED = True
UNSCALED = False

def load_reconstruction_data():
    """
    Loads the validation data (Mixed Input and Clean Ground Truth).
    Matches your requested structure using config paths.
    """
    # 1. Define Paths
    mix_file = cfg.DISEASE_GENES_PATH
    # Assuming 'pure_disease_truth.csv' exists in your data folder
    truth_file = cfg.DATA_SUB / 'pure_disease_truth.csv' 

    # 2. Validation
    if not mix_file.exists() or not truth_file.exists():
        print(f"⚠️ Warning: Reconstruction data not found:\n {mix_file}\n {truth_file}")
        return None, None
        
    # 3. Load & Transpose (Genes should be columns for the model)
    # Using 'T' because typically gene files are (Genes x Samples), but models expect (Samples x Genes)
    df_mixed = pd.read_csv(mix_file, index_col=0).T
    df_pure  = pd.read_csv(truth_file, index_col=0).T
    
    return df_mixed, df_pure


def analyze_reconstruction_grid(labels_dict, phase, scale_bool, save_path):
    """
    One function to rule them all. 
    If phase='disease', it handles 'mix' parsing. 
    If phase='healthy', it handles standalone parsing.
    """
    from core.models.model_factory import ModelFactory
    from sklearn.decomposition import PCA
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. Load Data & Tensors
    input_df, truth_df = load_reconstruction_data()
    if input_df is None: return
    
    tag = "scaled" if scale_bool else "unscaled"
    input_size = input_df.shape[1]
    input_tensor = torch.tensor(input_df.values).float().to("cpu")
    log_truth = np.log1p(truth_df.values).flatten()

    # 2. Determine if we are looping through Tournament Bases (Disease) or just Labels (Healthy)
    # This detects if labels_dict is {Base: {Models}} or just {Models}
    if phase == 'disease':
        iterator = labels_dict.items()
    else:
        # Wrap healthy labels in a dummy base so the loop structure is the same
        iterator = [("Standalone", labels_dict)]

    for base_name, models in iterator:
        n_rows = len(cfg.ENCODING_SIZES)
        n_cols = len(models)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
        fig.suptitle(f"Tournament Results: Healthy Base = {base_name.upper()}\nPhase: {phase.capitalize()} | Data: {tag.capitalize()}", 
                     fontsize=20, fontweight='bold', y=0.98)
                     
        for row_idx, enc in enumerate(cfg.ENCODING_SIZES):
            for col_idx, (model_label, folder_tag) in enumerate(models.items()):
                ax = axes[row_idx, col_idx]
                
                try:
                    # --- UNIFIED LOADING LOGIC ---
                    if "mix" in folder_tag:
                        # Disease Mix Logic
                        parts = folder_tag.split('_H-')
                        h_and_d = parts[1].split('_D-')
                        h_type, d_type = h_and_d[0], h_and_d[1]

                        def prepare_obj(m_type):
                            if m_type.lower() == 'pca':
                                obj = PCA(n_components=enc)
                                obj.mean_, obj.n_components_ = np.zeros(input_size), enc
                                obj.components_ = np.zeros((enc, input_size))
                                return obj
                            return ModelFactory.create_model(m_type, input_size, enc)

                        model = ModelFactory.create_mix_model(prepare_obj(h_type), prepare_obj(d_type))
                    else:
                        # Healthy / Standalone Logic
                        model = ModelFactory.create_model(folder_tag, input_size, enc)

                    # --- LOAD WEIGHTS & INFER ---
                    model_path = cfg.get_path(phase, tag, folder_tag, enc, cfg.MODELS_SUBFOLDER) / "model.pt"
                    if not model_path.exists():
                        ax.text(0.5, 0.5, "Model Not Found", ha='center'); continue

                    checkpoint = torch.load(model_path, map_location="cpu")
                    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
                    model.load_state_dict(state_dict)
                    model.eval()

                    with torch.no_grad():
                        output = model(input_tensor)
                        reconstructed = output[0] if isinstance(output, (tuple, list)) else output
                    
                    log_recon = np.log1p(reconstructed.numpy()).flatten()
                    recon_raw = reconstructed.numpy().flatten()
                    truth_raw = truth_df.values.flatten()
                    # corr = np.corrcoef(log_truth, log_recon)[0, 1]
                    corr = np.corrcoef(truth_raw, recon_raw)[0, 1]
                    # --- PLOTTING ---
                    # ax.hexbin(log_truth, log_recon, gridsize=70, cmap='YlGnBu', mincnt=1, bins='log')
                    # ax.plot([log_truth.min(), log_truth.max()], [log_truth.min(), log_truth.max()], 'r--', lw=1)
                    ax.hexbin(truth_raw, recon_raw, gridsize=70, cmap='YlGnBu', mincnt=1)

                    # Identity line based on raw min/max
                    ax.plot([truth_raw.min(), truth_raw.max()], [truth_raw.min(), truth_raw.max()], 'r--', lw=1)
                    if row_idx == 0: ax.set_title(f"{model_label}", fontsize=12, fontweight='bold')
                    if col_idx == 0: ax.set_ylabel(f"Enc: {enc}", fontsize=10, fontweight='bold')
                    ax.text(0.05, 0.95, f"R: {corr:.3f}", transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                except Exception as e:
                    ax.text(0.5, 0.5, f"Error Loading Model", ha='center', color='red')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        folder = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER) / f"Tournament_H-{base_name}"
        os.makedirs(folder, exist_ok=True)
        full_path = os.path.join(folder, f"{save_path}_{tag}")
        plt.savefig(full_path, dpi=150)
        plt.close()

def collect_phase_data(phase, model_labels):
    """
    getting trained models history and data from models
    
    :param phase: 'heathy', 'disease' or whatever option
    :param model_labels: possible models to retreive data from
    """

    # data dictionaries to collect
    data_s = {}
    data_u = {}

    for label, model_tag in model_labels.items():
        data_s[label] = au.load_data_for_analysis(SCALED, model_tag, phase)
        data_u[label] = au.load_data_for_analysis(UNSCALED, model_tag, phase)

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




def analyze_disease_mix(phase='disease'):

    # all possible combinations of healthy baselines with disease
    disease_mix_labels = {
        'PCA':
        {
            "basic": "mix_H-pca_D-ae_basic",
            "layered": "mix_H-pca_D-ae_layered",
            "pca-based": "mix_H-pca_D-pca"
        },
        'AE-Basic':
        {
            "basic": "mix_H-ae_basic_D-ae_basic",
            "layered": "mix_H-ae_basic_D-ae_layered",
            "pca-based": "mix_H-ae_basic_D-pca"
        },
        'AE-Layered':
        {
            "basic": "mix_H-ae_layered_D-ae_basic",
            "layered": "mix_H-ae_layered_D-ae_layered",
            "pca-based": "mix_H-ae_layered_D-pca"
        }
    }


    for baseline, labels in disease_mix_labels.items():

        # collect data for the current baseline
        data_s, data_u = collect_phase_data(phase, model_labels=labels)

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
        group_save_path = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER) / f"Tournament_H-{baseline}"
        group_save_path.mkdir(parents=True, exist_ok=True)

        zoom = {
            'last_n_epochs': 300,
            'ylim_top': None
        }

        # generating plots once per baseline
        print(f"Generating Group Plots for: {baseline}")
        # pu.plot_test_mse_bars(data_s, data_u, f'mse_bar_plot_H-{baseline}', group_save_path)
        # pu.plot_mse_vs_encoding(data_s, data_u, f'mse_vs_enc_size_H-{baseline}', group_save_path)
        # pu.plot_learning_curves(data_s, data_u, f'learning_curve_H-{baseline}', group_save_path, zoom_params=zoom)
        # pu.plot_training_vs_pca(data_s, data_u, 'training_vs_pca', group_save_path)

                    ## Sarina plot additions ##
        pu.plot_train_eval_curves(data_s, data_u, save_name=f'tournament_H-{baseline}', folder_path=group_save_path, include_pca=False, zoom_params=None)    # <--- No zoom
        
        ## with pca train/test vals too
        pu.plot_train_eval_curves(data_s, data_u, save_name=f'tournament_H-{baseline}', folder_path=group_save_path, include_pca=True, zoom_params=None)    # <--- No zoom

        pu.plot_test_mse_comparison_lines(data_s, data_u, cfg.ENCODING_SIZES, f'MSE Performance: H-{baseline}', f'mse_line_comparison_H-{baseline}.png', group_save_path)

        pu.plot_comprehensive_comparison_bars(
            data_s, data_u,
            encoding_sizes=cfg.ENCODING_SIZES,
            title=f"Disease Tournament: Impact of AE vs PCA (Healthy Base = {baseline})",
            save_path=f"{baseline.lower()}_base_tournament_bars.png",
            folder_path=group_save_path,
#            labels=["Disease Basic AE", "Disease Layered AE", "Disease PCA"]
        )
        ##unscaled data reconstructions
        analyze_reconstruction_grid(disease_mix_labels, phase='disease', 
                                    scale_bool=False, save_path="reconstructed_grid", 
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
    data_s, data_u = collect_phase_data(phase, model_labels=model_labels)
    save_path = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER)


    # plotting
    # pu.plot_test_mse_bars(data_s, data_u, 'mse_bar_plot', save_path)
    # pu.plot_mse_vs_encoding(data_s, data_u, 'mse_vs_enc_size', save_path)
    # pu.plot_learning_curves(data_s, data_u, 'learning_curve_plot', save_path, zoom_params=zoom)
    # pu.plot_training_vs_pca(data_s, data_u, 'training_vs_pca', save_path)
                ## Sarina plot additions ##
    pu.plot_train_eval_curves(data_s, data_u, save_name='healthy_train_history', folder_path=save_path, include_pca=False, zoom_params=None)    # <--- No zoom
    
    ## with pca train/test vals too
    pu.plot_train_eval_curves(data_s, data_u, save_name='healthy_train_history', folder_path=save_path, include_pca=True, zoom_params=None)    # <--- No zoom

    pu.plot_test_mse_comparison_lines(data_s, data_u, cfg.ENCODING_SIZES, 'Healthy Model Performance', 'mse_line_comparison.png', save_path)
    pu.plot_comprehensive_comparison_bars(data_s, data_u, cfg.ENCODING_SIZES, title="Performance Tournament: Scaled vs Raw Pipeline (Original Units)",
                                                          save_path="final_architecture_vs_scaling_bars.png",
                                                          folder_path=save_path)
   
    analyze_reconstruction_grid(model_labels, phase='healthy', 
                                scale_bool=False, 
                                save_path="reconstructed_grid")




if __name__ == '__main__':

    # TODO: fix logic, maybe from command lines arguments or something
    print(f'model type is: {'synthetic' if cfg.SYNTHETIC_DATA else 'synthetic'}\n\n')
    analyze_healthy_model()
    analyze_disease_mix()

    # if cfg.SYNTHETIC_DATA:    
    #     analyze_reconstruction()


