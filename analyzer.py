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

# FIXME: Reconstruction Plot Logic
# - cant load the model for reconstruction, not sure im doing it correctly
# - not sure if its taking best value or what, cant remember what it was suppose to be
def analyze_reconstruction(phase='disease'):
    """
    Loads one specific model (e.g., Best Basic-AE) and plots its reconstruction
    vs the Ground Truth.
    """
    print(f"\n{'='*20} Generating Reconstruction Scatter {'='*20}")

    # --- IMPORT MODEL CLASS HERE ---
    # Adjust this line if your class is in a different file (e.g. 'from models import AEBasic')
    from core.models.ae_architectures import Basic_AE

    # 1. Load Data
    input_df, truth_df = load_reconstruction_data()
    if input_df is None: return

    # Convert to Tensor 
    # We take the first 500 samples for the scatter plot
    n_samples = 500
    input_tensor = torch.tensor(input_df.iloc[:n_samples].values).float().to(cfg.DEVICE)
    truth_tensor = torch.tensor(truth_df.iloc[:n_samples].values).float().to(cfg.DEVICE)

    # 2. Select Model to Visualize
    target_enc = 8
    model_type = 'mix_H-ae_basic_D-ae_basic'
    scale_tag  = 'scaled' # Warning: Ensure folder name matches exactly (Case Sensitive)

    # Construct path to the saved model file
    model_path = cfg.get_path(phase, scale_tag, model_type, target_enc, cfg.MODELS_SUBFOLDER) / "model.pt"
    
    if not model_path.exists():
        print(f"Skipping Reconstruction: Model not found at {model_path}")
        return

    # 3. Load Model Architecture & Weights (THE FIX)
    # We must create the empty model structure first!
    model = Basic_AE(input_size=input_tensor.shape[1], encoding_size=target_enc)#, h1=cfg.H1, h2=cfg.H2)
    model.to(cfg.DEVICE)

    # Load the weights from the file (which is a dictionary, not a model object)
    checkpoint = torch.load(model_path, map_location=cfg.DEVICE)
    
    # Handle both full checkpoint dicts and direct state_dicts
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 4. Run Inference
    with torch.no_grad():
        reconstructed, _ = model(input_tensor)

    # 5. Plot
    save_path = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER) / f"reconstruction_scatter_{model_type}_{target_enc}.png"
    
    pu.plot_reconstruction_scatter(
        original=truth_tensor.cpu().numpy(), 
        reconstructed=reconstructed.cpu().numpy(), 
        title=f"Reconstruction: {model_type} (Enc {target_enc})", 
        save_path=save_path
    )
    print(f"Saved reconstruction plot to {save_path}")


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
        group_save_path = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER) / f"Tournament_{baseline}"
        group_save_path.mkdir(parents=True, exist_ok=True)

        zoom = {
            'last_n_epochs': 300,
            'ylim_top': None
        }

        # generating plots once per baseline
        print(f"Generating Group Plots for: {baseline}")
        pu.plot_test_mse_bars(data_s, data_u, f'mse_bar_plot_{baseline}', group_save_path)
        pu.plot_mse_vs_encoding(data_s, data_u, f'mse_vs_enc_size_{baseline}', group_save_path)
        pu.plot_learning_curves(data_s, data_u, f'learning_curve_{baseline}', group_save_path, zoom_params=zoom)
        pu.plot_training_vs_pca(data_s, data_u, 'training_vs_pca', group_save_path)





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
    data_s, data_u = collect_phase_data(phase='healthy', model_labels=model_labels)
    save_path = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER)


    # plotting
    pu.plot_test_mse_bars(data_s, data_u, 'mse_bar_plot', save_path)
    pu.plot_mse_vs_encoding(data_s, data_u, 'mse_vs_enc_size', save_path)
    pu.plot_learning_curves(data_s, data_u, 'learning_curve_plot', save_path, zoom_params=zoom)
    pu.plot_training_vs_pca(data_s, data_u, 'training_vs_pca', save_path)




if __name__ == '__main__':

    # TODO: fix logic, maybe from command lines arguments or something
    print(f'model type is: {'synthetic' if cfg.SYNTHETIC_DATA else 'synthetic'}\n\n')
    # analyze_healthy_model()
    # analyze_disease_mix()

    if cfg.SYNTHETIC_DATA:    
        analyze_reconstruction()


