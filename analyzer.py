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
    input_slice = input_df.iloc[:n_samples]
    truth_slice = truth_df.iloc[:n_samples]

    input_tensor = torch.tensor(input_slice.values).float().to(cfg.DEVICE)
    truth_tensor = torch.tensor(truth_slice.values).float().to(cfg.DEVICE)

    # 2. Select Model to Visualize
    target_enc = 8
    model_type = 'mix_H-pca_D-ae_basic'
    scale_tag  = 'unscaled' # Warning: Ensure folder name matches exactly (Case Sensitive)

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
    
    original_np = truth_tensor.cpu().detach().numpy()
    reconstructed_np = reconstructed.cpu().detach().numpy()
    # 5. Plot
    save_filename = f"reconstruction_hex_{model_type}_{target_enc}.png"
    save_path = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER) / save_filename
    
    pu.plot_io_scatter(
        original=original_np, 
        reconstructed=reconstructed_np, 
        title=f"Global Reconstruction: {model_type} (Enc {target_enc})", 
        save_path=save_path,
        log_scale=False  # Recommended for gene expression
    )
    
    print(f"Saved reconstruction plot to {save_path}")

def analyze_reconstruction_grid(labels_dict, phase='healthy', scale_bool=True):
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
        fig.suptitle(f"{phase.capitalize()} Reconstruction: {base_name} ({tag})", fontsize=16, fontweight='bold')

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
        save_path = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER) / f"grid_recon_{base_name}_{tag}.png"
        plt.savefig(save_path, dpi=150); plt.close()


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
        analyze_reconstruction_grid(disease_mix_labels, phase='disease', scale_bool=False)

    



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
   
    analyze_reconstruction_grid(model_labels, phase='healthy', scale_bool=False)




if __name__ == '__main__':

    # TODO: fix logic, maybe from command lines arguments or something
    print(f'model type is: {'synthetic' if cfg.SYNTHETIC_DATA else 'synthetic'}\n\n')
    analyze_healthy_model()
    analyze_disease_mix()

    # if cfg.SYNTHETIC_DATA:    
    #     analyze_reconstruction()


