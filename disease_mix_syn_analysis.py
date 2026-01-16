import os
import torch
import pandas as pd
import numpy as np
import joblib
import config as cfg
from utils import plots_utils_v1 as pu
from utils import analysis_utils as au
from utils.analysis_utils import TRAIN_LOSS_IDX, EVAL_LOSS_IDX, TEST_MSE_IDX

from core.models.model_factory import ModelFactory

def load_reconstruction_data(mode_val):
    """Loads the CSVs needed for scatter validation based on the experiment mode."""
    # Check if we are in fixed or uniform mode to pick the right input file
    
    # mix_file = 'disease_data_theta05.csv' if cfg.FIXED_THETA_EXP else 'disease_data_uniform_theta.csv'
    mix_file = cfg.DISEASE_GENES_PATH
    truth_file = cfg.DATA_SUB / 'pure_disease_truth.csv'

    if not os.path.exists(mix_file) or not os.path.exists(truth_file):
        print(f"⚠️ Warning: Required CSVs ({mix_file} or {truth_file}) not found.")
        return None, None
        
    df_mixed = pd.read_csv(mix_file, index_col=0).T
    df_pure  = pd.read_csv(truth_file, index_col=0).T
    return df_mixed, df_pure



def get_model_reconstruction(arch_name, enc, tag, mixed_sample_np, h_base="pca"):
    input_dim = mixed_sample_np.shape[0]
    label = f"mix_H-{h_base}_D-{arch_name}"
    model_dir = cfg.get_path("disease", tag, label, enc, folder_type=cfg.MODELS_SUBFOLDER)
    
    # --- CASE 1: PCA-PCA ARCHITECTURE (Joblib) ---
    if arch_name == "pca" and h_base == "pca":
        model_path = model_dir / "model.joblib"
        if not model_path.exists(): return None
        mix_obj = joblib.load(model_path)
        x = mixed_sample_np.reshape(1, -1)
        # Assuming mix_obj has a .pca_d attribute which is an sklearn PCA
        encoded = mix_obj.pca_d.transform(x)
        return mix_obj.pca_d.inverse_transform(encoded).flatten()

    # --- CASE 2: HYBRID OR AE MODELS (PyTorch) ---
    else:
        model_path = model_dir / "model.pt"
        if not model_path.exists(): return None
        from core.models.components import PCAComponent, AEComponent
        # 1. Initialize Healthy Base
        if h_base == "pca":
            from sklearn.decomposition import PCA
            # We create a dummy object that looks like what PCAComponent expects
            class DummyPCA:
                def __init__(self, in_dim, latent_dim):
                    self.mean_ = np.zeros(in_dim)
                    self.components_ = np.zeros((latent_dim, in_dim))
                    self.n_components_ = latent_dim # Add the underscore version
            
            dummy_pca = DummyPCA(input_dim, enc)
            h_component = PCAComponent(dummy_pca)
        else:
            base_ae = ModelFactory.create_model(h_base, input_dim, enc)
            h_component = AEComponent(base_ae)

        # 2. Build Disease Architecture
        # Note: We wrap the raw model in your AEComponent
        disease_raw = ModelFactory.create_model(arch_name, input_dim, enc)
        d_component = AEComponent(disease_raw)
        
        # 3. Assemble and Load
        # This matches the structure: mix_model.healthy and mix_model.disease_ae
        mix_model = ModelFactory.create_mix_model(h_component, d_component)
        
        state_dict = torch.load(model_path, weights_only=True, map_location='cpu')
        mix_model.load_state_dict(state_dict)
        mix_model.eval()

        with torch.no_grad():
            x = torch.tensor(mixed_sample_np, dtype=torch.float32).unsqueeze(0)
            # Use your AEComponent routing: .ae is the actual model
            z_d = mix_model.disease.encoder(x)
            recon_d = mix_model.disease.ae.decoder(z_d)
            return recon_d.squeeze().numpy()






def main(mode_val):
    print(f"\n>>> PROCESSING PLOTS FOR: {mode_val.upper()}")
    phase = "disease"
    tag = "scaled" 
    
    # 1. Define folder mapping for Bar Plots
    labels = {
        "basic": "mix_H-pca_D-ae_basic",
        "layered": "mix_H-pca_D-ae_layered",
        "benchmark": "mix_H-pca_D-pca"
    }

    # Load numerical results for bar charts
    data_s = {k: au.load_data_for_analysis(True, v, phase=phase) for k, v in labels.items()}
    data_u = {k: au.load_data_for_analysis(False, v, phase=phase) for k, v in labels.items()}

    # 1. Plot the Bar Grid for this Base Category
    pu.plot_bar_grid(data_s, data_u, cfg.ENCODING_SIZES, name, list(labels.keys()))
    
    # 2. Plot the Line Curves (One Figure per AE model vs PCA)
    pu.plot_learning_curves(data_s, data_u, cfg.ENCODING_SIZES, name, list(labels.keys()))

    # --- Setup Plotting Path ---
    plot_root = cfg.get_path(phase, folder_type=cfg.PLOTS_SUBFOLDER) #/ "synthtic_reconstruction"
    plot_root.mkdir(parents=True, exist_ok=True)
    plot_folder_str = str(plot_root) + os.sep

    # 2. Plot the Bar Comparison (MSE Performance)
    pu.plot_comprehensive_comparison_bars(
        m1_s=data_s["basic"][TEST_MSE_IDX],
        m2_s=data_s["layered"][TEST_MSE_IDX],
        pca_s=data_s["benchmark"][TEST_MSE_IDX],
        m1_u=data_u["basic"][TEST_MSE_IDX],
        m2_u=data_u["layered"][TEST_MSE_IDX],
        pca_u=data_u["benchmark"][TEST_MSE_IDX],
        encoding_sizes=cfg.ENCODING_SIZES,
        title=f"Disease Tournament ({mode_val.upper()}): Impact of Architecture",
        save_path=f"tournament_bars_{mode_val}.png",
        folder_path=plot_folder_str,
        labels=["Disease Basic AE", "Disease Layered AE", "Disease PCA"]
    )

    pu.plot_test_mse_comparison_lines(
            m1_s=data_s["basic"][au.TEST_MSE_IDX],
            m2_s=data_s["layered"][au.TEST_MSE_IDX],
            pca_s=data_s["benchmark"][TEST_MSE_IDX],
            m1_u=data_u["basic"][au.TEST_MSE_IDX],
            m2_u=data_u["layered"][au.TEST_MSE_IDX],
            pca_u=data_u["benchmark"][TEST_MSE_IDX],
            encoding_sizes=cfg.ENCODING_SIZES,
            title=f"Disease Tournament: AE Basic vs AE Layered vs PCA (Healthy Base = PCA)",
            save_path="pca_base_tournament_lines.png",
            folder_path=plot_folder_str
            # folder_path=plot_folder_str,
            # labels=["Disease Basic AE", "Disease Layered AE", "Disease PCA"]
        )

    for tag in ["scaled", "unscaled"]:
        if tag == "scaled":
            data = data_s
        else:
            data = data_u
        # pu.compare_models_side_by_side(
        #         losses_ae_basic=data["basic"][au.TRAIN_LOSS_IDX],     # Training curves
        #         losses_ae_layered=data["layered"][au.TRAIN_LOSS_IDX], # Training curves
        #         losses_pca=data["benchmark"][au.TRAIN_LOSS_IDX],      # Final MSE lines, FIXME: WAS EVAL_LOSS_IDX
        #         encoding_sizes=cfg.ENCODING_SIZES,
        #         save_path=f"dynamics_on_pca_base_test",
        #         folder_path=plot_folder_str,
        #         runtag=f"e{cfg.EPOCHS_NUM}",
        #         ylim_top=100, 
        #         zoom_x=100,
        #         name1=f"D-Basic (H-PCA)",
        #         name2=f"D-Layered (H-PCA)"
        #     )
        pu.plot_training_convergence_subplots(
            losses_ae_basic=data["basic"][au.TEST_MSE_IDX],     # Training curves
            losses_ae_layered=data["layered"][au.TEST_MSE_IDX], # Training curves
            losses_pca=data["benchmark"][au.TEST_MSE_IDX],      # Final MSE lines, FIXME: WAS EVAL_LOSS_IDX
            encoding_sizes=cfg.ENCODING_SIZES,
            save_path=f"dynamics_on_pca_base_test_try",
            folder_path=plot_folder_str
        )
    

    # 3. Multi-Model Reconstruction Scatter Comparison
    df_mixed, df_pure = load_reconstruction_data(mode_val)
    if df_mixed is not None:
        for enc in cfg.ENCODING_SIZES:
            # Pick a sample to visualize (e.g., the first one)
            idx = 0 
            pure_truth = df_pure.iloc[idx].values
            mixed_input = df_mixed.iloc[idx].values
            
            reconstructions = {}
            
            # Loop through all 3 types to compare side-by-side
            for arch in ["pca", "ae_basic", "ae_layered"]:
                recon = get_model_reconstruction(arch, enc, tag, mixed_input, h_base="pca")
                if recon is not None:
                    name = {"pca": "PCA-PCA", "ae_basic": "Basic AE", "ae_layered": "Layered AE"}[arch]
                    reconstructions[name] = recon
            
            if reconstructions:
                pu.plot_multi_model_reconstruction(
                    pure_truth=pure_truth,
                    mixed_input=mixed_input,
                    recon_dict=reconstructions,
                    sample_idx=idx,
                    folder_path=plot_folder_str,
                    runtag=mode_val,
                    enc=enc
                )

if __name__ == "__main__":
    # Ensure SYNTHETIC_DATA is True in config
    
    for mode in ["true", "fixed"]:
        # Update Global Configs dynamically
        if mode == "true":
            cfg.RANDOM_THETA_EXP = False
            cfg.FIXED_THETA_EXP = False
        elif mode == "fixed":
            cfg.RANDOM_THETA_EXP = False
            cfg.FIXED_THETA_EXP = True
        
        main(mode)