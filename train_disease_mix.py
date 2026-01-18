import torch
import config as cfg
from core.models.model_factory import ModelFactory
from core.engine import Trainer
from core.trainer_utils import ExperimentIO as io
from utils import data_utils, pca_utils

def run_cross_architecture_tournament(mode_val):
    print("\n>>> STARTING PHASE 2: CROSS-ARCHITECTURE TOURNAMENT")
    is_random = cfg.RANDOM_THETA_EXP
    for scale in cfg.SCALING_OPTIONS:
        tag = "scaled" if scale else "unscaled"
        train_d, test_d, scaler_d = data_utils.get_ready_tensors(
            cfg.DISEASE_GENES_PATH,
            split_path=cfg.get_split_path("disease", tag), use_scaling=scale,
             theta_path=cfg.THETA_PATH,
             mode=mode_val
        )

        # Move disease tensors to GPU
        train_d = train_d.to(cfg.DEVICE)
        test_d = test_d.to(cfg.DEVICE)

        input_dim = train_d.shape[1] - 1 
        # --- AUDIT PRINT ---
        print(f"\n[DATA AUDIT] Experiment Mode (Random={is_random})")
        print(f"Tensor Shape: {train_d.shape}") # Should be (Samples, 20007)

        # Show the last 5 columns of the first 5 rows
        # Columns 0 to 20005 are Genes, Column 20006 is Theta
        audit_slice = train_d[:5, -5:] 
        print("First 5 samples (Last 4 Genes + Theta Column):")
        print(audit_slice.cpu().numpy())

        # Check if Theta is within [0, 1]
        thetas = train_d[:, -1]
        print(f"Theta Statistics -> Mean: {thetas.mean():.4f}, Min: {thetas.min():.4f}, Max: {thetas.max():.4f}")
        print("-" * 50)

        for enc in cfg.ENCODING_SIZES:
            # 1. LOAD HEALTHY PCA BASELINE
            pca_path = cfg.get_path("healthy", tag, "pca", enc, folder_type=cfg.MODELS_SUBFOLDER) / "model.joblib"
            pca_h_obj = pca_utils.load_pca_model(pca_path) if pca_path.exists() else None
            
            # 2. COLLECT ALL HEALTHY AE BASES
            healthy_library = []
            if pca_h_obj:
                healthy_library.append(("pca", pca_h_obj))
            
            for h_arch in cfg.MODEL_TYPES:
                h_path = cfg.get_path("healthy", tag, h_arch, enc, folder_type=cfg.MODELS_SUBFOLDER) / "model.pt"
                if h_path.exists():
                    h_model = ModelFactory.create_model(h_arch, input_dim, enc)
                    h_model.load_state_dict(torch.load(h_path, weights_only=True))                   
                    healthy_library.append((h_arch, h_model))
            
            # --- BENCHMARK: PCA-PCA MIX ---
            if pca_h_obj is not None:
                print(f"Calculating PCA-PCA Benchmark | {tag} | Enc: {enc}")

                # Filter for disease-only samples for the PCA FIT ONLY
                disease_mask = (train_d[:, -1] > 0)
                disease_only_features = train_d[disease_mask, :-1]
                # Train a standard PCA on Disease data (gene portion only)
                # Note: pca_utils needs this function to return an sklearn PCA object
                pca_d_obj = pca_utils.train_single_pca(disease_only_features, enc) 
                
                full_pca_mix = ModelFactory.create_mix_model(pca_h_obj, pca_d_obj)
                bench_trainer = Trainer(full_pca_mix, scaler=scaler_d, device=cfg.DEVICE)
                pca_bench_val_mse = bench_trainer.get_mse(test_d)
                pca_bench_train_mse = bench_trainer.get_mse(train_d) 
                
                out_dir = cfg.get_path("disease", tag, "mix_H-pca_D-pca", enc, folder_type=cfg.MODELS_SUBFOLDER)
                io.save_results(
                    {"val_mse": pca_bench_val_mse, "train_mse": pca_bench_train_mse},
                    out_dir, "results.json"
                    )

            # --- TOURNAMENT: CROSS-ARCHITECTURE AE MIX ---
            for d_arch in cfg.MODEL_TYPES:
                for h_name, h_obj in healthy_library:
                    label = f"mix_H-{h_name}_D-{d_arch}"
                    print(f"Testing: {label} | {tag} | Enc: {enc}")
                    
                    disease_model = ModelFactory.create_model(d_arch, input_dim, enc)
                    mix_model = ModelFactory.create_mix_model(h_obj, disease_model)
                    
                    trainer = Trainer(mix_model, scaler=scaler_d, lr=cfg.LR, device=cfg.DEVICE)
                    history, best_info = trainer.fit(train_d, test_d, epochs=cfg.EPOCHS_NUM)
                    
                    out_dir = cfg.get_path("disease", tag, label, enc, folder_type=cfg.MODELS_SUBFOLDER)
                    io.save_checkpoint(best_info['best_state'], out_dir)
                    io.save_results(history, out_dir, "history.json")
                    
                    meta = {k:v for k,v in best_info.items() if k!='best_state'}
                    io.save_results(meta, out_dir, "best_meta.json")

if __name__ == "__main__":
    for mode in ["true", "fixed"]:
        print(f"\n" + "="*40)
        print(f">>> STARTING SYNTHETIC EXPERIMENT: {mode.upper()}")
        print("="*40)
        
        # Set the flags so get_path and get_ready_tensors behave correctly
        # if mode == "true":
        #     cfg.RANDOM_THETA_EXP = False
        #     cfg.FIXED_THETA_EXP = False
        # elif mode == "random":
        #     cfg.RANDOM_THETA_EXP = True
        #     cfg.FIXED_THETA_EXP = False
        # elif mode == "fixed":
        #     cfg.RANDOM_THETA_EXP = False
        #     cfg.FIXED_THETA_EXP = True
            
        run_cross_architecture_tournament('random')
