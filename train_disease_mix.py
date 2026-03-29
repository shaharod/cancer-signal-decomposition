import joblib
import pandas as pd
import torch
import config as cfg
from core.models.model_factory import ModelFactory
from core.engine import Trainer
from core.trainer_utils import ExperimentIO as io
from utils import data_utils, pca_utils

def run_cross_architecture_tournament(mode_val, is_mixed):
    print("\n>>> STARTING PHASE 2: CROSS-ARCHITECTURE TOURNAMENT")

    disease_gene_path = cfg.get_disease_gene_path(mode_val)
    for scale in cfg.SCALING_OPTIONS:
        tag = "scaled" if scale else "unscaled"

        ########### USE BOTH HEALTHY AND DISEASE SAMPLES ##############
        if is_mixed:
            df_healthy = data_utils.prepare_and_align_data(cfg.HEALTHY_GENES_PATH, theta_path=None)
            # 2. Load Disease Data (Theta > 0)
            df_disease = data_utils.prepare_and_align_data(disease_gene_path, theta_path=cfg.get_theta_path(mode_val), mode=mode_val) #cfg.DISEASE_GENES_PATH
            
            # 3. Concatenate 
            df_combined = pd.concat([df_healthy, df_disease]) #.sample(frac=1, random_state=42)
            df_combined['disease_type'] = df_combined['disease_type'].fillna(0)

            train_df, test_df = data_utils.get_split_data(df_combined, split_path=cfg.get_split_path("disease", tag, is_mixed=is_mixed)) #TODO need to make sure when running real data we delete the splits that was there before, it is wrong
            
            train_t, test_t, scaler = data_utils.get_ready_tensors_df(train_df, test_df, scale, phase="disease", is_mixed=is_mixed, theta=mode_val)
            # raise ValueError(f"train is {train_t.shape} and test is {test_t.shape}")

            # Move disease tensors to GPU
            train_d = train_t.to(cfg.DEVICE)
            test_d = test_t.to(cfg.DEVICE)
        else:
            ############ USE ONLY DISEASE SAMPLES ##############
            train_d, test_d, scaler = data_utils.get_ready_tensors(
                disease_gene_path,
                split_path=cfg.get_split_path("disease", tag, is_mixed), use_scaling=scale,
                theta_path=cfg.get_theta_path(mode_val),
                mode=mode_val,
                phase="disease",
                is_mixed=is_mixed,
                theta=mode_val
            )

        # train_d = train_d[:, :-1]
        # test_d = test_d[:, :-1]
        input_dim = train_d.shape[1] - 1 
        numpy_array = train_d.detach().cpu().numpy()

        # 2. Convert to a Pandas DataFrame
        df = pd.DataFrame(numpy_array)
        # --- AUDIT PRINT ---
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
            pca_path = cfg.get_path("healthy", tag, "pca", enc, folder_type=cfg.MODELS_SUBFOLDER, is_mixed=is_mixed) / "model.joblib"
            pca_h_obj = pca_utils.load_pca_model(pca_path) if pca_path.exists() else None
            
            # 2. COLLECT ALL HEALTHY AE BASES
            healthy_library = []
            if pca_h_obj:
                healthy_library.append(("pca", pca_h_obj))
            
            for h_arch in cfg.MODEL_TYPES:
                h_path = cfg.get_path("healthy", tag, h_arch, enc, folder_type=cfg.MODELS_SUBFOLDER, is_mixed=is_mixed) / "model.pt"
                if h_path.exists():
                    h_model = ModelFactory.create_model(h_arch, input_dim, enc, cfg.H1, cfg.H2, scale)
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
                bench_trainer = Trainer(full_pca_mix, scaler=scaler, device=cfg.DEVICE)
                pca_bench_val_mse = bench_trainer.get_mse(test_d)
                pca_bench_train_mse = bench_trainer.get_mse(train_d) 
                
                out_dir = cfg.get_path("disease", tag, "mix_H-pca_D-pca", enc, folder_type=cfg.MODELS_SUBFOLDER, is_mixed=is_mixed)
                pca_d_path = out_dir / "model.joblib"
                joblib.dump(pca_d_obj, pca_d_path) #
                io.save_results(
                    {"val_mse": pca_bench_val_mse, "train_mse": pca_bench_train_mse},
                    out_dir, "results.json"
                    )
            # --- HYBRID TOURNAMENT: Healthy AE + Disease PCA ---
            for h_name, h_obj in healthy_library:
                if h_name == "pca": continue # Already handled by benchmark
                
                label = f"mix_H-{h_name}_D-pca"
                print(f"Testing Hybrid: {label} | {tag} | Enc: {enc}")
                
                # Use the same disease PCA we just trained for this encoding size
                mix_model = ModelFactory.create_mix_model(h_obj, pca_d_obj)
                
                # Evaluate
                bench_trainer = Trainer(mix_model, scaler=scaler, device=cfg.DEVICE)
                val_mse = bench_trainer.get_mse(test_d)
                train_mse = bench_trainer.get_mse(train_d)
                
                # Save folder
                out_dir = cfg.get_path("disease", tag, label, enc, folder_type=cfg.MODELS_SUBFOLDER, is_mixed=is_mixed)
                
                # Save results for load_data_for_analysis
                io.save_results({"val_mse": val_mse, "train_mse": train_mse}, out_dir, "results.json")
                
                # Save the Disease PCA object so the grid can load it
                joblib.dump(pca_d_obj, out_dir / "model.joblib")

            # --- TOURNAMENT: CROSS-ARCHITECTURE AE MIX ---
            for d_arch in cfg.MODEL_TYPES:
                for h_name, h_obj in healthy_library:
                    if h_name != "pca" and h_name != "PCA": continue
                    label = f"mix_H-{h_name}_D-{d_arch}"

                    print(f"Testing: {label} | {tag} | Enc: {enc}")
                    
                    disease_model = ModelFactory.create_model(d_arch, input_dim, enc, cfg.H1, cfg.H2, scale)
                    mix_model = ModelFactory.create_mix_model(h_obj, disease_model)
                    
                    trainer = Trainer(mix_model, scaler=scaler, lr=cfg.LR, device=cfg.DEVICE)
                    history, best_info = trainer.fit(train_d, test_d, epochs=cfg.EPOCHS_NUM)
                    
                    out_dir = cfg.get_path("disease", tag, label, enc, folder_type=cfg.MODELS_SUBFOLDER, is_mixed=is_mixed)
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
        if mode == "true":
            cfg.RANDOM_THETA_EXP = False
            cfg.FIXED_THETA_EXP = False
        elif mode == "random":
            cfg.RANDOM_THETA_EXP = True
            cfg.FIXED_THETA_EXP = False
        elif mode == "fixed":
            cfg.RANDOM_THETA_EXP = False
            cfg.FIXED_THETA_EXP = True
            
        run_cross_architecture_tournament(mode, is_mixed=True)
        run_cross_architecture_tournament(mode, is_mixed=False)

