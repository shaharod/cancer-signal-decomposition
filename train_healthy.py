import config as cfg
from core.models.model_factory import ModelFactory
from core.engine import Trainer
from core.trainer_utils import ExperimentIO as io
from utils import data_utils, pca_utils
import os

def train_all_healthy():
    print(">>> PHASE 1: BUILDING HEALTHY MODEL LIBRARY")

    for scale in cfg.SCALING_OPTIONS:
        tag = "scaled" if scale else "unscaled"
        
        # 1. Prepare Data & Split
        train_h, test_h, scaler = data_utils.get_ready_tensors(
            cfg.HEALTHY_GENES_PATH, 
            split_path=cfg.get_split_path("healthy", tag),
            use_scaling=scale
        )

        # Add these lines to move data to the Mac GPU
        train_h = train_h.to(cfg.DEVICE)
        test_h = test_h.to(cfg.DEVICE)

        # 2. PCA Baselines (Train and save immediately)
        pca_models = pca_utils.train_pca_collection(train_h, cfg.ENCODING_SIZES)
        for enc, mod in pca_models.items():
            path = cfg.get_path(phase="healthy", scale_tag=tag, model_type="pca", enc=enc, folder_type=cfg.MODELS_SUBFOLDER)
            pca_utils.save_pca_model(pca_model=mod, folder=path)
            train_mse = pca_utils.get_pca_mse(mod, train_h, scaler)
            val_mse = pca_utils.get_pca_mse(mod, test_h, scaler)

            io.save_results({
                "train_mse": train_mse,"val_mse": val_mse}, 
                path
                )

        # 3. AE Tournament
        for arch in cfg.MODEL_TYPES:
            for enc in cfg.ENCODING_SIZES:
                print(f"Training: {arch} | {tag} | Enc: {enc}")
                
                path = cfg.get_path("healthy", tag, arch, enc, folder_type=cfg.MODELS_SUBFOLDER)
                model = ModelFactory.create_model(arch, train_h.shape[1]-1, enc).to(cfg.DEVICE)
                
                trainer = Trainer(model, scaler=scaler, lr=cfg.LR, device=cfg.DEVICE)
                history, best_info = trainer.fit(train_h, test_h, epochs=cfg.EPOCHS_NUM)
                
                # Save the "Best State" found during training
                io.save_checkpoint(best_info['best_state'], path)
                io.save_results(history, path, "history.json")
                meta = {k: v for k, v in best_info.items() if k != 'best_state'}
                io.save_results(meta, path, "best_meta.json")

def fix_missing_meta():
    print(">>> Retroactively creating best_meta.json files...")
    
    # We only need to fix the Autoencoders (PCA is already handled by results.json)
    for scale in cfg.SCALING_OPTIONS:
        tag = "scaled" if scale else "unscaled"
        
        for arch in cfg.MODEL_TYPES:
            for enc in cfg.ENCODING_SIZES:
                path = cfg.get_path("healthy", tag, arch, enc, folder_type=cfg.MODELS_SUBFOLDER)
                history_path = os.path.join(path, "history.json")
                meta_path = os.path.join(path, "best_meta.json")

                # If history exists but meta doesn't...
                if os.path.exists(history_path) and not os.path.exists(meta_path):
                    history = io.load_results(path, "history.json")
                    
                    if history and "val" in history and len(history["val"]) > 0:
                        val_curve = history["val"]
                        best_val = min(val_curve)
                        best_epoch = val_curve.index(best_val) + 1
                        
                        # Create the meta dictionary
                        meta = {
                            "best_val": best_val,
                            "best_epoch": best_epoch,
                            "note": "Retroactively generated from history.json"
                        }
                        
                        # Save it!
                        io.save_results(meta, path, "best_meta.json")
                        print(f"Fixed: {arch} | {tag} | Enc {enc} (Best Val: {best_val:.4f})")

if __name__ == "__main__":
    # fix_missing_meta()
    train_all_healthy()
    # print(cfg.HEALTHY_GENES_PATH)
