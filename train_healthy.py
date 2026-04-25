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
        
        # prepare data & split
        train_h, val_h, test_h, scaler = data_utils.get_ready_tensors(
            cfg.HEALTHY_GENES_PATH, 
            split_path=cfg.get_split_path("healthy", tag, is_mixed=False),
            use_scaling=scale,
            phase="healthy",
            is_mixed=False
        )
        ## here, if we had disease type col - its removed
        
        # move data to the mac GPU
        train_h = train_h.to(cfg.DEVICE)
        val_h = val_h.to(cfg.DEVICE)
        test_h = test_h.to(cfg.DEVICE)

        # PCA Baselines (train and save)
        pca_models = pca_utils.train_pca_collection(train_h, cfg.ENCODING_SIZES)
        for enc, mod in pca_models.items():
            path = cfg.get_path(phase="healthy", scale_tag=tag, model_type="pca", enc=enc, folder_type=cfg.MODELS_SUBFOLDER, is_mixed=False)
            pca_utils.save_pca_model(pca_model=mod, folder=path)
            train_mse = pca_utils.get_pca_mse(mod, train_h, scaler)
            val_mse = pca_utils.get_pca_mse(mod, val_h, scaler)
            test_mse = pca_utils.get_pca_mse(mod, test_h, scaler)
            io.save_results({
                "train_mse": train_mse,"val_mse": val_mse, "test_mse":test_mse}, 
                path
                )

        # AE Tournament
        for arch in cfg.MODEL_TYPES:
            for enc in cfg.ENCODING_SIZES:
                print(f"Training: {arch} | {tag} | Enc: {enc}")
                
                path = cfg.get_path("healthy", tag, arch, enc, folder_type=cfg.MODELS_SUBFOLDER, is_mixed=False)
                model = ModelFactory.create_model(arch, train_h.shape[1]-1, enc, cfg.H1, cfg.H2, scale).to(cfg.DEVICE)
                print(model)
                trainer = Trainer(model, scaler=scaler, lr=cfg.LR, device=cfg.DEVICE)
                history, best_info = trainer.fit(train_h, val_h, epochs=cfg.EPOCHS_NUM)
                
                #### adding logic to calculate test mse now to save time later for plotting ###
                model.load_state_dict(best_info['best_state'])
                test_mse = trainer.get_mse(test_h)

                # Save the "Best State" found during training
                io.save_checkpoint(best_info['best_state'], path)
                io.save_results(history, path, "history.json")

                meta['test_mse'] = test_mse
                meta = {k: v for k, v in best_info.items() if k != 'best_state'}
                io.save_results(meta, path, "best_meta.json")


if __name__ == "__main__":
    # fix_missing_meta()
    print(f"currrrnt run is with {cfg.BASE_EXP_DIR}")
    train_all_healthy()
    # print(cfg.HEALTHY_GENES_PATH)
