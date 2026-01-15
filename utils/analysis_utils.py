from core.trainer_utils import ExperimentIO as io
import config as cfg

TRAIN_LOSS_IDX = 0
EVAL_LOSS_IDX  = 1
TEST_MSE_IDX   = 2


def load_data_for_analysis(scale_bool, model_tag, phase="healthy") -> tuple[dict, dict, dict]:
    """Loads history and meta for all encodings for a specific model tag."""
    tag = "scaled" if scale_bool else "unscaled"

    train_loss = {}
    eval_loss = {}
    test_mse = {}

    for enc in cfg.ENCODING_SIZES:
        path = cfg.get_path(phase, tag, model_tag, enc, folder_type=cfg.MODELS_SUBFOLDER)

        if model_tag == "pca" or model_tag == "mix_H-pca_D-pca":
            res = io.load_results(path, "results.json")
            if res:
                t_mse = res.get("train_mse", 0)
                v_mse = res.get("val_mse", 0)

                # storing values
                train_loss[enc] = [t_mse] 

                # same for both, pca is calculated once
                eval_loss[enc] = [v_mse]
                test_mse[enc] = v_mse
                
            continue
            
        history = io.load_results(path, "history.json")
        meta = io.load_results(path, "best_meta.json")

        if history:
            train_loss[enc] = history.get("train", [])
            eval_loss[enc] = history.get("val", [])
        
        if meta:
            test_mse[enc] = meta.get('best_val', 0)
            # test_mse.append(meta.get("best_val", 0))
        else:
            # fallback - take last value from validation losses
            test_mse[enc] = eval_loss[enc][-1] if eval_loss.get(enc) else 0
            # test_mse.append(eval_loss[enc][-1] if eval_loss.get(enc) else 0)


    return train_loss, eval_loss, test_mse
