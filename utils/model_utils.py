import joblib
import torch

from core.models.model_factory import ModelFactory
import config as cfg

def create_load_mix_model(folder_tag, test_set, gene_size, enc, scale_tag):
    parts = folder_tag.split('_H-')
    h_and_d = parts[1].split('_D-')
    h_type, d_type = h_and_d[0], h_and_d[1]
    is_pca = "pca" in d_type.lower()
    is_mix = "mix" in folder_tag
    if is_mix:
        h_model = ModelFactory.create_model(h_type, gene_size, enc, cfg.H1, cfg.H2)
        d_model = ModelFactory.create_model(d_type, gene_size, enc, cfg.H1, cfg.H2)
        model = ModelFactory.create_mix_model(h_model, d_model)
    else:
        model = ModelFactory.create_model(folder_tag, gene_size, enc, cfg.H1, cfg.H2)
    print(f"curr model is {h_and_d} and has:\n {model}")        
    ext = "model.joblib" if is_pca else "model.pt"
    model_path = cfg.get_path('disease', scale_tag, folder_tag, enc, cfg.MODELS_SUBFOLDER, is_mixed=True) / ext
    if not model_path.exists():
        return None, None, None, None
    if is_pca:
        pca_sk = joblib.load(model_path)
        print(f"DEBUG: Model Disease Layer: {model.disease}")
        print(f"DEBUG: PCA Components Shape: {pca_sk.components_.shape}")   
        print(f"DEBUG: Target Components Shape: {model.disease.components.shape}")
        # model.disease.mean.data = torch.tensor(pca_sk.mean_, dtype=torch.float32)
        # model.disease.components.data = torch.tensor(pca_sk.components_, dtype=torch.float32)
        # Use copy_ for in-place data transfer
        model.disease.mean.data.copy_(torch.from_numpy(pca_sk.mean_).float())
        model.disease.components.data.copy_(torch.from_numpy(pca_sk.components_).float())
        
        # Final Verification
        comp_sum = model.disease.components.sum().item()
        print(f"DEBUG: PCA Load Check - Components Sum: {comp_sum:.4f}")
        if comp_sum == 0:
            print("!!! WARNING: PCA components are still zero after loading!")
            raise ValueError()
    else:

        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', 
                        checkpoint.get('best_state', 
                        checkpoint))
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
    model.eval()
    print("--- [Weight Check] ---")
    has_healthy_weights = any(p.sum() != 0 for p in model.healthy.parameters())
    has_disease_weights = any(p.sum() != 0 for p in model.disease.parameters())

    print(f"Healthy Branch has non-zero weights: {has_healthy_weights}")
    print(f"Disease Branch has non-zero weights: {has_disease_weights}")

    # List the first few keys to ensure they match the 'healthy.' and 'disease.' prefix
    print(f"First 3 state_dict keys: {list(model.state_dict().keys())[:3]}")
        
    with torch.no_grad():
        model_outputs = model(test_set)        
    return model_outputs


def create_load_standalone_model(phase, m_type, enc, scale_bool, input_size, test_t):
    """
    Loader for Phase 1 (Healthy) or standalone Disease models.
    """
    tag = "scaled" if scale_bool else "unscaled"
    path = cfg.get_path(phase, tag, m_type, enc, cfg.MODELS_SUBFOLDER, is_mixed=False)
    
    is_pca = (m_type.lower() == 'pca')
    ext = "model.joblib" if is_pca else "model.pt"
    full_path = path / ext

    if not full_path.exists():
        print(f"--> [Error] Standalone model not found: {full_path}")
        return None

    model = ModelFactory.create_model(m_type, input_size, enc, cfg.H1, cfg.H2)

    if is_pca:
        pca_sk = joblib.load(full_path)
        model.mean.data.copy_(torch.from_numpy(pca_sk.mean_).float())
        model.components.data.copy_(torch.from_numpy(pca_sk.components_).float())
    else:
        ckpt = torch.load(full_path, map_location="cpu")
        state_dict = ckpt.get('model_state_dict', ckpt.get('best_state', ckpt))
        model.load_state_dict(state_dict)

    with torch.no_grad():
       return model(test_t)
