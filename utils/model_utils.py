import joblib
import torch

from core.models.model_factory import ModelFactory
import config as cfg

# def create_load_mix_model(folder_tag, test_set, gene_size, enc, scale_tag):
#     scale_bool = True if scale_tag == "scaled" else False
#     parts = folder_tag.split('_H-')
#     h_and_d = parts[1].split('_D-')
#     h_type, d_type = h_and_d[0], h_and_d[1]
#     is_pca = "pca" in d_type.lower()
#     is_mix = "mix" in folder_tag
#     if is_mix:
#         h_model = ModelFactory.create_model(h_type, gene_size, enc, cfg.H1, cfg.H2, scale_bool)
#         d_model = ModelFactory.create_model(d_type, gene_size, enc, cfg.H1, cfg.H2, scale_bool)
#         model = ModelFactory.create_mix_model(h_model, d_model)
#     else:
#         model = ModelFactory.create_model(folder_tag, gene_size, enc, cfg.H1, cfg.H2, scale_bool)
#     # print(f"curr model is {h_and_d} and has:\n {model}")        
#     ext = "model.joblib" if is_pca else "model.pt"
#     model_path = cfg.get_path('disease', scale_tag, folder_tag, enc, cfg.MODELS_SUBFOLDER, is_mixed=True) / ext
#     if not model_path.exists():
#         return None, None, None, None
#     if is_pca:
#         pca_sk = joblib.load(model_path)
#         # print(f"DEBUG: Model Disease Layer: {model.disease}")
#         # print(f"DEBUG: PCA Components Shape: {pca_sk.components_.shape}")   
#         # print(f"DEBUG: Target Components Shape: {model.disease.components.shape}")
#         # model.disease.mean.data = torch.tensor(pca_sk.mean_, dtype=torch.float32)
#         # model.disease.components.data = torch.tensor(pca_sk.components_, dtype=torch.float32)
#         # Use copy_ for in-place data transfer
#         model.disease.mean.data.copy_(torch.from_numpy(pca_sk.mean_).float())
#         model.disease.components.data.copy_(torch.from_numpy(pca_sk.components_).float())
        
#         # Final Verification
#         comp_sum = model.disease.components.sum().item()
#         print(f"DEBUG: PCA Load Check - Components Sum: {comp_sum:.4f}")
#         if comp_sum == 0:
#             print("!!! WARNING: PCA components are still zero after loading!")
#             raise ValueError()
#     else:

#         checkpoint = torch.load(model_path, map_location="cpu")
#         if isinstance(checkpoint, dict):
#             state_dict = checkpoint.get('model_state_dict', 
#                         checkpoint.get('best_state', 
#                         checkpoint))
#         else:
#             state_dict = checkpoint
#         model.load_state_dict(state_dict)
#     model.eval()
#     # print("--- [Weight Check] ---")
#     # has_healthy_weights = any(p.sum() != 0 for p in model.healthy.parameters())
#     # has_disease_weights = any(p.sum() != 0 for p in model.disease.parameters())

#     # print(f"Healthy Branch has non-zero weights: {has_healthy_weights}")
#     # print(f"Disease Branch has non-zero weights: {has_disease_weights}")

#     # List the first few keys to ensure they match the 'healthy.' and 'disease.' prefix
#     # print(f"First 3 state_dict keys: {list(model.state_dict().keys())[:3]}")
        
#     with torch.no_grad():
#         model_outputs = model(test_set)        
#     return model_outputs

def create_load_mix_model(folder_tag, test_set, gene_size, enc, scale_tag):
    scale_bool = True if scale_tag == "scaled" else False
    is_mix = "mix" in folder_tag
    
    if not is_mix:
        # Fallback for standard/non-mix baselines
        model = ModelFactory.create_model(folder_tag, gene_size, enc, cfg.H1, cfg.H2, scale_bool)
        model_path = cfg.get_path('disease', scale_tag, folder_tag, enc, cfg.MODELS_SUBFOLDER, is_mixed=False) / "model.pt"
        if not model_path.exists(): 
            return None, None, None, None
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            return model(test_set)

    # Parse Mix model types
    parts = folder_tag.split('_H-')
    h_and_d = parts[1].split('_D-')
    h_type, d_type = h_and_d[0], h_and_d[1]
    
    disease_model_folder = cfg.get_path('disease', scale_tag, folder_tag, enc, cfg.MODELS_SUBFOLDER, is_mixed=True)

    # ==========================================
    # 1. LOAD/CREATE HEALTHY MODEL
    # ==========================================
    if h_type.lower() == "pca":
        # Load the actual fitted SKLearn object from Phase 1 (Healthy)
        h_pca_folder = cfg.get_path("healthy", scale_tag, "pca", enc, cfg.MODELS_SUBFOLDER, is_mixed=False)
        h_pca_path = h_pca_folder / "model.joblib"
        if not h_pca_path.exists():
            print(f"⚠️ Missing Phase 1 Healthy PCA: {h_pca_path}")
            return None, None, None, None
        h_sk = joblib.load(h_pca_path)
        h_model = ModelFactory.wrap_component(h_sk) # Safely wraps real weights!
    else:
        h_model = ModelFactory.create_model(h_type, gene_size, enc, cfg.H1, cfg.H2, scale_bool)

    # ==========================================
    # 2. LOAD/CREATE DISEASE MODEL
    # ==========================================
    if d_type.lower() == "pca":
        # Load the actual fitted SKLearn object from Phase 2 (Disease)
        d_pca_path = disease_model_folder / "model.joblib"
        if not d_pca_path.exists():
            print(f"⚠️ Missing Phase 2 Disease PCA: {d_pca_path}")
            return None, None, None, None
        d_sk = joblib.load(d_pca_path)
        d_model = ModelFactory.wrap_component(d_sk) # Safely wraps real weights!
    else:
        d_model = ModelFactory.create_model(d_type, gene_size, enc, cfg.H1, cfg.H2, scale_bool)

    # ==========================================
    # 3. BUILD THE MIX MODEL
    # ==========================================
    model = ModelFactory.create_mix_model(h_model, d_model)

    # ==========================================
    # 4. LOAD PYTORCH WEIGHTS (If Applicable)
    # ==========================================
    # If the disease branch is an Autoencoder, PyTorch saved a .pt file during Phase 2
    # If it is pure pca-pca, there is no .pt file (and we already loaded the joblibs anyway!)
    if d_type.lower() != "pca":
        pt_path = disease_model_folder / "model.pt"
        if not pt_path.exists():
            print(f"⚠️ Missing Phase 2 PyTorch Weights: {pt_path}")
            return None, None, None, None
            
        checkpoint = torch.load(pt_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('best_state', checkpoint))
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)

    model.eval()
    
    # --- Final Weight Verification ---
    if h_type.lower() == "pca" and model.healthy.components.sum().item() == 0:
        raise ValueError("Healthy PCA components are still zero! Check Phase 1 training.")
    if d_type.lower() == "pca" and model.disease.components.sum().item() == 0:
        raise ValueError("Disease PCA components are still zero! Check Phase 2 training.")

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

    model = ModelFactory.create_model(m_type, input_size, enc, cfg.H1, cfg.H2, scale_bool)

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
