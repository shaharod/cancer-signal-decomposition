import torch
import os
from pathlib import Path

# --- CONFIGURATION ---
ROOT = Path(r"C:\Users\saris\OneDrive\University\Year3\project\new_git\cancer-signal-decomposition\outputs\synthetic_experiments")

SCALINGS = ["scaled", "unscaled"]
ENCODINGS = ["enc_8", "enc_16"]
MODELS = [
    "mix_H-pca_D-ae_basic",
    "mix_H-pca_D-ae_layered",
    "mix_H-pca_D-pca"
]

def audit_model_stats(path):
    if not path.exists():
        return None
    
    try:
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get('model_state_dict', ckpt.get('best_state', ckpt))
        
        stats = {}
        
        # 1. Capture exact layer shapes (The "Width" of the model)
        # We look specifically at the disease encoder weights
        shape_list = []
        weight_keys = [k for k in sd.keys() if 'disease' in k and 'weight' in k and 'encoder' in k]
        
        for k in sorted(weight_keys):
            out_f, in_f = sd[k].shape
            shape_list.append(f"{in_f}->{out_f}")
        
        stats['architecture'] = " | ".join(shape_list) if shape_list else "No Weights Found"
        stats['depth'] = len(weight_keys)
        
        # 2. Capture Mean Weight (to see if one is 'dead')
        if weight_keys:
            stats['weight_mean'] = sd[weight_keys[0]].mean().item()
            
        return stats
    except Exception as e:
        print(f"Error reading {path.name}: {e}")
        return None

def main():
    header = f"{'Condition':<45} | {'Old Arch':<25} | {'New Arch':<25} | {'Match'}"
    print(header)
    print("-" * len(header))

    for model_name in MODELS:
        for scale in SCALINGS:
            for enc in ENCODINGS:
                # PATHS (Switching to match your previous run order)
                old_path = ROOT / "disease_mix_neg_arch" / "disease_mix_uniform_theta" / "trained_models" / scale / model_name / enc / "model.pt"
                new_path = ROOT / "disease_mix" / "disease_mix_uniform_theta" / "trained_models" / scale / model_name / enc / "model.pt"
                
                old_stats = audit_model_stats(old_path)
                new_stats = audit_model_stats(new_path)
                
                condition_label = f"{model_name.split('_D-')[1]} | {scale} | {enc}"
                
                # SAFE CHECK: Ensure both exist before accessing keys
                if old_stats is not None and new_stats is not None:
                    old_arch = old_stats['architecture']
                    new_arch = new_stats['architecture']
                    match = "✅" if old_arch == new_arch else "🚨 DIFF"
                    
                    print(f"{condition_label:<45} | {old_arch:<25} | {new_arch:<25} | {match}")
                else:
                    # Identify which one is missing
                    status = "MISSING OLD" if old_stats is None else "MISSING NEW"
                    if old_stats is None and new_stats is None: status = "BOTH MISSING"
                    print(f"{condition_label:<45} | {status:<25} | {'-':<25} | -")

if __name__ == "__main__":
    main()