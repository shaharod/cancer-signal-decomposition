import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import joblib
import os
import json
from sklearn.preprocessing import StandardScaler
import config as cfg

def get_scaler_path(phase, is_mixed=False, theta=""):
    """
    Forces the scaler to be shared across all theta-experiments 
    within the same base phase.
    """
    
    # Get the base directory for the phase (Healthy, Mix, or Mix_All)
    if phase == "healthy":
        base_dir = cfg.HEALTHY_OUT_DIR
    elif phase == "disease":
        # Anchors to theta path so we match to data currently used
        base_dir = cfg.BASE_EXP_DIR / 'disease_mix_all' if is_mixed else cfg.BASE_EXP_DIR / 'disease_mix'
        if theta == "fixed":
            base_dir = base_dir / "disease_mix_fixed_0.5"
        elif theta == "true":
            theta_type = 'uniform' if cfg.SYNTHETIC_DATA else 'true'
            base_dir = base_dir / f"disease_mix_{theta_type}_theta"
        else:
            raise ValueError()
    else:
        raise ValueError(f"Unknown phase: {phase}")

    # Return the path inside the 'scaled/unscaled' root of that phase
    path = base_dir / "scaler.joblib"
    
    # Ensure the directory exists so we can save/load there
    os.makedirs(path.parent, exist_ok=True)
    return path


def fit_and_scale(train_df, test_df, phase, is_mixed=False, theta=""):
    """
    Smarter Scaling: Anchors the scaler to the phase root to ensure 
    consistency across different theta-experiments.
    """
    scaler_path = get_scaler_path(phase, is_mixed, theta)
    cols_to_drop = [col for col in train_df.columns if 'theta' in col]
    # We must drop theta so the scaler doesn't treat it as a gene feature
    train_genes = train_df.drop(columns=cols_to_drop)
    test_genes = test_df.drop(columns=cols_to_drop)
    
    scaler = None

    # Check for existing global scaler
    if scaler_path.exists():
        print(f"✅ Loading GLOBAL scaler for {phase} (scaled): {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        print(f"🚀 No scaler found for {phase}. Fitting NEW global scaler...")
        scaler = StandardScaler()
        # Only fit on training genes to prevent data leakage from the test set
        scaler.fit(train_genes)
        
        # Save it to the phase root so other theta experiments can use it
        os.makedirs(scaler_path.parent, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"💾 Global scaler saved to: {scaler_path}")

    # Transform data
    train_scaled = scaler.transform(train_genes)
    test_scaled = scaler.transform(test_genes)
    
    return (
        torch.tensor(train_scaled, dtype=torch.float32),
        torch.tensor(test_scaled, dtype=torch.float32),
        scaler
    )

def inverse_scale(scaler, tensor):
    """Converts a tensor back to original units."""

    if scaler is None: return tensor
    print('############### IM HERE TO INVERSE SCALE ###############')
    array = tensor.detach().cpu().numpy()
    unscaled = scaler.inverse_transform(array)
    return torch.tensor(unscaled, dtype=torch.float32)

def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures each patient is represented only once by randomly 
    selecting one sample per patient ID (prefix before '_').
    """
    patient_ids = df.index.to_series().apply(lambda x: x.split('_')[0])
    return df.loc[patient_ids.groupby(patient_ids).apply(lambda g: g.index[0])]

def get_theta_cols(train_df, test_df):
    theta_cols = [col for col in train_df.columns if 'theta' in col]
    # Separate genes from theta
    train_theta = torch.tensor(train_df[theta_cols].values, dtype=torch.float32)
    test_theta = torch.tensor(test_df[theta_cols].values, dtype=torch.float32)
    return train_theta, test_theta

def prepare_and_align_data(gene_path, theta_path=None, mode="true"):
    """
    Loads data and aligns indices. 
    If theta_path is None, it treats data as Healthy (Theta=0).
    """
    df_genes = pd.read_csv(gene_path, index_col=0).T
    
    # Remove technical duplicate patient samples
    df_genes = clean_rows(df_genes)
    print(f"   -> After clean_rows: {df_genes.shape}")
    if theta_path:
        # Combined DF with theta as last column
        # df_genes['theta_value'] = df_theta.iloc[:, 0]
        if mode in ["fixed", "true"]:
            df_theta = pd.read_csv(theta_path, index_col=0)
            print(f"theta {mode} used from path {theta_path}")
            # Align indices
            common_idx = df_genes.index.intersection(df_theta.index)
            df_genes = df_genes.loc[common_idx]
            df_theta = df_theta.loc[common_idx]
            df_genes['theta_value'] = df_theta.iloc[:, 0] # True values

        elif mode == 'random':
            print(">>> [EXPERIMENT] Overwriting real thetas with rnad values U(0,1)")
            df_genes['theta_value'] = np.random.rand(len(df_genes))
        else:
            raise ValueError(f"what mode are we at: {mode} and why did we not read path well")

    else:
        # Healthy data: Add a column of zeros for theta
        df_genes['theta_value'] = 0.0
        
    return df_genes

def load_and_prep_tensors(phase, mode, scale_bool, is_mixed):
    """
    Unified Pipeline: 
    1. Loads (Healthy/Disease/Mixed) 
    2. Aligns with Theta (Mode: true/fixed/random)
    3. Splits (Tournament Paths)
    4. Scales (Global Scaler check)
    5. Tensorizes
    """
    tag = "scaled" if scale_bool else "unscaled"
    
    # 1. Load Core Data
    if phase == "healthy":
        # Healthy only case
        df_target = prepare_and_align_data(cfg.HEALTHY_GENES_PATH, theta_path=None)

    else:
        # Disease case (might be mixed with healthy)
        df_d = prepare_and_align_data(cfg.get_disease_gene_path(mode), theta_path=cfg.get_theta_path(mode), mode=mode)
        
        if is_mixed:
            df_h = prepare_and_align_data(cfg.HEALTHY_GENES_PATH, theta_path=None)
            df_target = pd.concat([df_h, df_d])
        else:
            df_target = df_d
    df_target = df_target.fillna(0.0)
    # 2. Handle Splits
    split_path = cfg.get_split_path(phase=phase, scale_tag=tag, is_mixed=is_mixed)
    train_df, test_df = get_split_data(df_target, split_path=split_path)
    info_dict = {
        'test_df_full': test_df.copy(), # Keep everything (genes, theta, type)
        'train_df_full': train_df.copy()
    }
    # 3. Clean Metadata (Drop disease_type but KEEP theta_value)
    train_df = train_df.drop(columns=['disease_type'], errors='ignore')
    test_df = test_df.drop(columns=['disease_type'], errors='ignore')

    # 4. Handle Scaling & Tensorization
    if not scale_bool:
        # Straight to tensors
        train_t = torch.tensor(train_df.values, dtype=torch.float32)
        test_t = torch.tensor(test_df.values, dtype=torch.float32)
        return train_t, test_t, None, info_dict

    # Separate genes from theta for scaling
    train_theta, test_theta = get_theta_cols(train_df, test_df)
    
    # Scale only the genes via the smart fit_and_scale
    train_genes_scaled, test_genes_scaled, scaler = fit_and_scale(
        train_df, test_df, phase, is_mixed, mode
    )
    
    # 5. Recombine [Genes | Theta]
    train_tensor = torch.cat([train_genes_scaled, train_theta], dim=1)
    test_tensor = torch.cat([test_genes_scaled, test_theta], dim=1)
    
    return train_tensor, test_tensor, scaler, info_dict

def get_split_data(df, split_path, test_size=0.2, seed=42):
    """
    Reproducible split: loads from JSON or creates a new one.
    """
    print(f"\n📂 [DEBUG] Checking Split Path: {split_path}")
    print(f"   -> DataFrame shape arriving at split: {df.shape}")
    if split_path and os.path.exists(split_path):
        print(f"   -> NOTE: Found an existing split file at this path!")
        with open(split_path, "r") as f:
            splits = json.load(f)

        print(f"--> Loaded split from {split_path}")
        return df.loc[splits["train_ids"]], df.loc[splits["test_ids"]]
    
    if 'disease_type' in df.columns:
        # like "healthy", "cancer_A", "cancer_B". we have in numbers 0 1 2
        strat_labels = df['disease_type']
    else:
        # Fallback: Just distinguish Healthy (theta=0) from Disease (theta>0)
        strat_labels = (df['theta_value'] > 0).astype(int)

    # New Split
    train_ids, test_ids = train_test_split(
        df.index.tolist(),
        test_size=test_size,
        random_state=seed,
        stratify=strat_labels
    )
    
    # Save for future runs
    if split_path:
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        with open(split_path, "w") as f:
            json.dump({"train_ids": train_ids, "test_ids": test_ids}, f, indent=2)
        print(f"--> Saved split to {split_path}")
        
    train_df = df.loc[train_ids]
    test_df = df.loc[test_ids]
    # Validation Print
    print("\n--- [Data Split Audit] ---")
    for name, df_s in [("Train", train_df), ("Test", test_df)]:
        h_count = (df_s['theta_value'] == 0).sum()
        d_count = (df_s['theta_value'] > 0).sum()
        print(f"{name} Set: {h_count} Healthy, {d_count} Disease (Ratio: {d_count/len(df):.2%})")


    return df.loc[train_ids], df.loc[test_ids]


def get_ready_tensors(gene_path, split_path=None, use_scaling=None, theta_path=None, mode="true", phase="healthy", is_mixed=False, theta=""):
    """
    Final Pipeline: Align -> Split -> Scale -> Tensor.
    Returns: (train_tensor, test_tensor, scaler)
    Used when we load the data too before readying the tensors
    """

    df_full = prepare_and_align_data(gene_path, theta_path, mode=mode)
    train_df, test_df = get_split_data(df_full, split_path)
    # update_sample_metadata(cfg.LOG_PATH, gene_path, train_df, test_df, mode)
    train_df = train_df.drop(columns=['disease_type'], errors='ignore')
    test_df = test_df.drop(columns=['disease_type'], errors='ignore')

    if not use_scaling:
        # We must use .values and specify dtype to create a valid PyTorch Tensor
        train_t = torch.tensor(train_df.values, dtype=torch.float32)
        test_t = torch.tensor(test_df.values, dtype=torch.float32)
        return train_t, test_t, None
    
    # Separate genes from theta
    train_theta, test_theta = get_theta_cols(train_df, test_df)

    
    # Scale only the genes
    train_genes_scaled, test_genes_scaled, scaler = fit_and_scale(train_df, test_df, phase, is_mixed, theta)
    
    # Combine back to [Genes | Theta]
    train_tensor = torch.cat([train_genes_scaled, train_theta], dim=1)
    test_tensor = torch.cat([test_genes_scaled, test_theta], dim=1)
    return train_tensor, test_tensor, scaler
    
def get_ready_tensors_df(train_df, test_df, use_scaling=None, phase="disease", is_mixed=False, theta=""):
    """
    function used when we had to fix df data beforehand (like when combining the healthy and disease)
    so we just pass the df without needing to load them
    """
    train_df = train_df.drop(columns=['disease_type'], errors='ignore')
    test_df = test_df.drop(columns=['disease_type'], errors='ignore')

    if not use_scaling:
        # We must use .values and specify dtype to create a valid PyTorch Tensor
        train_t = torch.tensor(train_df.values, dtype=torch.float32)
        test_t = torch.tensor(test_df.values, dtype=torch.float32)
        return train_t, test_t, None
    
    train_theta, test_theta = get_theta_cols(train_df, test_df)
    
    
    # Scale only the genes
    train_genes_scaled, test_genes_scaled, scaler = fit_and_scale(train_df, test_df, phase, is_mixed, theta)
    
    # Combine back to [Genes | Theta]
    train_tensor = torch.cat([train_genes_scaled, train_theta], dim=1)
    test_tensor = torch.cat([test_genes_scaled, test_theta], dim=1)
    return train_tensor, test_tensor, scaler



############ added so i can use in new analyzer stuff from interpretability

def load_reconstruction_data(phase, mode):
    """
    Loads the validation data (Mixed Input and Clean Ground Truth).
    Matches your requested structure using config paths.
    """
    if phase == "healthy":
        mix_file = cfg.HEALTHY_GENES_PATH  # Input is pure healthy data
        truth_file = cfg.HEALTHY_GENES_PATH
    else:
        mix_file =cfg.get_disease_gene_path(mode)  # Input is mixed data
        truth_file = cfg.get_data_dir() / 'pure_disease_truth.csv' # Truth is pure disease
    print(f"truth_file: {truth_file}")
    print(f"mix file: {mix_file}")
    # 2. Validation
    if not mix_file.exists():
        print(f"⚠️ Warning: Reconstruction data not found:\n {mix_file}")
        return None, None
    if not truth_file.exists():
        print(f"⚠️ Warning: truth file data not found:\n {truth_file}")
        return None, None
        
    # Load & Transpose (Genes should be columns for the model)
    # Using 'T' because typically gene files are (Genes x Samples), but models expect (Samples x Genes)
    df_mixed = pd.read_csv(mix_file, index_col=0).T
    df_pure  = pd.read_csv(truth_file, index_col=0).T
    
    return df_mixed, df_pure

def choose_sig_list(phase, disease_type):
    basic_l = ["Megakaryocyte", "Neutrophils"]
    if phase == "synthetic_complex":

        match disease_type:
            case "Healthy":
                return basic_l
            case "DiseaseA": #crc
                return ["Colon", "Rectum", "Colorectal", "Digestive", "GI", "GI Mucosa"] + basic_l
            case "DiseaseB":
                return ["Lung"] + basic_l
