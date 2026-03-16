import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import joblib
import os
import json
from sklearn.preprocessing import StandardScaler
import config as cfg

def fit_and_scale2(train_df, test_df):
    """Scaler fitted only on training genes."""
    scaler = StandardScaler()
    # Remove theta and disease type if there is before scaling
    
    train_genes = train_df.drop(columns=['theta_value'])
    test_genes = test_df.drop(columns=['theta_value'])
    
    train_scaled = scaler.fit_transform(train_genes)
    test_scaled = scaler.transform(test_genes)
    
    return (
        torch.tensor(train_scaled, dtype=torch.float32),
        torch.tensor(test_scaled, dtype=torch.float32),
        scaler
    )

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
    
    # We must drop theta so the scaler doesn't treat it as a gene feature
    train_genes = train_df.drop(columns=['theta_value'])
    test_genes = test_df.drop(columns=['theta_value'])
    
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
        if mode == 'random':
            print(">>> [EXPERIMENT] Overwriting real thetas with rnad values U(0,1)")
            df_genes['theta_value'] = np.random.rand(len(df_genes))
        elif mode == 'fixed':
            print(">>> [EXPERIMENT] Overwriting real thetas with fixed 0.5 values U(0,1)")
            df_genes['theta_value'] = 0.5 # Every sample is a "perfect mix"
        else:
            print("real thetas are used")
            df_theta = pd.read_csv(theta_path, index_col=0)
            # Align indices
            common_idx = df_genes.index.intersection(df_theta.index)
            df_genes = df_genes.loc[common_idx]
            df_theta = df_theta.loc[common_idx]
            df_genes['theta_value'] = df_theta.iloc[:, 0] # True values

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
        df_d = prepare_and_align_data(cfg.get_disease_gene_path(mode), theta_path=cfg.THETA_PATH, mode=mode)
        
        if is_mixed:
            df_h = prepare_and_align_data(cfg.HEALTHY_GENES_PATH, theta_path=None)
            df_target = pd.concat([df_h, df_d])
        else:
            df_target = df_d

    # 2. Handle Splits
    split_path = cfg.get_split_path(phase=phase, scale_tag=tag, is_mixed=is_mixed)
    train_df, test_df = get_split_data(df_target, split_path=split_path)
    info_dict = {
        'test_df_full': test_df.copy().fillna(0.0), # Keep everything (genes, theta, type)
        'train_df_full': train_df.copy().fillna(0.0)
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
    train_theta = torch.tensor(train_df[['theta_value']].values, dtype=torch.float32)
    test_theta = torch.tensor(test_df[['theta_value']].values, dtype=torch.float32)
    
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
    dropped_cols_train = torch.tensor(train_df[['theta_value']].values, dtype=torch.float32)
    dropped_cols_test = torch.tensor(test_df[['theta_value']].values, dtype=torch.float32)
    
    # Scale only the genes
    train_genes_scaled, test_genes_scaled, scaler = fit_and_scale(train_df, test_df, phase, is_mixed, theta)
    
    # Combine back to [Genes | Theta]
    train_tensor = torch.cat([train_genes_scaled, dropped_cols_train], dim=1)
    test_tensor = torch.cat([test_genes_scaled, dropped_cols_test], dim=1)
    return train_tensor, test_tensor, scaler
    
def get_ready_tensors_df(train_df, test_df, use_scaling=None, phase="disease", is_mixed=False, theta=""):
    train_df = train_df.drop(columns=['disease_type'], errors='ignore')
    test_df = test_df.drop(columns=['disease_type'], errors='ignore')

    if not use_scaling:
        # We must use .values and specify dtype to create a valid PyTorch Tensor
        train_t = torch.tensor(train_df.values, dtype=torch.float32)
        test_t = torch.tensor(test_df.values, dtype=torch.float32)
        return train_t, test_t, None
    
    # Separate genes from theta
    train_theta = torch.tensor(train_df[['theta_value']].values, dtype=torch.float32)
    test_theta = torch.tensor(test_df[['theta_value']].values, dtype=torch.float32)
    
    # Scale only the genes
    train_genes_scaled, test_genes_scaled, scaler = fit_and_scale(train_df, test_df, phase, is_mixed, theta)
    
    # Combine back to [Genes | Theta]
    train_tensor = torch.cat([train_genes_scaled, train_theta], dim=1)
    test_tensor = torch.cat([test_genes_scaled, test_theta], dim=1)
    return train_tensor, test_tensor, scaler

def fix_df_data(scale_bool, mode, is_mixed):
    """
    function to deal with disease data, it returns split data with theta, without disease type
    """
    tag = "scaled" if scale_bool else "unscaled"
    
    # 1. Load the core Disease Data
    df_d = prepare_and_align_data(cfg.get_disease_gene_path(mode), theta_path=cfg.THETA_PATH, mode=mode)
    
    if is_mixed:
        # Scenario A: Mixed Dataset (Healthy + Disease)
        df_h = prepare_and_align_data(cfg.HEALTHY_GENES_PATH, theta_path=None)
        df_target = pd.concat([df_h, df_d]) #.sample(frac=1, random_state=42)
    else:
        # Scenario B: Disease Samples Only
        df_target = df_d
    
    # 2. Get the correct split path based on the is_mixed flag
    tournament_split_path = cfg.get_split_path(
        phase="disease", 
        scale_tag=tag, 
        is_mixed=is_mixed # This ensures you use the correct split file
    )
    
    # 3. Get the train/test split
    train_df, test_df = get_split_data(df_target, split_path=tournament_split_path)

    train_df = train_df.drop(columns=['disease_type'], errors='ignore')
    test_df = test_df.drop(columns=['disease_type'], errors='ignore')
    
    return train_df, test_df

# def update_sample_metadata(log_path, gene_path, train_df, test_df, mode):
    """
    Saves counts into a single JSON file.
    Key: Filename + Mode
    """
    # Get just the filename (e.g., 'BRCA_data' from 'path/to/BRCA_data.csv')
    dataset_name = os.path.splitext(os.path.basename(gene_path))[0]
    entry_key = f"{dataset_name}_{mode}"

    stats = {}
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            stats = json.load(f)

    # Calculate Healthy vs Disease based on theta_value
    # (Assuming 0 is healthy, >0 is disease)
    total_df = pd.concat([train_df, test_df])
    healthy_count = int((total_df['theta_value'] == 0).sum())
    disease_count = int((total_df['theta_value'] > 0).sum())

    stats[entry_key] = {
        "Train": len(train_df),
        "Test": len(test_df),
        "Healthy": healthy_count,
        "Disease": disease_count,
        "Total": len(total_df)
    }

    with open(log_path, "w") as f:
        json.dump(stats, f, indent=4)
