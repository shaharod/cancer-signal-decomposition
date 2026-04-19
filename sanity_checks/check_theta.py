import torch
import pandas as pd
import config as cfg
from utils import data_utils


def check_theta_integrity():
    print(">>> STARTING THETA INTEGRITY CHECK\n")
    
    # 1. Load Real Data directly from CSV for comparison
    true_theta_df = pd.read_csv(cfg.THETA_PATH, index_col=0)
    
    # 2. Load Tensors using your pipeline (Experiment Mode)
    # Ensure cfg.RANDOM_THETA_EXP = True before this
    train_t, test_t, _ = data_utils.get_ready_tensors(
        cfg.get_disease_gene_path(mode_val="true"),
        theta_path=cfg.THETA_PATH,
        randomize_theta=True,
        use_scaling=True
    )

    # 3. Extract the theta column (the last one)
    random_thetas = train_t[:, -1].numpy()
    
    # --- CHECK A: Statistical Distribution ---
    print(f"Check A: Distribution of Randomized Theta")
    print(f"  - Mean: {random_thetas.mean():.4f} (Should be ~0.5)")
    print(f"  - Max:  {random_thetas.max():.4f}  (Should be ~1.0)")
    print(f"  - Min:  {random_thetas.min():.4f}  (Should be ~0.0)")
    
    # --- CHECK B: Direct Comparison ---
    # Pick a few sample IDs and compare tensor value vs CSV value
    print(f"\nCheck B: Direct Value Comparison")
    # Note: This assumes train_t rows match true_theta_df rows 
    # (Check your split_ids if this looks off)
    csv_sample = true_theta_df.iloc[0, 0]
    tensor_sample = random_thetas[0]
    
    print(f"  - Sample 0 | CSV: {csv_sample:.4f} | Tensor: {tensor_sample:.4f}")
    if abs(csv_sample - tensor_sample) < 1e-5:
        print("  !! ALERT: Tensor matches CSV. Randomization FAILED.")
    else:
        print("  >> SUCCESS: Tensor differs from CSV. Randomization confirmed.")


    # --- CHECK C: Correlation ---
    # If randomized, the correlation between real and fake should be ~0
    # This is the "Gold Standard" check.



check_theta_integrity()

