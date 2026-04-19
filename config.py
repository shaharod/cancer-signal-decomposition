from pathlib import Path
import os

# ---- Project Root ----
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / 'data'

## --------------- EXPERIMENT CONTROLS ----------------- ##

# Choose Data Source: 'real' or 'synthetic'
DATA_SOURCE = 'real'  

# Choose Complexity (Only applies if DATA_SOURCE is 'synthetic'): 'simple' or 'complex'
SYNTHETIC_TYPE = 'complex' 

# Choose Variant. 
# Synthetic Options: 'dif_dp', 'dif_hp', 'theta_0.1', 'theta_0.01', 'theta_0.001', 'theta_0.005', 'theta_lim_0.7', 'theta_no_lim'
# Real Options: 'basic', 'none'
VARIANT = 'basic' 

# Choose Theta Mode: 'true', 'fixed', 'random'
THETA_EXP_MODE = 'true' 

DEVICE = 'cpu' # 'cuda', 'mps'

# Helper boolean for the rest of your app to maintain compatibility
SYNTHETIC_DATA = (DATA_SOURCE == 'synthetic')

def build_data_sub_path() -> Path:
    """Dynamically routes the data path based on the high-level controls."""
    base = DATA_PATH / DATA_SOURCE # e.g., data/synthetic or data/real
    
    if SYNTHETIC_DATA:
        # e.g., data/synthetic/synthetic_complex/dif_dp
        return base / f"synthetic_{SYNTHETIC_TYPE}" / VARIANT
    else:
        # e.g., data/real/real_basic
        return base / f"real_{VARIANT}" if VARIANT != 'none' else base

DATA_SUB = build_data_sub_path()

## names of these paths are the same for synthetic and real data, after row cleaning
HEALTHY_GENES_PATH = DATA_SUB / "healthy_data.csv"
THETA_PATH         = DATA_SUB / "theta_values.csv"

SIG_PATH = DATA_SUB / "SIGpassClinicalQC_H3K4me3.csv"

def get_theta_path(mode):
    if not SYNTHETIC_DATA:
        return DATA_SUB / 'theta_values.csv'
    match mode:
        case 'true': return DATA_SUB / "theta_values.csv"
        case 'fixed': return DATA_SUB / "theta_values05.csv"
        case _: raise ValueError(f"Unknown mode: {mode}")

def get_disease_gene_path(mode_val):
    if mode_val == "true":
        return DATA_SUB / ("disease_data_uniform_theta.csv" if SYNTHETIC_DATA else "disease_data.csv")
    elif mode_val == "fixed":
        if not SYNTHETIC_DATA: raise ValueError("Still haven't fixed this for real data")
        return DATA_SUB / "disease_data_theta05.csv"
    

def get_data_dir():
    return DATA_SUB

def change_data_dir(suffix):
    DATA_SUB = DATA_PATH
    if not SYNTHETIC_DATA:
        DATA_SUB = DATA_SUB / 'real'
    
# ---- Hyperparameters ----
EPOCHS_NUM     = 300
BATCH_SIZE     = 32
LR             = 0.001
EPOCH_JUMP     = 5

def choose_enc_layers():
    if not SYNTHETIC_DATA or (SYNTHETIC_DATA and SYNTHETIC_TYPE == 'complex'):
        # return [16, 32, 64, 128], 512, 128
        return [2, 4, 8, 16], 512, 64 #, 4, 8, 16
    else:
        return [8, 16], 32, 16

ENCODING_SIZES, H1, H2 = choose_enc_layers() 

SCALING_OPTIONS = [False, True]
MODEL_TYPES     = ['ae_basic', 'ae_layered']
SIG_LIST        = ['Megakaryocyte', 'Neutrophils']

# ---- Output Directories ----

MODELS_SUBFOLDER = 'trained_models'
PLOTS_SUBFOLDER = 'plots'

def get_experiment_suffix() -> str:
    """Dynamically grabs the correct suffix for the output folder."""
    if not SYNTHETIC_DATA:
        return f"_{VARIANT}" if VARIANT != 'none' else ""
    return f"_{VARIANT}"

# base
_base_name = 'synthetic_experiments' if SYNTHETIC_DATA else 'real_experiments'
BASE_EXP_DIR = PROJECT_ROOT / 'outputs' / f"{_base_name}{get_experiment_suffix()}"

HEALTHY_OUT_DIR = BASE_EXP_DIR / 'healthy'
DISEASE_OUT_DIR = BASE_EXP_DIR / 'disease_mix'
LATENT_ROOT = PROJECT_ROOT / 'latent'
  
# sub-folders types
HEALTHY_OUT_DIR = BASE_EXP_DIR / 'healthy'
DISEASE_OUT_DIR = BASE_EXP_DIR / 'disease_mix'


def get_path(phase, scale_tag=None, model_type=None, enc=None, folder_type=MODELS_SUBFOLDER, is_mixed=False):
    if phase == "healthy":
        root = HEALTHY_OUT_DIR
    elif phase == "disease":
        base_dir = BASE_EXP_DIR / ('disease_mix_all' if is_mixed else 'disease_mix')
        
        if THETA_EXP_MODE == 'fixed':
            root = base_dir / 'disease_mix_fixed_0.5'
        elif THETA_EXP_MODE == 'random':
            root = base_dir / 'disease_mix_random_theta'
        else:
            theta_type = 'uniform' if SYNTHETIC_DATA else 'true'
            root = base_dir / f'disease_mix_{theta_type}_theta'
    else:
        raise ValueError(f"Unknown phase: {phase}")
        
    root = root / folder_type
    
    if scale_tag and model_type and enc:
        path = root / scale_tag / model_type / f"enc_{enc}"
    else:
        path = root

    path.mkdir(parents=True, exist_ok=True)
    return path

def get_split_path(phase, scale_tag, is_mixed):
    """
    Different than regular one, in the case of disease where we train with
    both healthy and disease samples or only disease. choose between disease_mix 
    and disease_mix_all
    """

    root = get_path(phase, folder_type=MODELS_SUBFOLDER, is_mixed=is_mixed)
    split_dir = root / "splits"
    os.makedirs(split_dir, exist_ok=True)
    return split_dir / f"split_{scale_tag}.json"

        



