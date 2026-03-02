from pathlib import Path
import os

# ---- Project Root ----
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / 'data'


# ---- Global Decisions ----
RANDOM_THETA_EXP = False
FIXED_THETA_EXP = True
SYNTHETIC_DATA = True
SYN_SIMPLE = False
SYN_CMPLX = not SYN_SIMPLE
DEVICE = 'cpu' # 'cuda' for Windows/Linux with NVIDIA, 'mps' for macOS

def get_theta_mode():
    if FIXED_THETA_EXP:
        return "fixed"
    if RANDOM_THETA_EXP: 
        return "random"
    return "true"

def get_disease_gene_path(mode_val):
    # mode_val = get_theta_mode()
    path = DATA_SUB
    if mode_val == "true":
        return path/ "disease_data_uniform_theta.csv" if SYNTHETIC_DATA else path / "GeneMatrix_H3K4me3_crc.csv"
    if mode_val == "fixed" and not SYNTHETIC_DATA:
        raise ValueError("Still havent fixed this")
    if mode_val == "fixed":
        return path /"disease_data_theta05.csv"


# ----- Input Data Paths ------
DATA_SUB = DATA_PATH 
if not SYNTHETIC_DATA:
    DATA_SUB = DATA_SUB / 'real'
else:
    if SYN_SIMPLE:
        DATA_SUB = DATA_SUB / 'synthetic_simple'
    elif SYN_CMPLX:
        DATA_SUB = DATA_SUB / 'synthetic_complex'


if SYNTHETIC_DATA:
    HEALTHY_GENES_PATH = DATA_SUB / "healthy_data.csv"
    # DISEASE_GENES_PATH = DATA_SUB / ("disease_data_theta05.csv" if FIXED_THETA_EXP else "disease_data_uniform_theta.csv")
    THETA_PATH         = DATA_SUB / "theta_values.csv"
else:
    HEALTHY_GENES_PATH = DATA_SUB / "GeneMatrix_H3K4me3_healthy.csv"
    # DISEASE_GENES_PATH = DATA_SUB / "GeneMatrix_H3K4me3_crc.csv"
    THETA_PATH         = DATA_SUB / "theta_CRC_passedQC.csv"
# DISEASE_GENES_PATH = get_disease_gene_path()

SYN_MIX_DISEASE_PART = DATA_SUB / "pure_disease_truth.csv"
SYN_MIX_HEALTHY_PART = DATA_SUB / "healthy_mix_basis.csv"
SIG_PATH = DATA_SUB / "SIGpassClinicalQC_H3K4me3.csv"

    
# ---- Hyperparameters ----
EPOCHS_NUM     = 300
BATCH_SIZE     = 32
LR             = 0.001
EPOCH_JUMP     = 5

def choose_enc():
    if not SYNTHETIC_DATA or (SYNTHETIC_DATA and SYN_CMPLX):
        return [16, 32, 64, 128], 512, 128
    else:
        return [8, 16], 32, 16
    
ENCODING_SIZES, H1, H2 = choose_enc() # [16, 32, 64, 128] if not SYNTHETIC_DATA else [8, 16]


# Layered AE architecture settings
# H1 = 512 if not SYNTHETIC_DATA else 32
# H2 = 128 if not SYNTHETIC_DATA else 16

# ---- Tournament Logic Helpers ----
SCALING_OPTIONS = [True, False]
MODEL_TYPES     = ['ae_basic', 'ae_layered']
SIG_LIST        = ['Megakaryocyte', 'Neutrophils']



# ---- Output Directories ----

# constants - subfolders names
MODELS_SUBFOLDER = 'trained_models'
PLOTS_SUBFOLDER = 'plots'

# base
BASE_EXP_DIR = PROJECT_ROOT / 'outputs' / ('synthetic_experiments' if SYNTHETIC_DATA else 'real_experiments')

# sub-folders types
HEALTHY_OUT_DIR = BASE_EXP_DIR / 'healthy'
DISEASE_OUT_DIR = BASE_EXP_DIR / 'disease_mix'


# latent representation
LATENT_ROOT = PROJECT_ROOT / 'latent'

def get_path(phase, scale_tag=None, model_type=None, enc=None, folder_type=MODELS_SUBFOLDER, is_mixed=False):
    # # getting phase root
    # if not is_mixed:
    #     return get_path(phase, scale_tag, model_type, enc, folder_type)
    if phase == "healthy":
        root = HEALTHY_OUT_DIR

    elif phase == "disease":
        base_dir = BASE_EXP_DIR / 'disease_mix_all' if is_mixed else BASE_EXP_DIR / 'disease_mix'
        if FIXED_THETA_EXP:
            root = base_dir / 'disease_mix_fixed_0.5'
        elif RANDOM_THETA_EXP:
            root = base_dir / 'disease_mix_random_theta'
        else:
            theta_type = 'uniform' if SYNTHETIC_DATA else 'true'
            root = base_dir / f'disease_mix_{theta_type}_theta'
    else:
        raise ValueError(f"What is the phase we passed??? - {phase}. I don't know you!")
    # usage category (Models vs Plots)
    root = root / folder_type
    
    # creating sub-folder path
    if scale_tag and model_type and enc:
        path = root / scale_tag / model_type / f"enc_{enc}"
    else:
        path = root

    os.makedirs(path, exist_ok=True)
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

        



