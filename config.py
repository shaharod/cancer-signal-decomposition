from pathlib import Path
import os

# TODO: Add missing logic:
# - try to add logs in training, like keeping track of how long it took, what were the values that were used, etc.
# - decide on path for that

# FIXME: theta issue
# - i remember three thetas checks, i see two: 0.5 and uniform/ true if its in real experiments
# also, why is theta 0.5 being tested in real experiments and not only in synthetic data


# ---- Project Root ----
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / 'data'


# ---- Global Decisions ----
RANDOM_THETA_EXP = False
FIXED_THETA_EXP = False
SYNTHETIC_DATA = False
DEVICE = 'mps' # 'cuda' for Windows/Linux with NVIDIA, 'mps' for macOS



# ----- Input Data Paths ------
DATA_SUB = DATA_PATH / ('synthetic' if SYNTHETIC_DATA else 'real')

if SYNTHETIC_DATA:
    HEALTHY_GENES_PATH = DATA_SUB / "healthy_data.csv"
    DISEASE_GENES_PATH = DATA_SUB / ("disease_data_theta05.csv" if FIXED_THETA_EXP else "disease_data_uniform_theta.csv")
    THETA_PATH         = DATA_SUB / "syn_theta_values.csv"
else:
    HEALTHY_GENES_PATH = DATA_SUB / "GeneMatrix_H3K4me3_healthy.csv"
    DISEASE_GENES_PATH = DATA_SUB / "GeneMatrix_H3K4me3_crc.csv"
    THETA_PATH         = DATA_SUB / "theta_CRC_passedQC.csv"

SIG_PATH = DATA_SUB / "SIGpassClinicalQC_H3K4me3.csv"

# ---- Hyperparameters ----
EPOCHS_NUM     = 300
BATCH_SIZE     = 32
LR             = 0.001
EPOCH_JUMP     = 5
ENCODING_SIZES = [16, 32, 64, 128] if not SYNTHETIC_DATA else [8, 16]


# Layered AE architecture settings
H1 = 512 if not SYNTHETIC_DATA else 32
H2 = 128 if not SYNTHETIC_DATA else 16

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



def get_path(phase, scale_tag=None, model_type=None, enc=None, folder_type=MODELS_SUBFOLDER):

    # getting phase root
    if phase == "disease":

        if FIXED_THETA_EXP:
            root = DISEASE_OUT_DIR / 'disease_mix_fixed_0.5'
        elif RANDOM_THETA_EXP:
            root = DISEASE_OUT_DIR / 'disease_mix_random_theta'
        else:
            theta_type = 'uniform' if SYNTHETIC_DATA else 'true'
            root = DISEASE_OUT_DIR / f'disease_mix_{theta_type}_theta'
    else:
        root = HEALTHY_OUT_DIR
    
    # usage category (Models vs Plots)
    root = root / folder_type
    
    # creating sub-folder path
    if scale_tag is None or model_type is None or enc is None:
        path = root
    else:
        path = root / scale_tag / model_type / f"enc_{enc}"

    os.makedirs(path, exist_ok=True)
    return path


def get_split_path(phase, scale_tag):
    """
    Ensures all models in a tournament share the same split for fairness.
    """

    # splits are always saved under 'trained_models'
    root = get_path(phase, folder_type=MODELS_SUBFOLDER)
    split_dir = root / "splits"

    os.makedirs(split_dir, exist_ok=True)
    return split_dir / f"split_{scale_tag}.json"

