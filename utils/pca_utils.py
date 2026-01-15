import numpy as np
import torch
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
import torch
import numpy as np

def train_single_pca(data_input, n_components):
    """
    Fits a single PCA model on the provided data.
    Ensures data is in NumPy format for sklearn.
    """
    # Convert Torch Tensor to NumPy if necessary
    if torch.is_tensor(data_input):
        data_np = data_input.detach().cpu().numpy()
    else:
        data_np = np.array(data_input)
        
    pca = PCA(n_components=n_components)
    pca.fit(data_np)
    return pca

def train_pca_collection(train_data, encoding_sizes):
    """
    Trains multiple PCA models. 
    train_tensor: [Samples, Genes + 1] - we slice out the Genes.
    Returns a dict {enc_size: pca_object}
    # """
    # # Slice out the theta column and convert to numpy for sklearn
    # train_np = train_tensor.detach().numpy()
    # 1. Convert to NumPy safely (sklearn needs NumPy, not Tensors)
    if torch.is_tensor(train_data):
        train_np = train_data.detach().cpu().numpy()
    elif hasattr(train_data, "values"):
        # This handles Pandas DataFrames
        train_np = train_data.values
    else:
        train_np = np.array(train_data)

    # 2. Slice to remove the theta column (last column)
    # We only want to train PCA on the genes
    train_features = train_np[:, :-1]

    models = {}
    for enc in encoding_sizes:
        pca = PCA(n_components=enc)
        pca.fit(train_features)
        models[enc] = pca
    return models

def get_pca_mse(pca_model, data_input, scaler=None):
    """
    Calculates reconstruction MSE.
    If scaler is provided, the result is in original unscaled units.
    """
    # 1. Prepare data (numpy, genes only)
    if torch.is_tensor(data_input):
        data_np = data_input.detach().cpu().numpy()
    elif hasattr(data_input, "values"):
        data_np = data_input.values
    else:
        data_np = np.array(data_input)

    # Slice out genes (exclude last theta column)
    genes_np = data_np[:, :-1]
    
    # 2. Reconstruct: X -> Latent -> X_hat
    latent = pca_model.transform(genes_np)
    recon_np = pca_model.inverse_transform(latent)
    
    # 3. Target
    target_np = genes_np
    
    # 4. Inverse Scale if necessary
    if scaler is not None:
        # We must use the scaler's inverse_transform on both
        target_np = scaler.inverse_transform(target_np)
        recon_np = scaler.inverse_transform(recon_np)
        
    mse = np.mean((target_np - recon_np) ** 2)
    return float(mse)

def save_pca_model(pca_model, folder, filename="model.joblib"):
    """Standardized save for PCA."""
    path = os.path.join(folder, filename)
    joblib.dump(pca_model, path)

def load_pca_model(path):
    """Standardized load for PCA."""
    return joblib.load(path)