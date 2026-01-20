import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA
from .ae_architectures import Basic_AE, Layered_AE
from .components import PCAComponent, AEComponent
from .mix_model import UniversalMixModel

class ModelFactory:
    @staticmethod
    def create_model(model_type, input_size, encoding_size, h1, h2):
        """
        Creates a standalone Autoencoder model.
        Used for Phase 1 (Healthy) and the Disease-side of Phase 2.
        """
        if model_type == "pca":
            pca_obj = PCA(n_components=encoding_size)
            pca_obj.mean_ = np.zeros(input_size)
            pca_obj.components_ = np.zeros((encoding_size, input_size))
            pca_obj.n_components_ = encoding_size
            return PCAComponent(pca_obj)
        if model_type == "ae_basic":
            return Basic_AE(input_size, encoding_size)
        elif model_type == "ae_layered":
            # Note: H1 and H2 are usually pulled from config
            return Layered_AE(input_size, encoding_size, h1, h2)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def wrap_component(obj):
        """
        Standardizes different model types into a common 'Component' interface.
        This ensures both PCA and AE have a .forward() that returns (recon, latent).
        """
        # If it's an sklearn PCA object
        if isinstance(obj, PCA):
            return PCAComponent(obj)
        
        # If it's a PyTorch AE model
        if isinstance(obj, nn.Module):
            # If it's already wrapped, don't double-wrap
            if isinstance(obj, (PCAComponent, AEComponent)):
                return obj
            return AEComponent(obj)
            
        raise ValueError(f"Cannot wrap object of type {type(obj)}. Expected PCA or nn.Module.")

    @classmethod
    def create_mix_model(cls, healthy_obj, disease_obj, freeze_healthy=True):
        """
        The Orchestrator for Phase 2.
        Takes a healthy baseline (PCA or AE) and a disease AE, 
        wraps them, freezes the healthy side, and returns a UniversalMixModel.
        """
        # 1. Standardize both sides
        h_comp = cls.wrap_component(healthy_obj)
        d_comp = cls.wrap_component(disease_obj)
        
        # 2. Freeze the Healthy Component
        # This is vital so the 'Normal' baseline doesn't shift during training
        if freeze_healthy:
            for param in h_comp.parameters():
                param.requires_grad = False
            h_comp.eval() 
            
        # 3. Return the integrated Mix Model
        return UniversalMixModel(h_comp, d_comp)