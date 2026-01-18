import torch
import torch.nn as nn

class PCAComponent(nn.Module):
    def __init__(self, pca_obj):
        super().__init__()
        self.in_features = pca_obj.mean_.shape[0]
        self.latent_dim = pca_obj.n_components_
        
        # Use register_buffer for non-trainable constants
        self.register_buffer("mean", torch.tensor(pca_obj.mean_, dtype=torch.float32))
        self.register_buffer("components", torch.tensor(pca_obj.components_, dtype=torch.float32))

    def encoder(self, x):
        # Explicit encoder method for get_latents()
        return torch.matmul(x - self.mean, self.components.t())

    def decoder(self, z):
        return torch.matmul(z, self.components) + self.mean

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

class AEComponent(nn.Module):
    def __init__(self, ae_model: nn.Module):
        super().__init__()
        self.ae = ae_model
        self.in_features = ae_model.in_features

    def encoder(self, x):
        # Routes directly to the internal AE encoder
        return self.ae.encoder(x)
    
    def decoder(self, z):
        return self.ae.decoder(z)

    def forward(self, x: torch.Tensor):
        # Standardizes all AE outputs to (recon, latent)
        # Assuming your Basic_AE/Layered_AE return (recon, latent)
        return self.ae(x)