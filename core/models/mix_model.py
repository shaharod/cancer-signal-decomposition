import torch
import torch.nn as nn

class UniversalMixModel(nn.Module):
    def __init__(self, healthy_comp: nn.Module, disease_comp: nn.Module):
        super().__init__()
        self.is_mix_model = True
        self.healthy = healthy_comp
        self.disease = disease_comp
        self.in_features = healthy_comp.in_features
        self.has_trainable_disease = any(p.requires_grad for p in disease_comp.parameters())

        # Guard: Ensure input dimensions match
        if self.healthy.in_features != self.disease.in_features:
            raise ValueError(f"Dim mismatch: Healthy({self.healthy.in_features}) != Disease({self.disease.in_features})")
        
        # Always freeze the healthy component
        for param in self.healthy.parameters():
            param.requires_grad = False
        self.healthy.eval()

    def _get_x(self, x_combined):
        """Standardizes input: always returns just the gene features."""
        print(f"x_combined shape is: {x_combined.shape}")
        # print(f"in_features of {self} is: {self.in_features}")
        if x_combined.shape[1] > self.in_features:
            return x_combined[:, :-1]
        return x_combined

    def forward(self, x_combined: torch.Tensor):
        x = self._get_x(x_combined)
        theta = x_combined[:, -1:]
        print(f"x size is {x.shape}")
        # # x_combined might be [genes | theta] or just [genes]
        # if x_combined.shape[1] > self.in_features:
        #     raise ValueError(f"why am i here")
        # Both components return (recon, latent)
        x_hat_h, _ = self.healthy(x)
        x_hat_d, z_d = self.disease(x)
        
        # Apply mixing: x_mix = theta * disease + (1 - theta) * healthy
        x_hat_mix = (theta * x_hat_d) + ((1 - theta) * x_hat_h)
        # --- DEBUG PRINT: HEALTHY SAMPLE RECONSTRUCTION ---
        # Find indices where theta is exactly 0
        healthy_indices = (theta == 0).nonzero(as_tuple=True)[0]

        if healthy_indices.numel() > 0:
            # Pick the first healthy sample found in this batch
            idx = healthy_indices[0]
            
            print(f"\n--- [DEBUG] Internal Values for Healthy Sample (Index {idx}) ---")
            print(f"Theta: {theta[idx].item()}")
            
            # Check a few genes from the Healthy Module (0-499)
            # They should be ~100 in x_hat_h and ~0 in x_hat_d
            print(f"Healthy Branch - Genes 0-5: {x_hat_h[idx, :5].detach().cpu().numpy()}")
            print(f"Disease Branch - Genes 0-5: {x_hat_d[idx, :5].detach().cpu().numpy()}")
            
            # Check a few genes from the Disease Module (500-999)
            # They should be ~0 in BOTH branches for a healthy sample
            print(f"Healthy Branch - Genes 500-505: {x_hat_h[idx, 500:505].detach().cpu().numpy()}")
            print(f"Disease Branch - Genes 500-505: {x_hat_d[idx, 500:505].detach().cpu().numpy()}")
            print("-----------------------------------------------------------\n")
        return x_hat_mix, x_hat_d, x_hat_h, z_d # NOTE i added healthy part to return, might lead to memory problems but shouldnt crash anything
    
    def get_latents(self, x_input):
        """
        Returns both the healthy representation and the disease-specific encoding.
        """
        x = self._get_x(x_input)
        # Pass through the frozen healthy part
        with torch.no_grad():
            h_latent = self.healthy.encoder(x)
            d_latent = self.disease.encoder(x)
            
        return h_latent, d_latent