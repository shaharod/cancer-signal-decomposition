import torch
import torch.nn as nn
import copy
from utils.data_utils import inverse_scale
from torch.utils.data import DataLoader, TensorDataset

class Trainer:
    def __init__(self, model, scaler=None, lr=1e-3, device=torch.device('cpu')):
        self.device = device
        self.model = model.to(self.device)
        self.scaler = scaler
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # collect trainable
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        if hasattr(model, 'is_mix_model'):
            # ff it's a MixModel, check if the Disease part was supposed to train
            if model.has_trainable_disease and len(trainable_params) == 0:
                raise ValueError(
                    "CRITICAL: Disease component has trainable layers, but no parameters "
                    "were passed to the optimizer! Did we freeze the model?"
                )
        
        # 
        if len(trainable_params) > 0:
            self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
            total_weights = sum(p.numel() for p in trainable_params)
            print(f">>> Trainer: Training mode ({total_weights:,} params)")
        else:
            self.optimizer = None
            print(">>> Trainer: Inference mode (Benchmarking PCA)")



    def _safe_forward(self, tensor):
        """Helper to handle MixModel vs Standard AE data flow."""

        # move input batch to gpu
        tensor = tensor.to(self.device)
        num_genes = self.model.in_features
        
        # Choosing input based on model type
        if hasattr(self.model, 'is_mix_model') and self.model.is_mix_model:
            outputs = self.model(tensor) # Sends 20,007
        else:
            outputs = self.model(tensor[:, :num_genes]) # Sends 20,006

        #flexible unpacking (handles recon or (recon, latent) cases)
        recon = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        return recon
    
    def get_mse(self, tensor):
        """
        Calculates MSE in the original (unscaled) domain.
        """
        self.model.eval()
        tensor = tensor.to(self.device)
        num_genes = self.model.in_features

        with torch.no_grad():
            # forward pass on gpu
            recon = self._safe_forward(tensor)
            target = tensor[:, :num_genes]  # exclude theta if present, target is always genes
            
            if self.scaler:
                recon = inverse_scale(self.scaler, recon.cpu())
                target = inverse_scale(self.scaler, target.cpu())
                
            return self.criterion(recon, target).item()


    def train_epoch(self, loader):
        """Standardized epoch loop."""
        self.model.train()
        num_genes = self.model.in_features

        for (batch,) in loader:
            # move batch to gpu
            batch = batch.to(self.device)
            target = batch[:, :num_genes]

            recon = self._safe_forward(batch)

            loss = self.criterion(recon, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Return loss on the whole training set (unscaled)
        return self.get_mse(loader.dataset.tensors[0])


    def fit(self, train_tensor, val_tensor, epochs, batch_size=32, jump=5):
        """
        Full training steps.
        Returns history and best_state.
        """
        loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
        history = {"train": [], "val": []}
        
        best_val = float('inf')
        best_state = None
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(loader)
            history["train"].append(train_loss)
            
            if epoch % jump == 0:
                val_loss = self.get_mse(val_tensor)
                history["val"].append(val_loss)
                
                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = epoch
                    # Save state dict on CPU for better portability
                    best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                    #best_state = copy.deepcopy(self.model.state_dict())

        return history, {"best_state": best_state, "best_val": best_val, "best_epoch": best_epoch}