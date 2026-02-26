import torch
import torch.nn as nn
from typing import Tuple

class Basic_AE(nn.Module):
    def __init__(self, input_size: int, encoding_size: int):
        super().__init__()
        self.in_features = input_size # Added for the Trainer to identify input width
        self.encoder = nn.Sequential(nn.Linear(input_size, encoding_size))
        self.decoder = nn.Sequential(nn.Linear(encoding_size, input_size)
                                     , nn.ReLU() 
                                     ) 
        nn.init.constant_(self.decoder[-2].bias, 50.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] > self.in_features:
            print("im here in basic and i had to cut off a column")
            x= x[:, :-1]
        z = self.encoder(x)
        return self.decoder(z), z
    
    def get_latents(self, x):
        """Returns the central bottleneck representation."""
        return self.encoder(x)


class Layered_AE(nn.Module):
    def __init__(self, input_size: int, encoding_size: int, h1: int, h2: int):
        super(Layered_AE, self).__init__()
        self.in_features = input_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.GELU(),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Linear(h2, encoding_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, h2),
            nn.GELU(),
            nn.Linear(h2, h1),
            nn.GELU(),
            nn.Linear(h1, input_size)
            , nn.ReLU() # Gradients never die       )
        )
        nn.init.constant_(self.decoder[-2].bias, 50.0)

        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] > self.in_features:
            print("im here in layered and i had to cut off a column")
            x= x[:, :-1]
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z

    def get_latents(self, x):
        """Returns the central bottleneck representation."""
        return self.encoder(x)