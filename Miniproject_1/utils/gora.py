import torch
import torch.nn as nn


class GORA(nn.Module):
    """
    G.O.R.A.: Gaussian nOise Restoring Autoencoder
        A Space Model
    - Yes, by.
    """
    def __init__(self) -> None:
        super(GORA, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
