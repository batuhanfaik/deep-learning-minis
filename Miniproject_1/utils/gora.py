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
        # Initialize weights
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)

    def _init_weights(self) -> None:
        # Weight initialization used in the Noise2Noise paper, proposed by
        # He, X. et al. (2015) https://arxiv.org/abs/1502.01852
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
