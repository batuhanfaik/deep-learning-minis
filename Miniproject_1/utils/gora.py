import torch
import torch.nn as nn


class GORA(nn.Module):
    """
    G.O.R.A.: Gaussian nOise Reducing Autoencoder
        A Space Model
    - Yes, by.
    """

    def __init__(self) -> None:
        super(GORA, self).__init__()
        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..3
        self._block2 = nn.Sequential(
            nn.Conv2d(24, 24, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv4, upsample3
        self._block3 = nn.Sequential(
            nn.Conv2d(24, 24, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, 24, 3, stride=2, padding=1, output_padding=1))

        # Layers: dec_conv3a, dec_conv3b, upsample2
        self._block4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=2
        self._block5 = nn.Sequential(
            nn.Conv2d(72, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(48 + 3, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, stride=1, padding=1),
            nn.ReLU(inplace=True))
        # Initialize weights
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Through encoder, then decoder by adding U-skip connections. """
        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)

        # Decoder
        upsample3 = self._block3(pool3)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block4(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block6(concat1)

    def _init_weights(self) -> None:
        # Weight initialization used in the Noise2Noise paper, proposed by
        # He, X. et al. (2015) https://arxiv.org/abs/1502.01852
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
