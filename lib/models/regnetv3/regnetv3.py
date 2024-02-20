import torch
from torch import nn
from lib.models.regnetv3.regnet_encoder import RegNetEncoder

class RegNetv3(nn.Module):
    def __init__(self, winsize, in_channels, stem_out_c, d, w, g=1, p_dropout=0, weights_file=None, freeze=False):
        super().__init__()

        self.e = RegNetEncoder(winsize, in_channels, stem_out_c, d, w, g, p_dropout, weights_file, freeze)
        self.o = nn.Sequential(
            nn.AvgPool1d(kernel_size=self.e.latent_dim), # Nxdims[-1]x1
            nn.Flatten(start_dim=1), # Nxdims[-1]
            nn.Linear(in_features=w[-1], out_features=5)
        )
    def forward(self, x):
        return self.o(self.e(x))