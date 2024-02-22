import torch
from torch import nn
from lib.models.regnetv3.regnet_encoder import RegNetEncoder

class RegNetv3(nn.Module):
    def __init__(
            self, 
            winsize=None, in_channels=3, stem_out_c=None, d=None, w=None, 
            g=1, p_dropout=0, 
            weights_file=None, freeze=False,
            CONFIG=None
        ):
        super().__init__()
        if CONFIG:
            winsize = CONFIG['WINDOW_SIZE']
            stem_out_c = CONFIG['WIDTHI'][0]
            d = CONFIG['DEPTHI']
            w = CONFIG['WIDTHI']
            weights_file = CONFIG.get('WEIGHTS_FILE', None)
            freeze = CONFIG.get('FROZEN', False)
        if not stem_out_c:
            stem_out_c = w[0]

        self.e = RegNetEncoder(winsize, in_channels, stem_out_c, d, w, g, p_dropout, weights_file, freeze)
        self.o = nn.Sequential(
            nn.AvgPool1d(kernel_size=self.e.latent_dim), # Nxdims[-1]x1
            nn.Flatten(start_dim=1), # Nxdims[-1]
            nn.Linear(in_features=w[-1], out_features=5)
        )
    def forward(self, x):
        return self.o(self.e(x))