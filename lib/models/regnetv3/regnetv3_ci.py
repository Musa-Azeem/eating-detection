from lib.models.regnetv3.regnet_encoder import RegNetEncoder
from lib.models.regnetv3.regnet_mae_v3 import RegNetMAEv3
from torch import nn
import torch

class RegNetv3Ci(nn.Module):
    def __init__(
        self, 
        winsize=None, in_channels=3, stem_out_c=None, d=None, w=None, 
        g=1, p_dropout=0, 
        d_model=64, ntrans=1, nhead=2, trans_dropout=0.01, tran_linear_dim=None,
        weights_file=None, freeze=False,
        CONFIG=None
    ):
        super().__init__()
        if CONFIG:
            winsize = CONFIG['WINDOW_SIZE']
            stem_out_c = CONFIG['WIDTHI'][0]
            d = CONFIG['DEPTHI']
            w = CONFIG['WIDTHI']
            d_model = CONFIG['DMODEL']
            ntrans = CONFIG['NTL']
            p_dropout = CONFIG['PDROPOUT']
            weights_file = CONFIG.get('WEIGHTS_FILE', None)
            freeze = CONFIG.get('FROZEN', False)
        else:
            CONFIG = {
                'WINDOW_SIZE': winsize,
                'MASK_PCT': 0.0,
                'WIDTHI': w,
                'DEPTHI': d,
                'DMODEL': d_model,
                'NTL': ntrans,
                'PDROPOUT': p_dropout,
            }
        if not stem_out_c:
            stem_out_c = w[0]
        
        mae = RegNetMAEv3(CONFIG=CONFIG)
        if weights_file:
            mae.load_state_dict(torch.load(weights_file))
        if freeze:
            for param in mae.parameters():
                param.requires_grad = False
        
        self.e = mae.e
        self.trans_skip = mae.trans_skip
        self.transformer_encoder = mae.transformer_encoder
        self.o = nn.Sequential(
            nn.AvgPool1d(kernel_size=self.e.latent_dim), # Nxdims[-1]x1
            nn.Flatten(start_dim=1), # Nxdims[-1]
            nn.Linear(in_features=d_model, out_features=5)
        )
    def forward(self, x):
        x = self.e(x)
        x = self.trans_skip(x) + self.transformer_encoder(x)
        x = self.o(x)
        return x