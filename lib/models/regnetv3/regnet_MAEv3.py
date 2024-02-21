import torch
from torch import nn
from lib.models.regnetv3.regnet_encoder import RegNetEncoder
from lib.models.regnetv3.modules import Permute, PositionalEncoding, Mask

class RegNetMAEv3(nn.Module):
    def __init__(
            self, 
            winsize, in_channels, stem_out_c, d, w, 
            g=1, p_dropout=0, 
            d_model=64, ntrans=1, nhead=2, trans_dropout=0.01, tran_linear_dim=2048,
            maskpct=0.15, n_mask_chunks=10, mask_type='zeros', 
        ):
        super().__init__()
        self.maskpct = maskpct

        # dont change the name of self.e
        self.e = RegNetEncoder(winsize, in_channels, stem_out_c, d, w, g, p_dropout)
        self.mask = Mask(maskpct, n_mask_chunks, mask_type)
        self.trans_skip = nn.Conv1d(w[-1], d_model, kernel_size=1)
        self.transformer_encoder = nn.Sequential(
            nn.Conv1d(w[-1], d_model, kernel_size=1),
            Permute(0,2,1),
            PositionalEncoding(d_model, seq_len=self.e.latent_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead, tran_linear_dim, trans_dropout, batch_first=True), 
                num_layers=ntrans,
                enable_nested_tensor=False
            ),
            Permute(0,2,1),
        )
        self.decoder_skip = nn.Sequential(
            nn.Conv1d(d_model, in_channels, kernel_size=1),
            nn.Upsample(size=winsize),
        )
        self.decoder = nn.Sequential()
        for i,width in enumerate(w):
            in_c = d_model if i==0 else w[i-1]
            s = 2 if in_c < width else 1
            self.decoder.add_module(
                f"decoder-{i}_w{width}",
                nn.ConvTranspose1d(in_c, width, kernel_size=3, stride=s),
            )
        print('latent dim:',self.e.latent_dim)
    def forward(self, x):
        x = self.e(x)
        x = self.mask(x)
        x = self.trans_skip(x) + self.transformer_encoder(x)
        x = self.decoder(x)
        return x