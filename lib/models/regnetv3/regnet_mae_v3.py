import torch
from torch import nn
from lib.models.regnetv3.regnet_encoder import RegNetEncoder
from lib.models.regnetv3.modules import Permute, PositionalEncoding, Mask

class RegNetMAEv3(nn.Module):
    def __init__(
            self, 
            winsize=None, in_channels=3, stem_out_c=None, d=None, w=None, 
            g=1, p_dropout=0, 
            d_model=64, ntrans=1, nhead=2, trans_dropout=0.01, tran_linear_dim=None,
            maskpct=0.15, n_mask_chunks=10, mask_type='zeros', 
            CONFIG=None
        ):
        super().__init__()
        if CONFIG:
            winsize = CONFIG['WINDOW_SIZE']
            stem_out_c = CONFIG['WIDTHI'][0]
            maskpct = CONFIG['MASKPCT']
            d = CONFIG['DEPTHI']
            w = CONFIG['WIDTHI']
            d_model = CONFIG['DMODEL']
            ntrans = CONFIG['NTL']
            p_dropout = CONFIG['PDROPOUT']
            trans_dropout = CONFIG['PDROPOUT']
            tran_linear_dim = CONFIG.get('TRAN_LINEAR_DIM', None)
        if not stem_out_c:
            stem_out_c = w[0]
        if not tran_linear_dim:
            tran_linear_dim = d_model*4

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
        for i,width in enumerate(w[::-1]):
            in_c = d_model if i==0 else w[-i]
            s = 2 if in_c < width else 1
            self.decoder.add_module(
                f"decoder-{i}_w{width}",
                nn.ConvTranspose1d(in_c, width, kernel_size=3, stride=s, groups=g),
            )
        self.decoder.add_module(
            "decoder-final",
            nn.Sequential(
                nn.Upsample(size=(winsize//2)),
                nn.ConvTranspose1d(w[0], in_channels, kernel_size=3, stride=2)
            )
        )
        print('latent dim:',self.e.latent_dim)
    def forward(self, x):
        x = self.e(x)
        x = self.mask(x)
        x = self.trans_skip(x) + self.transformer_encoder(x)
        x = self.decoder(x) + self.decoder_skip(x)
        return x