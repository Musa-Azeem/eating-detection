from torch import nn
import torch

class ResBlockMAE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, seq_len, relu=True, p_dropout=None):
        super().__init__()
        self.use_relu = relu
        self.c = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LayerNorm((out_channels, seq_len)),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LayerNorm((out_channels, seq_len)),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LayerNorm((out_channels, seq_len)),
        )
        if self.use_relu:
            self.c.add_module('relu', nn.ReLU())
        if p_dropout is not None:
            self.c.add_module('dropout', nn.Dropout(p=p_dropout))

        self.identity = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.LayerNorm((out_channels, seq_len))
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.c(x) + self.identity(x)
        return self.relu(x) if self.use_relu else x
    
class MAEGamma(nn.Module):
    def __init__(self, winsize, in_channels, mask_chunk_size=11, enc_dims=(8,16,32,64,96,128), rec_dims=(128,160,192,224,256), maskpct=0.75):
        super().__init__()
        self.winsize = winsize
        self.in_channels = in_channels
        self.enc_dims = enc_dims
        self.rec_dims = rec_dims
        self.dec_dims = [rec_dims[-1]] + list(enc_dims)[::-1]
        self.maskpct = maskpct
        self.mask_chunk_size = mask_chunk_size
        p_dropout = 0.025

        self.e = nn.Sequential(
            ResBlockMAE(in_channels, enc_dims[0], 9, 'same', winsize),
            *[ResBlockMAE(self.enc_dims[i], self.enc_dims[i+1], 3, 'same', winsize, p_dropout=p_dropout) for i in range(len(self.enc_dims)-1)]
        )
        self.r = nn.Sequential(
            *[ResBlockMAE(self.rec_dims[i], self.rec_dims[i+1], 3, 'same', winsize, p_dropout=p_dropout) for i in range(len(self.rec_dims)-1)],
        )
        self.d = nn.Sequential(
            *[ResBlockMAE(self.dec_dims[i], self.dec_dims[i+1], 3, 'same', winsize, p_dropout=p_dropout) for i in range(len(self.dec_dims)-1)],
            ResBlockMAE(self.dec_dims[-1], in_channels, 9, 'same', winsize, relu=False)
        )

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.winsize)
        x = self.e(x)
        x = self.mask(x)
        x = self.r(x)
        x = self.d(x)
        return x.flatten(start_dim=1)
    
    def mask(self, x):
        # Mask: split X into chunks and randomly set maskpct% of chunks 
        # (all 64 dims) to values from a normal distribution
        x = x.view(x.shape[0], x.shape[1], x.shape[2]//self.mask_chunk_size, -1).clone()
        mask = torch.rand(x.shape[0], 1, x.shape[2]) < self.maskpct # maskpct% of values are True
        mask = mask.expand(-1, x.shape[1], -1)                      # expand to all 64 dims
        x[mask] = torch.randn(x.shape, device=x.device)[mask]       # set masked chunks to random values
        x = x.flatten(start_dim=2)                                  # get rid of chunk dim
        return x

class MAEGammaClassifier(nn.Module):
    pass