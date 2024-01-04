import torch
from torch import nn

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

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        position = torch.arange(seq_len).unsqueeze(1)
        div_term =  torch.pow(10000.0, torch.arange(0, d_model, 2) / d_model)
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return x
    
class MAEDelta(nn.Module):
    def __init__(self, winsize, in_channels, mask_chunk_size=11, enc_dims=(8,16,32,64,96,128), d_model=192, maskpct=0.75):
        super().__init__()
        self.winsize = winsize
        self.enc_dims = enc_dims
        self.d_model = d_model
        self.mask_chunk_size = mask_chunk_size
        self.maskpct = maskpct
        p_dropout = 0.01
        
        self.e = nn.Sequential(
            ResBlockMAE(in_channels, enc_dims[0], 5, 'same', winsize),
            *[ResBlockMAE(self.enc_dims[i], self.enc_dims[i+1], 3, 'same', winsize, p_dropout=p_dropout) for i in range(len(self.enc_dims)-1)]
        )
        self.transformer_encoder = nn.Sequential(
            nn.Conv1d(enc_dims[-1], d_model, 1),
            Permute(0,2,1),
            PositionalEncoding(d_model, seq_len=winsize),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, 1, 2048, 0.1, batch_first=True), 
                1,
                enable_nested_tensor=False
            ),
            Permute(0,2,1),
        )
        self.d = nn.Sequential(
            ResBlockMAE(d_model, enc_dims[-1], 3, 'same', winsize),
            nn.Conv1d(enc_dims[-1], in_channels, 3, padding='same'),
        )

    def forward(self, x):
        x = x.view(-1, 3, self.winsize)
        x = self.e(x)
        x = self.mask(x)
        x = self.transformer_encoder(x)
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