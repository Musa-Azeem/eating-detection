import torch
from torch import nn

def get_lout(lin, k, s, p):
    return int(((lin-k+2*p)/s) + 1)

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