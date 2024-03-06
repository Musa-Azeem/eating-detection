import torch
from torch import nn
from torch.nn import functional as F

def get_lout(lin, k, s, p):
    return int(((lin-k+2*p)/s) + 1)

class XBlockV3(nn.Module):
    def __init__(self, in_channels, out_channels, k, lin, g, p_dropout):
        super().__init__()
        if k%2==0:
            raise ValueError("k must be odd")
        
        s = 2 if out_channels > in_channels else 1
        p = k // 2
        lout = get_lout(lin, k, s, p)
        
        self.c = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=p, groups=g, stride=s),
            nn.LayerNorm((lout), elementwise_affine=False),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
        )
        self.identity = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=s),
        ) if out_channels != in_channels else nn.Identity()

    def forward(self, x):
        return F.relu(self.c(x) + self.identity(x))

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
    
class Mask(nn.Module):
    def __init__(self, maskpct, n_chunks, mask_type='zeros'):
        super().__init__()
        self.maskpct = maskpct
        self.n_chunks = n_chunks
        self.mask_type = mask_type
    def forward(self, x):
        # Mask: split X into chunks of mask_len size and randomly set maskpct% 
        # of chunks (all channels) to values from a normal distribution
        chunk_len = x.shape[2] // self.n_chunks
        chunked = list(torch.split(x, chunk_len, dim=2))
        mask = torch.rand(len(chunked), x.shape[0]) < self.maskpct # maskpct% of values are True
        for i,mi in enumerate(mask):
            chunked[i] = chunked[i].clone()
            match self.mask_type:
                case 'zeros':
                    chunked[i][mi] = torch.zeros_like(chunked[i][mi], device=x.device)
                case 'normal':
                    chunked[i][mi] = torch.randn_like(chunked[i][mi], device=x.device)
                case _:
                    raise ValueError(f"mask_type {self.mask_type} not recognized")
        x = torch.cat(chunked, dim=2)
        return x