import torch
from torch import nn
from torch.nn import functional as F
from lib.models.regnetv3.modules import get_lout

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
            nn.LayerNorm((lout)),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
        )
        self.identity = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=s),
        ) if out_channels > in_channels else nn.Identity()

    def forward(self, x):
        return F.relu(self.c(x) + self.identity(x))