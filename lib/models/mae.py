from torch import nn
import torch

class ResBlockMAE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, seq_len):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LayerNorm((out_channels, seq_len)),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LayerNorm((out_channels, seq_len)),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LayerNorm((out_channels, seq_len)),
            nn.ReLU()
        )
        self.identity = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.LayerNorm((out_channels, seq_len))
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.identity(x) + self.c(x))

class MAE(nn.Module):
    def __init__(self, winsize, in_channels, dims, maskpct=0.25):
        super().__init__()
        self.winsize = winsize
        self.in_channels = in_channels
        self.dims = dims
        self.maskpct = maskpct
        # self.dims = [self.in_channels] + list(dims)

        self.e = nn.Sequential(
            nn.Conv1d(in_channels, dims[0], kernel_size=9, padding='same'),
            nn.LayerNorm((dims[0], winsize)),
            nn.ReLU(),
            *[ResBlockMAE(self.dims[i], self.dims[i+1], 3, 'same', winsize) for i in range(len(self.dims)-1)]
        )

        self.d = nn.Sequential(
            *[ResBlockMAE(self.dims[i], self.dims[i-1], 3, 'same', winsize) for i in range(len(self.dims)-1, 0, -1)],
            nn.Conv1d(self.dims[0], in_channels, kernel_size=9, padding='same'),
        )

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.winsize).clone()
        # Mask: randomly set maskpct% of X (xyz pairs) to values from a normal distribution
        mask = torch.rand(x.shape[0], 1, x.shape[2]) < self.maskpct # maskpct% of values are True
        mask = mask.expand(-1,3,-1)                         # expand to all xyz pairs
        x[mask] = torch.randn(x.shape)[mask].to(x.device)   # set masked values to random normal
        x = self.e(x)
        x = self.d(x)
        return x.flatten(start_dim=1)