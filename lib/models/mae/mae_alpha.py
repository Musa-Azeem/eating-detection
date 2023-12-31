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

class MAEAlpha(nn.Module):
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
    
class MAEAlphaClassifier(nn.Module):
    def __init__(self, winsize, in_channels, dims, n_hl=10, weights_file=None, freeze=False):
        super().__init__()

        self.winsize = winsize
        self.in_channels = in_channels
        self.dims = dims
        self.weights_file = weights_file
        self.freeze = freeze

        self.encoder = self.get_encoder()
        self.classifier = nn.Sequential(
            nn.AvgPool1d(kernel_size=winsize), # Nxenc_dimx1
            nn.Flatten(start_dim=1), # Nxenc_dim
            nn.Linear(in_features=self.dims[-1], out_features=n_hl),
            nn.ReLU(),
            nn.Linear(in_features=n_hl, out_features=1)
        )

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.winsize)
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def get_encoder(self):
        autoencoder = MAEAlpha(self.winsize, self.in_channels, self.dims)

        if self.weights_file:
            print("Model is loading pretrained encoder")
            autoencoder.load_state_dict(torch.load(self.weights_file))
        
        encoder = autoencoder.e

        if self.freeze:
            print("Model is freezing encoder")
            for p in encoder.parameters():
                p.requires_grad = False
        
        return encoder