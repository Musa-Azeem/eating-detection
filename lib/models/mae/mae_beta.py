from torch import nn
import torch

class ResBlockMAE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, seq_len, relu=True):
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
        
        self.identity = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.LayerNorm((out_channels, seq_len))
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.c(x) + self.identity(x)
        return self.relu(x) if self.use_relu else x
    
class MAEBeta(nn.Module):
    def __init__(self, winsize, in_channels, enc_dims=(8,16,32,64), rec_dims=(64,128,192), maskpct=0.75):
        super().__init__()
        self.winsize = winsize
        self.in_channels = in_channels
        self.enc_dims = enc_dims
        self.rec_dims = rec_dims
        self.dec_dims = [rec_dims[-1]] + list(enc_dims)[::-1]
        self.maskpct = maskpct

        self.e = nn.Sequential(
            ResBlockMAE(in_channels, enc_dims[0], 9, 'same', winsize),
            *[ResBlockMAE(self.enc_dims[i], self.enc_dims[i+1], 3, 'same', winsize) for i in range(len(self.enc_dims)-1)]
        )
        self.r = nn.Sequential(
            *[ResBlockMAE(self.rec_dims[i], self.rec_dims[i+1], 3, 'same', winsize) for i in range(len(self.rec_dims)-1)],
        )
        self.d = nn.Sequential(
            *[ResBlockMAE(self.dec_dims[i], self.dec_dims[i+1], 3, 'same', winsize) for i in range(len(self.dec_dims)-1)],
            ResBlockMAE(self.dec_dims[-1], in_channels, 9, 'same', winsize, relu=False)
        )

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.winsize)
        x = self.e(x)
        # Mask: randomly set maskpct% of X (all 64 dims) to values from a normal distribution
        mask = torch.rand(x.shape[0], 1, x.shape[2]) < self.maskpct # maskpct% of values are True
        mask = mask.expand(-1,self.enc_dims[-1],-1)                         # expand to all 64 dims
        x_masked = x.clone()
        x_masked[mask] = torch.randn(x.shape)[mask].to(x.device)
        x = self.r(x_masked)
        x = self.d(x)
        return x.flatten(start_dim=1)

class MAEBetaClassifier(nn.Module):
    def __init__(self, winsize, in_channels, dims=(8,16,32,64), n_hl=100, weights_file=None, freeze=False):
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
        autoencoder = MAEBeta(self.winsize, self.in_channels)

        if self.weights_file:
            print("Model is loading pretrained encoder")
            autoencoder.load_state_dict(torch.load(self.weights_file))
        
        encoder = autoencoder.e

        if self.freeze:
            print("Model is freezing encoder")
            for p in encoder.parameters():
                p.requires_grad = False
        
        return encoder