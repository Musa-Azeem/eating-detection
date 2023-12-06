import torch
from torch import nn

# Model 5
# =============================================================================
# =============================================================================
# =============================================================================

class ResBlock(nn.Module):
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

class DecoderResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, seq_len, use_relu=True):
        super().__init__()
        self.use_relu = use_relu
        self.c = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LayerNorm((out_channels, seq_len)),
            nn.ReLU(),
            nn.ConvTranspose1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LayerNorm((out_channels, seq_len)),
            nn.ReLU(),
            nn.ConvTranspose1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LayerNorm((out_channels, seq_len)),
        )
        if use_relu:
            self.c.add_module('relu', nn.ReLU())
        
        self.identity = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1),
            nn.LayerNorm((out_channels, seq_len))
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.identity(x) + self.c(x)) if self.use_relu else self.identity(x) + self.c(x)

class ResAutoEncoder(nn.Module):
    def __init__(self, winsize, in_channels):
        super().__init__()
        self.winsize = winsize
        self.in_channels = in_channels

        self.encoder = nn.Sequential(
            ResBlock(in_channels, 16, 15, 7, winsize), # Nx16x101
            nn.MaxPool1d(kernel_size=3, stride=3), # Nx16x33
            ResBlock(16, 8, 9, 4, 33), # Nx8x33
            nn.MaxPool1d(kernel_size=3, stride=3), # Nx8x11
            ResBlock(8, 4, 5, 2, 11), # Nx4x11
        )

        self.decoder = nn.Sequential(
            DecoderResBlock(4, 8, 5, 2, 11), # Nx8x11
            nn.Upsample(scale_factor=3, mode='nearest'), # Nx8x33
            DecoderResBlock(8, 16, 9, 4, 33), # Nx16x33
            nn.Upsample(scale_factor=3.09, mode='nearest'), # Nx16x101
            DecoderResBlock(16, 3, 15, 7, 101, use_relu=False) # Nx3x101
        )

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.winsize)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.flatten(start_dim=1)

class ResEncoderClassifier(nn.Module):
    def __init__(self, winsize, weights_file=None, freeze=False):
        super().__init__()

        self.winsize = winsize
        self.weights_file = weights_file
        self.freeze = freeze

        self.encoder = self.get_encoder()
        
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5, stride=1, padding=2), # Nx4x11 -> Nx4x11
            nn.ReLU(),
            nn.LayerNorm((4, 11)),
            nn.Flatten(start_dim=1), # Nx4x11 -> Nx44
            nn.Linear(in_features=44, out_features=100), # Nx44 -> Nx11
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=1) # Nx11 -> Nx1
        )

    def forward(self, x):
        x = x.view(-1,3,self.winsize)
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def get_encoder(self):
        autoencoder = ResAutoEncoder(self.winsize, 3)

        if self.weights_file:
            print("Model is loading pretrained encoder")
            autoencoder.load_state_dict(torch.load(self.weights_file))
        
        encoder = autoencoder.encoder

        if self.freeze:
            print("Model is freezing encoder")
            for p in encoder.parameters():
                p.requires_grad = False
        
        return encoder

class ResEncoderClassifierAve(nn.Module):
    def __init__(self, winsize, weights_file=None, freeze=False):
        super().__init__()

        self.winsize = winsize
        self.weights_file = weights_file
        self.freeze = freeze

        self.encoder = self.get_encoder()

        self.classifier = nn.Sequential(
            nn.AvgPool1d(kernel_size=11), # Nx4x11 -> Nx4x1
            nn.Flatten(start_dim=1), # Nx4x1 -> Nx4
            nn.Linear(in_features=4, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=1)
        )
        
    def forward(self, x):
        x = x.view(-1,3,self.winsize)
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    
    def get_encoder(self):
        autoencoder = ResAutoEncoder(self.winsize, 3)

        if self.weights_file:
            print("Model is loading pretrained encoder")
            autoencoder.load_state_dict(torch.load(self.weights_file))
        
        encoder = autoencoder.encoder

        if self.freeze:
            print("Model is freezing encoder")
            for p in encoder.parameters():
                p.requires_grad = False
        
        return encoder