import torch
from torch import nn
from . import ResBlock, DecoderResBlock

# Model 6 -- Bigger Resnet Autoencoder

class BiggerResNetAE(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.winsize = 505
        self.in_channels = in_channels

        self.encoder = nn.Sequential(
            # inchannels, outchannels, kernel, padding, seq_len
            ResBlock(in_channels, 64, 9, 4, 505),   # Nx3x505  -> Nx64x505
            nn.MaxPool1d(kernel_size=2, stride=2),  # Nx64x505 -> Nx64x252
            ResBlock(64, 48, 5, 2, 252),            # Nx64x252 -> Nx48x252
            nn.MaxPool1d(kernel_size=2, stride=2),  # Nx48x252 -> Nx48x126
            ResBlock(48, 32, 3, 1, 126),            # Nx48x126 -> Nx32x126
            nn.MaxPool1d(kernel_size=2, stride=2),  # Nx32x126 -> Nx32x63
            ResBlock(32, 16, 3, 1, 63),             # Nx32x63  -> Nx16x63
            nn.MaxPool1d(kernel_size=2, stride=2),  # Nx16x63  -> Nx16x31
            ResBlock(16, 8, 3, 1, 31),              # Nx16x31  -> Nx8x31
            nn.MaxPool1d(kernel_size=2, stride=2),  # Nx8x31   -> Nx8x15
            ResBlock(8, 4, 3, 1, 15),               # Nx8x15   -> Nx4x15
        )
        self.decoder = nn.Sequential(
            DecoderResBlock(4, 8, 3, 1, 15),        # Nx4x15   -> Nx8x15
            nn.Upsample(scale_factor=2.1),          # Nx8x15   -> Nx8x31
            DecoderResBlock(8, 16, 3, 1, 31),       # Nx8x31   -> Nx16x31
            nn.Upsample(scale_factor=2.04),         # Nx16x31  -> Nx16x63
            DecoderResBlock(16, 32, 3, 1, 63),      # Nx16x63  -> Nx32x63
            nn.Upsample(scale_factor=2),            # Nx32x63  -> Nx32x126
            DecoderResBlock(32, 48, 3, 1, 126),     # Nx32x126 -> Nx48x126
            nn.Upsample(scale_factor=2),            # Nx48x126 -> Nx48x252
            DecoderResBlock(48, 64, 5, 2, 252),     # Nx48x252 -> Nx64x252
            nn.Upsample(scale_factor=2.005),        # Nx64x252 -> Nx64x505
            DecoderResBlock(64, 3, 9, 4, 505, use_relu=False), # Nx64x505 -> Nx3x505
        )
    
    def forward(self, x):
        x = x.view(-1, self.in_channels, self.winsize)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.flatten(start_dim=1)
