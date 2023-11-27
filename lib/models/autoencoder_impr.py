import torch
from torch import nn

class ConvAutoencoderImproved(nn.Module):
    def __init__(self, winsize):
        super().__init__()
        self.winsize = winsize
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=15, padding=7), # Nx3x101 -> Nx16x101
            nn.MaxPool1d(kernel_size=3, stride=3), # Nx16x101 -> Nx16x33
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=9, padding=4), # Nx16x33 -> Nx8x33
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3), # Nx8x33 -> Nx8x11,
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=5, padding=2), # Nx8x11 -> Nx4x11
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=4, out_channels=8, kernel_size=5, padding=2), # Nx4x11 -> Nx8x11
            nn.ReLU(),
            nn.Upsample(scale_factor=3, mode='nearest'), # Nx8x11 -> Nx8x33
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=8, out_channels=16, kernel_size=9, padding=4), # Nx8x33 -> Nx16x33
            nn.Upsample(scale_factor=3.09, mode='nearest'), # Nx16x33 -> Nx16x101
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=3, kernel_size=15, padding=7), # Nx16x101 -> Nx3x101
        )

    def forward(self, x):
        x = x.view(-1,3,self.winsize)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.flatten(start_dim=1)


class EncoderClassifierImproved(nn.Module):
    def __init__(self, winsize, weights_file=None, freeze=False):
        super().__init__()

        self.winsize = winsize
        self.weights_file = weights_file
        self.freeze = freeze

        self.encoder = self.get_encoder()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=44, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x):
        x = x.view(-1,3,self.winsize)
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

    def get_encoder(self):
        autoencoder = ConvAutoencoderImproved(self.winsize)

        if self.weights_file:
            print("Model is loading pretrained encoder")
            autoencoder.load_state_dict(torch.load(self.weights_file))
        
        encoder = autoencoder.encoder

        if self.freeze:
            print("Model is freezing encoder")
            for p in encoder.parameters():
                p.requires_grad = False
        
        return encoder
    

class ConvEncoderClassifierImproved(nn.Module):
    def __init__(self, winsize, weights_file=None, freeze=False):
        super().__init__()

        self.winsize = winsize
        self.weights_file = weights_file
        self.freeze = freeze

        self.encoder = self.get_encoder()
        
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5, stride=1, padding=2), # Nx4x11 -> Nx4x11
            nn.ReLU(),
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
        autoencoder = ConvAutoencoderImproved(self.winsize)

        if self.weights_file:
            print("Model is loading pretrained encoder")
            autoencoder.load_state_dict(torch.load(self.weights_file))
        
        encoder = autoencoder.encoder

        if self.freeze:
            print("Model is freezing encoder")
            for p in encoder.parameters():
                p.requires_grad = False
        
        return encoder