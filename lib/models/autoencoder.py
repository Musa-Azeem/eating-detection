from torch import nn

class LinearAutoencoder(nn.Module):
    def __init__(self, winsize):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(winsize*3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, 12), 
            nn.ReLU(True), 
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(), 
            nn.Linear(128, winsize*3)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

# =============================================================================
# =============================================================================
# =============================================================================

class ConvAutoencoder(nn.Module):
    def __init__(self, winsize):
        super().__init__()
        self.winsize = winsize
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=8, kernel_size=3, stride=3, padding=2), # Nx3x101 -> Nx8x35
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, stride=3, padding=2), # Nx8x35 -> Nx4x13
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=4, out_channels=8, kernel_size=3, stride=3, padding=2), # Nx4x13 -> Nx8x35
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=8, out_channels=3, kernel_size=3, stride=3, padding=2), # Nx8x25 -> Nx3x101
        )

    def forward(self, x):
        x = x.view(-1,3,self.winsize)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.flatten(start_dim=1)
    
class EncoderClassifier(nn.Module):
    def __init__(self, autoencoder: ConvAutoencoder):
        super().__init__()
        self.winsize = autoencoder.winsize
        self.encoder = autoencoder.encoder
        self.encoder.requires_grad_ = False
        self.classifier = nn.Sequential(
            nn.Linear(in_features=52, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x):
        x = x.view(-1,3,self.winsize)
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x