from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self, n_hl, winsize):
        super().__init__()
        self.h1 = nn.Linear(in_features=winsize*3, out_features=n_hl)
        self.relu = nn.ReLU()
        self.out = nn.Linear(in_features=n_hl, out_features=1)
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.h1(x)
        x = self.relu(x)
        x = self.out(x)
        return x
    
class MLP2hl(nn.Module):
    def __init__(self, n_hl, winsize):
        super().__init__()
        self.h1 = nn.Linear(in_features=winsize*3, out_features=n_hl[0])
        self.h2 = nn.Linear(in_features=n_hl[0], out_features=n_hl[1])
        self.out = nn.Linear(in_features=n_hl[1], out_features=1)
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.h1(x)
        x = nn.ReLU()(x)
        x = self.h2(x)
        x = nn.ReLU()(x)
        x = self.out(x)
        return x

class LSTM(nn.Module):
    def __init__(self, winsize, subwinsize):
        super().__init__()
        self.winsize = winsize
        self.subwinsize = subwinsize
        self.n_channels = 3

        self.lstm = nn.LSTM(
            input_size=self.subwinsize*self.n_channels,
            hidden_size=16,
            bias=False,
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features=16, out_features=5),
            nn.ReLU(),
            nn.Linear(in_features=5, out_features=1)
        )

    def partition_window(self, x, p_size, n_channels):
        x = x.view(-1, n_channels, self.winsize)
        x = x.transpose(1,2).view(-1, self.winsize//p_size, p_size, n_channels)
        x = x.transpose(2,3).flatten(start_dim=2)
        return x
    
    def window_window(self, x):
        # Seperate x,y,z
        x = x.reshape(-1, 3, self.winsize)

        # Pad x,y, and z on both sides with 0s
        x = nn.functional.pad(x, (self.subwinsize//2, self.subwinsize//2), 'constant', 0)

        # Window x,y, and z
        xacc = x[:,0].unsqueeze(2)
        yacc = x[:,1].unsqueeze(2)
        zacc = x[:,2].unsqueeze(2)

        w = self.subwinsize - 1
        xs = [xacc[:, :-w]]
        ys = [yacc[:, :-w]]
        zs = [zacc[:, :-w]]

        for i in range(1,w):
            xs.append(xacc[:, i:i-w])
            ys.append(yacc[:, i:i-w])
            zs.append(zacc[:, i:i-w])
        
        xs.append(xacc[:, w:])
        ys.append(yacc[:, w:])
        zs.append(zacc[:, w:])

        xs = torch.cat(xs, dim=2)
        ys = torch.cat(ys, dim=2)
        zs = torch.cat(zs, dim=2)

        x = torch.cat([xs, ys, zs], dim=2)
        return x

    def forward(self, x):
        x = self.partition_window(x, p_size=self.subwinsize, n_channels=self.n_channels)
        # x = self.window_window(x)
        o, (h,c) = self.lstm(x) # o is shape (batch_size, p_size, 64)
        o = nn.ReLU()(o[:,-1,:])
        logits = self.mlp(o)

        return logits



class ResNetClassifier(nn.Module):
    def __init__(self, in_channels, winsize):
        super().__init__()
        self.winsize = winsize

        self.resnet_conv = ResNetConv(in_channels=in_channels, winsize=self.winsize)
        # Global Pooling and output
        self.gp = nn.AvgPool1d(kernel_size=self.winsize)    # Take mean across each feature map (N, C, L) => (N,C)
        self.output = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        # Run convolutional layers of ResNet
        y = self.resnet_conv(x)

        # Run Global pooling and classifier
        y = self.gp(y).squeeze(2)
        logits = self.output(y)
        return logits


class ResNetConv(nn.Module):
    def conv_block(self, in_channels, out_channels, kernel_size, use_relu=True):
        if use_relu:
            return nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
                nn.BatchNorm1d(num_features=out_channels)
            )
    
    def inner_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            self.conv_block(in_channels=in_channels, out_channels=out_channels, kernel_size=8),
            self.conv_block(in_channels=out_channels, out_channels=out_channels, kernel_size=5),
            self.conv_block(in_channels=out_channels, out_channels=out_channels, kernel_size=3, use_relu=False)
        )
    
    def shortcut(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm1d(num_features=out_channels)
        )

    def __init__(self, in_channels, winsize):
        super().__init__()
        self.winsize = winsize

        # First ResNet Block components
        self.shortcut1 = self.shortcut(in_channels=in_channels, out_channels=3)
        self.res1 = self.inner_res_block(in_channels=in_channels, out_channels=3)
        self.relu1 = nn.ReLU()

        # # Second Res Block components
        # self.shortcut2 = self.shortcut(in_channels=8, out_channels=16)
        # self.res2 = self.inner_res_block(in_channels=8, out_channels=16)
        # self.relu2 = nn.ReLU()

        # Third Res Block components
        self.shortcut3 = self.shortcut(in_channels=3, out_channels=5)
        self.res3 = self.inner_res_block(in_channels=3, out_channels=5)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        # Reshape x: (batch_size, 303) -> (batch_size, 3, 101)
        x = x.view(-1, 3, self.winsize)
        
        # First Res Block
        x_shortcut = self.shortcut1(x)
        h = self.res1(x)
        y = h + x_shortcut
        y = self.relu1(y)

        # # Second Res Block
        # y_shortcut = self.shortcut2(y)
        # h = self.res2(y)
        # y = h + y_shortcut
        # y = self.relu2(y)

        # Third Res Block
        y_shortcut = self.shortcut3(y)
        h = self.res3(y)
        y = h + y_shortcut
        y = self.relu3(y)

        return y




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
    

# =============================================================================
# =============================================================================
# =============================================================================


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
            ResBlock(in_channels, 16, 15, 7, winsize), # Nx3x101
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