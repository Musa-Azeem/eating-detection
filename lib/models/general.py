from torch import nn
import torch
from . import ResBlock

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

class ClassifierResBlock(nn.Module):
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
    
class ResNetClassifier(nn.Module):
    def __init__(self, winsize, in_channels, dims):
        super().__init__()
        self.winsize = winsize
        self.in_channels = in_channels
        self.dims = dims
        # self.dims = [self.in_channels] + list(dims)

        self.e = nn.Sequential(
            nn.Conv1d(in_channels, dims[0], kernel_size=7, padding='same'),
            nn.LayerNorm((dims[0], winsize)),
            nn.ReLU(),
            *[ClassifierResBlock(self.dims[i], self.dims[i+1], 3, 'same', winsize) for i in range(len(self.dims)-1)]
        )

        self.o = nn.Sequential(
            nn.AvgPool1d(kernel_size=winsize), # Nxdims[-1]x1
            nn.Flatten(start_dim=1), # Nxdims[-1]
            nn.Linear(in_features=self.dims[-1], out_features=1)
        )

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.winsize)
        x = self.e(x)
        x = self.o(x)
        return x

# Resnet old
# class ResNetClassifier(nn.Module):
#     def __init__(self, winsize, in_channels):
#         super().__init__()
#         self.winsize = winsize
#         self.in_channels = in_channels

#         self.r = nn.Sequential(
#             ResBlock(in_channels, 32, 9, 4, winsize), # Nx32x101
#             nn.MaxPool1d(kernel_size=3, stride=3), # Nx32x33
#             ResBlock(32, 16, 3, 1, 33), # Nx16x33
#             nn.MaxPool1d(kernel_size=3, stride=3), # Nx16x11
#             ResBlock(16, 8, 3, 1, 11), # Nx8x11
#         )
#         self.o = nn.Sequential(
#             nn.AvgPool1d(kernel_size=11), # Nx8x1
#             nn.Flatten(start_dim=1), # Nx8
#             nn.Linear(in_features=8, out_features=1)
#         )

#     def forward(self, x):
#         x = x.view(-1, self.in_channels, self.winsize)
#         x = self.r(x)
#         x = self.o(x)
#         return x