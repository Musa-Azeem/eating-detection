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



class ResNetClassifier(nn.Module):
    def __init__(self, winsize, in_channels):
        super().__init__()
        self.winsize = winsize
        self.in_channels = in_channels

        self.r = nn.Sequential(
            ResBlock(in_channels, 32, 9, 4, winsize), # Nx32x101
            nn.MaxPool1d(kernel_size=3, stride=3), # Nx32x33
            ResBlock(32, 16, 3, 1, 33), # Nx16x33
            nn.MaxPool1d(kernel_size=3, stride=3), # Nx16x11
            ResBlock(16, 8, 3, 1, 11), # Nx8x11
        )
        self.o = nn.Sequential(
            nn.AvgPool1d(kernel_size=11), # Nx8x1
            nn.Flatten(start_dim=1), # Nx8
            nn.Linear(in_features=8, out_features=1)
        )

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.winsize)
        x = self.r(x)
        x = self.o(x)
        return x


# class ResNetClassifier(nn.Module):
#     def __init__(self, in_channels, winsize):
#         super().__init__()
#         self.winsize = winsize

#         self.resnet_conv = ResNetConv(in_channels=in_channels, winsize=self.winsize)
#         # Global Pooling and output
#         self.gp = nn.AvgPool1d(kernel_size=self.winsize)    # Take mean across each feature map (N, C, L) => (N,C)
#         self.output = nn.Linear(in_features=5, out_features=1)

#     def forward(self, x):
#         # Run convolutional layers of ResNet
#         y = self.resnet_conv(x)

#         # Run Global pooling and classifier
#         y = self.gp(y).squeeze(2)
#         logits = self.output(y)
#         return logits


# class ResNetConv(nn.Module):
#     def conv_block(self, in_channels, out_channels, kernel_size, use_relu=True):
#         if use_relu:
#             return nn.Sequential(
#                 nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
#                 nn.BatchNorm1d(num_features=out_channels),
#                 nn.ReLU()
#             )
#         else:
#             return nn.Sequential(
#                 nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
#                 nn.BatchNorm1d(num_features=out_channels)
#             )
    
#     def inner_res_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             self.conv_block(in_channels=in_channels, out_channels=out_channels, kernel_size=8),
#             self.conv_block(in_channels=out_channels, out_channels=out_channels, kernel_size=5),
#             self.conv_block(in_channels=out_channels, out_channels=out_channels, kernel_size=3, use_relu=False)
#         )
    
#     def shortcut(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
#             nn.BatchNorm1d(num_features=out_channels)
#         )

#     def __init__(self, in_channels, winsize):
#         super().__init__()
#         self.winsize = winsize

#         # First ResNet Block components
#         self.shortcut1 = self.shortcut(in_channels=in_channels, out_channels=3)
#         self.res1 = self.inner_res_block(in_channels=in_channels, out_channels=3)
#         self.relu1 = nn.ReLU()

#         # # Second Res Block components
#         # self.shortcut2 = self.shortcut(in_channels=8, out_channels=16)
#         # self.res2 = self.inner_res_block(in_channels=8, out_channels=16)
#         # self.relu2 = nn.ReLU()

#         # Third Res Block components
#         self.shortcut3 = self.shortcut(in_channels=3, out_channels=5)
#         self.res3 = self.inner_res_block(in_channels=3, out_channels=5)
#         self.relu3 = nn.ReLU()

#     def forward(self, x):
#         # Reshape x: (batch_size, 303) -> (batch_size, 3, 101)
#         x = x.view(-1, 3, self.winsize)
        
#         # First Res Block
#         x_shortcut = self.shortcut1(x)
#         h = self.res1(x)
#         y = h + x_shortcut
#         y = self.relu1(y)

#         # # Second Res Block
#         # y_shortcut = self.shortcut2(y)
#         # h = self.res2(y)
#         # y = h + y_shortcut
#         # y = self.relu2(y)

#         # Third Res Block
#         y_shortcut = self.shortcut3(y)
#         h = self.res3(y)
#         y = h + y_shortcut
#         y = self.relu3(y)

#         return y