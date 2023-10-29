from torch import nn

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
    def __init__(self, winsize):
        super().__init__()
        self.winsize = winsize
        self.p_size = 101
        self.n_channels = 3

        self.lstm = nn.LSTM(
            input_size=self.p_size*self.n_channels,
            hidden_size=64,
            bias=False,
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features=64, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1)
        )

    def partition_window(self, x, p_size, n_channels):
        x = x.view(-1, n_channels, self.winsize)
        x = x.transpose(1,2).view(-1, self.winsize//p_size, p_size, n_channels)
        x = x.transpose(2,3).flatten(start_dim=2)
        return x

    def forward(self, x):
        x = self.partition_window(x, p_size=self.p_size, n_channels=self.n_channels)
        o, (h,c) = self.lstm(x) # o is shape (batch_size, p_size, 64)
        o = nn.ReLU()(o[:,-1,:])
        logits = self.mlp(o)

        return logits
