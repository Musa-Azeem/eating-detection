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
