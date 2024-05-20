from lib.models.regnetv3.regnetv3 import RegNetv3
import torch
from torch import nn

class ClassifierLSTM(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.weights_file = CONFIG.get('CLASS_WEIGHTS_FILE', None)
        if not self.weights_file:
            raise ValueError('No weights file provided')
        self.winsize = CONFIG['WINDOW_SIZE']
        self.seq_len = CONFIG['LSTM_SEQLEN']
        self.w = CONFIG['WIDTHI']
        hidden_dim = CONFIG.get('HIDDEN_DIM', 64)
        dropout = CONFIG.get('LSTM_DROP', 0.0)
        num_layers = CONFIG.get('LSTM_LAYERS', 1)

        # self.regnet = RegNetv3(CONFIG=CONFIG)
        regnet = RegNetv3(CONFIG=CONFIG)
        print(f'Loading weights from {self.weights_file}')
        weights = torch.load(self.weights_file)
        if list(weights.keys())[0].startswith('module'):
            weights = {k[7:]:v for k,v in weights.items()}
        regnet.load_state_dict(weights)
        self.e = regnet.e
        for p in self.e.parameters():
            p.requires_grad = False
        self.gp = nn.AvgPool1d(kernel_size=self.e.latent_dim)
        self.lstm = nn.LSTM(
            input_size=self.w[-1],   # number of classes
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, 5)
        )
        
    def forward(self, x):
        x = x.view(x.shape[0], 3, self.seq_len, self.winsize).permute(0,2,1,3)
        b,s,_,_ = x.shape
        x = x.flatten(0,1)
        x = self.e(x)
        x = self.gp(x)
        x = x.view(b,s,-1)
        o, (h,c) = self.lstm(x)
        # get middle hidden state
        x = self.out(o[:,self.seq_len//2])
        return x