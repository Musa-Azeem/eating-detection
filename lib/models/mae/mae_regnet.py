import torch
from torch import nn

def get_padding(l,out_l,k,s,d=1):
    if l % 2 == 0:
        return ((out_l-1)*s - l + d*(k-1))//2 + 1
    return ((out_l-1)*s - l + d*(k-1) + 1)//2 

class XBlockMAE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, seq_len, b, g, p_dropout=None, relu=True):
        super().__init__()

        stride = 1
        self.seq_len = seq_len
        if out_channels > in_channels:
            stride = 2
            self.seq_len = seq_len // 2 if seq_len % 2 == 0 else seq_len // 2 + 1

        padding = get_padding(seq_len, self.seq_len, kernel_size, stride)

        self.use_relu = relu
        self.c = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // b, kernel_size=1, padding=0),
            nn.LayerNorm((seq_len)),
            nn.ReLU(),
            nn.Conv1d(in_channels // b, in_channels // b, kernel_size=kernel_size, padding=padding, groups=g, stride=stride),
            nn.LayerNorm((self.seq_len)),
            nn.ReLU(),
            nn.Conv1d(in_channels // b, out_channels, kernel_size=1, padding=0),
            nn.LayerNorm((self.seq_len)),
            nn.ReLU()
        )
        if self.use_relu:
            self.c.add_module('relu', nn.ReLU())
        if p_dropout is not None:
            self.c.add_module('dropout', nn.Dropout(p=p_dropout))

        self.identity = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride),
            nn.LayerNorm((self.seq_len)),
            nn.ReLU()
        ) if out_channels > in_channels else nn.Identity()

        self.outrelu = nn.ReLU() if relu else nn.Identity() 

    def forward(self, x):
        return self.outrelu(self.c(x) + self.identity(x))

class XDecoderBlockMAE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, seq_len, out_seq, b, g, p_dropout=None, relu=True):
        super().__init__()

        stride = 1
        self.seq_len = out_seq
        p = 0
        if out_channels < in_channels:
            stride = 2
            if seq_len % 2 == 0:
                p = 1 if out_seq % 2 == 0 else 0
            else:
                p = 1 if out_seq % 2 == 0 else 0

        self.use_relu = relu
        self.c = nn.Sequential(
            nn.ConvTranspose1d(in_channels, in_channels // b, kernel_size=1, padding=0),
            nn.LayerNorm((seq_len)),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels // b, in_channels // b, kernel_size=kernel_size, padding=kernel_size//2, groups=g, stride=stride, output_padding=p),
            nn.LayerNorm((self.seq_len)),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels // b, out_channels, kernel_size=1, padding=0),
            nn.LayerNorm((self.seq_len)),
            nn.ReLU()
        )
        if self.use_relu:
            self.c.add_module('relu', nn.ReLU())
        if p_dropout is not None:
            self.c.add_module('dropout', nn.Dropout(p=p_dropout))

        self.identity = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1, stride=stride, output_padding=p),
            nn.LayerNorm((self.seq_len)),
            nn.ReLU()
        ) if out_channels < in_channels else nn.Identity()

        self.outrelu = nn.ReLU() if relu else nn.Identity() 

    def forward(self, x):
        return self.outrelu(self.c(x) + self.identity(x))

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        position = torch.arange(seq_len).unsqueeze(1)
        div_term =  torch.pow(10000.0, torch.arange(0, d_model, 2) / d_model)
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return x
def get_out_padding(in_seq, out_seq):
    if in_seq % 2 == 0:
        return 1 if out_seq % 2 == 0 else 0
    else:
        return 1 if out_seq % 2 == 0 else 0
    
import math
class RegNetMAE(nn.Module):
    def __init__(self, winsize, in_channels, stem_out_c, d: tuple, w: tuple, d_model, b=1, g=1, p_dropout=None, maskpct=0.75, ntrans=1, nhead=1):
        super().__init__()
        if len(w) != len(d):
            raise ValueError('d and w must have same length')
        
        self.winsize = winsize
        self.in_channels = in_channels
        self.stem_out_c = stem_out_c
        self.n_stage = len(d)
        self.d_str = '-'.join([str(di) for di in d])
        self.w_str = '-'.join([str(wi) for wi in w])
        self.p_dropout = p_dropout
        self.b = b
        self.g = g
        self.ntrans = ntrans
        self.nhead = nhead

        self.d_model = d_model
        self.maskpct = maskpct
        
        w = [stem_out_c] + list(w)
        stem_pre_ln = math.floor(((winsize-1))/2+1)
        stem_out_len = math.floor(((stem_pre_ln-3))/2+1)

        s = nn.Sequential()
        for i in range(self.n_stage):
            rs = nn.Sequential()
            for j in range(d[i]):
                rs.add_module(f'e_stage-{i}_block-{j}', XBlockMAE(
                    w[i] if j==0 else w[i+1], 
                    w[i+1], 
                    kernel_size=3, 
                    seq_len=rs[-1].seq_len if j>0 else s[-1][-1].seq_len if i>0 else stem_out_len, 
                    b=b, 
                    g=g, 
                    relu=True,
                    p_dropout=p_dropout
                ))
            s.add_module(f'e_stage-{i}', rs)

        self.e = nn.Sequential(
            nn.Conv1d(in_channels, stem_out_c, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm((stem_pre_ln)),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.ReLU(),
            s
        )
        self.trans_seq_len = s[-1][-1].seq_len
        print(f'latent dims: {self.trans_seq_len}')
        self.transformer_encoder = nn.Sequential(
            nn.Conv1d(w[-1], d_model, 1),
            Permute(0,2,1),
            PositionalEncoding(d_model, seq_len=self.trans_seq_len),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead, 2048, 0.1, batch_first=True), 
                num_layers=ntrans,
                enable_nested_tensor=False
            ),
            Permute(0,2,1),
            nn.Conv1d(d_model, w[-1], 1),
        )

        ds = nn.Sequential()
        for i in range(self.n_stage):
            rs = nn.Sequential()
            for j in range(d[-i-1]):
                in_seq = rs[-1].seq_len if j>0 else ds[-1][-1].seq_len if i>0 else self.trans_seq_len
                out_seq = in_seq if j < d[-i-1]-1 else s[-i-2][0].seq_len if i < self.n_stage-1 else stem_out_len

                rs.add_module(f'd-stage-{len(d)-i-1}_block-{d[-i-1]-j-1}', XDecoderBlockMAE(
                    w[-i-1], 
                    w[-i-2] if j == d[-i-1]-1 else w[-i-1], 
                    kernel_size=3, 
                    seq_len=in_seq,
                    out_seq=out_seq,
                    b=1, 
                    g=1, 
                    relu=True,
                    p_dropout=0.01
                ))
            ds.add_module(f'd_stage-{len(d)-i-1}', rs)

        self.dec = nn.Sequential(
            ds,
            nn.Upsample(size=stem_pre_ln, mode='linear'),
            nn.ConvTranspose1d(w[0], in_channels, kernel_size=3, stride=2, padding=1, output_padding=get_out_padding(stem_out_len, winsize)),
        )

    def forward(self, x):
        x = self.e(x)
        x = self.mask(x)
        x = self.transformer_encoder(x)
        x = self.dec(x)
        return x
    
    def mask(self, x):
        # Mask: split X into chunks of mask_len size and randomly set maskpct% 
        # of chunks (all channels) to values from a normal distribution
        mask_len = 10
        n_chunks = x.shape[2] // mask_len
        chunked = list(torch.split(x, n_chunks, dim=2))
        mask = torch.rand(len(chunked), x.shape[0]) < self.maskpct # maskpct% of values are True
        for i,mi in enumerate(mask):
            chunked[i] = chunked[i].clone()
            chunked[i][mi] = torch.zeros_like(chunked[i][mi], device=x.device)
        x = torch.cat(chunked, dim=2)
        return x

class RegNetClassifier(nn.Module):
    def __init__(self, winsize, in_channels, stem_out_c, d: tuple, w: tuple, d_model=192, b=1, g=1, p_dropout=None, maskpct=0.75, ntrans=1, nhead=1, weights_file=None, freeze=False):
        """
            stem_out_c: out channels of stem before first stage
            d: tuple of num blocks in each stage
            w: tuple of num channels in each stage
            can leave d_model, b, g, maskpct, ntrans, nhead as default if no weights file
        """        
        super().__init__()
        self.winsize = winsize
        self.in_channels = in_channels
        self.stem_out_c = stem_out_c
        self.n_stage = len(d)
        if len(w) != len(d):
            raise ValueError('d and w must have same length')
        
        self.d_str = '-'.join([str(di) for di in d])
        self.w_str = '-'.join([str(wi) for wi in w])
        self.g = g
        self.b = b
        self.p_dropout = p_dropout

        self.autoencoder_params = dict(
            winsize=winsize, 
            in_channels=in_channels, 
            stem_out_c=stem_out_c, 
            d=d, 
            w=w, 
            d_model=d_model, 
            b=b, 
            g=g, 
            p_dropout=p_dropout, 
            maskpct=maskpct, 
            ntrans=ntrans, 
            nhead=nhead
        )

        self.weights_file = weights_file
        self.freeze = freeze

        s = nn.Sequential()
        w = [stem_out_c] + list(w)

        stem_pre_ln = math.floor(((winsize-1))/2+1)
        stem_out_len = math.floor(((stem_pre_ln-3))/2+1)
        
        self.e, encoder_outdims = self.get_encoder()

        self.o = nn.Sequential(
            nn.AvgPool1d(kernel_size=encoder_outdims), # Nxdims[-1]x1
            nn.Flatten(start_dim=1), # Nxdims[-1]
            nn.Linear(in_features=w[-1], out_features=5)
        )

    def forward(self, x):
        x = self.e(x)
        x = self.o(x)
        return x
    
    def get_encoder(self):
        autoencoder = RegNetMAE(**self.autoencoder_params)

        if self.weights_file:
            print("Model is loading pretrained encoder")
            autoencoder.load_state_dict(torch.load(self.weights_file))
        
        encoder = autoencoder.e

        if self.freeze:
            print("Model is freezing encoder")
            for p in encoder.parameters():
                p.requires_grad = False
        
        return encoder, autoencoder.trans_seq_len