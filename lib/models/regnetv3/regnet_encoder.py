import torch
from torch import nn
from lib.models.regnetv3.modules import get_lout, XBlockV3, Downsample
import math

class RegNetEncoder(nn.Module):
    def __init__(self, lin, in_channels, stem_out_c, d, w, g=1, p_dropout=0, weights_file=None, freeze=False):
        super().__init__()
        if len(w) != len(d):
            raise ValueError('d and w must have same length')
        if stem_out_c > w[0]:
            raise ValueError('stem_out_c must be less than or equal to w[0]')
        if lin % 2 == 0:
            raise ValueError('winsize must be odd')
        
        k = 3
        s = 2
        p = 3//2
        stem_pre_ln = get_lout(lin, k, s, p)
        mp_s = 2
        mp_k = 2
        stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_out_c, kernel_size=k, stride=s, padding=p),
            nn.LayerNorm((stem_pre_ln)),
            nn.MaxPool1d(kernel_size=mp_k,stride=mp_s),
            nn.ReLU(),
        )
        seq_len = math.floor((stem_pre_ln-mp_k)/mp_s+1) # maxpool formula
        in_c = stem_out_c
        out_c = w[0]
        encoder_stages = nn.Sequential()
        for i,stage in enumerate(d):
            encoder_stage = nn.Sequential()
            for j in range(stage):
                encoder_stage.add_module(
                    f"e_stage-{i}_block-{j}", 
                    XBlockV3(in_c, out_c, 3, seq_len, g, p_dropout))
                in_c = out_c
                # if stem out was less wide than w[0], update seq_len
                if i==0 and j==0 and stem_out_c < w[0]:
                    seq_len = get_lout(seq_len, 3, 2, 3//2)
                # if going to 2nd block of new stage and width is not wider, update seq_len
                if i>0 and j==0 and w[i-1]<w[i]:
                    seq_len = get_lout(seq_len, 3, 2, 3//2)
            # if going to next stage update out_c
            out_c = w[i+1] if i<len(d)-1 else out_c
            encoder_stages.add_module(f"e_stage-{i}", encoder_stage)
        
        # dont change the name of self.e
        self.e = nn.Sequential(
            stem,
            encoder_stages
        )
        self.latent_dim = seq_len

        # self.skip_e = nn.Sequential(
        #     nn.Conv1d(in_channels, w[-1], kernel_size=1),
        #     Downsample(self.latent_dim),
        # )

        if weights_file:
            print("Model is loading pretrained encoder")
            weights = {k[2:]:v for k,v in torch.load(weights_file).items() if k.startswith('e.')}
            self.load_state_dict(weights)
        if freeze:
            print("Freezing encoder")
            for p in self.e.parameters():
                p.requires_grad = False
            for p in self.skip_e.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.e(x)