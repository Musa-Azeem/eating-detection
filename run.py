from pathlib import Path
from lib.models import RegNetMAEv2
from lib.data.dataloading import load_raw
from lib.modules import optimization_loop_xonly
from lib.config import RAW_DIR
import torch
from torch import nn
import numpy as np
import sys
import os

def sample_regnet():
    initial_width = np.round(int(np.clip(np.exp(np.random.uniform(np.log(8),np.log(64))),0,64)))
    slope = np.round(int(np.clip(np.exp(np.random.uniform(np.log(8),np.log(64))),0,64)))
    network_depth = int(np.clip(np.exp(np.random.uniform(np.log(1),np.log(20)+1)),0,20))
    quantized_param = np.random.uniform(2,3)
    # We need to derive block width and number of blocks from initial parameters.
    parameterized_width = initial_width + slope * np.arange(network_depth)  # From equation 2
    parameterized_block = np.log(parameterized_width / initial_width) / np.log(quantized_param)  # From equation 3

    parameterized_block = np.round(parameterized_block)
    quantized_width = initial_width * np.power(quantized_param, parameterized_block)
    # We need to convert quantized_width to make sure that it is divisible by 8
    quantized_width = 8 * np.round(quantized_width / 8)

    w, d = np.unique(quantized_width.astype(int), return_counts=True)
    if len(d) != 4:
        return sample_regnet()
    else:
        return [int(di) for di in d],[int(wi) for wi in w],[wi for wi,di in zip(w,d) for i in range(di)]
    
def train_mae_9(CONFIG, project_dir, epochs=1000, patience=200, label='', outdirlabel=''):
    p_dropout = 0.1
    model = RegNetMAEv2(
        winsize=CONFIG['WINDOW_SIZE'], 
        in_channels=3, 
        stem_out_c=CONFIG['WIDTHI'][0], 
        d=CONFIG['DEPTHI'], 
        w=CONFIG['WIDTHI'], 
        d_model=CONFIG['DMODEL'], 
        b=1, 
        g=1, 
        p_dropout=p_dropout, 
        ntrans=CONFIG['NTL'], 
        nhead=2,
        maskpct=CONFIG['MASKPCT']
    ).to(CONFIG['DEVICE'])

    criterion = nn.MSELoss()
    # criterion = CosineEmbeddingLossPositive()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    trainloader, testloader = load_raw(
        RAW_DIR,
        CONFIG['WINDOW_SIZE'],
        # n_hours=20,
        test_size=CONFIG['TEST_SIZE'],
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle_test=True,
        chunk_len_hrs=0.25,
        stride=CONFIG['WINDOW_STRIDE']
    )

    outdir = (
        f'{str(type(model)).split(".")[-1][:-2]}'
        f'_lossfn-{str(type(criterion)).split(".")[-1][:-2]}'
        f"_win{CONFIG['WINDOW_SIZE']}"
        f"_stride{CONFIG['WINDOW_STRIDE']}"
        f"_so{model.stem_out_c}_d{model.d_str}_w{model.w_str}"
        f"_b{model.b}_g{model.g}"
        f'_p{p_dropout}'
        f'_ntl{model.ntrans}_nth{model.nhead}_dmodel{model.d_model}'
        f'_maskpct{model.maskpct}{outdirlabel}'
    )
    print('here')
    optimization_loop_xonly(
        model,
        trainloader,
        testloader,
        criterion,
        optimizer,
        epochs=epochs,
        patience=patience,
        config=CONFIG,
        continue_training=False,
        device=CONFIG['DEVICE'],
        outdir=f'dev/{project_dir}/{outdir}',
        writer=f'runs/{project_dir}/{outdir}',
        label=label
    )


def try_wrapper(*args, **kwargs):
    try:
        train_mae_9(*args, **kwargs)
    except RuntimeError as e:
        if not 'CUDA out of memory' in str(e):
            raise e
        print(e)
        CONFIG = (args[0] if len(args) > 0 else kwargs['CONFIG']).copy()
        for bi in [64,32]:
            CONFIG['BATCH_SIZE'] = bi
            args = list(args)
            args[0] = CONFIG
            print(args)
            try:
                train_mae_9(*args, **kwargs)
                break
            except RuntimeError as e:
                if not 'CUDA out of memory' in str(e):
                    raise e
                print(e)

if __name__ == '__main__':
    for i in range(1000):
        CONFIG = {
            'WINDOW_SIZE':3901,
            'WINDOW_STRIDE':3901,
            'BATCH_SIZE':128,
            'LEARNING_RATE':3e-4,
            'TEST_SIZE':0.2,
            'DEVICE':'cuda:0',
            'DEPTHI': [],
            'WIDTHI': [],
            'NTL': None,
            'DMODEL': None,
            'MASKPCT': 0.15
        }
        CONFIG['DMODEL'] = int(np.random.choice([64,128,256]))
        CONFIG['NTL'] = int(np.random.choice([1,2,3]))
        while True:
            d,w,_ = sample_regnet()
            sys.stdout = open(os.devnull, 'w')
            params = sum([p.numel() for p in RegNetMAEv2(winsize=CONFIG['WINDOW_SIZE'],in_channels=3,stem_out_c=w[0],d=d,w=w,d_model=CONFIG['DMODEL'],b=1,g=1,p_dropout=0.1,ntrans=CONFIG['NTL'],nhead=2,maskpct=CONFIG['MASKPCT']).parameters()])
            sys.stdout = sys.__stdout__
            print(params)
            if params < 3000000:
                break
        CONFIG['DEPTHI'] = d
        CONFIG['WIDTHI'] = w
        try:
            try_wrapper(
                CONFIG, 
                project_dir='9_regnet-mae/mae-search',
                epochs=200, 
                patience=50, 
                label=f'{i}:w{CONFIG["WINDOW_SIZE"]}-s{CONFIG["WINDOW_STRIDE"]}-d{d}-w{w}'
            )
        except FileExistsError:
            pass
