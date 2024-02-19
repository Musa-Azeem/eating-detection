from pathlib import Path
from lib.models import RegNetClassifier
from lib.data.dataloading import load_nursing_5_class
from lib.modules import optimization_loop_multi_class
from lib.config import RAW_DIR
import torch
from torch import nn
import json
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

def train_mae_9_class(CONFIG, weights_file, freeze, epochs=1000, patience=500, project_dir='', label='', outdirlabel=''):
    CONFIG['FROZEN'] = freeze
    CONFIG['PRETRAINED'] = weights_file is not None

    p_dropout = 0.1
    model = RegNetClassifier(
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
        maskpct=CONFIG['MASKPCT'],
        weights_file=weights_file,
        freeze=freeze
    ).to(CONFIG['DEVICE'])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    nursing_trainloader, nursing_testloader = load_nursing_5_class(
        range(11,71), 
        CONFIG['WINDOW_SIZE'], 
        test_size=CONFIG['TEST_SIZE'], 
        batch_size=CONFIG['BATCH_SIZE'],
        stride=CONFIG['WINDOW_STRIDE'],
    )

    pretrained = 'pretrained' if weights_file else 'random'
    frozen = 'frozen' if freeze else 'unfrozen'
    outdir = (
        f'{str(type(model)).split(".")[-1][:-2]}'
        f'_lossfn-{str(type(criterion)).split(".")[-1][:-2]}'
        f"_win{CONFIG['WINDOW_SIZE']}"
        f"_stride{CONFIG['WINDOW_STRIDE']}"
        f"_so{model.stem_out_c}_d{model.d_str}_w{model.w_str}"
        f"_b{model.b}_g{model.g}"
        f'_p{p_dropout}'
        f'_ntl{model.autoencoder_params["ntrans"]}'
        f'_nth{model.autoencoder_params["nhead"]}'
        f'_dmodel{model.autoencoder_params["d_model"]}'
        f'_maskpct{model.autoencoder_params["maskpct"]}'
        f'_{pretrained}_{frozen}{outdirlabel}'
    )
    optimization_loop_multi_class(
        model,
        nursing_trainloader,
        nursing_testloader,
        criterion,
        optimizer,
        epochs=epochs,
        patience=patience,
        device=CONFIG['DEVICE'],
        outdir=f'dev/{project_dir}/{outdir}',
        writer=f'runs/{project_dir}/{outdir}',
        config=CONFIG,
        label=label
    )

def try_wrapper(*args, **kwargs):
    try:
        train_mae_9_class(*args, **kwargs)
    except RuntimeError as e:
        if not 'CUDA out of memory' in str(e):
            raise e
        print(e)
        CONFIG = (args[0] if len(args) > 0 else kwargs['CONFIG']).copy()
        # CONFIG = CONFIG.copy()
        for bi in [64,32]:
            CONFIG['BATCH_SIZE'] = bi
            args = list(args)
            args[0] = CONFIG
            try:
                train_mae_9_class(*args, **kwargs)
                break
            except RuntimeError as e:
                if not 'CUDA out of memory' in str(e):
                    raise e
                print(e)


def train_pretrained_models():
    # all:
    for autoencoder_dir in Path('/home/musa/eating/eating-detection/dev/9_regnet-mae/dev-2-5-24').iterdir():
        CONFIG = json.load((autoencoder_dir / 'config.json').open())
        CONFIG['DEVICE'] = 'cuda:0'
        weights_file = autoencoder_dir / 'best_model.pt'

        try_wrapper(CONFIG, weights_file, True, 20000, 1000)
        try_wrapper(CONFIG, weights_file, False, 20000, 1000)
        try_wrapper(CONFIG, None, False, 20000, 1000)


def train_random_models():
    CONFIG = {
        'WINDOW_SIZE':3901,
        'WINDOW_STRIDE':3901,
        'BATCH_SIZE':256,
        'LEARNING_RATE':3e-4,
        'TEST_SIZE':0.2,
        'DEVICE':'cuda:1',
        'DEPTHI': [2],
        'WIDTHI': [64],
        'NTL': 1,
        'DMODEL': 32,
        'MASKPCT': 0.0
    }
    try_wrapper(CONFIG, None, False, 1000, 200)

def stride_search():
    CONFIG = {
        'WINDOW_SIZE':3901,
        'WINDOW_STRIDE':0,
        'BATCH_SIZE':128,
        'LEARNING_RATE':3e-4,
        'TEST_SIZE':0.2,
        'DEVICE':'cuda:1',
        'DEPTHI': [],
        'WIDTHI': [],
        'NTL': 1,
        'DMODEL': 2,
        'MASKPCT': 0.0
    }
    winsizes = [101, 501, 1001, 2001, 3001, 3901]
    stride_pcnts = [1, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]

    for i in range(1000):
        winsize = np.random.choice(winsizes)
        stride_pcnt = np.random.choice(stride_pcnts)
        CONFIG['WINDOW_SIZE'] = int(winsize)
        CONFIG['WINDOW_STRIDE'] = int(np.ceil(winsize * stride_pcnt))
        while True:
            d,w,_ = sample_regnet()
            sys.stdout = open(os.devnull, 'w')
            params = sum([p.numel() for p in RegNetClassifier(winsize=CONFIG['WINDOW_SIZE'],in_channels=3,stem_out_c=w[0],d=d,w=w,d_model=CONFIG['DMODEL'],b=1,g=1,p_dropout=0.1,ntrans=CONFIG['NTL'],nhead=2,maskpct=CONFIG['MASKPCT'],weights_file=None,freeze=True).parameters()])
            sys.stdout = sys.__stdout__
            if params < 1000000:
                break
        CONFIG['DEPTHI'] = d
        CONFIG['WIDTHI'] = w
        try:
            try_wrapper(
                CONFIG=CONFIG, 
                weights_file=None, 
                freeze=False, 
                epochs=200, 
                patience=50, 
                project_dir='stride-search-rand', 
                label=f'{i}:w{winsize}-s{CONFIG["WINDOW_STRIDE"]}-d{d}-w{w}'
            )
        except FileExistsError:
            pass

if __name__ == '__main__':
    if len(sys.argv) == 1:
        train_mae_9_class()
        exit(0)
    
    locals()[sys.argv[1]]()