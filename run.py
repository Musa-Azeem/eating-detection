from pathlib import Path
from lib.models import RegNetMAEv3, CosineMSELoss
from lib.data.dataloading import load_raw
from lib.modules import optimization_loop_xonly, sample_regnet
from lib.config import RAW_DIR
import torch
from torch import nn
import numpy as np
import sys
import os
from lib.models import RegNetv3
from lib.modules import optimization_loop_multi_class
from lib.data.dataloading import load_nursing_5_class
    
def train_mae_9(CONFIG, outdir, epochs=1000, patience=200, label=''):
    model = RegNetMAEv3(CONFIG=CONFIG).to(CONFIG['DEVICE'])
    criterion = CosineMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    trainloader, testloader = load_raw(
        RAW_DIR,
        CONFIG['WINDOW_SIZE'],
        test_size=CONFIG['TEST_SIZE'],
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle_test=True,
        chunk_len_hrs=0.25,
        stride=CONFIG['WINDOW_STRIDE']
    )

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
        outdir=outdir,
        writer=outdir,
        label=label
    )
def train_multi_class(CONFIG, outdir, epochs=1000, patience=200, weights_file=None, freeze=False, label=''):
    CONFIG['FROZEN'] = freeze
    CONFIG['PRETRAINED'] = weights_file is not None 

    nursing_trainloader, nursing_testloader = load_nursing_5_class(
        range(11,71), 
        CONFIG['WINDOW_SIZE'], 
        test_size=CONFIG['NURSING_TEST_SIZE'], 
        batch_size=CONFIG['BATCH_SIZE'],
        stride=CONFIG['NURSING_STRIDE'],
    )

    model = RegNetv3(CONFIG=CONFIG).to(CONFIG['DEVICE'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    optimization_loop_multi_class(
        model,
        nursing_trainloader,
        nursing_testloader,
        criterion,
        optimizer,
        epochs=epochs,
        device=CONFIG['DEVICE'],
        patience=patience,
        outdir=outdir,
        writer=outdir,
        config=CONFIG,
        label=label
    )


def random_search():
    for i in range(1000):
        CONFIG = {
            'WINDOW_SIZE':3901,
            'WINDOW_STRIDE':3901 // 4,
            'NURSING_STRIDE': 3901,
            'BATCH_SIZE':128,
            'LEARNING_RATE':3e-4,
            'TEST_SIZE':0.1,
            'NURSING_TEST_SIZE': 0.25,
            'DEVICE':'cuda:0',
            'DEPTHI': [2],
            'WIDTHI': [64],
            'NTL': 1,
            'DMODEL': 64,
            'MASKPCT': 0.15,
            'PDROPOUT': 0.01,
        }
        CONFIG['DMODEL'] = int(np.random.choice([64,128,256]))
        CONFIG['NTL'] = int(np.random.choice([1,2,3]))
        while True:
            d,w,_ = sample_regnet()
            sys.stdout = open(os.devnull, 'w')
            params = sum([p.numel() for p in RegNetMAEv3(CONFIG=CONFIG).parameters()])
            sys.stdout = sys.__stdout__
            print(params)
            if params < 3000000:
                break
        CONFIG['DEPTHI'] = d
        CONFIG['WIDTHI'] = w
        try:
            train_mae_9(
                CONFIG, 
                project_dir='9_regnet-mae/mae-search',
                epochs=200, 
                patience=50,
                label=f'{i}:w{CONFIG["WINDOW_SIZE"]}-s{CONFIG["WINDOW_STRIDE"]}-d{d}-w{w}'
            )
        except FileExistsError:
            pass

def train_ae():
    CONFIG = {
        'WINDOW_SIZE':3901,
        'WINDOW_STRIDE':3901 // 16,
        'NURSING_STRIDE': 3901,
        'BATCH_SIZE':256,
        'LEARNING_RATE':3e-4,
        'TEST_SIZE':0.1,
        'NURSING_TEST_SIZE': 0.25,
        'DEVICE':'cuda:0',
        'DEPTHI': [2],
        'WIDTHI': [64],
        'NTL': 1,
        'DMODEL': 64,
        'MASKPCT': 0.15,
        'PDROPOUT': 0.01,
    }
    train_mae_9(
        CONFIG, 
        project_dir='9_regnet-mae/more-data',
        epochs=3500, 
        patience=50,
        outdir='2024-02-26'
    )

def train_ae_and_class():
    CONFIG = {
        'WINDOW_SIZE': 0,
        'WINDOW_STRIDE': 0,
        'NURSING_STRIDE': 0,
        'BATCH_SIZE':256,
        'LEARNING_RATE': 3e-4,
        'TEST_SIZE': 0.1,
        'NURSING_TEST_SIZE': 0.25,
        'DEVICE':'cuda:1',
        'DEPTHI': [2],
        'WIDTHI': [64],
        'NTL': 1,
        'DMODEL': 64,
        'MASKPCT': 0.0,
        'PDROPOUT': 0.01,
    }
    for winsize in [1001,2001,3001]:
        CONFIG['WINDOW_SIZE'] = winsize
        CONFIG['WINDOW_STRIDE'] = winsize // 4
        CONFIG['NURSING_STRIDE'] = winsize // 4

        outdir = f'dev/9_regnet-mae/winsize-search/w{winsize}-nopretrain'
        train_multi_class(
            CONFIG,
            outdir=outdir.replace('winsize-search','winsize-search-class'),
            epochs=500,
            patience=50,
            weights_file=None,
            freeze=False
        )
        for mask_pct in [0.0, 0.4]:
            CONFIG['MASKPCT'] = mask_pct
            outdir = f'dev/9_regnet-mae/winsize-search/w{winsize}-maskpct{mask_pct}'
            train_mae_9(
                CONFIG, 
                outdir=outdir,
                epochs=100, 
                patience=5,
                label=f'w{winsize}-maskpct{mask_pct}_ae'
            )
            train_multi_class(
                CONFIG,
                outdir=outdir.replace('winsize-search','winsize-search-class'),
                epochs=500,
                patience=50,
                weights_file=f'{outdir}/best_model.pt',
                freeze=False,
                label=f'w{winsize}-maskpct{mask_pct}_class'
            )
import json
if __name__ == '__main__':
    for ae_dir in Path('dev/9_regnet-mae/winsize-search').iterdir():
        outdir = str(ae_dir).replace('winsize-search','winsize-search-class') + '-unfrozen'
        CONFIG = json.load(open(ae_dir / 'config.json'))
        CONFIG['DEVICE'] = 'cuda:0'
        train_multi_class(
            CONFIG,
            outdir=outdir,
            epochs=500,
            patience=50,
            weights_file=f'{outdir}/best_model.pt',
            freeze=False,
            label=f'w{CONFIG["WINDOW_SIZE"]}-maskpct{CONFIG["MASKPCT"]}_class'
        )