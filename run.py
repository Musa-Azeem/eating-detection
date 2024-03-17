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
from lib.models import RegNetv3, RegNetv3Ci
from lib.modules import optimization_loop_multi_class
from lib.data.dataloading import load_nursing_5_class
    
def train_mae_9(CONFIG, outdir, epochs=1000, patience=200, label=''):
    model = RegNetMAEv3(CONFIG=CONFIG).to(CONFIG['DEVICE'])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    sys.stdout = open(os.devnull, 'w')
    trainloader, testloader = load_raw(
        RAW_DIR,
        CONFIG['WINDOW_SIZE'],
        test_size=CONFIG['TEST_SIZE'],
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle_test=True,
        chunk_len_hrs=0.25,
        stride=CONFIG['WINDOW_STRIDE']
    )
    sys.stdout = sys.__stdout__

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
    CONFIG['WEIGHTS_FILE'] = weights_file

    nursing_trainloader, nursing_testloader = load_nursing_5_class(
        range(11,71), 
        CONFIG['WINDOW_SIZE'], 
        test_size=CONFIG['NURSING_TEST_SIZE'], 
        batch_size=CONFIG['BATCH_SIZE'],
        stride=CONFIG['NURSING_STRIDE'],
    )

    model = RegNetv3(CONFIG=CONFIG).to(CONFIG['DEVICE'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [
            {"params": model.o.parameters()},
            {"params": model.e.parameters(), "lr": CONFIG['ENC_LEARNING_RATE']},
        ],
        lr=CONFIG['CLASS_LR']
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

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
def train_multi_class_ci(CONFIG, outdir, epochs=1000, patience=200, weights_file=None, freeze=False, label=''):
    CONFIG['FROZEN'] = freeze
    CONFIG['PRETRAINED'] = weights_file is not None 
    CONFIG['WEIGHTS_FILE'] = weights_file

    nursing_trainloader, nursing_testloader = load_nursing_5_class(
        range(11,71), 
        CONFIG['WINDOW_SIZE'], 
        test_size=CONFIG['NURSING_TEST_SIZE'], 
        batch_size=CONFIG['BATCH_SIZE'],
        stride=CONFIG['NURSING_STRIDE'],
    )

    model = RegNetv3Ci(CONFIG=CONFIG).to(CONFIG['DEVICE'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [
            {"params": model.o.parameters()},
            {"params": [
                *model.e.parameters(),
                *model.trans_skip.parameters(), 
                *model.transformer_encoder.parameters()
            ], "lr": CONFIG['ENC_LEARNING_RATE']},
        ],
        lr=CONFIG['CLASS_LR']
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

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
            'WINDOW_SIZE':2001,
            'WINDOW_STRIDE':2001 // 16,
            'NURSING_STRIDE': 2001 // 16,
            'BATCH_SIZE': 128,
            'LEARNING_RATE': 1e-3,
            'CLASS_LR': 3e-4,
            'ENC_LEARNING_RATE': 5e-5,
            'TEST_SIZE': 0.1,
            'NURSING_TEST_SIZE': 0.25,
            'DEVICE': 'cuda:1',
            'DEPTHI': [],
            'WIDTHI': [],
            'NTL': 1,
            'DMODEL': 0,
            'MASKPCT': 0.5,
            'PDROPOUT': 0.0,
        }
        while True:
            np.random.seed()
            d,w,_ = sample_regnet()
            CONFIG['DEPTHI'] = d
            CONFIG['WIDTHI'] = w
            CONFIG['DMODEL'] = w[-1]
            sys.stdout = open(os.devnull, 'w')
            params = sum([p.numel() for p in RegNetMAEv3(CONFIG=CONFIG).parameters()])
            sys.stdout = sys.__stdout__
            print(d,w,params)
            if params > 10_000_000 and params < 30_000_000:
                break
        outdir = f'dev/9_regnet-mae/random-search-2/mae/{d}-{w}'

        try:
            train_mae_9(
                CONFIG, 
                outdir=outdir,
                epochs=2000, 
                patience=50,
                label=f'{i}:d{d}-w{w}'
            )
            train_multi_class(
                CONFIG,
                outdir=outdir.replace('mae','class') + '-nopretrain-unfrozen',
                epochs=500,
                patience=50,
                weights_file=None,
                freeze=False,
                label=f'{i}:{d}-{w}_class-nopretrain'
            )            
            train_multi_class(
                CONFIG,
                outdir=outdir.replace('mae','class') + '-pretrained-unfrozen',
                epochs=500,
                patience=50,
                weights_file=f'{outdir}/best_model.pt',
                freeze=False,
                label=f'{i}:{d}-{w}_class'
            )
            train_multi_class_ci(
                CONFIG,
                outdir=outdir.replace('mae','class') + '-pretrained-ci-unfrozen',
                epochs=500,
                patience=50,
                weights_file=f'{outdir}/best_model.pt',
                freeze=False,
                label=f'{i}:{d}-{w}_class'
            )
        except FileExistsError:
            print('File exists')
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
        'WINDOW_SIZE': 2001,
        'WINDOW_STRIDE': 2001 // 16,
        'NURSING_STRIDE': 2001 // 16,
        'BATCH_SIZE': 512,
        'LEARNING_RATE': 3e-4,
        'TEST_SIZE': 0.1,
        'NURSING_TEST_SIZE': 0.25,
        'DEVICE':'cuda:0',
        'DEPTHI': [4, 13, 3],
        'WIDTHI': [48, 120, 304],
        'NTL': 1,
        'DMODEL': 304,
        'MASKPCT': 0.05,
        'PDROPOUT': 0.0,
    }

    outdir = f'/home/musa/eating-detection/dev/9_regnet-mae/prototyping-3-16-24/mae2'
    train_mae_9(
        CONFIG, 
        outdir=outdir,
        epochs=100, 
        patience=5,
        label=f'mae'
    )
    train_multi_class(
        CONFIG,
        outdir=outdir.replace('mae2','class') + '-nopretrain',
        epochs=500,
        patience=50,
        weights_file=None,
        freeze=False,
        label=f'no-pretrain'
    )
    train_multi_class(
        CONFIG,
        outdir=outdir.replace('mae2','class') + '-pretrained',
        epochs=500,
        patience=50,
        weights_file=f'{outdir}/best_model.pt',
        freeze=False,
        label=f'pretrained'
        )
    train_multi_class_ci(
        CONFIG,
        outdir=outdir.replace('mae2','class') + '-nopretrain-ci',
        epochs=500,
        patience=50,
        weights_file=None,
        freeze=False,
        label=f'no-pretained-ci'
    )
    train_multi_class_ci(
        CONFIG,
        outdir=outdir.replace('mae2','class') + '-pretrain-ci',
        epochs=500,
        patience=50,
        weights_file=f'{outdir}/best_model.pt',
        freeze=False,
        label=f'pretained-ci'
    )
import json
if __name__ == '__main__':
    train_ae_and_class()