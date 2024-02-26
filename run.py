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
    
def train_mae_9(CONFIG, project_dir, epochs=1000, patience=200, label='', outdir=''):
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
        outdir=f'dev/{project_dir}/{outdir}',
        writer=f'runs/{project_dir}/{outdir}',
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

def train_one():
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
if __name__ == '__main__':
    train_one()