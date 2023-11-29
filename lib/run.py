import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from lib.models import MLP2hl, ResAutoEncoder, ResEncoderClassifier, ResEncoderClassifierAve
from lib.modules import optimization_loop_xonly, optimization_loop, pad_for_windowing, window_session, evaluate_loop
from lib.utils import plot_and_save_losses, plot_and_save_cm, summary
from datetime import datetime

def pipeline():
    WINSIZE = 101
    DEVICE = 'cuda:1'

    data_dir = Path('../data/andrew/new')
    acceleration = pd.read_csv(data_dir / 'acceleration.csv',skiprows=1).rename({'x':'x_acc', 'y':'y_acc', 'z':'z_acc'}, axis=1)
    acceleration_start_time_seconds = float(pd.read_csv(data_dir / 'acceleration.csv',nrows=1,header=None).iloc[0,0].split()[-1])/1000
    acceleration.timestamp = ((acceleration.timestamp - acceleration.timestamp[0])*1e-9)+acceleration_start_time_seconds

    start = int(datetime.datetime(2023, 10, 26, 16, 20, 0).strftime('%s'))
    end = int(datetime.datetime(2023, 10, 26, 16, 37, 0).strftime('%s'))
    acceleration['label'] = 0
    acceleration.loc[(acceleration.timestamp > start) & (acceleration.timestamp < end),'label'] = 1

    Xte = torch.Tensor(acceleration[['x_acc','y_acc','z_acc']].values)
    yte = torch.Tensor(acceleration['label'].values).unsqueeze(1)

    Xte = pad_for_windowing(Xte, WINSIZE)
    Xte = window_session(Xte, WINSIZE)

    model = MLP2hl([20,20], WINSIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    model.load_state_dict(torch.load(Path('dev/mlp2hl/best_model.pt')))

    testloader = DataLoader(TensorDataset(Xte,yte), batch_size=64)
    ys,metrics = evaluate_loop(model, criterion, testloader, DEVICE)
    plot_and_save_cm(ys['true'], ys['pred'])
    summary(metrics)

    torch.save(ys['pred'], 'pred.pt')
    pd.DataFrame({'pred': ys['pred'].flatten().numpy()}).to_csv('pred.csv', index=False)


def train_autoencoder(epochs, outdir, device, label=''):
    WINSIZE = 101
    DEVICE = device
    autoencoder_dir = Path(outdir)

    model = ResAutoEncoder(winsize=WINSIZE, in_channels=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()

    trainloader = torch.load('pytorch_datasets/trainloader_11-25-23.pt')
    testloader = torch.load('pytorch_datasets/testloader_11-25-23.pt')

    optimization_loop_xonly(
        model, 
        trainloader, 
        testloader, 
        criterion, 
        optimizer, 
        epochs, 
        DEVICE, 
        patience=20,
        min_delta=0.0001,
        outdir=autoencoder_dir,
        label=label
    )

def train_encoderclassifier(epochs, outdir, device, autoencoder_dir=None, freeze=True, label=''):
    WINSIZE = 101
    DEVICE = device
    autoencoder_dir = Path(autoencoder_dir) if autoencoder_dir else None
    encoderclass_dir = Path(outdir)

    nursing_trainloader = torch.load('pytorch_datasets/nursing_trainloader_11-25-23.pt')
    nursing_testloader = torch.load('pytorch_datasets/nursing_testloader_11-25-23.pt')

    model = ResEncoderClassifier(
        WINSIZE, 
        weights_file=autoencoder_dir / 'best_model.pt' if autoencoder_dir else None, 
        freeze=freeze
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()

    optimization_loop(
        model, 
        nursing_trainloader, 
        nursing_testloader, 
        criterion, optimizer, 
        epochs, 
        DEVICE, 
        patience=20,
        min_delta=0.0001,
        outdir=encoderclass_dir,
        label=label
    )


def train_encoderclassifier_avgpool(epochs, outdir, device, autoencoder_dir=None, freeze=True, label=''):
    WINSIZE = 101
    DEVICE = device
    autoencoder_dir = Path(autoencoder_dir) if autoencoder_dir else None
    encoderclass_dir = Path(outdir)

    nursing_trainloader = torch.load('pytorch_datasets/nursing_trainloader_11-25-23.pt')
    nursing_testloader = torch.load('pytorch_datasets/nursing_testloader_11-25-23.pt')

    model = ResEncoderClassifierAve(
        WINSIZE, 
        weights_file=autoencoder_dir / 'best_model.pt' if autoencoder_dir else None, 
        freeze=freeze
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()

    optimization_loop(
        model, 
        nursing_trainloader, 
        nursing_testloader, 
        criterion, optimizer, 
        epochs, 
        DEVICE, 
        patience=20,
        min_delta=0.0001,
        outdir=encoderclass_dir,
        label=label
    )