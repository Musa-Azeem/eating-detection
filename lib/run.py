import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from lib.models import *
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

def train_resnet(epochs, outdir, device, label=''):
    WINSIZE = 101
    DEVICE = device

    model = ResNetClassifier(winsize=WINSIZE, in_channels=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()

    nursing_trainloader = torch.load('pytorch_datasets/nursing_trainloader_11-25-23.pt')
    nursing_testloader = torch.load('pytorch_datasets/nursing_testloader_11-25-23.pt')

    optimization_loop(
        model, 
        nursing_trainloader, 
        nursing_testloader, 
        criterion, 
        optimizer, 
        epochs, 
        DEVICE, 
        patience=30,
        min_delta=0.0001,
        outdir=outdir,
        label=label
    )


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
        patience=30,
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
        patience=30,
        min_delta=0.0001,
        outdir=encoderclass_dir,
        label=label
    )

def train_autoencoder_6(epochs, outdir, device, label=''):
    DEVICE = device
    autoencoder_dir = Path(outdir)

    model = BiggerResNetAE(in_channels=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()

    trainloader = torch.load('pytorch_datasets/trainloader_12-5-23.pt')
    testloader = torch.load('pytorch_datasets/testloader_12-5-23.pt')

    optimization_loop_xonly(
        model, 
        trainloader, 
        testloader, 
        criterion, 
        optimizer, 
        epochs, 
        DEVICE, 
        patience=10,
        min_delta=0.001,
        outdir=autoencoder_dir,
        label=label
    )

from lib.models import MAEAlpha, MAEAlphaClassifier, MAEBeta, MAEBetaClassifier
from lib.config import RAW_DIR, NURSING_RAW_DIR, NURSING_LABEL_DIR
from lib.data.dataloading import load_raw
def train_mae_7(epochs, outdir, device, label=''):
    winsize = 1001
    autoencoder_dir = Path(outdir)

    model = MAEBeta(winsize, 3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    trainloader, testloader = load_raw(
        RAW_DIR,
        winsize,
        n_hours=2,
        test_size=0.5,
        batch_size=256,
        chunk_len_hrs=0.25
    )
    optimization_loop_xonly(
        model, 
        trainloader, 
        testloader, 
        criterion, 
        optimizer, 
        epochs, 
        device, 
        patience=10,
        min_delta=0.001,
        outdir=autoencoder_dir,
        label=label
    )
def train_mae_class_7(epochs, outdir, device, autoencoder_dir=None, freeze=True, label=''):
    torch.multiprocessing.set_sharing_strategy('file_system')
    DEVICE = device
    autoencoder_dir = Path(autoencoder_dir) if autoencoder_dir else None
    encoderclass_dir = Path(outdir)

    winsize = 1001
    nursing_trainloader, nursing_testloader = load_nursing(
        NURSING_RAW_DIR, 
        NURSING_LABEL_DIR, 
        winsize=winsize, 
        n_sessions=16,
        test_size=0.5, 
        batch_size=256,
    )

    model = MAEBetaClassifier(
        winsize, 
        in_channels=3,
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
        patience=10,
        min_delta=0.0001,
        outdir=encoderclass_dir,
        label=label
    )

from lib.models import ResNetClassifier, ResBlock
from torch import nn
from lib.data.dataloading import load_nursing
from pathlib import Path
from lib.modules import optimization_loop, evaluate_loop
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from lib.utils import plot_and_save_cm
import seaborn as sns
from tabulate import tabulate
from tqdm import tqdm

def train_classifier():
    nursing_raw_dir = Path("/home/musa/datasets/nursingv1")
    nursing_label_dir = Path("/home/musa/datasets/eating_labels")
    DEVICE = 'cuda:0'

    winsize = 1001
    nursing_trainloader, nursing_testloader = load_nursing(
        nursing_raw_dir, 
        nursing_label_dir, 
        winsize=winsize, 
        # session_idxs=[50, 52],
        # n_sessions=24, 
        test_size=0.5, 
        batch_size=256,
        # shuffle_test=True
    )
    model = ResNetClassifier(winsize, 3, (8,8,16,32,64,128,256,512)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()
    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))

    train_loss_batch = []
    test_loss_batch = []
    confs = []
    train_stats = {'loss': [], 'prec': [], 'recall': [], 'f1score': []}
    test_stats = {'loss': [], 'prec': [], 'recall': [], 'f1score': []}

    # optimization_loop(model, nursing_trainloader, nursing_testloader, criterion, optimizer, 10, DEVICE)

    for epoch in tqdm(range(1)):
        model.train()
        for X,y in (pbar := tqdm(nursing_trainloader, leave=False)):
            pbar.set_description('Optimizing')
            X,y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        model.eval()
        loss = 0
        ypred = []
        for X,y in (pbar := tqdm(nursing_trainloader, leave=False)):
            pbar.set_description('Evaluating Train')
            X,y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss_i = criterion(logits, y).item()
            train_loss_batch.append(loss_i)
            loss += loss_i
            ypred += torch.round(torch.sigmoid(logits.detach().cpu())).tolist()
        train_stats['loss'].append(loss / len(nursing_trainloader))
        prec, recall, f1score, _ = precision_recall_fscore_support(
            nursing_trainloader.dataset.y.squeeze(), 
            ypred, 
            zero_division='warn', pos_label=1, average='binary'
        )
        train_stats['prec'].append(prec)
        train_stats['recall'].append(recall)    
        train_stats['f1score'].append(f1score)


        loss = 0
        test_ypred = []
        for X,y in (pbar := tqdm(nursing_testloader, leave=False)):
            pbar.set_description('Evaluation Test')
            X,y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss_i = criterion(logits, y).item()
            test_loss_batch.append(loss_i)
            loss += loss_i
            test_ypred += torch.round(torch.sigmoid(logits.detach().cpu())).tolist()
            confs += torch.sigmoid(logits.detach().cpu()).tolist()

        test_stats['loss'].append(loss / len(nursing_testloader))
        prec, recall, f1score, _ = precision_recall_fscore_support(
            nursing_testloader.dataset.y, 
            test_ypred, 
            zero_division=0.0, pos_label=1, average='binary'
        )
        test_stats['prec'].append(prec)
        test_stats['recall'].append(recall)
        test_stats['f1score'].append(f1score)

        torch.save(model.state_dict(), 'dev/test-model.pt')


        print("Train\n", tabulate([["Metric", "Value"], *[[k,v] for k,v in train_stats.items()]], headers="firstrow"))
        print("\nDev\n", tabulate([["Metric", "Value"], *[[k,v] for k,v in test_stats.items()]], headers="firstrow"))


        fig,axes = plt.subplots(1, 3, figsize=(25,10))

        axes[0].plot(train_stats['loss'])
        axes[0].plot(test_stats['loss'])
        axes[0].plot(train_stats['f1score'])
        axes[0].plot(test_stats['f1score'])
        axes[0].legend(['Train Loss', 'Dev Loss', 'Train F1', 'Dev F1'])
        axes[0].set_title('Metrics')

        axes[1].plot(train_loss_batch)
        axes[1].plot(test_loss_batch)
        axes[1].legend(['Train Loss', 'Dev Loss'])

        sns.heatmap(confusion_matrix(y_true=nursing_testloader.dataset.y, y_pred=test_ypred), annot=True, ax=axes[2], cbar=False, fmt='.2f')
        axes[2].set_title('Dev Confusion Matrix')
        axes[2].set(xlabel='Predicted', ylabel='True')
        plt.show()
        plt.savefig('dev/test.png')
