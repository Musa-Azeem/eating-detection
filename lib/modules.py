import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics import precision_recall_fscore_support, f1_score,recall_score,precision_score,confusion_matrix
import pandas as pd
import json

def read_and_window_session(session_idx, winsize, datapath, labelpath):
    df = pd.read_csv(
        Path(datapath, f'{session_idx}', 'raw_data.csv'), 
        header=None,
        usecols=[2,3,4]
    )

    labels = json.load(
        Path(labelpath, f'{session_idx}_data.json').open()
    )

    X = torch.Tensor(df.values)
    y = torch.zeros(len(X), 1)
    y[labels['start']:labels['end']] = 1

    X = pad_for_windowing(X, winsize)
    X = window_session(X, winsize)

    return X,y

def read_session(session_idx, datapath):
    df = pd.read_csv(
        Path(datapath, f'{session_idx}', 'raw_data.csv'), 
        header=None,
        usecols=[2,3,4],
        names=['x_acc','y_acc','z_acc']
    )
    return df

def pad_for_windowing(X: torch.Tensor, winsize: int) -> torch.Tensor:
    if winsize % 2 != 1:
        print("winsize must be odd")
        return
    
    p = winsize // 2
    X = nn.functional.pad(X, (0, 0, p, p), "constant", 0)
    return X



def window_session(X: torch.Tensor, winsize: int) -> torch.Tensor:
    # Input shape: L x 3
    # Output shape: L x 3*WINSIZE

    # Window session
    x_acc = X[:,0].unsqueeze(1)
    y_acc = X[:,1].unsqueeze(1)
    z_acc = X[:,2].unsqueeze(1)

    w = winsize-1

    xs = [x_acc[:-w]]
    ys = [y_acc[:-w]]
    zs = [z_acc[:-w]]

    for i in range(1,w):
        xs.append(x_acc[i:i-w])
        ys.append(y_acc[i:i-w])
        zs.append(z_acc[i:i-w])

    xs.append(x_acc[w:])
    ys.append(y_acc[w:])
    zs.append(z_acc[w:])

    xs = torch.cat(xs,axis=1).float()
    ys = torch.cat(ys,axis=1).float()
    zs = torch.cat(zs,axis=1).float()

    X = torch.cat([xs,ys,zs], axis=1)
    return X

def metrics(y_true,y_pred):
    return {
        'precision':precision_score(y_true=y_true,y_pred=y_pred,average='macro'),
        'recall':recall_score(y_true=y_true,y_pred=y_pred,average='macro'),
        'f1':f1_score(y_true=y_true,y_pred=y_pred,average='macro')
    }

def evaluate_loop(
    model: nn.Module, 
    criterion: nn.Module, 
    devloader: DataLoader, 
    device: str,
    metrics: bool = False,
    outdir: Path = None,

) -> any:

    y_true, y_pred, dev_lossi = inner_evaluate_loop(model, devloader, criterion, device)
    dev_loss = sum(dev_lossi) / len(devloader)

    if outdir:
        plot_and_save_cm(y_true, y_pred, outdir)

    if metrics:
        prec, recall, f1score, _ = precision_recall_fscore_support(
            y_true, y_pred, zero_division='warn', pos_label=1, average='binary'
        )

        return y_true, y_pred, dev_loss, prec, recall, f1score
    
    return y_true, y_pred, dev_loss

def inner_evaluate_loop(
    model: nn.Module,
    devloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:

    y_preds = []
    y_true = []
    dev_lossi = []

    model.eval()
    for X,y in devloader:
        y_true.append(y)
        X,y = X.to(device), y.to(device)
        logits = model(X)
        dev_lossi.append(criterion(logits, y).item())
        pred = torch.round(nn.Sigmoid()(logits)).detach().to('cpu')
        y_preds.append(pred)

    return (torch.cat(y_true), torch.cat(y_preds), dev_lossi)

def plot_and_save_cm(
    y_true, 
    y_pred, 
    filename: str = None
) -> None:
    """ 
        Plot and save confusion matrix (recall, precision, and total) for 
        given true labels and predictions. Saves plot to image with given 
        filename

    Args:
        y_true: True labels, 0 or 1 for each example
        y_pred: Predictions - same length as y_true
        filename (str): file name and path to save image as
    """

    fig,axes = plt.subplots(1,3,sharey=True,figsize=(10,5))

    sns.heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred,normalize='true'),annot=True,ax=axes[0],cbar=False,fmt='.2f')
    sns.heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred,normalize='pred'),annot=True,ax=axes[1],cbar=False,fmt='.2f')
    sns.heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred),annot=True,ax=axes[2],cbar=False,fmt='.2f')

    axes[0].set_title('Recall')
    axes[1].set_title('Precision')
    axes[2].set_title('Count')
    fig.set_size_inches(16, 9)
    
    if filename:
        plt.savefig(filename, dpi=400, bbox_inches='tight')
        plt.close()
    else:
        plt.show()