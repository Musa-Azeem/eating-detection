from torch import nn
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import torch

def metrics(y_true,y_pred):
    return {
        'true': y_true,
        'pred': y_pred,
        'loss': nn.BCELoss(y_pred, y_true),
        'precision': precision_score(y_true=y_true,y_pred=y_pred,average='macro'),
        'recall': recall_score(y_true=y_true,y_pred=y_pred,average='macro'),
        'f1': f1_score(y_true=y_true,y_pred=y_pred,average='macro')
    }

def summary(metrics: dict):
    m = [["Metric", "Value"]]
    for key in metrics:
        if isinstance(metrics[key], torch.Tensor):
            continue
        m.append([key, metrics[key]])
    print(tabulate(m, headers="firstrow"))

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

def plot_and_save_loss(
    losses: list[float], 
    n_epochs: int,
    filename: str
) -> None:
    """
        Plots a loss curve for given losses.

    Args:
        losses (list[float]): losses
        n_epochs (int): number of epochs that generated these results
        filename (str): file name and path to save image as
    """

    plt.style.use('ggplot')
    plt.figure(figsize=(16,9))
    plt.plot(torch.tensor(losses), label='Train Loss')
    plt.ylabel("Loss")
    plt.xlabel('Epochs')
    plt.legend(loc='lower left')
    plt.title(f'Loss over {n_epochs} Epochs')
    plt.savefig(filename, dpi=400)
    plt.close()

def plot_and_save_losses(
    train_losses: list[float], 
    test_losses: list[float], 
    n_epochs: int,
    filename: str,
    f1: list[float] = None
) -> None:
    """
        Plots a loss curve for given train and test losses.

    Args:
        train_losses (list[float]): train losses
        test_losses (list[float]): test losses
        n_epochs (int): number of epochs that generated these results
        filename (str): file name and path to save image as
    """

    plt.style.use('ggplot')
    plt.figure(figsize=(16,9))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.ylabel("Loss")
    plt.xlabel('Epochs')
    plt.title(f'Train and Test Loss over {n_epochs} Epochs')
    if f1:
        plt.plot(f1, label='F1')
        plt.ylabel("Loss/F1")
        plt.title(f'Train and Test Loss/F1 over {n_epochs} Epochs')
    plt.legend(loc='lower left')
    plt.savefig(filename, dpi=400)
    plt.close()

def get_bouts(y_pred) -> list[dict]:
    last_y = 0
    bouts = []
    for i,y in enumerate(y_pred):
        if y == 0:
            if last_y == 0:
                last_y = y.item()
                continue
            if last_y == 1:
                bouts[-1]['end'] = i
        if y == 1:
            if last_y == 0:
                bouts.append({'start': i, 'end': len(y_pred)})
            if last_y == 1:
                last_y = y.item()
                continue
        last_y = y.item()
    return bouts

def get_bouts_smoothed(y_pred) -> list[dict]:
    last_y = 0
    state = 0
    bouts = []
    for i,y in enumerate(y_pred):
        state = last_y
        last_y = y.item()
        if y == 0:
            if state == 0:
                continue
            if state == 1:
                bouts[-1]['end'] = i
        if y == 1:
            if state == 0:
                if len(bouts) != 0 and i - bouts[-1]['end'] < 1500:
                    continue
                else:
                    bouts.append({'start': i, 'end': len(y_pred)})
            if state == 1:
                continue
    return bouts