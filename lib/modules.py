import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import json
from .utils import (
    plot_and_save_cm, 
    plot_and_save_loss, 
    plot_and_save_losses,
    summary,
    get_bouts,
    get_bouts_smoothed
)
from tqdm import tqdm
import plotly.express as px
import os


# =============================================================================
# =================== Nursing Data Loading and Processing =====================
# =============================================================================

def read_nursing_session(session_idx, datapath):
    df = pd.read_csv(
        Path(datapath, f'{session_idx}', 'raw_data.csv'), 
        header=None,
        usecols=[2,3,4],
        names=['x_acc','y_acc','z_acc']
    )
    return df

def read_nursing_labels(session_idx, labelpath):
    labels_file = Path(labelpath, f'{session_idx}_data.json')
    if not labels_file.is_file():
        print(f'Error - label file for participant {session_idx} does not exist')
        return None

    labels = json.load(labels_file.open())
    if not ('start' in labels and 'end' in labels):
        print(f'Error - labels for participant {session_idx} do not exist')
        return None

    return labels

def read_and_window_nursing_session(session_idx, winsize, datapath, labelpath):
    df = read_session(session_idx, datapath)
    labels = read_labels(session_idx, labelpath)

    X = torch.Tensor(df.values)
    y = torch.zeros(len(X), 1)
    y[labels['start']:labels['end']] = 1

    X = pad_for_windowing(X, winsize)
    X = window_session(X, winsize)

    return X,y



# =============================================================================
# ==================== Delta Data Loading and Processing ========================
# =============================================================================

# def read_delta_session(filepath):
#     df = pd.read_csv(
#         Path(filepath), 
#         header=None,
#         usecols=[2,3,4],
#         names=['x_acc','y_acc','z_acc']
#     )
#     return df


# =============================================================================
# =========================== Tensor Processing ===============================
# =============================================================================

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



# =============================================================================
# =================== Training and Evaluating Modules =========================
# =============================================================================

def evaluate_loop(
    model: nn.Module, 
    criterion: nn.Module, 
    devloader: DataLoader, 
    device: str,
    outdir: Path = None,

) -> any:

    y_true, y_pred, confs, dev_lossi = inner_evaluate_loop(model, devloader, criterion, device)
    dev_loss = sum(dev_lossi) / len(devloader)

    if outdir:
        plot_and_save_cm(y_true, y_pred, outdir)

    prec, recall, f1score, _ = precision_recall_fscore_support(
        y_true, y_pred, zero_division='warn', pos_label=1, average='binary'
    )

    return ({
        "true": y_true, 
        "pred": y_pred, 
        "conf": confs,
    }, {
        "loss": dev_loss, 
        "precision": prec, 
        "recall": recall, 
        "f1": f1score
    })
    
def inner_evaluate_loop(
    model: nn.Module,
    devloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:

    y_preds = []
    y_true = []
    all_confs = []
    dev_lossi = []

    model.eval()
    for X,y in devloader:
        y_true.append(y)
        X,y = X.to(device), y.to(device)
        logits = model(X)
        dev_lossi.append(criterion(logits, y).item())
        confs = torch.sigmoid(logits).detach().cpu()
        all_confs.append(confs)
        y_preds.append(torch.round(confs))

    return (
        torch.cat(y_true), 
        torch.cat(y_preds), 
        torch.cat(all_confs), 
        dev_lossi
    )


def train_loop(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer, 
    epochs: int,
    device: str,
    outdir: Path = None,
):
    if outdir:
        model_outdir = outdir / 'model'
        model_outdir.mkdir(parents=True)

    train_loss = []

    pbar = tqdm(range(epochs))
    for epoch in pbar:

        # Train Loop
        train_lossi = inner_train_loop(model, trainloader, criterion, optimizer, device)
        train_loss.append(sum(train_lossi) / len(trainloader))

        pbar.set_description(f'Epoch {epoch}: Train Loss: {train_loss[-1]:.5}')

        # Plot loss
        plt.plot(train_loss)
        plt.savefig('running_loss.jpg')

        if outdir:
            torch.save(model.state_dict(), model_outdir / f'{epoch}.pt')
            plot_and_save_loss(train_loss, epochs, str(outdir / 'loss.jpg'))
        
        if epoch != epochs-1:
            plt.close()

    plt.show()

def inner_train_loop(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> list[float]:

    model.train()
    lossi = []
    for Xtr,ytr in trainloader:
        Xtr,ytr = Xtr.to(device),ytr.to(device)

        # Forward pass
        logits = model(Xtr)
        loss = criterion(logits, ytr)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lossi.append(loss.item())
    
    return lossi

def optimization_loop(
    model: nn.Module,
    trainloader: DataLoader,
    devloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer, 
    epochs: int,
    device: str,
    outdir: Path = None,
):
    if outdir:
        model_outdir = outdir / 'model'
        model_outdir.mkdir(parents=True)

    train_loss = []
    dev_loss = []

    lowest_loss = -1

    pbar = tqdm(range(epochs))
    for epoch in pbar:

        # Train Loop
        train_lossi = inner_train_loop(model, trainloader, criterion, optimizer, device)
        train_loss.append(sum(train_lossi) / len(trainloader))            

        # Dev Loop
        y_true, y_pred, confs, dev_lossi = inner_evaluate_loop(model, devloader, criterion, device)
        dev_loss.append(sum(dev_lossi) / len(devloader))

        pbar.set_description(f'Epoch {epoch}: Train Loss: {train_loss[-1]:.5}: Dev Loss: {dev_loss[-1]:.5}')

        # Plot loss
        plt.plot(train_loss)
        plt.plot(dev_loss)
        plt.savefig('running_loss.jpg')

        if outdir:
            torch.save(model.state_dict(), model_outdir / f'{epoch}.pt')
            plot_and_save_losses(train_loss, dev_loss, epochs, str(outdir / 'loss.jpg'))

            # Save model with lowest loss
            if lowest_loss < 0 or dev_loss[-1] < lowest_loss:
                lowest_loss = dev_loss[-1]
                torch.save(model.state_dict(), outdir / f'best_model.pt')
                os.system(f'touch {outdir / str(epoch)}.txt') 
        plt.close()





# =============================================================================
# ================================ Misc =======================================
# =============================================================================

def predict_and_plot_pretty_session(
    session_idx, 
    dim_factor, 
    datapath, 
    labelpath,
    winsize,
    model,
    criterion,
    batch_size,
    device
):
    session = read_session(session_idx, datapath)
    labels = read_labels(session_idx, labelpath)
    X,y = read_and_window_session(session_idx, winsize, datapath, labelpath)
    ys, metrics = evaluate_loop(
        model, 
        criterion, 
        DataLoader(TensorDataset(X, y), batch_size), 
        device
    )
    pred_bouts = get_bouts_smoothed(ys['pred'])

    summary(metrics)

    session['Predicted Eating'] = ys['pred']
    session['Confidence'] = ys['conf']

    fig = px.line(
        session[::dim_factor], 
        x=session.index[::dim_factor], 
        y=['x_acc','y_acc','z_acc', 'Predicted Eating', 'Confidence']
    )
    fig.add_vrect(
        x0=labels['start'], 
        x1=labels['end'], 
        fillcolor='black', 
        opacity=.2,
        line_width=0,
        layer="below"
    )
    for bout in pred_bouts:
        fig.add_vrect(
            x0=bout['start'], 
            x1=bout['end'], 
            fillcolor='red', 
            opacity=.2,
            line_width=0,
            layer="below"
        )
    fig.show(renderer='browser')