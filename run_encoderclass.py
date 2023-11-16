import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from lib.modules import (
    evaluate_loop, 
    read_and_window_nursing_session,
    read_nursing_session,
    train_loop,
    optimization_loop,
    predict_and_plot_pretty_session
)
from lib.utils import (
    plot_and_save_cm,
    summary
)
from lib.models import  ConvAutoencoder, EncoderClassifier
from tqdm import tqdm
import plotly.express as px
from tabulate import tabulate
import numpy as np
from sklearn.model_selection import train_test_split


raw_dir = Path("/home/musa/datasets/nursingv1")
label_dir = Path("/home/musa/datasets/eating_labels")
WINSIZE = 101
DEVICE = 'cuda:1'

all_sessions = set(range(71))
not_labeled = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 34, 70}
reserved = {15, 43, 45, 51, 55}
sessions = list(all_sessions - not_labeled - reserved)
np.random.seed(10)
train_sessions, test_sessions = train_test_split(sessions, test_size=0.25)

Xs = []
ys = []

for session_idx in train_sessions:
    X,y = read_and_window_nursing_session(session_idx, WINSIZE, raw_dir, label_dir)

    Xs.append(X)
    ys.append(y)

Xtr = torch.cat(Xs)
ytr = torch.cat(ys)

Xs = []
ys = []

for session_idx in test_sessions:
    X,y = read_and_window_nursing_session(session_idx, WINSIZE, raw_dir, label_dir)

    Xs.append(X)
    ys.append(y)

Xte = torch.cat(Xs)
yte = torch.cat(ys)

model = EncoderClassifier(WINSIZE, weights_file='dev/autoencoder3/best_model.pt', freeze=True).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.BCEWithLogitsLoss()

trainloader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
testloader = DataLoader(TensorDataset(Xte,yte), batch_size=64)

optimization_loop(model, trainloader, testloader, criterion, optimizer, 100, DEVICE, Path('dev/encoderclass3'))