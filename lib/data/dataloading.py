from lib.data.datasets import AccAndLabelsDataset, AccRawDataset
import numpy as np
from sklearn.model_selection import train_test_split
from lib.modules import pad_for_windowing, read_nursing_session, read_nursing_labels
import torch
from torch.utils.data import DataLoader, TensorDataset
import random

def load_nursing(raw_dir, label_dir, winsize, n_sessions=None, session_idxs=None, test_size=0.25, batch_size=64, shuffle_test=False):
    not_labeled = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 34, 70}
    reserved = {15, 43, 45, 51, 55}

    if session_idxs:
        unlabled_sessions = set.intersection(set(session_idxs), not_labeled)
        sessions_in_reserved = set.intersection(set(session_idxs), reserved)
        out_of_bounds = set(session_idxs) - set(range(71))
        if unlabled_sessions:
            raise ValueError(f"Session indexes {unlabled_sessions} are not labled")
        if sessions_in_reserved:
            raise ValueError(f"Session indexes {sessions_in_reserved} are in reserved set")
        if out_of_bounds:
            raise ValueError(f"Session indexes {out_of_bounds} do not exist")
        sessions = session_idxs
        
    elif n_sessions:
        sessions = set(range(71))
        sessions = list(sessions - not_labeled - reserved)
        random.seed(10)
        sessions = random.sample(sessions, n_sessions)
        print("Selected sessions:", sessions)
    else:
        sessions = set(range(71))
        sessions = list(sessions - not_labeled - reserved)
        print("Using all available sessions:", sessions)

    np.random.seed(10)
    train_sessions, test_sessions = train_test_split(sessions, test_size=test_size)

    Xtr = torch.Tensor()
    Xte = torch.Tensor()
    ytr = torch.Tensor()
    yte = torch.Tensor()

    for session_idx in sessions:
        df = read_nursing_session(session_idx, raw_dir)
        labels = read_nursing_labels(session_idx, label_dir)
        X = torch.Tensor(df.values)
        y = torch.zeros(len(X), 1)
        y[labels['start']:labels['end']] = 1

        if session_idx in train_sessions:
            Xtr = torch.cat((Xtr, X))
            ytr = torch.cat((ytr, y))
        else:
            Xte = torch.cat((Xte, X))
            yte = torch.cat((yte, y))

    tr = AccAndLabelsDataset(pad_for_windowing(Xtr, winsize), ytr, winsize)
    te = AccAndLabelsDataset(pad_for_windowing(Xte, winsize), yte, winsize)

    print(f"Train sessions: {train_sessions} - total length: {len(tr)}")
    print(f"Test sessions: {test_sessions} - total length: {len(te)}")

    trainloader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(te, batch_size=batch_size, shuffle=shuffle_test, num_workers=2)

    return trainloader, testloader

