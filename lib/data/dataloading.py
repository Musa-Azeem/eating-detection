from lib.data.datasets import AccAndLabelsDataset, AccRawDataset
import numpy as np
from sklearn.model_selection import train_test_split
from lib.modules import pad_for_windowing, read_nursing_session, read_nursing_labels, read_delta_session
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
from pathlib import Path
import pandas as pd
from datetime import timedelta

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

def load_raw(raw_dir, winsize, n_hours=None, sessions=None, chunk_len_hrs=5, test_size=0.25, batch_size=64, shuffle_test=False):
    """
        1. Get list of N raw recording directories (from delta app, no labels)
        2. Read all raw data into N DataFrames of lengths l. Reset their timestamp to be seconds from the start. Print the length of each session
        3. Read and pad N recordings into N tensors of sizes (l+100 x 3). Then concatonate all recordings into a single tensor of size (L x 3)
        4. Cut off end of tensor so that C|L. Split the single tensor into chunks of, when chunk_len_hrs=5, C=1.8e6 samples (5 hours) to get a tensor of size (L/C x C x 3).
        5. Pad each chunk to get a tensor of size (L/C x C+100 x 3). This is so that no windows overlap two chunks
        6. If desired, select n_hours worth of chunks randomly
        7. Take 75% of chunks to be training chunks, and take the other 25% to be test chunks
        8. Seperately flatten the first two dims of train and test chunks to get tensors of size (L1 x 3) and (L2 x 3)
        9. Create an AccRawDataset for train and test chunks. This dataset will create windows dynamically to return tensors of size (303). It will have size L-101.
        10. Create a DataLoader for train and test datasets
    """
    
    raw_dir = Path(raw_dir)
    if n_hours and n_hours < chunk_len_hrs*2:
        raise ValueError(f"n_hours must be at least chunk_len_hrs*2 ({chunk_len_hrs*2})")

    # Use all available sessions if none provided
    if sessions:
        sessions = [raw_dir / Path(session) for session in sessions]
    else:
        print("Using all available sessions")
        sessions = list(raw_dir.iterdir())
    print("Using Directories: "+str([session.name for session in sessions]))

    # Read all sessions from CSVs
    accelerations = []
    for session_dir in sessions:
        accelerations.append(read_delta_session(raw_dir, session_dir.name))
        print(f'Index: {len(accelerations)-1}, Date: {session_dir.name}, nSamples: {len(accelerations[-1])}, Time Elapsed: {timedelta(seconds=accelerations[-1].timestamp.iloc[-1] - accelerations[-1].timestamp.iloc[0])}, Time Recorded: {timedelta(seconds=len(accelerations[-1]) / 100)}')

    # Convert to Tensors and pad all sessions for windowing
    accs = []
    for acc in accelerations:
        accs.append(pad_for_windowing(torch.Tensor(acc[['x_acc','y_acc','z_acc']].values), winsize))

    # Concatenate all sessions and split into chunks of length chunk_len_hrs
    chunk_len = chunk_len_hrs * 60 * 60 * 100 # number of samples at 100 Hz to get chunk_len_hrs hours
    all_acc = torch.cat(accs, axis=0)
    all_acc = all_acc[:len(all_acc) - len(all_acc) % chunk_len] # cut off very last part
    all_acc = all_acc.view(-1, chunk_len, 3)
    print(f"Created {len(all_acc)} chunks of length {chunk_len} samples each")

    if n_hours:
        # Randomly Select n_hours worth of chunks
        n_chunks = n_hours // chunk_len_hrs # if chunk_len_hrs is 5: 5 hours = 1 chunk, 10 hours = 2 chunks, etc.
        if n_chunks > len(all_acc):
            raise ValueError(f"n_hours ({n_hours}) is greater than the total number of hours ({len(all_acc)*chunk_len_hrs})")
        random.seed(10)
        idxs = random.sample(list(range(len(all_acc))), n_chunks)
        all_acc = all_acc[idxs]
        print(f"Randomly selected {n_chunks} chunks")

    # Split into train and test
    def proc(x):
        x = pad_for_windowing(x, winsize) # pad second dimension
        x = x.flatten(end_dim=1)
        return x
    np.random.seed(10)
    acctr, accte = map(proc, train_test_split(all_acc, test_size=test_size))

    print(f"Total train length: {timedelta(seconds=len(acctr) / 100)} ({len(acctr)} Samples)")
    print(f"Total test length: {timedelta(seconds=len(accte) / 100)} ({len(accte)} Samples)")

    Xtr = AccRawDataset(acctr, winsize)
    Xte = AccRawDataset(accte, winsize)

    trainloader = DataLoader(Xtr, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(Xte, batch_size=batch_size, shuffle=shuffle_test, num_workers=4)

    return trainloader, testloader