import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from lib.models import  ConvAutoencoderImproved
from tqdm import tqdm
import matplotlib.pyplot as plt
from lib.utils import plot_and_save_losses
from pathlib import Path
import os

WINSIZE = 101
DEVICE = 'cuda:0'

model = ConvAutoencoderImproved(winsize=WINSIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.MSELoss()

trainloader = torch.load('pytorch_datasets/trainloader_11-15-23.pt')
testloader = torch.load('pytorch_datasets/testloader_11-15-23.pt')

epochs = 100

losses = []
test_losses = []
lowest_loss = -1

pbar = tqdm(range(epochs))
for epoch in pbar:
    lossi = 0
    test_lossi = 0

    for X in trainloader:
        X = X[0].to(DEVICE)

        # Foward Pass
        logits = model(X)
        loss = criterion(logits, X)

        # Backwards Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Sum Loss
        lossi += loss.item()
    
    # Test model
    for X in testloader:
        X = X[0].to(DEVICE)
        logits = model(X)
        test_lossi += criterion(logits, X).item()

    losses.append(lossi/len(trainloader))
    test_losses.append(test_lossi/len(testloader))
    pbar.set_description(f'Epoch {epoch}: Train Loss: {losses[-1]:.5}, Test Loss: {test_losses[-1]:.5}')

    plt.plot(losses)
    plt.plot(test_losses)
    plt.savefig('running_loss.png')
    torch.save(model.state_dict(), f'dev/autoencoder3/model/{epoch}.pt')
    plot_and_save_losses(losses, test_losses, epoch, Path('dev/autoencoder3/loss.jpg'))

    # Save model with lowest loss
    if lowest_loss < 0 or test_losses[-1] < lowest_loss:
        lowest_loss = test_losses[-1]
        torch.save(model.state_dict(), f'dev/autoencoder3/best_model.pt')
        os.system(f'touch dev/autoencoder3/{str(epoch)}.txt')
