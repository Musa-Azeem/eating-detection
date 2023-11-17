import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from lib.models import  ConvAutoencoderImproved, EncoderClassifierImproved
from lib.modules import optimization_loop_xonly, optimization_loop
from tqdm import tqdm
import matplotlib.pyplot as plt
from lib.utils import plot_and_save_losses
from pathlib import Path
from lib.datasets import AccRawDataset

WINSIZE = 101
DEVICE = 'cuda:0'
RAW_DIR = Path('/home/musa/datasets/eating_raw/')
nursing_raw_dir = Path("/home/musa/datasets/nursingv1")
nursing_label_dir = Path("/home/musa/datasets/eating_labels")
autoencoder_dir = Path('dev/autoencoder4')
encoderclass_dir = Path('dev/encoderclass4')
epochs = 100


model = ConvAutoencoderImproved(winsize=WINSIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.MSELoss()

trainloader = torch.load('pytorch_datasets/trainloader_11-16-23.pt')
testloader = torch.load('pytorch_datasets/testloader_11-16-23.pt')

optimization_loop_xonly(
    model, 
    trainloader, 
    testloader, 
    criterion, 
    optimizer, 
    epochs, 
    DEVICE, 
    autoencoder_dir
)

# Classifier

model = EncoderClassifierImproved(WINSIZE, weights_file=autoencoder_dir / 'best_model.pt', freeze=True).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.BCEWithLogitsLoss()

nursing_trainloader = torch.load('pytorch_datasets/nursing_trainloader_11-16-23.pt')
nursing_testloader = torch.load('pytorch_datasets/nursing_testloader_11-16-23.pt')

optimization_loop(
    model, 
    nursing_trainloader, 
    nursing_testloader, 
    criterion, optimizer, 
    epochs, 
    DEVICE, 
    encoderclass_dir
)