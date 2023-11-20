import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from lib.modules import optimization_loop
from lib.models import  ConvEncoderClassifierImproved

WINSIZE = 101
DEVICE = 'cuda:0'
autoencoder_dir = Path('dev/autoencoder4_conv-impr')
encoderclass_dir = Path('dev/encoderclass4_conv-nofreeze')
epochs = 100

model = ConvEncoderClassifierImproved(WINSIZE, weights_file=None, freeze=False).to(DEVICE)
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