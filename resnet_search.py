from lib.data.dataloading import load_nursing_5_class
import torch
from torch import nn
from lib.config import *
import matplotlib.pyplot as plt
from lib.models import ResNetClassifierFiveClass,ResNetClassifierFiveClassXBlock
from lib.modules import optimization_loop_multi_class

DEVICE = 'cuda:0'

def train(model,w,g=None,b=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
    
    optimization_loop_multi_class(
        model,
        nursing_trainloader,
        nursing_testloader,
        criterion,
        optimizer,
        epochs=250,
        device=DEVICE,
        patience=50,
        writer=f'runs/test-xblock/{str(type(model)).split(".")[-1][:-2]}_w{w}_{model.dims_str}_b{b}_g{g}',
        label=f'w{w}_d{model.dims_str}'
)

w = 3001
nursing_trainloader, nursing_testloader = load_nursing_5_class(range(11,71), w, test_size=0.2, batch_size=128, window=False)    

# model = ResNetClassifierFiveClass(w, 3, (64,1028)).to(DEVICE)
# train(model,w)

for b in [2,4,8,16]:
    for g in [2,4,8,16]:
        try:
            model = ResNetClassifierFiveClassXBlock(w, 3, (64,1028), b, g).to(DEVICE)
            train(model,w,g,b)

        except Exception as e:
            print(e)
            for b in [128,64,32,16]:
                nursing_trainloader, nursing_testloader = load_nursing_5_class(range(11,71), w, test_size=0.2, batch_size=b, window=False)
                try:
                    model = ResNetClassifierFiveClassXBlock(w, 3, (64,1028), b, g).to(DEVICE)
                    train(model,w,g,b)
                    break
                except Exception as e:
                    print(e)
