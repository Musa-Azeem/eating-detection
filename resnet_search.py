from lib.data.dataloading import load_nursing_5_class
import torch
from torch import nn
from lib.config import *
import matplotlib.pyplot as plt
from lib.models import ResNetClassifierFiveClass
from lib.modules import optimization_loop_multi_class

DEVICE = 'cuda:0'

def train(model,w):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
    
    optimization_loop_multi_class(
        model,
        nursing_trainloader,
        nursing_testloader,
        criterion,
        optimizer,
        epochs=100,
        device=DEVICE,
        patience=30,
        writer=f'runs/resnet-search/_w{w}_d{model.dims_str}',
        label=f'w{w}_d{model.dims_str}'
)

for w in [1001,101,501,2001,3001]:
    nursing_trainloader, nursing_testloader = load_nursing_5_class(range(11,71), w, test_size=0.2, batch_size=256, window=False)
    
    # search for optimal resnet architecture
    for i in [2,4,8,16,32,64]:
        # bc it stopped
        if w == 1001 and i < 8:
            continue
        for j in [2,4,8,16,32,64,128,256]:
            for k in [2,4,8,16,32,64,128,256,512,1028]:
                if j < i or k < j or k < i:
                    continue
                
                try:
                    model = ResNetClassifierFiveClass(w, 3, (i,j,k)).to(DEVICE)
                    train(model,w)

                except Exception as e:
                    print(e)
                    for b in [128,64,32,16]:
                        nursing_trainloader, nursing_testloader = load_nursing_5_class(range(11,71), w, test_size=0.2, batch_size=b, window=False)
                        try:
                            model = ResNetClassifierFiveClass(w, 3, (i,j,k)).to(DEVICE)
                            train(model,w)
                            break
                        except Exception as e:
                            print(e)
