from lib.data.dataloading import load_nursing_5_class
import torch
from torch import nn
from lib.config import *
import matplotlib.pyplot as plt
from lib.models import ResNetClassifierFiveClass
from lib.modules import optimization_loop_multi_class

for w in [101,501,1001,2001,3001]:
    nursing_trainloader, nursing_testloader = load_nursing_5_class(range(11,71), w, test_size=0.2, batch_size=256, window=False)

    # search for optimal resnet architecture
    for i in [2,4,8,16,32,64]:
        for j in [2,4,8,16,32,64,128,256]:
            for k in [2,4,8,16,32,64,128,256,512,1028]:
                if j < i or k < j or k < i:
                    continue
                model = ResNetClassifierFiveClass(w, 3, (i,j,k)).to(DEVICE)
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
                    # outdir='dev/test',
                    writer=f'runs/resnet-search/_w{w}_d{model.dims_str}'
                )