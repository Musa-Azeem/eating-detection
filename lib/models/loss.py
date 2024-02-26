from torch import nn
import torch

class CosineEmbeddingLossPositive(nn.Module):
    def __init__(self, margin=0.0, reduction='mean', batched=True):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

        self.pre_sim = nn.Flatten(start_dim=1) if batched else nn.Identity() 
        self.sim = nn.CosineEmbeddingLoss(margin=self.margin, reduction=self.reduction)

    def forward(self, input1, input2):
        return self.sim(
            self.pre_sim(input1),
            self.pre_sim(input2),
            torch.ones(input1.shape[0]).to(input1.device),
        )
    
class CosineMSELoss(nn.Module):
    def __init__(self, margin=0.0, reduction='mean', batched=True):
        super().__init__()
        self.sim = CosineEmbeddingLossPositive(margin, reduction, batched)
        self.mse = nn.MSELoss(reduction=reduction)
    def forward(self, x1, x2):
        return self.sim(x1, x2) + self.mse(x1, x2)
