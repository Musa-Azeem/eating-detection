from torch import nn
import torch

class CosineEmbeddingLossPositive(nn.Module):
    def __init__(self, margin=0.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, input1, input2):
        if len(input1.shape) > 2:
            return nn.CosineEmbeddingLoss(margin=self.margin, reduction=self.reduction)(
                input1.flatten(start_dim=1),
                input2.flatten(start_dim=1),
                torch.ones(input1.shape[0]).to(input1.device),
            )
        
        return nn.CosineEmbeddingLoss(margin=self.margin, reduction=self.reduction)(
            input1,
            input2,
            torch.ones(input1.shape[0]).to(input1.device),
        )