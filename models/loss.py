import torch
import torch.nn as nn
import numpy as np


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.BCELoss = nn.BCELoss()
    # funcat cellcycle hierarchical shape [99,178,142,77,4]
    def forward(self,Yg,Yl,Yghat,Ylhat):
        global_losses = self.BCELoss(Yghat,Yg.float())
        local_losses = 0.
        for yl,ylhat in zip(Yl,Ylhat):
            local_losses += self.BCELoss(ylhat,yl.float())
        # Hierarchical violation Loss
        return local_losses + global_losses