import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented!")

    def optimizerAndScheduler( self ):
        # Define optimizer and loss
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
        return optimizer,scheduler
