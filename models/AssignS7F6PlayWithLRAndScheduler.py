# -*- coding: utf-8 -*-
"""# 
File:  AssignS7F6PlayWithLRAndScheduler.py
Play with LR and schedulers.
Results:
Parameters: 7628
Best Train Accuracy: 98.81
Best Test Accuracy: 99.25 ( epoch 14 )
Analysis:
The model also fits in the parameter requirement i.e. < 8k and epoch <= 15
Model is underfit now. Training if for more epoch might reach to expected accuracy.
Accuracy 99.4 is not achieved.
"""
from models.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

class Net(BaseModel):
    def __init__(self):
        super(Net, self).__init__()
        dropout_val = 0.1
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),  # Batch Normalization
            nn.ReLU(),
            #nn.Dropout(dropout_val)
        ) # output_size = 26 | RF = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),  # Batch Normalization
            nn.ReLU(),
            #nn.Dropout(dropout_val)
        ) # output_size = 24 | RF = 5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(12),  # Batch Normalization
            nn.ReLU(),
            #nn.Dropout(dropout_val)
        ) # output_size = 22 | RF = 7

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11 | RF = 8
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),  # Batch Normalization
            nn.ReLU(),
        ) # output_size = 11 | RF = 8

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),  # Batch Normalization
            nn.ReLU(),
            nn.Dropout(dropout_val)
        ) # output_size = 9 | RF = 12
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(12),  # Batch Normalization
            nn.ReLU(),
            nn.Dropout(dropout_val)  # Dropout with 25% probability
        ) # output_size = 7 | RF = 16

        # ADDITIONAL LAYERS TO INCREASE RF
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(12),  # Batch Normalization
            nn.ReLU(),
            nn.Dropout(dropout_val)
        ) # output_size = 5 | RF = 20
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(12),  # Batch Normalization
            nn.ReLU(),
            nn.Dropout(dropout_val)
        ) # output_size = 3 | RF = 24

        # OUTPUT BLOCK
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),  # Batch Normalization
            nn.ReLU(),
        ) # output_size = 1 | RF = 30
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
    def optimizerAndScheduler( self ):
        # Define optimizer and loss
        optimizer = optim.SGD(self.parameters(), lr=0.014, momentum=0.9, weight_decay=0.005)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.2)
        return optimizer,scheduler