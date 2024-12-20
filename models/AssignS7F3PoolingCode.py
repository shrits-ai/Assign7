# -*- coding: utf-8 -*-
"""# 
File:  AssignS7F3PoolingCode.py
Added 2nd transition block maxpool so as to increase receptive field to atleast input size.
Results:
Parameters: 5116
Best Train Accuracy: 99.42
Best Test Accuracy: 99.13
Analysis:
The model also fits in the parameter requirement i.e. < 8k and epoch<15
Overfitting is still an issue. 
Accuracy 99.4 is not achieved.
"""
from models.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F

class Net( BaseModel ):

    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )  # output_size = 26 | RF = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )  # output_size = 24 | RF = 5

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 12 | RF = 6

        # CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )  # output_size = 10 | RF = 10

        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2)  # output_size = 5 | RF = 12

        # OUTPUT BLOCK
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )  # output_size = 3 | RF = 20
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )  # output_size = 1 | RF = 30

        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.pool2(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)