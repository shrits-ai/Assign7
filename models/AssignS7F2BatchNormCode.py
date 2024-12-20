# -*- coding: utf-8 -*-
"""# 
File:  AssignS7F2BatchNormCode.py
Added batch normalization to the model to stabilize training process
Reduced number of parameters by reducing number of channel 
Added gap layer to remove 7*7 kernels but receptive field is only 16.
Receptive field will not reach overall image size. 
Results:
Parameters: 5708
Best Train Accuracy: 98.86
Best Test Accuracy: 98.91
Analysis:
The model fits in the parameter requirement i.e. < 8k and epoch<15
Convergence was faster.
Slight overfitting is still seen.
Good part is even after reducing the parameter not much loss in test accuracy is seen.
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
            nn.BatchNorm2d(8),  # Batch Normalization
            nn.ReLU()
        ) # output_size = 26 | RF = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(12),  # Batch Normalization
            nn.ReLU()
        ) # output_size = 24 | RF = 5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),  # Batch Normalization
            nn.ReLU()
        ) # output_size = 22 | RF = 7

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11 | RF = 8
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),  # Batch Normalization
            nn.ReLU()
        ) # output_size = 11 | RF = 8

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(12),  # Batch Normalization
            nn.ReLU()
        ) # output_size = 9 | RF = 12
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),  # Batch Normalization
            nn.ReLU()
        ) # output_size = 7 | RF = 16

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),  # Batch Normalization
            nn.ReLU()
        ) # output_size = 7 | RF = 16
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
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
