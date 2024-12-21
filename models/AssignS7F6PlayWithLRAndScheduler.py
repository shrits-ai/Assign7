# -*- coding: utf-8 -*-
"""# 
File:  AssignS7F5RegularizationCode.py
Model was highly underfit. Removed dropout layer. Added extra layer and max pool to reduce underfitting and keep parameters under 8k.
changed optimizer to adam for faster convergence and 
scheduler to ReduceLROnPlateau to reduce learning rate when test accuracy plateaus.
tune image augmentation parameter to increase accuracy.
reduced batch size to 64 to increase accuracy.
Results:
Parameters: 7710
Best Train Accuracy: 98.67
Best Test Accuracy: 99.41 ( epoch 14 )
Analysis:
The model also fits in the parameter requirement i.e. < 8k and epoch <= 15
training and test accuracy was converging faster and underfit issue is not seen now.
test accuracy was around 99.2 from epoch 10 and gradually increased to 99.41 in epoch 14.
"""
from models.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.optim as optim

class Net(BaseModel):
    def __init__(self):
        super(Net, self).__init__()
        dropout_val = 0.001

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), padding=1, bias=False),  # Reduce channels further
            nn.ReLU(),
            nn.BatchNorm2d(4),  # Batch Normalization
        )  # output_size = 28 | RF = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),  # Further reduced filters
            nn.ReLU(),
            nn.BatchNorm2d(8),  # Batch Normalization
        )  # output_size = 26 | RF = 5

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),  # Reduced filters further
            nn.ReLU(),
            nn.BatchNorm2d(16),  # Batch Normalization
        )  # output_size = 24 | RF = 7

        # TRANSITION BLOCK1
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 12 | RF = 10
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),  # Reduced channels again
            nn.ReLU(),
            nn.BatchNorm2d(8),  # Batch Normalization
        )  # output_size = 12 | RF = 10

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),  # Increased filters a little
            nn.ReLU(),
            nn.BatchNorm2d(16),  # Batch Normalization
            nn.Dropout(dropout_val)
        )  # output_size = 10 | RF = 14

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=1, bias=False),  # Still reducing channels
            nn.ReLU(),
            nn.BatchNorm2d(20),  # Batch Normalization
        )  # output_size = 8 | RF = 18

        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2)  # output_size = 4 | RF = 26

        # OUTPUT BLOCK (final layer)
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),  # Reduced to 8 channels
            nn.ReLU(),
            nn.BatchNorm2d(10),  # Batch Normalization
        )  # output_size = 2 | RF = 30

        # Global Average Pooling to reduce spatial dimensions
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling (final spatial output = 1)

        # Classification Layer 
        self.fc = nn.Linear(10, 10)  # 10 output classes for MNIST

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = x.view(-1, 10)  # Flatten for classification
        x = self.fc(x)  # Fully connected layer for classification
        return F.log_softmax(x, dim=-1)

    def optimizerAndScheduler(self):
        # Define optimizer and loss
        #optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
            # Use Adam optimizer for better convergence
        optimizer = optim.Adam(self.parameters(), lr=0.0011, weight_decay=0.0001)
    
        # Reduce learning rate when validation accuracy plateaus
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.2, verbose=True)
    
        #scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        return optimizer, scheduler