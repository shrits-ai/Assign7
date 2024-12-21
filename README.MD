Assignment 7 
```
Objective: With 8k parameters and 15 epoch , try to reach accuracy of 99.4

This is done step by step. We have started from base model and then added batch normalization, pooling , regularization and normalization techniques and play with lr at the end to achieve the model.
```

```
File:  AssignS7F1SkeletonCode.py
Copied the basic skeleton code from the assignment. 
Tried reducing parameter count by reducing the number of channels. 
Reduced to required number of epochs (15).
Results:
Parameters: 17,068
Best Train Accuracy: 98.83
Best Test Accuracy: 98.68
Analysis:
The model is large as compared to expected in the assignment. 
Accuracy 99.4 is not achieved.
Overfitting is an issue. 
```

```
File:  AssignS7F1SkeletonCode.py
Copied the basic skeleton code as discussed in the class. 
Tried reducing parameter count by reducing the number of channels. (As it is required for assignment)
Reduced to required number of epochs (15).
Results:
Parameters: 17,068
Best Train Accuracy: 98.83
Best Test Accuracy: 98.68
Analysis:
The model is large as compared to expected in the assignment. 
Accuracy 99.4 is not achieved.
Overfitting is an issue. 
```

```
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
```

```
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
```


```
File:  AssignS7F4RemPoolAddConvLayer.py
Added another convolution layer to increase parameters and instead of second transition block (max pooling)
Results:
Parameters: 7628
Best Train Accuracy: 99.37
Best Test Accuracy: 99.25 ( epoch 12)
Analysis:
The model also fits in the parameter requirement i.e. < 8k and epoch<15
Overfitting is still an issue. 
Slight increase in accuracy as compared to additional transition block.
Accuracy 99.4 is not achieved.
```

```
File:  AssignS7F5RegularizationCode.py
Added dropout in middle layer to reduce overfitting. 
Added image augmentation as well 
Results:
Parameters: 7628
Best Train Accuracy: 98.81
Best Test Accuracy: 99.25 ( epoch 14 )
Analysis:
The model also fits in the parameter requirement i.e. < 8k and epoch <= 15
Model is underfit now. Training if for more epoch might reach to expected accuracy.
Accuracy 99.4 is not achieved.
```

```
File:  AssignS7F5RegularizationCode.py
Model was highly underfit. Removed dropout layer. Added extra layer and max pool to reduce underfitting and keep parameters under 8k.
changed optimizer to adam for faster convergence and 
scheduler to ReduceLROnPlateau to reduce learning rate when test accuracy plateaus.
tune image augmentation parameter to increase accuracy.
reduced batch size to 64 to increase accuracy.
Results:
Parameters: 7710
Best Train Accuracy: 98.59
Best Test Accuracy: 99.47 ( epoch 13 )
Analysis:
The model also fits in the parameter requirement i.e. < 8k and epoch <= 15
training and test accuracy was converging faster and underfit issue is not seen now.
test accuracy was around 99.4 from epoch 12 and was consistent.
```

Final training logs : 
```
(venv) shriti@Shritis-MacBook-Pro Assgn7_ERAV3 % python3 main.py
cpu
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 28, 28]              36
              ReLU-2            [-1, 4, 28, 28]               0
       BatchNorm2d-3            [-1, 4, 28, 28]               8
            Conv2d-4            [-1, 8, 28, 28]             288
              ReLU-5            [-1, 8, 28, 28]               0
       BatchNorm2d-6            [-1, 8, 28, 28]              16
            Conv2d-7           [-1, 16, 28, 28]           1,152
              ReLU-8           [-1, 16, 28, 28]               0
       BatchNorm2d-9           [-1, 16, 28, 28]              32
        MaxPool2d-10           [-1, 16, 14, 14]               0
           Conv2d-11            [-1, 8, 14, 14]             128
             ReLU-12            [-1, 8, 14, 14]               0
      BatchNorm2d-13            [-1, 8, 14, 14]              16
           Conv2d-14           [-1, 16, 14, 14]           1,152
             ReLU-15           [-1, 16, 14, 14]               0
      BatchNorm2d-16           [-1, 16, 14, 14]              32
          Dropout-17           [-1, 16, 14, 14]               0
           Conv2d-18           [-1, 20, 14, 14]           2,880
             ReLU-19           [-1, 20, 14, 14]               0
      BatchNorm2d-20           [-1, 20, 14, 14]              40
        MaxPool2d-21             [-1, 20, 7, 7]               0
           Conv2d-22             [-1, 10, 5, 5]           1,800
             ReLU-23             [-1, 10, 5, 5]               0
      BatchNorm2d-24             [-1, 10, 5, 5]              20
AdaptiveAvgPool2d-25             [-1, 10, 1, 1]               0
           Linear-26                   [-1, 10]             110
================================================================
Total params: 7,710
Trainable params: 7,710
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.76
Params size (MB): 0.03
Estimated Total Size (MB): 0.79
----------------------------------------------------------------
EPOCH 1
Loss=0.2854807674884796 Accuracy=84.75: 100%|███████████████████████████████████████████████| 938/938 [01:01<00:00, 15.26it/s]

Test set: Average loss: 0.0014, Accuracy: 98.01%

EPOCH 2
Loss=0.07878708094358444 Accuracy=95.92: 100%|██████████████████████████████████████████████| 938/938 [01:02<00:00, 15.09it/s]

Test set: Average loss: 0.0008, Accuracy: 98.57%

EPOCH 3
Loss=0.04576168581843376 Accuracy=96.71: 100%|██████████████████████████████████████████████| 938/938 [01:02<00:00, 15.10it/s]

Test set: Average loss: 0.0007, Accuracy: 98.53%

EPOCH 4
Loss=0.08128917962312698 Accuracy=97.11: 100%|██████████████████████████████████████████████| 938/938 [01:01<00:00, 15.16it/s]

Test set: Average loss: 0.0006, Accuracy: 98.99%

EPOCH 5
Loss=0.049138978123664856 Accuracy=97.57: 100%|█████████████████████████████████████████████| 938/938 [01:02<00:00, 15.05it/s]

Test set: Average loss: 0.0005, Accuracy: 99.02%

EPOCH 6
Loss=0.06235263869166374 Accuracy=97.66: 100%|██████████████████████████████████████████████| 938/938 [01:02<00:00, 15.07it/s]

Test set: Average loss: 0.0004, Accuracy: 99.06%

EPOCH 7
Loss=0.03590230643749237 Accuracy=97.75: 100%|██████████████████████████████████████████████| 938/938 [01:02<00:00, 15.03it/s]

Test set: Average loss: 0.0004, Accuracy: 99.11%

EPOCH 8
Loss=0.2680085599422455 Accuracy=97.88: 100%|███████████████████████████████████████████████| 938/938 [01:02<00:00, 15.11it/s]

Test set: Average loss: 0.0005, Accuracy: 99.04%

EPOCH 9
Loss=0.01134796254336834 Accuracy=97.90: 100%|██████████████████████████████████████████████| 938/938 [01:02<00:00, 15.06it/s]

Test set: Average loss: 0.0004, Accuracy: 99.12%

EPOCH 10
Loss=0.006671663373708725 Accuracy=98.06: 100%|█████████████████████████████████████████████| 938/938 [01:02<00:00, 15.02it/s]

Test set: Average loss: 0.0004, Accuracy: 99.12%

EPOCH 11
Loss=0.013377211056649685 Accuracy=98.07: 100%|█████████████████████████████████████████████| 938/938 [01:02<00:00, 14.91it/s]

Test set: Average loss: 0.0004, Accuracy: 99.13%

Epoch 00011: reducing learning rate of group 0 to 2.2000e-04.
EPOCH 12
Loss=0.0052712890319526196 Accuracy=98.41: 100%|████████████████████████████████████████████| 938/938 [01:02<00:00, 15.07it/s]

Test set: Average loss: 0.0003, Accuracy: 99.41%

EPOCH 13
Loss=0.010838045738637447 Accuracy=98.59: 100%|█████████████████████████████████████████████| 938/938 [01:02<00:00, 15.06it/s]

Test set: Average loss: 0.0003, Accuracy: 99.47%

EPOCH 14
Loss=0.08827409893274307 Accuracy=98.54: 100%|██████████████████████████████████████████████| 938/938 [01:01<00:00, 15.13it/s]

Test set: Average loss: 0.0003, Accuracy: 99.40%

EPOCH 15
Loss=0.135578915476799 Accuracy=98.59: 100%|████████████████████████████████████████████████| 938/938 [01:02<00:00, 15.00it/s]

Test set: Average loss: 0.0003, Accuracy: 99.42%

2024-12-21 15:44:10.729 Python[10676:5572901] +[IMKClient subclass]: chose IMKClient_Modern
2024-12-21 15:44:10.729 Python[10676:5572901] +[IMKInputSession subclass]: chose IMKInputSession_Modern
```

