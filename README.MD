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
```

