import torch
from models.AssignS7F6PlayWithLRAndScheduler import Net  # Import your models here
from data.dataloader import get_dataloaders
from train import train
from test import test
from utils.analysis import plot_metrics
from torchsummary import summary
import numpy as np
from torch.optim.lr_scheduler import StepLR, OneCycleLR

def main():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(device)
    
    # Load data
    batch_size = 64
    num_workers = 4
    train_loader, test_loader = get_dataloaders(batch_size, num_workers, cuda)

    # Initialize model
    model = Net().to(device)
    summary(model, input_size=(1, 28, 28))

    # Define optimizer and loss
    optimizer,scheduler = model.optimizerAndScheduler()
    criterion = torch.nn.CrossEntropyLoss()
    # Training and testing loop
    epochs = 15
    train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []
    for epoch in range(epochs):
        print(f"EPOCH {epoch + 1}")
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        # Step the learning rate scheduler
       
        test_loss, test_acc = test(model, device, test_loader, criterion)
        scheduler.step(test_loss)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

# Convert only PyTorch tensors to NumPy
    train_losses = np.array([loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in train_losses])
    train_acc = np.array(train_accuracies)  # No processing needed as they are already floats
    test_losses = np.array([loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in test_losses])
    test_acc = np.array(test_accuracies)  # No processing needed as they are already floats
    # Plot metrics
    plot_metrics(train_losses, train_acc, test_losses, test_acc)

if __name__ == "__main__":
    main()

