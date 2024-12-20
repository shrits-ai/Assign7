from utils.metrics import calculate_accuracy, confusion_matrix
import torch

def test(model, device, test_loader, criterion, num_classes=10):
    model.eval()
    test_loss = 0
    correct = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            all_predictions.append(output)
            all_targets.append(target)

    # Combine all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate accuracy
    accuracy = calculate_accuracy(all_predictions, all_targets)
    conf_matrix = confusion_matrix(all_predictions, all_targets, num_classes)

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n')
    #print(f'Confusion Matrix:\n{conf_matrix}')
    
    return test_loss, accuracy

