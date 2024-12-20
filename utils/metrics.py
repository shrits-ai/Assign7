import torch

def calculate_accuracy(predictions, targets):
    """
    Calculate the accuracy of predictions compared to targets.
    Args:
        predictions (torch.Tensor): Model predictions (logits or probabilities).
        targets (torch.Tensor): Ground truth labels.
    Returns:
        float: Accuracy as a percentage.
    """
    pred_labels = predictions.argmax(dim=1)
    correct = pred_labels.eq(targets).sum().item()
    accuracy = 100.0 * correct / len(targets)
    return accuracy

def calculate_precision_recall_f1(predictions, targets, num_classes):
    """
    Calculate precision, recall, and F1 score per class.
    Args:
        predictions (torch.Tensor): Model predictions (logits or probabilities).
        targets (torch.Tensor): Ground truth labels.
        num_classes (int): Number of classes in the dataset.
    Returns:
        dict: Precision, recall, and F1 score per class.
    """
    pred_labels = predictions.argmax(dim=1)
    precision = []
    recall = []
    f1_score = []
    
    for cls in range(num_classes):
        true_positive = ((pred_labels == cls) & (targets == cls)).sum().item()
        false_positive = ((pred_labels == cls) & (targets != cls)).sum().item()
        false_negative = ((pred_labels != cls) & (targets == cls)).sum().item()

        prec = true_positive / (true_positive + false_positive + 1e-8)
        rec = true_positive / (true_positive + false_negative + 1e-8)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-8)

        precision.append(prec)
        recall.append(rec)
        f1_score.append(f1)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

def confusion_matrix(predictions, targets, num_classes):
    """
    Generate a confusion matrix.
    Args:
        predictions (torch.Tensor): Model predictions (logits or probabilities).
        targets (torch.Tensor): Ground truth labels.
        num_classes (int): Number of classes in the dataset.
    Returns:
        torch.Tensor: Confusion matrix of size (num_classes, num_classes).
    """
    pred_labels = predictions.argmax(dim=1)
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    
    for t, p in zip(targets, pred_labels):
        conf_matrix[t.long(), p.long()] += 1

    return conf_matrix

