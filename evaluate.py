import torch
import numpy as np

# Computes accuracy, false negatives, precision, recall, and F1 score
# Requires: label and predictions are equal length pytorch tensors
def evaluate(labels, predicts):
    accuracy = (predicts == labels).sum().item() / labels.size(0)

    # 1 = Spam, 0 = Ham?
    tp = torch.logical_and(predicts == 1, labels == 1).sum().item()
    fp = torch.logical_and(predicts == 1, labels != 1).sum().item()
    fn = torch.logical_and(predicts != 1, labels == 1).sum().item()

    precision = 0.0
    recall = 0.0
    f1 = 0.0
    if (tp + fp > 0.0): precision = tp / (tp + fp)
    if (tp + fn > 0.0): recall = tp / (tp + fn)
    if (precision + recall > 0.0): f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1, fn
    
