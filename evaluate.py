import torch
import numpy as np
from matplotlib import pyplot as plt

# Splits a dataset into t=0 test/train portions containing half the data
# Requires: dataset is temporally sorted, split is a decimal in [0, 1]
def get_t0_split(dataset, split=0.75):
    t0 = dataset[:dataset.size/2]
    t0_train = t0[:split * t0.size]
    t0_test = t0[split * t0.size + 1:]
    return [t0_train, t0_test]
    
# Splits a dataset into test/train portions then t=1,...n,
#    each contain the data from one month
# Requires: dataset is temporally sorted, split is a decimal in [0, 1]
def get_monthly_split(dataset, split=0.75):
    pass
    

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


# Plots a bar graph showing accuracy, precision, recall, and f1
def plot_results(accuracy, precision, recall, f1):
    _, ax = plt.subplots()

    metrics = ['accuracy', 'precision', 'recall', 'f1 score']
    values = [accuracy, precision, recall, f1]
    bar_colors = ['tab:pink', 'tab:orange', 'tab:red', 'tab:purple']

    ax.bar(metrics, values, color=bar_colors)

    ax.set_title('Model Performance')
    ax.legend(title='Metric')

    plt.show()
    return


def main():
    # initialize models
    # set up continual learning setting
    # evaluate
    # plot
    pass
    
if __name__ == "__main__":
    main()
    