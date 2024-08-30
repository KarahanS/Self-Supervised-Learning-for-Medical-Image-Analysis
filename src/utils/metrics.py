# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Dict, List, Sequence

import torch
from torchmetrics import AUROC
from torchmetrics.classification import MultilabelAUROC, MultilabelRecall
from torch.utils import data
from src.utils.enums import MedMNISTCategory
import torch


def accuracy_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: Sequence[int] = (1, 5)
) -> Sequence[int]:
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        outputs (torch.Tensor): output of a classifier (logits or probabilities).
        targets (torch.Tensor): ground truth labels.
        top_k (Sequence[int], optional): sequence of top k values to compute the accuracy over.
            Defaults to (1, 5).

    Returns:
        Sequence[int]:  accuracies at the desired k.
    """

    with torch.no_grad():
        maxk = max(top_k)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def weighted_mean(outputs: List[Dict], key: str, batch_size_key: str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.

    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the metric of interest.
        batch_size_key (str): key of batch size values.

    Returns:
        float: weighted mean of the values of a key
    """

    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value.squeeze(0)


def get_auroc_metric(model, test_loader, num_classes, task):
    """
    Compute the AUROC (Area Under the Receiver Operating Characteristic) metric
    for a multiclass classification task.

    Args:
        model (torch.nn.Module): -
        test_loader (torch.utils.data.DataLoader): The data loader for the test
            dataset used to compute the metric.
        num_classes (int): The number of classes in the classification task.

    Returns:
        float: The AUROC metric value.
    """
    y_true = []
    y_pred = []

    for batch in test_loader:
        x, y = batch
        y_true.extend(y)
        y_pred.extend(model(x)["logits"])  # forward is called - auroc works with logits

    y_true = torch.stack(y_true).squeeze()
    y_pred = torch.stack(y_pred)

    if task == "multilabel":
        # macro: Calculate score for each label and average them
        auroc_metric = MultilabelAUROC(num_labels=num_classes, average="macro")
        return auroc_metric(y_pred, y_true).item()
    else:
        auroc_metric = AUROC(task="multiclass", num_classes=num_classes)
        return auroc_metric(y_pred, y_true).item()

def get_balanced_accuracy_metric(model, test_loader, num_classes, task='multiclass'):
    """
    Compute the Balanced Accuracy metric for a multiclass or multilabel classification task.

    Balanced Accuracy is calculated as the average of recall obtained on each class or label.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        num_classes (int): The number of classes or labels in the classification task.
        task (str): The type of classification task ('multiclass' or 'multilabel').

    Returns:
        float: The balanced accuracy as a percentage.
    """
    y_true = []
    y_pred = []

    # making sure that model is in eval mode and on gpu
    model.eval()
    model = model.cuda()

    # Collect predictions and true labels
    for batch in test_loader:
        batch = [b.cuda() for b in batch]
        inputs, labels = batch
        outputs = model(inputs)

        if task == 'multiclass':
            # Multiclass case: get the predicted class index
            _, predicted_labels = torch.max(outputs["logits"], 1)
            y_true.extend(labels)
            y_pred.extend(predicted_labels)
        elif task == 'multilabel':
            # Multilabel case: apply sigmoid and threshold
            y_true.append(labels)
            y_pred.append(outputs["logits"].sigmoid() > 0.5)  # threshold at 0.5 for multilabel

    # Convert lists to tensors
    y_true = torch.cat(y_true) if task == 'multilabel' else torch.stack(y_true).squeeze()
    y_pred = torch.cat(y_pred) if task == 'multilabel' else torch.stack(y_pred)

    if task == 'multiclass':
        # Multiclass: Calculate TP and Nc for each class
        TP_c = torch.zeros(num_classes)
        Nc = torch.zeros(num_classes)

        for c in range(num_classes):
            TP_c[c] = ((y_pred == c) & (y_true == c)).sum().float()
            Nc[c] = (y_true == c).sum().float()

        # Calculate balanced accuracy for multiclass
        class_recalls = TP_c / Nc
        balanced_acc = class_recalls.mean().item() * 100

    elif task == 'multilabel':
        # Multilabel: Calculate balanced accuracy using multilabel recall
        recall_metric = MultilabelRecall(num_labels=num_classes, average='macro')
        balanced_acc = recall_metric(y_pred, y_true) * 100  # convert to percentage

    return balanced_acc