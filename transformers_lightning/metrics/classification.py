import torch
import numpy as np


# TODO: switch to pytorch and remove numpy
def get_tp_and_fp(preds: torch.Tensor, labels: torch.Tensor, threshold: float):
    """
    Compute number of true and false positives
    """
    indexes = np.argsort(preds, dim=-1, descending=True)
    labels = labels[indexes]
    preds = preds[indexes]
    return (
        int(labels[0] == 1 and preds[0] >= threshold), # TP
        int(labels[0] == 0 and preds[0] >= threshold) # FP
    )

def get_tp_and_fn(preds: torch.Tensor, labels: torch.Tensor, threshold: float):
    """
    Compute number of true positives and false negatives
    """
    indexes = np.argsort(preds, dim=-1, descending=True)
    labels = labels[indexes]
    preds = preds[indexes]
    return (
        int(labels[0] == 1 and preds[0] >= threshold), # TP
        int(preds[0] < threshold) # FN
    )