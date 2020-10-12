import torch
import numpy as np


def _get_mini_groups(idx: torch.Tensor) -> list:
    indexes = dict()
    for i, _id in enumerate(idx):
        _id = _id.item()
        if _id in indexes:
            indexes[_id] += [i]
        else:
            indexes[_id] = [i]
    return indexes.values()

def _prepare(x: torch.Tensor, do_softmax: bool = False, do_argmax: bool = False) -> torch.Tensor:
    # softmax and argmax to obtain discrete predictions probabilities
    if do_softmax:
        x = torch.nn.functional.softmax(x, dim=-1)
    if do_argmax:
        x = torch.argmax(x, dim=-1)
    return x

def _rr(preds: torch.Tensor, labels: torch.Tensor):
    # reciprocal rank over a single group
    labels = labels[torch.argsort(preds, dim=-1, descending=True)]
    position = torch.where(labels == 1)[0]
    return 1.0 / (position[0] + 1) if (len(position.shape) > 0) and (position.shape[0] > 0) else torch.tensor([0.0])

def _ap(preds: torch.Tensor, labels: torch.Tensor):
    # average precision over a single group
    labels = labels[torch.argsort(preds, dim=-1, descending=True)]
    positions = (torch.arange(len(labels), device=labels.device) + 1) * labels
    denominators = positions[torch.where(positions > 0)[0]]
    res = torch.true_divide((torch.arange(len(denominators), device=denominators.device) + 1), denominators).mean()
    return res

def _patk(preds: torch.Tensor, labels: torch.Tensor, precision_at: int):
    # prediction at k over a single group
    if labels.sum() == 0:
        return torch.Tensor(1).type_as(preds)
    return torch.true_divide(labels[torch.argsort(preds, descending=True)][:precision_at].sum(), labels.sum())

def _true_and_false_positives(preds: torch.Tensor, labels: torch.Tensor, threshold: float):
    # tp and fp
    indexes = np.argsort(preds)[::-1]
    labels = labels[indexes]
    preds = preds[indexes]
    return [
        int(labels[0] == 1 and preds[0] >= threshold), # TP
        int(labels[0] == 0 and preds[0] >= threshold) # FP
    ]

def _true_positive_and_false_negative(preds: torch.Tensor, labels: torch.Tensor, threshold: float):
    indexes = np.argsort(preds)[::-1]
    labels = labels[indexes]
    preds = preds[indexes]
    return [
        int(labels[0] == 1 and preds[0] >= threshold), # TP
        int(preds[0] < threshold) # FN
    ]


from pytorch_lightning.metrics.functional import accuracy
def masked_accuracy(labels=None, predictions=None, logits=None, exclude=-100):
    """
    Compute accuracy when there are labels that should not be considered.
    """
    assert labels is not None, "labels should be provided to compute accuracy"
    assert (predictions is None) != (logits is None), \
        "only one between `predictions` and `logits` should be provided"
    
    if logits is not None:
        predictions = torch.argmax(logits, dim=-1)

    # do not compute performance when label is equal to exclue
    valid_indexes = (labels != exclude) 
    predictions = predictions[valid_indexes]
    labels = labels[valid_indexes]

    return accuracy(predictions.view(-1), labels.view(-1))
