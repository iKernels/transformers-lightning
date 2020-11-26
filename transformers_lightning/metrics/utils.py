import torch
from typing import List

from transformers_lightning.utils import IGNORE_IDX


def get_mini_groups(idx: torch.Tensor) -> List[torch.Tensor]:
    """
    Return a list of lists where each sub-list contains the indexes of each value of `idx`
    :params idx: a tensor of integer indexes

    Example:
    >>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    >>> groups = get_mini_groups(indexes)
    >>> groups
    ... [torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5, 6])]
    """

    indexes = dict()
    for i, _id in enumerate(idx):
        _id = _id.item()
        if _id in indexes:
            indexes[_id] += [i]
        else:
            indexes[_id] = [i]
    res = [torch.tensor(x, dtype=torch.int64) for x in indexes.values()]
    return res

def masked_metric(predictions=None, logits=None, labels=None, exclude=IGNORE_IDX, metric=None, args=[], kwargs={}):
    """
    Compute a metric by considering only entries where the labels are different from `exclude` parameter.
    """
    assert metric is not None, "A non-None metric to compute should be provided"
    assert labels is not None, "labels should be provided to compute accuracy"
    assert (predictions is None) != (logits is None), \
        "only one between `predictions` and `logits` should be provided"
    
    if logits is not None:
        predictions = torch.argmax(logits, dim=-1)

    # do not compute performance when label is equal to exclue
    valid_indexes = (labels != exclude) 
    predictions = predictions[valid_indexes]
    labels = labels[valid_indexes]

    return metric(predictions.view(-1), labels.view(-1), *args, **kwargs)
