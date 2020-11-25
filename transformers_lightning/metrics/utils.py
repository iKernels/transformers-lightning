import torch

from transformers_lightning.utils import IGNORE_IDX


def masked_metric(predictions=None, logits=None, labels=None, exclude=IGNORE_IDX, metric=None, args=[], kwargs={}):
    """
    Compute a metric only not taking into account labels that should not be considered.
    """
    assert metric is not None, "A non-None metric to compute on predictions and labels should be provided"
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
