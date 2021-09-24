import torch
from torch import Tensor, tensor
from torchmetrics.utilities.checks import _check_retrieval_functional_inputs


def retrieval_hit_rate(preds: Tensor, target: Tensor, k: int = None) -> Tensor:
    """
    Computes the hit rate (for information retrieval).
    The hit rate is 1.0 if there is at least one relevant document among all the top `k` retrieved documents.

    ``preds`` and ``target`` should be of the same shape and live on the same device. If no ``target`` is ``True``,
    ``0`` is returned. ``target`` must be either `bool` or `integers` and ``preds`` must be `float`,
    otherwise an error is raised. If you want to measure Precision@K, ``k`` must be a positive integer.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant or not.
        k: consider only the top k elements (default: None)

    Returns:
        a single-value tensor with the precision (at ``k``) of the predictions ``preds`` w.r.t. the labels ``target``.

    Example:
        >>> preds = tensor([0.2, 0.3, 0.5])
        >>> target = tensor([True, False, True])
        >>> retrieval_hit_rate(preds, target, k=2)
        tensor(0.5000)
    """
    preds, target = _check_retrieval_functional_inputs(preds, target)

    if not (isinstance(k, int) and k > 0):
        raise ValueError("`k` has to be a positive integer")

    if not target.sum():
        return tensor(0.0, device=preds.device)

    relevant = target[torch.argsort(preds, dim=-1, descending=True)][:k].sum()
    return (relevant > 0).float()
