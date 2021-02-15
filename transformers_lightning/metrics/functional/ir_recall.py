import torch


def recall(preds: torch.Tensor, target: torch.Tensor, k: int = None):
    r"""
    Computes the recall metric (for information retrieval),
    as explained `here <https://en.wikipedia.org/wiki/Precision_and_recall#Recall>`_.
    Recall at K is the fraction of relevant documents in top K among all the relevant documents.

    `preds` and `target` should be of the same shape and live on the same device. If no `target` is ``True``,
    0 is returned. Target must be of type `bool` or `int`, otherwise an error is raised.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant or not. Requires `bool` or `int` tensor.
        k: consider only the top k elements.

    Returns:
        a single-value tensor with the recall at k (R@K) of the predictions `preds` wrt the labels `target`.

    Example:
        >>> preds = torch.tensor([0.2, 0.3, 0.5])
        >>> target = torch.tensor([True, False, True])
        >>> recall(preds, target, k=2)
        ... tensor(0.5)
    """

    if preds.shape != target.shape or preds.device != target.device:
        raise ValueError("`preds` and `target` must have the same shape and live on the same device")

    if target.dtype not in (torch.bool, torch.int16, torch.int32, torch.int64):
        raise ValueError("`target` must be a tensor of booleans or integers")

    if target.dtype is not torch.bool:
        target = target.bool()

    if target.sum() == 0:
        return torch.tensor(0, device=preds.device)

    relevant = target[torch.argsort(preds, dim=-1, descending=True)][:k].sum()
    return torch.true_divide(relevant, target.sum())
