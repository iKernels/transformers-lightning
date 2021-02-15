import torch


def hit_rate(preds: torch.Tensor, target: torch.Tensor, k: int = 1):
    r"""
    Computes hit rate (for information retrieval).
    Hir Rate is 1.0 iff there is at least one relevant documents among the top `k`.

    `preds` and `target` should be of the same shape and live on the same device. If no `target` is ``True``,
    0 is returned. Target must be of type `bool` or `int`, otherwise an error is raised.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant or not. Requires `bool` or `int` tensor.
        k: consider only the top k elements.

    Return:
        a single-value tensor with the hit rate (HR) of the predictions `preds` wrt the labels `target`.

    Example:
        >>> preds = torch.tensor([0.2, 0.3, 0.5])
        >>> target = torch.tensor([True, False, True])
        >>> hit_rate(preds, target)
        tensor(1.0)
    """

    if k <= 0:
        raise ValueError("`k` must be an integer greater or equal to `0`")

    if preds.shape != target.shape or preds.device != target.device:
        raise ValueError("`preds` and `target` must have the same shape and live on the same device")

    if target.dtype not in (torch.bool, torch.int16, torch.int32, torch.int64):
        raise ValueError("`target` must be a tensor of booleans or integers")

    if target.dtype is not torch.bool:
        target = target.bool()

    relevant = target[torch.argsort(preds, dim=-1, descending=True)][:k].sum()
    return (relevant > 0).float()
